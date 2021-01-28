import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from args import read_args
import numpy as np
import string
import re
import math
import sys

args = read_args()


class HetAgg(nn.Module):
    def __init__(self, args, feature_list, neigh_list_train, train_id_list, dl, input_data, device):
        super(HetAgg, self).__init__()
        embed_d = args.embed_d
        in_f_d = args.in_f_d
        self.args = args
        self.node_n = dl.nodes['count']
        self.attr_n = dict()
        self.input_data = input_data
        self.node_types = list(dl.nodes['attr'].keys())
        for node_type in self.node_types:
            self.attr_n[node_type] = dl.nodes['count'][node_type]
        self.feature_list = feature_list
        self.neigh_list_train = neigh_list_train
        self.train_id_list = train_id_list
        self.node_min_size = input_data.top_k
        self.agg_fcs = nn.ModuleDict()
        layers = []
        for node_type in dl.nodes['attr'].keys():
            self.agg_fcs[f'agg_fc{node_type}'] = nn.Linear(self.feature_list[node_type].shape[1], embed_d).to(device)

        self.content_rnns = nn.ModuleDict()
        for node_type in dl.nodes['attr'].keys():
            self.content_rnns[f'content_rnn{node_type}'] = nn.LSTM(embed_d, int(embed_d / 2), 1, bidirectional=True)

        self.neigh_rnns = nn.ModuleDict()
        for node_type in dl.nodes['attr'].keys():
            self.neigh_rnns[f'neigh_rnn{node_type}'] = nn.LSTM(embed_d, int(embed_d / 2), 1, bidirectional=True)

        self.neigh_atts = nn.ModuleDict()
        for node_type in dl.nodes['attr'].keys():
            temp = nn.ParameterList()
            temp.append(nn.Parameter(torch.ones(embed_d * 2, 1), requires_grad=True))
            self.neigh_atts[f'neigh_att{node_type}'] = temp

        self.softmax = nn.Softmax(dim=1)
        self.act = nn.LeakyReLU()
        self.drop = nn.Dropout(p=0.5)
        self.bn = nn.BatchNorm1d(embed_d)
        self.embed_d = embed_d
        self.layers = layers

    def init_weights(self):
        ms = self.modules()
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Parameter):
                nn.init.xavier_normal_(m.weight.data)
                # nn.init.normal_(m.weight.data)
                m.bias.data.fill_(0.1)

    def conteng_agg(self, id_batch, node_type):
        if args.feat_type==2:
            embed_batch = self.feature_list[node_type][id_batch]
            embed_batch = self.agg_fcs[f'agg_fc{node_type}'](embed_batch)
        else:
            embed_batch = self.feature_list[node_type]
            embed_batch = self.agg_fcs[f'agg_fc{node_type}'](embed_batch)
            embed_batch = embed_batch[id_batch]
        concate_embed = embed_batch.view(len(id_batch[0]), 1, self.embed_d)
        concate_embed = torch.transpose(concate_embed, 0, 1)
        all_state, last_state = self.content_rnns[f'content_rnn{node_type}'](concate_embed)
        return torch.mean(all_state, 0)

    # 聚合id_batch的node_type类型的节点
    def node_neigh_agg(self, id_batch, node_type):
        batch_s = int(len(id_batch[0]) / self.node_min_size[node_type])
        agg = self.conteng_agg(id_batch=id_batch, node_type=node_type).view(batch_s, self.node_min_size[node_type],
                                                                            self.embed_d)
        agg = torch.transpose(agg, 0, 1)
        all_state, last_state = self.neigh_rnns[f'neigh_rnn{node_type}'](agg)
        agg = torch.mean(all_state, 0).view(batch_s, self.embed_d)
        return agg

    # heterogeneous neighbor aggregation
    def node_het_agg(self, id_batch, node_type):
        node_types = self.node_types
        neigh_batch = dict()
        for n_type in node_types:
            neigh_batch[n_type] = [[0] * self.node_min_size[n_type]] * len(id_batch)
        for i in range(len(id_batch)):
            for n_type in node_types:
                neigh_batch[n_type][i] = self.neigh_list_train[node_type][n_type][id_batch[i]]
        ''''neigh's embed of each type node'''
        agg_batch = dict()
        for n_type in node_types:
            neigh_batch[n_type] = np.reshape(neigh_batch[n_type], (1, -1))
            agg_batch[n_type] = self.node_neigh_agg(neigh_batch[n_type], n_type)
        # attention module
        id_batch = np.reshape(id_batch, (1, -1))
        c_agg_batch = self.conteng_agg(id_batch=id_batch, node_type=node_type)
        c_agg_batch_2 = dict()
        c_agg_batch_2[-1] = torch.cat((c_agg_batch, c_agg_batch), 1).view(len(c_agg_batch), self.embed_d * 2)
        for n_type in node_types:
            c_agg_batch_2[n_type] = torch.cat((c_agg_batch, agg_batch[n_type]), 1).view(len(c_agg_batch),
                                                                                        self.embed_d * 2)

        # compute weights
        node_type_n = len(node_types)
        concat_embed_tuple = list()
        for k in sorted(c_agg_batch_2.keys()):
            concat_embed_tuple.append(c_agg_batch_2[k])
        concat_embed_tuple = tuple(concat_embed_tuple)
        concate_embed = torch.cat(concat_embed_tuple, 1).view(len(c_agg_batch), node_type_n + 1, self.embed_d * 2)
        atten_w = self.act(torch.bmm(concate_embed,
                                     self.neigh_atts[f'neigh_att{node_type}'][0].unsqueeze(0).expand(len(c_agg_batch),
                                                                                                  *self.neigh_atts[
                                                                                                      f'neigh_att{node_type}'][0].size())))
        atten_w = self.softmax(atten_w).view(len(c_agg_batch), 1, node_type_n + 1)

        # weighted combination
        agg_batch_tuple = [c_agg_batch]
        for n_type in node_types:
            agg_batch_tuple.append(agg_batch[n_type])
        agg_batch_tuple = tuple(agg_batch_tuple)

        concate_embed = torch.cat(agg_batch_tuple, 1).view(len(c_agg_batch), node_type_n + 1, self.embed_d)
        weight_agg_batch = torch.bmm(atten_w, concate_embed).view(len(c_agg_batch), self.embed_d)
        return weight_agg_batch

    def het_agg(self, triple_pair, c_id_batch, pos_id_batch, neg_id_batch, ):
        # batch processing
        # nine cases for academic data (author, paper, venue)
        c_agg = self.node_het_agg(c_id_batch, triple_pair[0])
        p_agg = self.node_het_agg(pos_id_batch, triple_pair[1])
        n_agg = self.node_het_agg(neg_id_batch, triple_pair[1])
        return c_agg, p_agg, n_agg

    def save_embed(self, file_):
        embed_file = open(file_, "w")
        save_batch_s = self.args.mini_batch_s

        for node_type in self.node_types:
            batch_number = int(len(self.train_id_list[node_type]) / save_batch_s)
            for j in range(batch_number):
                id_batch = self.train_id_list[node_type][j * save_batch_s: (j + 1) * save_batch_s]
                out_temp = self.node_het_agg(id_batch=id_batch, node_type=node_type)
                out_temp = out_temp.data.cpu().numpy()
                for k in range(len(id_batch)):
                    index = id_batch[k]
                    embed_file.write(self.input_data.node_type2name[node_type] + str(index) + " ")
                    for l in range(self.args.embed_d - 1):
                        embed_file.write(str(out_temp[k][l]) + " ")
                    embed_file.write(str(out_temp[k][-1]) + "\n")
        embed_file.close()
        return [], [], []

    def aggregate_all(self, triple_list_batch, triple_pair):
        c_id_batch = [x[0] for x in triple_list_batch]
        pos_id_batch = [x[1] for x in triple_list_batch]
        neg_id_batch = [x[2] for x in triple_list_batch]

        c_agg, pos_agg, neg_agg = self.het_agg(triple_pair, c_id_batch, pos_id_batch, neg_id_batch, )

        return c_agg, pos_agg, neg_agg

    def forward(self, triple_list_batch, triple_pair, ):
        c_out, p_out, n_out = self.aggregate_all(triple_list_batch, triple_pair)
        return c_out, p_out, n_out


def cross_entropy_loss(c_embed_batch, pos_embed_batch, neg_embed_batch, embed_d):
    batch_size = c_embed_batch.shape[0] * c_embed_batch.shape[1]

    c_embed = c_embed_batch.view(batch_size, 1, embed_d)
    pos_embed = pos_embed_batch.view(batch_size, embed_d, 1)
    neg_embed = neg_embed_batch.view(batch_size, embed_d, 1)

    out_p = torch.bmm(c_embed, pos_embed)
    out_n = - torch.bmm(c_embed, neg_embed)

    sum_p = F.logsigmoid(out_p)
    sum_n = F.logsigmoid(out_n)
    loss_sum = - (sum_p + sum_n)

    # loss_sum = loss_sum.sum() / batch_size

    return loss_sum.mean()
