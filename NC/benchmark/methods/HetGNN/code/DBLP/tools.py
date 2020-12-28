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
    def __init__(self, args, feature_list, a_neigh_list_train, p_neigh_list_train, t_neigh_list_train,
                 v_neigh_list_train,
                 a_train_id_list, p_train_id_list, t_train_id_list, v_train_id_list, dl):
        super(HetAgg, self).__init__()
        embed_d = args.embed_d
        in_f_d = args.in_f_d
        self.args = args
        self.node_n = dl.nodes['count']
        self.attr_n = []
        for node_type in dl.nodes['attr'].keys():
            self.attr_n.append(dl.nodes['attr'][node_type].shape[1])
        self.feature_list = feature_list
        self.a_neigh_list_train = a_neigh_list_train
        self.p_neigh_list_train = p_neigh_list_train
        self.t_neigh_list_train = t_neigh_list_train
        self.v_neigh_list_train = v_neigh_list_train
        self.a_train_id_list = a_train_id_list
        self.p_train_id_list = p_train_id_list
        self.t_train_id_list = t_train_id_list
        self.v_train_id_list = v_train_id_list

        self.fc_a_agg = nn.Linear(self.attr_n[0], embed_d)  # 334->128
        self.fc_p_agg = nn.Linear(self.attr_n[1], embed_d)  # 4231->128
        self.fc_t_agg = nn.Linear(self.attr_n[2], embed_d)  # 50->128
        self.fc_v_agg = nn.Linear(self.attr_n[3], embed_d)  # 20->128

        self.a_content_rnn = nn.LSTM(embed_d, int(embed_d / 2), 1, bidirectional=True)
        self.p_content_rnn = nn.LSTM(embed_d, int(embed_d / 2), 1, bidirectional=True)
        self.t_content_rnn = nn.LSTM(embed_d, int(embed_d / 2), 1, bidirectional=True)
        self.v_content_rnn = nn.LSTM(embed_d, int(embed_d / 2), 1, bidirectional=True)

        self.a_neigh_rnn = nn.LSTM(embed_d, int(embed_d / 2), 1, bidirectional=True)
        self.p_neigh_rnn = nn.LSTM(embed_d, int(embed_d / 2), 1, bidirectional=True)
        self.t_neigh_rnn = nn.LSTM(embed_d, int(embed_d / 2), 1, bidirectional=True)
        self.v_neigh_rnn = nn.LSTM(embed_d, int(embed_d / 2), 1, bidirectional=True)

        self.a_neigh_att = nn.Parameter(torch.ones(embed_d * 2, 1), requires_grad=True)
        self.p_neigh_att = nn.Parameter(torch.ones(embed_d * 2, 1), requires_grad=True)
        self.t_neigh_att = nn.Parameter(torch.ones(embed_d * 2, 1), requires_grad=True)
        self.v_neigh_att = nn.Parameter(torch.ones(embed_d * 2, 1), requires_grad=True)

        self.softmax = nn.Softmax(dim=1)
        self.act = nn.LeakyReLU()
        self.drop = nn.Dropout(p=0.5)
        self.bn = nn.BatchNorm1d(embed_d)
        self.embed_d = embed_d

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Parameter):
                nn.init.xavier_normal_(m.weight.data)
                # nn.init.normal_(m.weight.data)
                m.bias.data.fill_(0.1)

    def a_content_agg(self, id_batch):  # heterogeneous content aggregation
        embed_d = self.embed_d
        # print len(id_batch)
        # embed_d = in_f_d, it is flexible to add feature transformer (e.g., FC) here
        # print (id_batch)
        a_embed_batch = self.feature_list[0][id_batch]
        a_embed_batch = self.fc_a_agg(a_embed_batch)
        # concate_embed = torch.cat((a_embed_batch), 1).view(len(id_batch[0]), 1, embed_d)
        concate_embed = a_embed_batch.view(len(id_batch[0]), 1, embed_d)
        concate_embed = torch.transpose(concate_embed, 0, 1)
        all_state, last_state = self.a_content_rnn(concate_embed)
        return torch.mean(all_state, 0)

    def p_content_agg(self, id_batch):
        embed_d = self.embed_d
        p_embed_batch = self.feature_list[1][id_batch]
        p_embed_batch = self.fc_p_agg(p_embed_batch)
        # concate_embed = torch.cat((p_embed_batch), 1).view(len(id_batch[0]), 1, embed_d)
        concate_embed = p_embed_batch.view(len(id_batch[0]), 1, embed_d)
        concate_embed = torch.transpose(concate_embed, 0, 1)
        all_state, last_state = self.p_content_rnn(concate_embed)
        return torch.mean(all_state, 0)

    def t_content_agg(self, id_batch):
        embed_d = self.embed_d
        if not isinstance(id_batch, np.ndarray):
            exit(1)
        try:
            t_embed_batch = self.feature_list[2][id_batch]
        except:
            exit(1)
        t_embed_batch = self.fc_t_agg(t_embed_batch)
        # concate_embed = torch.cat((t_embed_batch), 1).view(len(id_batch[0]), 1, embed_d)
        concate_embed = t_embed_batch.view(len(id_batch[0]), 1, embed_d)
        concate_embed = torch.transpose(concate_embed, 0, 1)
        all_state, last_state = self.t_content_rnn(concate_embed)
        return torch.mean(all_state, 0)

    def v_content_agg(self, id_batch):
        embed_d = self.embed_d
        v_embed_batch = self.feature_list[3][id_batch]
        v_embed_batch = self.fc_v_agg(v_embed_batch)
        # concate_embed = torch.cat((v_embed_batch), 1).view(len(id_batch[0]), 1, embed_d)
        concate_embed = v_embed_batch.view(len(id_batch[0]), 1, embed_d)
        concate_embed = torch.transpose(concate_embed, 0, 1)
        all_state, last_state = self.v_content_rnn(concate_embed)
        return torch.mean(all_state, 0)

    # 聚合id_batch的node_type类型的节点
    def node_neigh_agg(self, id_batch, node_type):  # type based neighbor aggregation with rnn
        embed_d = self.embed_d

        if node_type == 1 or node_type == 2 or node_type == 3:
            batch_s = int(len(id_batch[0]) / 10)
        else:
            # print (len(id_batch[0]))
            batch_s = int(len(id_batch[0]) / 3)

        if node_type == 1:
            neigh_agg = self.a_content_agg(id_batch).view(batch_s, 10, embed_d)
            neigh_agg = torch.transpose(neigh_agg, 0, 1)
            all_state, last_state = self.a_neigh_rnn(neigh_agg)
        elif node_type == 2:
            neigh_agg = self.p_content_agg(id_batch).view(batch_s, 10, embed_d)
            neigh_agg = torch.transpose(neigh_agg, 0, 1)
            all_state, last_state = self.p_neigh_rnn(neigh_agg)
        elif node_type == 3:
            neigh_agg = self.t_content_agg(id_batch).view(batch_s, 10, embed_d)
            neigh_agg = torch.transpose(neigh_agg, 0, 1)
            all_state, last_state = self.t_neigh_rnn(neigh_agg)
        else:
            neigh_agg = self.v_content_agg(id_batch).view(batch_s, 3, embed_d)
            neigh_agg = torch.transpose(neigh_agg, 0, 1)
            all_state, last_state = self.v_neigh_rnn(neigh_agg)
        neigh_agg = torch.mean(all_state, 0).view(batch_s, embed_d)
        '''input shape of LSTM(seq_len, batch, input_size)'''
        return neigh_agg

    def node_het_agg(self, id_batch, node_type):  # heterogeneous neighbor aggregation
        a_neigh_batch = [[0] * 10] * len(id_batch)
        p_neigh_batch = [[0] * 10] * len(id_batch)
        t_neigh_batch = [[0] * 10] * len(id_batch)
        v_neigh_batch = [[0] * 3] * len(id_batch)
        for i in range(len(id_batch)):
            if node_type == 1:
                a_neigh_batch[i] = self.a_neigh_list_train[0][id_batch[i]]
                p_neigh_batch[i] = self.a_neigh_list_train[1][id_batch[i]]
                t_neigh_batch[i] = self.a_neigh_list_train[2][id_batch[i]]
                v_neigh_batch[i] = self.a_neigh_list_train[3][id_batch[i]]
            elif node_type == 2:
                a_neigh_batch[i] = self.p_neigh_list_train[0][id_batch[i]]
                p_neigh_batch[i] = self.p_neigh_list_train[1][id_batch[i]]
                t_neigh_batch[i] = self.p_neigh_list_train[2][id_batch[i]]
                v_neigh_batch[i] = self.p_neigh_list_train[3][id_batch[i]]
            elif node_type == 3:
                a_neigh_batch[i] = self.t_neigh_list_train[0][id_batch[i]]
                p_neigh_batch[i] = self.t_neigh_list_train[1][id_batch[i]]
                t_neigh_batch[i] = self.t_neigh_list_train[2][id_batch[i]]
                v_neigh_batch[i] = self.t_neigh_list_train[3][id_batch[i]]
            else:
                a_neigh_batch[i] = self.v_neigh_list_train[0][id_batch[i]]
                p_neigh_batch[i] = self.v_neigh_list_train[1][id_batch[i]]
                t_neigh_batch[i] = self.v_neigh_list_train[2][id_batch[i]]
                v_neigh_batch[i] = self.v_neigh_list_train[3][id_batch[i]]
        '''a,p,t,v四种邻居的embed'''
        a_neigh_batch = np.reshape(a_neigh_batch, (1, -1))
        a_agg_batch = self.node_neigh_agg(a_neigh_batch, 1)
        p_neigh_batch = np.reshape(p_neigh_batch, (1, -1))
        p_agg_batch = self.node_neigh_agg(p_neigh_batch, 2)
        t_neigh_batch = np.reshape(t_neigh_batch, (1, -1))
        t_agg_batch = self.node_neigh_agg(t_neigh_batch, 3)
        v_neigh_batch = np.reshape(v_neigh_batch, (1, -1))
        v_agg_batch = self.node_neigh_agg(v_neigh_batch, 4)

        # attention module
        id_batch = np.reshape(id_batch, (1, -1))
        if node_type == 1:
            c_agg_batch = self.a_content_agg(id_batch)
        elif node_type == 2:
            c_agg_batch = self.p_content_agg(id_batch)
        elif node_type == 3:
            c_agg_batch = self.t_content_agg(id_batch)
        else:
            c_agg_batch = self.v_content_agg(id_batch)

        c_agg_batch_2 = torch.cat((c_agg_batch, c_agg_batch), 1).view(len(c_agg_batch), self.embed_d * 2)
        a_agg_batch_2 = torch.cat((c_agg_batch, a_agg_batch), 1).view(len(c_agg_batch), self.embed_d * 2)
        p_agg_batch_2 = torch.cat((c_agg_batch, p_agg_batch), 1).view(len(c_agg_batch), self.embed_d * 2)
        t_agg_batch_2 = torch.cat((c_agg_batch, t_agg_batch), 1).view(len(c_agg_batch), self.embed_d * 2)
        v_agg_batch_2 = torch.cat((c_agg_batch, v_agg_batch), 1).view(len(c_agg_batch), self.embed_d * 2)

        # compute weights
        node_type_n = 4
        concate_embed = torch.cat((c_agg_batch_2, a_agg_batch_2, p_agg_batch_2, t_agg_batch_2,
                                   v_agg_batch_2), 1).view(len(c_agg_batch), node_type_n+1, self.embed_d * 2)
        if node_type == 1:
            '''bmm(mat1, mat2 ,size) mat1(z,x,y), mat2(z,y,c) out(z,x,c)'''
            atten_w = self.act(torch.bmm(concate_embed, self.a_neigh_att.unsqueeze(0).expand(len(c_agg_batch),
                                                                                             *self.a_neigh_att.size())))
        elif node_type == 2:
            atten_w = self.act(torch.bmm(concate_embed, self.p_neigh_att.unsqueeze(0).expand(len(c_agg_batch),
                                                                                             *self.p_neigh_att.size())))
        elif node_type == 3:
            atten_w = self.act(torch.bmm(concate_embed, self.p_neigh_att.unsqueeze(0).expand(len(c_agg_batch),
                                                                                             *self.p_neigh_att.size())))
        else:
            atten_w = self.act(torch.bmm(concate_embed, self.v_neigh_att.unsqueeze(0).expand(len(c_agg_batch),
                                                                                             *self.v_neigh_att.size())))
        atten_w = self.softmax(atten_w).view(len(c_agg_batch), 1, node_type_n+1)

        # weighted combination
        concate_embed = torch.cat((c_agg_batch, a_agg_batch, p_agg_batch, t_agg_batch,
                                   v_agg_batch), 1).view(len(c_agg_batch), node_type_n+1, self.embed_d)
        weight_agg_batch = torch.bmm(atten_w, concate_embed).view(len(c_agg_batch), self.embed_d)

        return weight_agg_batch

    def het_agg(self, triple_index, c_id_batch, pos_id_batch, neg_id_batch):
        embed_d = self.embed_d
        # batch processing
        # nine cases for academic data (author, paper, venue)
        node_type_n = 4
        c_agg = self.node_het_agg(c_id_batch, int(triple_index / node_type_n) + 1)
        p_agg = self.node_het_agg(pos_id_batch, triple_index % 4 + 1)
        n_agg = self.node_het_agg(neg_id_batch, triple_index % 4 + 1)
        return c_agg, p_agg, n_agg

    def save_embed(self, file_):
        embed_file = open(file_, "w")
        save_batch_s = self.args.mini_batch_s
        node_type_n=4
        for i in range(node_type_n):
            if i == 0:
                batch_number = int(len(self.a_train_id_list) / save_batch_s)
            elif i == 1:
                batch_number = int(len(self.p_train_id_list) / save_batch_s)
            elif i == 2:
                batch_number = int(len(self.t_train_id_list) / save_batch_s)
            else:
                batch_number = int(len(self.v_train_id_list) / save_batch_s)
            for j in range(batch_number):
                if i == 0:
                    id_batch = self.a_train_id_list[j * save_batch_s: (j + 1) * save_batch_s]
                    out_temp = self.node_het_agg(id_batch, 1)
                elif i == 1:
                    id_batch = self.p_train_id_list[j * save_batch_s: (j + 1) * save_batch_s]
                    out_temp = self.node_het_agg(id_batch, 2)
                elif i == 2:
                    id_batch = self.t_train_id_list[j * save_batch_s: (j + 1) * save_batch_s]
                    out_temp = self.node_het_agg(id_batch, 3)
                else:
                    id_batch = self.v_train_id_list[j * save_batch_s: (j + 1) * save_batch_s]
                    out_temp = self.node_het_agg(id_batch, 4)
                out_temp = out_temp.data.cpu().numpy()
                for k in range(len(id_batch)):
                    index = id_batch[k]
                    if i == 0:
                        embed_file.write('a' + str(index) + " ")
                    elif i == 1:
                        embed_file.write('p' + str(index) + " ")
                    elif i == 2:
                        embed_file.write('t' + str(index) + " ")
                    else:
                        embed_file.write('v' + str(index) + " ")
                    for l in range(self.args.embed_d - 1):
                        embed_file.write(str(out_temp[k][l]) + " ")
                    embed_file.write(str(out_temp[k][-1]) + "\n")

            if i == 0:
                id_batch = self.a_train_id_list[batch_number * save_batch_s: -1]
                out_temp = self.node_het_agg(id_batch, 1)
            elif i == 1:
                id_batch = self.p_train_id_list[batch_number * save_batch_s: -1]
                out_temp = self.node_het_agg(id_batch, 2)
            elif i == 2:
                id_batch = self.t_train_id_list[batch_number * save_batch_s: -1]
                out_temp = self.node_het_agg(id_batch, 3)
            else:
                id_batch = self.v_train_id_list[batch_number * save_batch_s: -1]
                out_temp = self.node_het_agg(id_batch, 4)
            out_temp = out_temp.data.cpu().numpy()
            for k in range(len(id_batch)):
                index = id_batch[k]
                if i == 0:
                    embed_file.write('a' + str(index) + " ")
                elif i == 1:
                    embed_file.write('p' + str(index) + " ")
                elif i == 2:
                    embed_file.write('t' + str(index) + " ")
                else:
                    embed_file.write('v' + str(index) + " ")
                for l in range(self.args.embed_d - 1):
                    embed_file.write(str(out_temp[k][l]) + " ")
                embed_file.write(str(out_temp[k][-1]) + "\n")
        embed_file.close()
        return [], [], []

    def aggregate_all(self, triple_list_batch, triple_index):
        c_id_batch = [x[0] for x in triple_list_batch]
        pos_id_batch = [x[1] for x in triple_list_batch]
        neg_id_batch = [x[2] for x in triple_list_batch]

        c_agg, pos_agg, neg_agg = self.het_agg(triple_index, c_id_batch, pos_id_batch, neg_id_batch)

        return c_agg, pos_agg, neg_agg

    def forward(self, triple_list_batch, triple_index):
        c_out, p_out, n_out = self.aggregate_all(triple_list_batch, triple_index)
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
