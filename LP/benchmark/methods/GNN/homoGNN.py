import sys
import random
import torch as th
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torch_geometric.nn import GCNConv, SAGEConv, TopKPooling, GATConv
import torch.nn.functional as F
from sklearn.metrics import (auc, f1_score, precision_recall_curve, roc_auc_score)
import os
import math
import pickle
import argparse
import sys
import scipy.sparse as sp
from collections import defaultdict

sys.path.append('../../')
from scripts.data_loader import data_loader

os.environ['CUDA_VISIBLE_DEVICES'] = ''

device = th.device('cuda:0' if th.cuda.is_available() else 'cpu')
print(f'use device: {device}')


def gen_edge_list(dl, reverse=True):
    edge_list = [list(), list()]
    types = list(dl.links['data'].keys())
    data = dl.links['data'][types[0]]
    for t in types[1:]:
        data += dl.links['data'][t]

    row, col = data.nonzero()
    for h_id, t_id in zip(row, col):
        edge_list[0].append(h_id)
        edge_list[1].append(t_id)
        if reverse:
            edge_list[1].append(h_id)
            edge_list[0].append(t_id)
    return th.LongTensor(edge_list).to(device)


def gen_feat(node_num, feat_type=0):
    feat = None
    if feat_type == 0:
        # sparse
        indices = np.vstack((np.arange(node_num), np.arange(node_num)))
        indices = th.LongTensor(indices)
        values = th.FloatTensor(np.ones(node_num))
        feat = th.sparse.FloatTensor(indices, values, th.Size([node_num, node_num])).to(device)

    elif feat_type == 1:
        # dense
        feat = th.FloatTensor(np.eye(node_num)).to(device)
    return feat


def gen_feat_list(dim_list):
    feat_list = list()
    for dim in dim_list:
        indices = np.vstack((np.arange(dim), np.arange(dim)))
        indices = th.LongTensor(indices)
        values = th.FloatTensor(np.ones(dim))
        feat = th.sparse.FloatTensor(indices, values, th.Size([dim, dim]))
        feat_list.append(feat)
    return feat_list


class hom_data(Dataset):
    def __init__(self, dl, eval_type):
        self.eval_type = eval_type
        train_datax4 = [dl.train_pos[eval_type][0],
                        dl.train_pos[eval_type][1],
                        dl.train_neg[eval_type][0],
                        dl.train_neg[eval_type][1]]

        shuffle_index = np.arange(len(train_datax4[0]))
        random.shuffle(shuffle_index)
        self.train_datax4 = [
            np.array(train_datax4[0])[shuffle_index],
            np.array(train_datax4[1])[shuffle_index],
            np.array(train_datax4[2])[shuffle_index],
            np.array(train_datax4[3])[shuffle_index]
        ]
        self.train_label = th.LongTensor([1, 0] * len(train_datax4[0]))
        self.train_data = th.LongTensor(self.trans_x4_x2(self.train_datax4))
        self.valid_pos = dl.valid_pos
        self.valid_data = [self.valid_pos[eval_type][0] + dl.valid_neg[eval_type][0],
                           self.valid_pos[eval_type][1] + dl.valid_neg[eval_type][1]]
        self.valid_label = [1] * len(self.valid_pos[eval_type][0]) + [0] * len(dl.valid_neg[eval_type][0])

    def __len__(self):
        return self.train_data[0].shape[0]

    def __getitem__(self, item):
        return [self.train_data[0][item], self.train_data[1][item], self.train_label[item]]

    def refresh_train_neg(self, edge_type):
        train_neg = dl.get_train_neg([eval_type])[edge_type]
        train_neg_dict = defaultdict(list)
        for i in range(len(train_neg[0])):
            train_neg_dict[train_neg[0][i]].append(train_neg[1][i])
        h_t_index = dict(zip(set(train_neg_dict.keys()), [0] * len(train_neg_dict.keys())))
        for i in range(len(self.train_datax4[0])):
            h_id = self.train_datax4[2][i]
            self.train_datax4[3][i] = train_neg_dict[h_id][h_t_index[h_id]]
            h_t_index[h_id] += 1
        self.train_data = th.LongTensor(self.trans_x4_x2(self.train_datax4))

    def refresh_valid_neg(self, edge_type):
        valid_neg_type = dl.get_valid_neg([edge_type])[edge_type]
        self.valid_data = [self.valid_pos[edge_type][0] + valid_neg_type[0],
                           self.valid_pos[eval_type][1] + valid_neg_type[1]]
        self.valid_label = [1] * len(self.valid_pos[eval_type][0]) + [0] * len(valid_neg_type[0])

    def trans_x4_x2(self, datax4: list):
        datax2 = [list(), list()]
        for row in range(len(datax4[0])):
            datax2[0].append(datax4[0][row])
            datax2[1].append(datax4[1][row])
            datax2[0].append(datax4[2][row])
            datax2[1].append(datax4[3][row])
        return datax2


class EarlyStop:
    def __init__(self, save_path='', patience=5):
        self.patience = patience
        self.min_loss = float('inf')
        self.patience_now = 0
        self.stop = False
        self.save_path = save_path

    def step(self, loss, model):
        if loss < self.min_loss:
            self.min_loss = loss
            self.patience_now = 0
            # update model
            if self.save_path != '':
                self.save_checkpoint(model)
        else:
            self.patience_now += 1
            if self.patience_now >= self.patience:
                self.stop = True
            else:
                print(f'EarlyStopping counter: {self.patience_now} out of {self.patience}')

    def get_best(self):
        return self.min_loss

    def save_checkpoint(self, model):
        print('Saving model')
        th.save(model.state_dict(), self.save_path)


class DisMult(th.nn.Module):
    def __init__(self, rel_num, dim):
        super(DisMult, self).__init__()
        self.dim = dim
        self.weights = nn.Parameter(th.FloatTensor(size=(rel_num, dim, dim)))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.weights, gain=1.414)

    def forward(self, r_list, input1, input2):
        w = self.weights[r_list]
        input1 = th.unsqueeze(input1, 1)
        input2 = th.unsqueeze(input2, 2)
        tmp = th.bmm(input1, w)
        re = th.bmm(tmp, input2).squeeze()
        return re


class Dot(nn.Module):
    def __init__(self):
        super(Dot, self).__init__()

    def forward(self, r_list, input1, input2):
        input1 = th.unsqueeze(input1, 1)
        input2 = th.unsqueeze(input2, 2)
        return th.bmm(input1, input2).squeeze()


class GCN(th.nn.Module):
    def __init__(self, in_feats, hid_feats, n_layers=2, dropout=0.5, decoder='dot', rel_num=1):
        super(GCN, self).__init__()
        self.fc_list = nn.ModuleList([nn.Linear(feats_dim, hid_feats, bias=True) for feats_dim in in_feats])
        for fc in self.fc_list:
            nn.init.xavier_normal_(fc.weight, gain=1.414)
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(GCNConv(hid_feats, hid_feats))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(GCNConv(hid_feats, hid_feats))
        self.dropout = nn.Dropout(p=dropout)
        for layer in self.layers:
            nn.init.xavier_normal_(layer.weight, gain=1.4)
        if decoder == 'dismult':
            self.decode = DisMult(rel_num=rel_num, dim=hid_feats)
        elif decoder == 'dot':
            self.decode = Dot()

    def encode(self, data):
        x, edge_list = data
        h = []
        for fc, feature in zip(self.fc_list, x):
            h.append(fc(feature))
        x = th.cat(h, 0)
        for i, layer in enumerate(self.layers):
            if i != 0:
                x = self.dropout(x)
            x = layer(x, edge_list)
            if i < len(self.layers) - 1:
                x = F.elu(x)
        return x


class GAT(th.nn.Module):
    def __init__(self, in_feats, hid_feats, n_layers=2, dropout=0.5, heads=[1], decoder="dot", rel_num=1):
        super(GAT, self).__init__()
        self.fc_list = nn.ModuleList([nn.Linear(feats_dim, hid_feats, bias=True) for feats_dim in in_feats])
        for fc in self.fc_list:
            nn.init.xavier_normal_(fc.weight, gain=1.414)
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(GATConv(hid_feats, hid_feats, heads[0]))
        # hidden layers
        for l in range(1, n_layers - 1):
            self.layers.append(GATConv(hid_feats * heads[l - 1], hid_feats, heads[l]))
        # output layer
        self.layers.append(GATConv(hid_feats * heads[-2], hid_feats, heads[-1]))
        self.dropout = nn.Dropout(p=dropout)
        for layer in self.layers:
            nn.init.xavier_normal_(layer.lin_l.weight, gain=1.4)
        if decoder == 'dismult':
            self.decode = DisMult(rel_num=rel_num, dim=hid_feats)
        elif decoder == 'dot':
            self.decode = Dot()

    def encode(self, data):
        x, edge_list = data
        h = []
        for fc, feature in zip(self.fc_list, x):
            h.append(fc(feature))
        x = th.cat(h, 0)
        for i, layer in enumerate(self.layers):
            if i != 0:
                x = self.dropout(x)
            x = layer(x, edge_list)
            if i < len(self.layers) - 1:
                x = F.elu(x)
        return x


def evaluate(out_feat, pos_num):
    neg_num = out_feat.size()[0] - pos_num
    true_list = np.ones(pos_num, dtype=int).tolist() + np.zeros(neg_num, dtype=int).tolist()
    prediction_list = out_feat.cpu().detach().numpy().tolist()

    sorted_pred = prediction_list[:]
    sorted_pred.sort()
    threshold = sorted_pred[-pos_num]

    y_pred = np.zeros(len(prediction_list), dtype=np.int32)
    for i in range(len(prediction_list)):
        if prediction_list[i] >= threshold:
            y_pred[i] = 1

    y_true = np.array(true_list)
    y_scores = np.array(prediction_list)
    ps, rs, _ = precision_recall_curve(y_true, y_scores)
    return roc_auc_score(y_true, y_scores), auc(rs, ps), f1_score(y_true, y_pred)


def file2list(file_name, eval_type):
    file_list = [[], [], []]
    pos_num = 0
    with open(file_name) as file_handle:
        for line in file_handle:
            edge_type, left, right, label = line[:-1].split('\t')
            if edge_type != eval_type:
                continue
            file_list[0].append(int(left))
            file_list[1].append(int(right))
            file_list[2].append(int(label))
            if int(label) == 1:
                pos_num += 1
    return file_list, pos_num


def main(edge_list, feat_list, train_data, args, model, dl):
    train_data_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=False, num_workers=0)
    early_stopping = EarlyStop(save_path=args.model_path, patience=args.patience)
    edge_num = len(edge_list)
    lossFun = nn.BCELoss()
    optimizer = th.optim.Adam([{'params': model.parameters()}], lr=args.lr, weight_decay=args.weight_decay)
    for epoch in range(args.epochs):
        train_data.refresh_train_neg(eval_type)
        train_data.refresh_valid_neg(eval_type)
        for index, train_data_bach in enumerate(train_data_loader):
            # train_data_bach = train_data_bach.to(device)
            model.train()
            if args.edge_sample_ratio < 1:
                edge_sample_index = np.random.choice(edge_num, int(edge_num * args.edge_sample_ratio), replace=False)
                edge_list_part = edge_list[:, edge_sample_index]
                hid_feat = model.encode([feat_list, edge_list_part])
            else:
                hid_feat = model.encode([feat_list, edge_list])
            r_list = [0] * train_data_bach[0].shape[0]
            out_feat = model.decode(r_list, hid_feat[train_data_bach[0]], hid_feat[train_data_bach[1]])
            out_feat = th.sigmoid(out_feat)
            # out_feat = model.decode(hid_feat, [train_data_bach[0], train_data_bach[1]]).to(device)
            target = train_data_bach[2].float().to(device)
            loss = lossFun(out_feat, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f'epoch {epoch} | batch {index} |train loss {loss.item()}')

            if index % 10 == 0:
                # valid
                valid_list, valid_label = train_data.valid_data, train_data.valid_label
                model.eval()
                if args.edge_sample_ratio < 1:
                    edge_sample_index = np.random.choice(edge_num, int(edge_num * args.edge_sample_ratio),
                                                         replace=False)
                    edge_list_part = edge_list[:, edge_sample_index]
                    hid_feat = model.encode([feat_list, edge_list_part])
                else:
                    hid_feat = model.encode([feat_list, edge_list])
                r_list = [0] * len(valid_label)
                out_feat = model.decode(r_list, hid_feat[valid_list[0]], hid_feat[valid_list[1]]).to(device)
                out_feat = th.sigmoid(out_feat)
                target = th.FloatTensor(valid_label).to(device)
                valid_loss = lossFun(out_feat, target)

                threshold = args.threshold if args.threshold != None else np.median(
                    out_feat.flatten().detach().cpu().numpy())
                valid_score = dl.evaluate(
                    edge_list=valid_list,
                    confidence=out_feat.flatten().detach().cpu().numpy(),
                    labels=target.flatten().detach().cpu().numpy(), threshold=threshold)
                print(
                    f"Valid: loss {valid_loss}, score:{valid_score}")
                early_stopping.step(valid_loss, model)
                if early_stopping.stop:
                    break
        if early_stopping.stop:
            break

    th.cuda.empty_cache()
    # random test
    test_list, test_label = dl.get_test_neigh_w_random([eval_type])
    test_list, test_label = test_list[eval_type], test_label[eval_type]
    model.load_state_dict(th.load(args.model_path))
    model.eval()
    target = th.FloatTensor(test_label).to(device)
    if args.test_with_CPU:
        model = model.cpu()
        for i in range(len(feat_list)):
            feat_list[i] = feat_list[i].cpu()
        edge_list = edge_list.cpu()
        target = target.cpu()
    hid_feat = model.encode([feat_list, edge_list])
    r_list = [0] * len(test_label)
    out_feat = model.decode(r_list, hid_feat[test_list[0]], hid_feat[test_list[1]])
    out_feat = th.sigmoid(out_feat)
    threshold = args.threshold if args.threshold != None else np.median(out_feat.flatten().detach().cpu().numpy())
    random_test_score = dl.evaluate(
        edge_list=test_list,
        confidence=out_feat.flatten().detach().cpu().numpy(),
        labels=target.flatten().detach().cpu().numpy(), threshold=threshold, eval_mrr=True)
    print(f"Random test:{random_test_score}")

    # sec neigh test
    test_list, test_label = dl.get_test_neigh([eval_type])
    test_list, test_label = test_list[eval_type], test_label[eval_type]
    model.load_state_dict(th.load(args.model_path))
    model.eval()
    target = th.FloatTensor(test_label).view(-1, 1).to(device)
    if args.test_with_CPU:
        model = model.cpu()
        for i in range(len(feat_list)):
            feat_list[i] = feat_list[i].cpu()
        edge_list = edge_list.cpu()
        target = target.cpu()

    hid_feat = model.encode([feat_list, edge_list])
    r_list = [0] * len(test_label)
    out_feat = model.decode(r_list, hid_feat[test_list[0]], hid_feat[test_list[1]])
    out_feat = th.sigmoid(out_feat)
    threshold = args.threshold if args.threshold != None else np.median(out_feat.flatten().detach().cpu().numpy())
    test_score = dl.evaluate(
        edge_list=test_list,
        confidence=out_feat.flatten().detach().cpu().numpy(),
        labels=target.flatten().detach().cpu().numpy(), threshold=threshold)
    print(f"Test:{test_score}")

    return random_test_score, test_score


def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='LastFM',
                        help='path to data')
    parser.add_argument('--model_path', type=str, default='model.pt',
                        help='path to save model')
    parser.add_argument('--train_iter_n', type=int, default=50,
                        help='max number of training iteration')
    parser.add_argument("--cuda", default=0, type=int)
    parser.add_argument("--checkpoint", default='', type=str)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--patience", default=3, type=str)
    parser.add_argument("--n_layers", default=2, type=int)
    parser.add_argument("--n_heads", default=[4], type=list)
    parser.add_argument("--dropout", default=0.5, type=float)
    parser.add_argument("--model", default='GCN', type=str)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--weight_decay', type=float, default=0.000)
    parser.add_argument('--batch_size', type=int, default=1000)
    parser.add_argument('--edge_sample_ratio', type=float, default=1)
    parser.add_argument('--test_with_CPU', type=bool, default=False)
    parser.add_argument('--decoder', type=str, default="dismult", choices=['dot', 'dismult'])
    parser.add_argument('--threshold', type=float, default=None)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = read_args()
    print(args)
    data_name = args.data
    data_dir = os.path.join('../../data', data_name)
    node_type_file = os.path.join('../../data', data_name, 'node.dat')
    dl_pickle_f = os.path.join(data_dir, 'dl_pickle')
    if os.path.exists(dl_pickle_f):
        dl = pickle.load(open(dl_pickle_f, 'rb'))
        print(f'Info: load {data_name} from {dl_pickle_f}')
    else:
        dl = data_loader(data_dir)
        pickle.dump(dl, open(dl_pickle_f, 'wb'))
        print(f'Info: load {data_name} from original data and generate {dl_pickle_f} ')
    model = None

    # feat = gen_feat(dl.nodes['total']).to(device)
    dim_list = list()
    for node_type in sorted(dl.nodes['count'].keys()):
        dim_list.append(dl.nodes['count'][node_type])
    feat_list = gen_feat_list(dim_list=dim_list)
    feat_list = [f.to(device) for f in feat_list]
    if args.model == 'GCN':
        model = GCN(in_feats=dim_list, hid_feats=64, n_layers=args.n_layers, dropout=args.dropout,
                    decoder=args.decoder)
    elif args.model == "GAT":
        heads = args.n_heads * args.n_layers + [1]
        model = GAT(in_feats=dim_list, hid_feats=64, n_layers=args.n_layers, heads=heads,
                    dropout=args.dropout, decoder=args.decoder)
    else:
        exit('please input true model_type within [GCN, GAT]')
    test_score = {'auc_score': list(), 'roc_auc': list(), 'F1': list(), 'MRR': list()}
    random_test_score = {'auc_score': list(), 'roc_auc': list(), 'F1': list(), 'MRR': list()}
    eval_types = list(dl.links_test['data'].keys())
    for eval_type in eval_types:
        train_data = hom_data(dl, eval_type)
        print(f'dataset: {data_name}, eval_type: {eval_type}')
        model = model.to(device)
        edge_list = gen_edge_list(dl, reverse=True)
        random_score, score = main(edge_list=edge_list, feat_list=feat_list, train_data=train_data, args=args,
                                   model=model,
                                   dl=dl)
        test_score['auc_score'].append(score['auc_score'])
        test_score['roc_auc'].append(score['roc_auc'])
        test_score['F1'].append(score['F1'])
        test_score['MRR'].append(score['MRR'])
        random_test_score['auc_score'].append(random_score['auc_score'])
        random_test_score['roc_auc'].append(random_score['roc_auc'])
        random_test_score['F1'].append(random_score['F1'])
        random_test_score['MRR'].append(random_score['MRR'])

    print(f'Sec neigh test: score list: {test_score}')
    print(f'Random test: score list: {random_test_score}')

    for s in test_score.keys():
        test_score[s] = np.mean(test_score[s])
    print(f'Sec neigh test:average score: {test_score}')

    for s in test_score.keys():
        random_test_score[s] = np.mean(random_test_score[s])
    print(f'Random test: average score: {random_test_score}')
