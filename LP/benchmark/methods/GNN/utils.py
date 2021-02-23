from sklearn.metrics import (auc, f1_score, precision_recall_curve, roc_auc_score)
import scipy.sparse as sp
import argparse
from torch.utils.data import Dataset
import torch as th
import numpy as np
import random
import sys
from collections import defaultdict

sys.path.append('../../')
from scripts.data_loader import data_loader


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
    """self loop"""
    for i in range(data.shape[0]):
        edge_list[1].append(i)
        edge_list[0].append(i)
    return th.LongTensor(edge_list)


def gen_feat(node_num, feat_type=0):
    feat = None
    if feat_type == 0:
        # sparse
        indices = np.vstack((np.arange(node_num), np.arange(node_num)))
        indices = th.LongTensor(indices)
        values = th.FloatTensor(np.ones(node_num))
        feat = th.sparse.FloatTensor(indices, values, th.Size([node_num, node_num]))

    elif feat_type == 1:
        # dense
        feat = th.FloatTensor(np.eye(node_num))
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
        np.random.shuffle(shuffle_index)
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

    def refresh_train_neg(self, edge_type, dl):
        train_neg = dl.get_train_neg([edge_type])[edge_type]
        train_neg_dict = defaultdict(list)
        for i in range(len(train_neg[0])):
            train_neg_dict[train_neg[0][i]].append(train_neg[1][i])
        h_t_index = dict(zip(set(train_neg_dict.keys()), [0] * len(train_neg_dict.keys())))
        for i in range(len(self.train_datax4[0])):
            h_id = self.train_datax4[2][i]
            self.train_datax4[3][i] = train_neg_dict[h_id][h_t_index[h_id]]
            h_t_index[h_id] += 1
        self.train_data = th.LongTensor(self.trans_x4_x2(self.train_datax4))

    def refresh_valid_neg(self, edge_type, dl):
        valid_neg_type = dl.get_valid_neg([edge_type])[edge_type]
        self.valid_data = [self.valid_pos[edge_type][0] + valid_neg_type[0],
                           self.valid_pos[edge_type][1] + valid_neg_type[1]]
        self.valid_label = [1] * len(self.valid_pos[edge_type][0]) + [0] * len(valid_neg_type[0])

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
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--edge_sample_ratio', type=float, default=1)
    parser.add_argument('--test_with_CPU', type=bool, default=False)
    parser.add_argument('--decoder', type=str, default="dismult", choices=['dot', 'dismult'])

    args = parser.parse_args()
    return args