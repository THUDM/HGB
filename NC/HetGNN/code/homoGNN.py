import numpy as np
import torch as th
from torch_geometric.data import Data
import torch.nn as nn
from torch_geometric.nn import GCNConv, SAGEConv, TopKPooling, GATConv
import torch.nn.functional as F
import re
import random
import csv
import argparse
import datetime
from sklearn.metrics import f1_score
from itertools import *
from scipy.sparse import coo_matrix, bmat

device = th.device('cuda:0' if th.cuda.is_available() else 'cpu')


class EarlyStopping(object):
    def __init__(self, patience=10):
        dt = datetime.datetime.now()
        self.filename = 'early_stop_{}_{:02d}-{:02d}-{:02d}.pth'.format(
            dt.date(), dt.hour, dt.minute, dt.second)
        self.patience = patience
        self.counter = 0
        self.best_acc = None
        self.best_loss = None
        self.early_stop = False

    def step(self, loss, acc, model):
        if self.best_loss is None:
            self.best_acc = acc
            self.best_loss = loss
            self.save_checkpoint(model)
        elif (loss > self.best_loss) and (acc < self.best_acc) and (acc > 0.8):
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            if (loss <= self.best_loss) and (acc >= self.best_acc):
                self.save_checkpoint(model)
            self.best_loss = np.min((loss, self.best_loss))
            self.best_acc = np.max((acc, self.best_acc))
            self.counter = 0
        return self.early_stop

    def save_checkpoint(self, model):
        """Saves model when validation loss decreases."""
        th.save(model.state_dict(), self.filename)

    def load_checkpoint(self, model):
        """Load the latest checkpoint."""
        model.load_state_dict(th.load(self.filename))


class GCN(th.nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, n_layers=2, dropout=0.5):
        print('initialing a GCN network...')
        super(GCN, self).__init__()
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(GCNConv(in_feats, hid_feats))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(GCNConv(hid_feats, hid_feats))
        # output layer
        self.layers.append(GCNConv(hid_feats, out_feats))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, data):
        x, edge_list = data
        for i, layer in enumerate(self.layers):
            if i != 0:
                x = self.dropout(x)
            x = layer(x, edge_list)
            x = F.leaky_relu(x)
        return x


class GAT(th.nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, n_layers=2, dropout=0.5, heads=[1]):
        print('initialing a GAT network...')
        super(GAT, self).__init__()
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(GATConv(in_feats, hid_feats, heads[0]))
        # hidden layers
        for l in range(1, n_layers):
            self.layers.append(GATConv(hid_feats * heads[l - 1], hid_feats, heads[l]))
        # output layer
        self.layers.append(GATConv(hid_feats * heads[-2], out_feats, heads[-1]))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, data):
        x, edge_list = data
        for i, layer in enumerate(self.layers):
            if i != 0:
                x = self.dropout(x)
            x = layer(x, edge_list)
            x = F.leaky_relu(x, negative_slope=0.01)
        return x


def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='../data/academic_test/',
                        help='path to data')
    parser.add_argument('--model_path', type=str, default='../model_save/',
                        help='path to save model')
    parser.add_argument('--A_n', type=int, default=28646,
                        help='number of author node')
    parser.add_argument('--P_n', type=int, default=21044,
                        help='number of paper node')
    parser.add_argument('--V_n', type=int, default=18,
                        help='number of venue node')
    parser.add_argument('--in_f_d', type=int, default=128,
                        help='input feature dimension')
    parser.add_argument('--embed_d', type=int, default=128,
                        help='embedding dimension')
    parser.add_argument('--train_iter_n', type=int, default=50,
                        help='max number of training iteration')
    parser.add_argument("--cuda", default=0, type=int)
    parser.add_argument("--checkpoint", default='', type=str)
    parser.add_argument("--epochs", default=500, type=str)
    parser.add_argument("--patience", default=50, type=str)
    parser.add_argument("--n_layers", default=3, type=int)
    parser.add_argument("--n_heads", default=[4], type=list)
    parser.add_argument("--dropout", default=0.5, type=float)
    parser.add_argument("--model", default='GAT', type=str)
    parser.add_argument('--lr', type=float, default=0.005, )
    parser.add_argument('--weight_decay', type=float, default=0.0)
    args = parser.parse_args()
    return args


def gen_embed(args, feat_type=1):
    p_abstract_embed = np.zeros((args.P_n, args.in_f_d))
    p_a_e_f = open(args.data_path + "p_abstract_embed.txt", "r")
    for line in islice(p_a_e_f, 1, None):
        values = line.split()
        index = int(values[0])
        embeds = np.asarray(values[1:], dtype='float32')
        p_abstract_embed[index] = embeds
    p_a_e_f.close()

    p_title_embed = np.zeros((args.P_n, args.in_f_d))
    p_t_e_f = open(args.data_path + "p_title_embed.txt", "r")
    for line in islice(p_t_e_f, 1, None):
        values = line.split()
        index = int(values[0])
        embeds = np.asarray(values[1:], dtype='float32')
        p_title_embed[index] = embeds
    p_t_e_f.close()
    p_embed = np.hstack((p_title_embed, p_abstract_embed))
    if feat_type == 0:
        dim = args.A_n
        indices_a = (np.arange(dim, dtype=np.int), np.arange(dim, dtype=np.int))
        values_a = np.ones(dim)
        a_sparse = coo_matrix((values_a, indices_a), shape=(dim, dim))
        p_sparse = coo_matrix(p_embed)
        dim = args.V_n
        indices_v = (np.arange(dim, dtype=np.int), np.arange(dim, dtype=np.int))
        values_v = np.ones(dim)
        v_sparse = coo_matrix((values_v, indices_v), shape=(dim, dim))

        ap_sparse = bmat([[a_sparse, None],
                          [None, p_sparse]])
        apv_sparse = bmat([[ap_sparse, None],
                           [None, v_sparse]])
        values = apv_sparse.data
        indices = np.vstack((apv_sparse.row, apv_sparse.col))
        indices = th.LongTensor(indices)
        values = th.FloatTensor(values)
        shape = apv_sparse.shape
        embeds = th.sparse.FloatTensor(indices, values, th.Size(shape))
    elif feat_type == 1:
        p_embed = coo_matrix(p_embed)
        row, col = p_embed.row, p_embed.col
        shape = p_embed.shape
        row = row + args.A_n
        indices = np.vstack((row, col))
        indices = th.LongTensor(indices)
        values = th.FloatTensor(p_embed.data)
        embeds = th.sparse.FloatTensor(indices, values, th.Size([shape[0]+args.A_n+args.V_n, shape[1]]))
    elif feat_type == 2:
        a_embed = np.zeros((args.A_n, np.shape(p_embed)[1]))
        v_embed = np.zeros((args.V_n, np.shape(p_embed)[1]))
        embeds = np.vstack((a_embed, p_embed, v_embed))
        embeds = th.FloatTensor(embeds)
    else:
        exit('invalid feat_type')
    return embeds


def gen_labels(args):
    labels = th.zeros(args.A_n + args.P_n + args.V_n).long()
    with open(args.data_path + "a_class_train.txt") as f:
        data_file = csv.reader(f)
        for i, d in enumerate(data_file):
            labels[int(d[0])] = int(d[1])
        f.close()
    with open(args.data_path + "a_class_test.txt") as f:
        data_file = csv.reader(f)
        for i, d in enumerate(data_file):
            labels[int(d[0])] = int(d[1])
        f.close()
    return labels


def gen_edge_list(args):
    a_p_list = [[], []]
    p_a_list = [[], []]
    p_p_list = [[], []]
    v_p_list = [[], []]
    p_v_list = [[], []]

    relation_f = ["a_p_list_train.txt", "p_a_list_train.txt", "p_p_cite_list_train.txt", "v_p_list_train.txt"]
    for i in range(len(relation_f)):
        f_name = relation_f[i]
        neigh_f = open(args.data_path + f_name, "r")
        for line in neigh_f:
            line = line.strip()
            node_id = int(re.split(':', line)[0])
            neigh_list = re.split(':', line)[1]
            neigh_list_id = re.split(',', neigh_list)
            if f_name == 'a_p_list_train.txt':
                for j in range(len(neigh_list_id)):
                    a_p_list[0].append(node_id)
                    a_p_list[1].append(int(neigh_list_id[j]))
            elif f_name == 'p_a_list_train.txt':
                for j in range(len(neigh_list_id)):
                    p_a_list[0].append(node_id)
                    p_a_list[1].append(int(neigh_list_id[j]))
            elif f_name == 'p_p_cite_list_train.txt':
                for j in range(len(neigh_list_id)):
                    p_p_list[0].append(node_id)
                    p_p_list[1].append(int(neigh_list_id[j]))
            elif f_name == 'v_p_list_train.txt':
                for j in range(len(neigh_list_id)):
                    v_p_list[0].append(node_id)
                    v_p_list[1].append(int(neigh_list_id[j]))
            else:
                print('Some errors occur.')
        neigh_f.close()
    # get paper-venue edge
    p_v = [0] * args.P_n

    p_v_f = open(args.data_path + 'p_v.txt', "r")
    for line in p_v_f:
        line = line.strip()
        p_id = int(re.split(',', line)[0])
        v_id = int(re.split(',', line)[1])
        p_v[p_id] = v_id
        p_v_list[0].append(p_id)
        p_v_list[1].append(v_id)
    p_v_f.close()
    # set range of a,p,v (0-28645,0-21043,0-17) -> (0-28645,28646-49689,49690-49707)
    for i in range(len(a_p_list[1])):
        a_p_list[1][i] += args.A_n
    for i in range(len(p_a_list[0])):
        p_a_list[0][i] += args.A_n
    for i in range(len(p_p_list[0])):
        p_p_list[0][i] += args.A_n
        p_p_list[1][i] += args.A_n
    for i in range(len(v_p_list[0])):
        v_p_list[0][i] += args.A_n + args.P_n
        v_p_list[1][i] += args.A_n
    for i in range(len(p_v_list[0])):
        p_v_list[0][i] += args.A_n
        p_v_list[1][i] += args.A_n + args.P_n
    # 42379 42379 ...
    start_list = a_p_list[0] + p_a_list[0] + p_p_list[0] + v_p_list[0] + p_v_list[0]
    end_list = a_p_list[1] + p_a_list[1] + p_p_list[1] + v_p_list[1] + p_v_list[1]
    return th.LongTensor([start_list, end_list])


def gen_mask(args, val_ratio_of_train=0.15):
    train_mask, val_mask, test_mask = th.zeros(args.A_n + args.P_n + args.V_n, dtype=th.uint8), \
                                      th.zeros(args.A_n + args.P_n + args.V_n, dtype=th.uint8), \
                                      th.zeros(args.A_n + args.P_n + args.V_n, dtype=th.uint8)
    each_num = {'train_num': 0, 'val_num': 0, 'test_num': 0}
    with open(args.data_path + "a_class_train.txt") as f:
        data_file = csv.reader(f)
        for i, d in enumerate(data_file):
            if random.random() < val_ratio_of_train:
                val_mask[int(d[0])] = 1
                each_num['val_num'] += 1
            else:
                train_mask[int(d[0])] = 1
                each_num['train_num'] += 1
        f.close()
    with open(args.data_path + "a_class_test.txt") as f:
        data_file = csv.reader(f)
        for i, d in enumerate(data_file):
            test_mask[int(d[0])] = 1
            each_num['test_num'] += 1
        f.close()
    print(each_num)
    return train_mask.bool(), val_mask.bool(), test_mask.bool()


def score(logits, labels):
    _, indices = th.max(logits, dim=1)
    prediction = indices.long().cpu().numpy()
    labels = labels.cpu().numpy()

    accuracy = (prediction == labels).sum() / len(prediction)
    micro_f1 = f1_score(labels, prediction, average='micro')
    macro_f1 = f1_score(labels, prediction, average='macro')

    return accuracy, micro_f1, macro_f1


def train(model, data, train_mask, val_mask, test_mask, labels):
    # train
    stopper = EarlyStopping(patience=args.patience)
    loss_func = th.nn.CrossEntropyLoss()
    optimizer = th.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    CosineLR = th.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=0.001)
    for epoch in range(args.epochs):
        model.train()
        logits = model(data)
        loss = loss_func(logits[train_mask], labels[train_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_acc, train_micro_f1, train_macro_f1 = score(logits[train_mask], labels[train_mask])
        val_loss, val_acc, val_micro_f1, val_macro_f1 = test(model, data, val_mask, labels)
        early_stop = stopper.step(val_loss.data.item(), val_macro_f1, model)

        print('Epoch {:d} | Train Loss {:.4f} | Train Micro f1 {:.4f} | Train Macro f1 {:.4f} | '
              'Val Loss {:.4f} | Val Micro f1 {:.4f} | Val Macro f1 {:.4f}'
            .format(
            epoch + 1, loss.item(), train_micro_f1, train_macro_f1,
            val_loss.item(), val_micro_f1, val_macro_f1
        ))
        if val_macro_f1 > 0.9:
            CosineLR.step()
        print(CosineLR.get_lr())
        if early_stop:
            break
    stopper.load_checkpoint(model)
    print('Score of test_data(accuracy, micro_f1, macro_f1):', test(model, data, test_mask, labels))


def test(model, data, test_mask, labels):
    model.eval()
    loss_func = th.nn.CrossEntropyLoss()
    with th.no_grad():
        logits = model(data)
    loss = loss_func(logits[test_mask], labels[test_mask])
    accuracy, micro_f1, macro_f1 = score(logits[test_mask], labels[test_mask])
    # print('Test loss {:.4f} | Test Micro f1 {:.4f} | Test Macro f1 {:.4f}'.format(
    #     loss.item(), micro_f1, macro_f1))
    return loss, accuracy, micro_f1, macro_f1


def main(args):
    if th.cuda.is_available():
        print("Use GPU(%s) success." % th.cuda.get_device_name(), ' model:', args.model)

    train_mask, val_mask, test_mask = gen_mask(args)
    train_mask, val_mask, test_mask = train_mask.to(device), val_mask.to(device), test_mask.to(device)
    labels = gen_labels(args).to(device)

    edge_list = gen_edge_list(args).to(device)
    embeds = gen_embed(args).to(device)
    in_feats = embeds.shape[1]
    if args.model == 'GCN':
        model = GCN(in_feats=in_feats, hid_feats=128, out_feats=4, n_layers=args.n_layers, dropout=args.dropout).to(
            device)
    else:
        heads = args.n_heads * args.n_layers + [1]
        model = GAT(in_feats=in_feats, hid_feats=128, out_feats=4, n_layers=args.n_layers, dropout=args.dropout,
                    heads=heads).to(device)
    train(model, [embeds, edge_list], train_mask, val_mask, test_mask, labels, )
    test(model, [embeds, edge_list], test_mask, labels)


if __name__ == '__main__':
    args = read_args()
    print(args)
    main(args)
