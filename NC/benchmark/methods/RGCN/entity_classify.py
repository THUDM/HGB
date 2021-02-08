"""
Modeling Relational Data with Graph Convolutional Networks
Paper: https://arxiv.org/abs/1703.06103
Code: https://github.com/tkipf/relational-gcn

Difference compared to tkipf/relation-gcn
* l2norm applied to all weights
* remove nodes that won't be touched
"""
from numpy.lib.function_base import append
from model import BaseRGCN
import json
from sklearn.metrics import f1_score
from scipy import sparse
from dgl.nn.pytorch import RelGraphConv
import dgl
import torch.nn.functional as F
import torch
import time
import numpy as np
from os import link
import argparse
import torch.nn as nn
import pynvml
import os
import gc
import psutil


import sys

sys.path.append('../../')
pynvml.nvmlInit()


def sp_to_spt(mat):
    coo = mat.tocoo()
    values = coo.data
    indices = np.vstack((coo.row, coo.col))

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = coo.shape

    return torch.sparse.FloatTensor(i, v, torch.Size(shape))


def mat2tensor(mat):
    if type(mat) is np.ndarray:
        return torch.from_numpy(mat).type(torch.FloatTensor)
    return sp_to_spt(mat)


def evaluate(model_pred, labels):
    # 注意这里的model_pred是经过sigmoid处理的，sigmoid处理后可以视为预测是这一类的概率
    # 预测结果，大于这个阈值则视为预测正确
    pred_result = model_pred.argmax(dim=1)
    labels = labels.cpu()
    pred_result = pred_result.cpu()

    micro = f1_score(labels, pred_result, average='micro')
    macro = f1_score(labels, pred_result, average='macro')

    return micro, macro


def multi_evaluate(model_pred, labels):
    # 注意这里的model_pred是经过sigmoid处理的，sigmoid处理后可以视为预测是这一类的概率
    # 预测结果，大于这个阈值则视为预测正确
    model_pred = torch.sigmoid(model_pred)
    accuracy_th = 0.5
    pred_result = model_pred > accuracy_th
    pred_result = pred_result.float()
    labels = labels.cpu()
    pred_result = pred_result.cpu()

    micro = f1_score(labels, pred_result, average='micro')
    macro = f1_score(labels, pred_result, average='macro')

    return micro, macro


class EntityClassify(BaseRGCN):
    def build_input_layer(self):
        return nn.ModuleList([nn.Linear(in_dim, self.h_dim, bias=True) for in_dim in self.in_dims])

    def build_hidden_layer(self, idx):
        return RelGraphConv(self.h_dim, self.h_dim, self.num_rels, "basis",
                            self.num_bases, activation=F.relu, self_loop=self.use_self_loop,
                            dropout=self.dropout)

    def build_output_layer(self):
        return RelGraphConv(self.h_dim, self.out_dim, self.num_rels, "basis",
                            self.num_bases, activation=None,
                            self_loop=self.use_self_loop)


def main(args):
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    if args.gpu >= 0:
        handle = pynvml.nvmlDeviceGetHandleByIndex(args.gpu)
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
    gpu_begin = meminfo.used
    print("begin:", gpu_begin)
    dataset = ['dblp', 'imdb', 'acm', 'freebase']
    if args.dataset in dataset:
        dataset = None
    else:
        raise ValueError()

    # Load from hetero-graph
    if args.dataset in ['imdb']:
        LOSS = F.binary_cross_entropy_with_logits
    else:
        LOSS = F.cross_entropy

    folder = './data/'+args.dataset.upper()
    from scripts.data_loader import data_loader
    dl = data_loader(folder)
    all_data = {}
    for etype in dl.links['meta']:
        etype_info = dl.links['meta'][etype]
        metrix = dl.links['data'][etype]
        all_data[(etype_info[0], 'link', etype_info[1])] = (
            sparse.find(metrix)[0]-dl.nodes['shift'][etype_info[0]], sparse.find(metrix)[1]-dl.nodes['shift'][etype_info[1]])
    hg = dgl.heterograph(all_data)
    category_id = list(dl.labels_train['count'].keys())[0]
    train_idx = np.nonzero(dl.labels_train['mask'])[0]
    test_idx = np.nonzero(dl.labels_test['mask'])[0]
    if args.dataset == 'imdb':
        labels = torch.FloatTensor(
            dl.labels_train['data']+dl.labels_test['data'])
    else:
        labels = torch.LongTensor(
            dl.labels_train['data']+dl.labels_test['data']).argmax(dim=1)
    num_classes = dl.labels_test['num_classes']

    num_rels = len(hg.canonical_etypes)
    if args.dataset in ['imdb']:
        EVALUATE = multi_evaluate
    else:
        EVALUATE = evaluate

    # split dataset into train, validate, test
    if args.validation:
        val_idx = train_idx[:len(train_idx) // 5]
        train_idx = train_idx[len(train_idx) // 5:]
    else:
        val_idx = train_idx

    # calculate norm for each edge type and store in edge
    for canonical_etype in hg.canonical_etypes:
        u, v, eid = hg.all_edges(form='all', etype=canonical_etype)
        _, inverse_index, count = torch.unique(
            v, return_inverse=True, return_counts=True)
        degrees = count[inverse_index]
        norm = torch.ones(eid.shape[0]).float() / degrees.float()
        norm = norm.unsqueeze(1)
        hg.edges[canonical_etype].data['norm'] = norm

    g = dgl.to_homogeneous(hg, edata=['norm'])
    num_nodes = g.number_of_nodes()
    node_ids = torch.arange(num_nodes)
    edge_norm = g.edata['norm']
    edge_type = g.edata[dgl.ETYPE].long()

    # find out the target node ids in g
    node_tids = g.ndata[dgl.NTYPE]
    loc = (node_tids == category_id)
    target_idx = node_ids[loc]

    # check cuda
    use_cuda = args.gpu >= 0 and torch.cuda.is_available()
    device = torch.device('cuda:'+str(args.gpu) if use_cuda else 'cpu')
    torch.cuda.set_device(args.gpu)
    edge_type = edge_type.to(device)
    edge_norm = edge_norm.to(device)
    labels = labels.to(device)

    features_list = []
    for i in range(len(dl.nodes['count'])):
        th = dl.nodes['attr'][i]
        if th is None:
            features_list.append(np.eye(dl.nodes['count'][i]))
        else:
            features_list.append(th)
    features_list = [mat2tensor(features).to(device)
                     for features in features_list]
    feats_type = args.feats_type
    in_dims = []
    if feats_type == 0:
        in_dims = [features.shape[1] for features in features_list]
    elif feats_type == 1 or feats_type == 5:
        save = 0 if feats_type == 1 else 2
        in_dims = []
        for i in range(0, len(features_list)):
            if i == save:
                in_dims.append(features_list[i].shape[1])
            else:
                in_dims.append(10)
                features_list[i] = torch.zeros(
                    (features_list[i].shape[0], 10)).to(device)
    elif feats_type == 2 or feats_type == 4:
        save = feats_type - 2
        in_dims = [features.shape[0] for features in features_list]
        for i in range(0, len(features_list)):
            if i == save:
                in_dims[i] = features_list[i].shape[1]
                continue
            dim = features_list[i].shape[0]
            indices = np.vstack((np.arange(dim), np.arange(dim)))
            indices = torch.LongTensor(indices)
            values = torch.FloatTensor(np.ones(dim))
            features_list[i] = torch.sparse.FloatTensor(
                indices, values, torch.Size([dim, dim])).to(device)
    elif feats_type == 3:
        in_dims = [features.shape[0] for features in features_list]
        for i in range(len(features_list)):
            dim = features_list[i].shape[0]
            indices = np.vstack((np.arange(dim), np.arange(dim)))
            indices = torch.LongTensor(indices)
            values = torch.FloatTensor(np.ones(dim))
            features_list[i] = torch.sparse.FloatTensor(
                indices, values, torch.Size([dim, dim])).to(device)

    model = EntityClassify(in_dims,
                           args.n_hidden,
                           num_classes,
                           num_rels,
                           num_bases=args.n_bases,
                           num_hidden_layers=args.n_layers - 2,
                           dropout=args.dropout,
                           use_self_loop=args.use_self_loop)

    model.to(device)
    g = g.to('cuda:%d' % args.gpu)

    # optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.l2norm)

    # training loop
    # print("start training...")
    forward_time = []
    backward_time = []
    save_dict_micro = {}
    save_dict_macro = {}
    best_result_micro = 0
    best_result_macro = 0
    best_epoch_micro = 0
    best_epoch_macro = 0

    model.train()
    for epoch in range(args.n_epochs):
        optimizer.zero_grad()
        t0 = time.time()
        logits = model(g, features_list, edge_type, edge_norm)
        logits = logits[target_idx]
        loss = LOSS(logits[train_idx], labels[train_idx])
        t1 = time.time()
        loss.backward()
        optimizer.step()
        t2 = time.time()

        forward_time.append(t1 - t0)
        backward_time.append(t2 - t1)
        # print("Epoch {:05d} | Train Forward Time(s) {:.4f} | Backward Time(s) {:.4f}".
        #       format(epoch, forward_time[-1], backward_time[-1]))
        # val_loss = LOSS(logits[val_idx], labels[val_idx])
        # train_micro, train_macro = EVALUATE(
        #     logits[train_idx], labels[train_idx])
        valid_micro, valid_macro = EVALUATE(
            logits[val_idx], labels[val_idx])
        if valid_micro > best_result_micro:
            save_dict_micro = model.state_dict()
            best_result_micro = valid_micro
            best_epoch_micro = epoch
        if valid_macro > best_result_macro:
            save_dict_macro = model.state_dict()
            best_result_macro = valid_macro
            best_epoch_macro = epoch

        # print("Train micro: {:.4f} | Train macro: {:.4f} | Train Loss: {:.4f} | Validation micro: {:.4f} | Validation macro: {:.4f} | Validation loss: {:.4f}".
        #     format(train_micro, train_macro, loss.item(), valid_micro, valid_macro, val_loss.item()))
    # print()

    model.eval()
    result = [save_dict_micro, save_dict_macro]
    torch.cuda.empty_cache()
    with torch.no_grad():
        for i in range(2):
            if i == 0:
                print("Best Micro At:"+str(best_epoch_micro))
            else:
                print("Best Macro At:"+str(best_epoch_macro))
            model.load_state_dict(result[i])
            t0 = time.time()
            logits = model.forward(g, features_list, edge_type, edge_norm)
            t1 = time.time()
            print("test time:"+str(t1-t0))
            logits = logits[target_idx]
            test_loss = LOSS(logits[test_idx], labels[test_idx])
            test_micro, test_macro = EVALUATE(
                logits[test_idx], labels[test_idx])
            print("Test micro: {:.4f} | Test macro: {:.4f} | Test loss: {:.4f}".format(
                test_micro, test_macro, test_loss.item()))
        # print("Mean forward time: {:4f}".format(
        #     np.mean(forward_time[len(forward_time) // 4:])))
        # print("Mean backward time: {:4f}".format(
        #     np.mean(backward_time[len(backward_time) // 4:])))
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
    gpu_end = meminfo.used
    print("test end:", gpu_end)
    print("net gpu usage:", (gpu_end-gpu_begin)/1024/1024, 'MB')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RGCN')
    parser.add_argument('--feats-type', type=int, default=3,
                        help='Type of the node features used. ' +
                        '0 - loaded features; ' +
                        '1 - only target node features (zero vec for others); ' +
                        '2 - only target node features (id vec for others); ' +
                        '3 - all id vec. Default is 2;' +
                        '4 - only term features (id vec for others);' +
                        '5 - only term features (zero vec for others).')
    parser.add_argument("--dropout", type=float, default=0,
                        help="dropout probability")
    parser.add_argument("--n-hidden", type=int, default=16,
                        help="number of hidden units")
    parser.add_argument("--gpu", type=int, default=-1,
                        help="gpu")
    parser.add_argument("--lr", type=float, default=1e-2,
                        help="learning rate")
    parser.add_argument("--n-bases", type=int, default=-1,
                        help="number of filter weight matrices, default: -1 [use all]")
    parser.add_argument("--n-layers", type=int, default=3,
                        help="number of propagation rounds")
    parser.add_argument("-e", "--n-epochs", type=int, default=150,
                        help="number of training epochs")
    parser.add_argument("-d", "--dataset", type=str, required=True,
                        help="dataset to use")
    parser.add_argument("--l2norm", type=float, default=0,
                        help="l2 norm coef")
    parser.add_argument("--use-self-loop", default=False, action='store_true',
                        help="include self feature as a special relation")
    fp = parser.add_mutually_exclusive_group(required=False)
    fp.add_argument('--validation', dest='validation', action='store_true')
    fp.add_argument('--testing', dest='validation', action='store_false')
    parser.set_defaults(validation=True)

    args = parser.parse_args()
    # print(args)
    main(args)
