from __future__ import division
from __future__ import print_function

import time
import argparse
from networkx.algorithms.centrality import trophic
from networkx.algorithms.cuts import edge_expansion
import numpy as np
import pickle as pkl
from copy import deepcopy
from random import shuffle

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from scipy.sparse import coo_matrix
import dgl

from utils import load_data, accuracy, dense_tensor_to_sparse, resample, makedirs
from models import HGAT, GCN
import os
import gc
import sys
from print_log import Logger
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

logdir = "log/"
savedir = 'model/'
embdir = 'embeddings/'
makedirs([logdir, savedir, embdir])


dataset = 'mr'


# Training settings
write_embeddings = True
LR = 0.01 if dataset == 'snippets' else 0.005
DP = 0.95 if dataset in ['agnews', 'tagmynews'] else 0.8
WD = 0 if dataset == 'snippets' else 5e-8
LR = 0.05 if 'multi' in dataset else LR
DP = 0.5 if 'multi' in dataset else DP
WD = 0 if 'multi' in dataset else WD
parser = argparse.ArgumentParser()
parser.add_argument('--no_cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=300,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=LR,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=WD,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=512,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=DP,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--dataset', type=str, default=dataset,
                    help='Dataset')
parser.add_argument('--repeat', type=int, default=1,
                    help='Number of repeated trials')
parser.add_argument('--node', action='store_false', default=True,
                    help='Use node-level attention or not. ')
parser.add_argument('--type', action='store_false', default=True,
                    help='Use type-level attention or not. ')
parser.add_argument('--baseline', action='store_true', default=False,
                    help='Use Baseline')
args = parser.parse_args()

dataset = args.dataset
args.cuda = not args.no_cuda and torch.cuda.is_available()
sys.stdout = Logger(logdir + "{}.log".format(dataset))

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

loss_list = dict()


def margin_loss(preds, y, weighted_sample=False):
    nclass = y.shape[1]
    preds = preds[:, :nclass]
    y = y.float()
    lam = 0.25
    m = nn.Threshold(0., 0.)
    loss = y * m(0.9 - preds) ** 2 + \
        lam * (1.0 - y) * (m(preds - 0.1) ** 2)

    if weighted_sample:
        n, N = y.sum(dim=0, keepdim=True), y.shape[0]
        weight = torch.where(y == 1, n, torch.zeros_like(loss))
        weight = torch.where(y != 1, N-n, weight)
        weight = N / weight / 2
        loss = torch.mul(loss, weight)

    loss = torch.mean(torch.sum(loss, dim=1))
    return loss


def nll_loss(preds, y):
    y = y.max(1)[1].type_as(labels)
    return F.nll_loss(preds, y)


def evaluate(preds_list, y_list):
    nclass = y_list.shape[1]
    preds_list = preds_list[:, :nclass]
    if not preds_list.device == 'cpu':
        preds_list, y_list = preds_list.cpu(), y_list.cpu()

    threshold = 0.5
    multi_label = 'multi' in dataset
    if multi_label:
        y_list = y_list.numpy()
        preds_probs = preds_list.detach().numpy()
        preds = deepcopy(preds_probs)
        preds[np.arange(preds.shape[0]), preds.argmax(1)] = 1.0
        preds[np.where(preds >= threshold)] = 1.0
        preds[np.where(preds < threshold)] = 0.0
        [precision, recall, F1, support] = \
            precision_recall_fscore_support(y_list[preds.sum(axis=1) != 0], preds[preds.sum(axis=1) != 0],
                                            average='micro')
        [precision_ma, recall_ma, F1_ma, support] = \
            precision_recall_fscore_support(y_list[preds.sum(axis=1) != 0], preds[preds.sum(axis=1) != 0],
                                            average='macro')
        ER = accuracy_score(y_list, preds) * 100

        report = classification_report(y_list, preds, digits=5)

        print(' ER: %6.2f' % ER,
              'mi-ma: P: %5.1f %5.1f' % (precision*100, precision_ma*100),
              'R: %5.1f %5.1f' % (recall*100, recall_ma*100),
              'F1: %5.1f %5.1f' % (F1*100, F1_ma*100),
              end="")
        return ER, report
    else:
        y_list = y_list.numpy()
        preds_probs = preds_list.detach().numpy()
        preds = deepcopy(preds_probs)
        preds[np.arange(preds.shape[0]), preds.argmax(1)] = 1.0
        preds[np.where(preds < 1)] = 0.0
        [precision, recall, F1, support] = \
            precision_recall_fscore_support(y_list, preds, average='macro')
        ER = accuracy_score(y_list, preds) * 100
        print(' Ac: %6.2f' % ER,
              'P: %5.1f' % (precision*100),
              'R: %5.1f' % (recall*100),
              'F1: %5.1f' % (F1*100),
              end="")
        return ER, F1


LOSS = margin_loss if 'multi' in dataset else nll_loss


def train(epoch,
          input_adj_train, input_features_train, idx_out_train, idx_train,
          input_adj_val, input_features_val, idx_out_val, idx_val):
    print('Epoch: {:04d}'.format(epoch+1), end='')
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(input_features_train, input_adj_train)

    if isinstance(output, list):
        O, L = output[0][idx_out_train], labels[idx_train]
    else:
        O, L = output[idx_out_train], labels[idx_train]
    loss_train = LOSS(O, L)
    print(' | loss: {:.4f}'.format(loss_train.item()), end='')
    acc_train, f1_train = evaluate(O, L)
    loss_train.backward()
    optimizer.step()

    model.eval()
    output = model(input_features_val, input_adj_val)
    if isinstance(output, list):
        loss_val = LOSS(output[0][idx_out_val], labels[idx_val])
        print(' | loss: {:.4f}'.format(loss_val.item()), end='')
        results = evaluate(output[0][idx_out_val], labels[idx_val])
    else:
        loss_val = LOSS(output[idx_out_val], labels[idx_val])
        print(output[idx_out_val])
        print(' | loss: {:.4f}'.format(loss_val.item()), end='')
        results = evaluate(output[idx_out_val], labels[idx_val])
    print(' | time: {:.4f}s'.format(time.time() - t))
    loss_list[epoch] = [loss_train.item()]

    if 'multi' in dataset:
        acc_val, res_line = results
        return float(acc_val.item()), res_line
    else:
        acc_val, f1_val = results
        return float(acc_val.item()), float(f1_val.item())


def test(epoch, input_adj_test, input_features_test, idx_out_test, idx_test):
    print(' '*90 if 'multi' in dataset else ' '*65, end='')
    t = time.time()
    model.eval()
    output = model(input_features_test, input_adj_test)

    if isinstance(output, list):
        loss_test = LOSS(output[0][idx_out_test], labels[idx_test])
        print(' | loss: {:.4f}'.format(loss_test.item()), end='')
        results = evaluate(output[0][idx_out_test], labels[idx_test])
    else:
        loss_test = LOSS(output[idx_out_test], labels[idx_test])
        print(' | loss: {:.4f}'.format(loss_test.item()), end='')
        results = evaluate(output[idx_out_test], labels[idx_test])
    print(' | time: {:.4f}s'.format(time.time() - t))
    loss_list[epoch] += [loss_test.item()]

    if 'multi' in dataset:
        acc_test, res_line = results
        return float(acc_test.item()), res_line
    else:
        acc_test, f1_test = results
        return float(acc_test.item()), float(f1_test.item())

# change to homo graph


def change_to_homo(input_adj_train, input_features_train,
                   input_adj_val, input_features_val):
    feature_len = [0, 0, 0]  # text, topic, entity
    node_num = [0, 0, 0]
    for i in range(3):
        feature_len[i] = input_features_train[i].shape[1]
        node_num[i] = input_features_train[i].shape[0]

    feature_row = torch.zeros([0], dtype=torch.int32)
    feature_col = torch.zeros([0], dtype=torch.int32)
    feature_val = torch.zeros([0], dtype=torch.float)
    adj_row = torch.zeros([0], dtype=torch.int32)
    adj_col = torch.zeros([0], dtype=torch.int32)
    adj_val = torch.zeros([0], dtype=torch.float)

    # add all types edges to homo graph
    # and stack features together
    for t1 in range(3):
        feture_map = input_features_train[t1]
        if t1 == 0:
            row_begin = 0
            feture_begin = 0
        elif t1 == 1:
            row_begin = node_num[0]
            feture_begin = feature_len[0]
        else:
            row_begin = node_num[0]+node_num[1]
            feture_begin = feature_len[0]+feature_len[1]
        feature_row = torch.cat((
            feature_row, feture_map.coalesce().indices()[0] + row_begin), 0)
        feature_col = torch.cat((
            feature_col, feture_map.coalesce().indices()[1] + feture_begin))
        feature_val = torch.cat(
            (feature_val, feture_map.coalesce().values()))
        for t2 in range(3):
            adj = input_adj_train[t1][t2]
            if t2 == 0:
                col_begin = 0
            elif t2 == 1:
                col_begin = node_num[0]
            else:
                col_begin = node_num[0]+node_num[1]
            adj_row = torch.cat((
                adj_row, adj.coalesce().indices()[0] + row_begin))
            adj_col = torch.cat((
                adj_col, adj.coalesce().indices()[1] + col_begin))
            adj_val = torch.cat((adj_val, adj.coalesce().values()))

    homo_adj_sci = coo_matrix((adj_val.tolist(), (adj_row.tolist(), adj_col.tolist())),
                              shape=(sum(node_num), sum(node_num)))
    hg = dgl.from_scipy(homo_adj_sci)
    edge_index = torch.stack((feature_row, feature_col), 0)
    homo_features = torch.sparse.FloatTensor(
        edge_index, feature_val, torch.Size([sum(node_num), sum(feature_len)]))

    return hg, homo_features, sum(feature_len)


def gcn_model(input_adj_train, input_features_train,
              input_adj_val, input_features_val):
    hg, homo_feature, feature_dim = change_to_homo(input_adj_train, input_features_train,
                                                   input_adj_val, input_features_val)
    model = GCN(g=hg,
                in_feats=feature_dim,
                n_hidden=args.hidden,
                n_classes=labels.shape[1],
                n_layers=2,
                activation=torch.sigmoid,
                dropout=args.dropout,
                sparse_input=False)
    return model, homo_feature.to_dense()


# train step on GCN
def baseline_train(epoch, input_features_train, idx_out_train, idx_train, input_features_val, idx_out_val, idx_val):
    print('Epoch: {:04d}'.format(epoch+1), end='')
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(input_features_train)
    O, L = output[idx_out_train], labels[idx_train]
    loss_train = LOSS(O, L)
    print(' | loss: {:.4f}'.format(loss_train.item()), end='')
    acc_train, f1_train = evaluate(O, L)
    loss_train.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        output = model(input_features_val)
        O, L = output[idx_out_val], labels[idx_val]
        loss_val = LOSS(O, L)
        print(' | loss: {:.4f}'.format(loss_val.item()), end='')
        results = evaluate(O, L)
        print(' | time: {:.4f}s'.format(time.time() - t))
        loss_list[epoch] = [loss_train.item()]
        acc_val, f1_val = results

    return float(acc_val.item()), float(f1_val.item())


# test step on GCN
def baseline_test(epoch, input_features_test, idx_out_test, idx_test):
    t = time.time()
    model.eval()
    output = model(input_features_test)

    loss_test = LOSS(output[idx_out_test], labels[idx_test])
    print(' | loss: {:.4f}'.format(loss_test.item()), end='')
    results = evaluate(output[idx_out_test], labels[idx_test])
    print(' | time: {:.4f}s'.format(time.time() - t))
    loss_list[epoch] += [loss_test.item()]

    acc_test, f1_test = results
    return float(acc_test.item()), float(f1_test.item())


path = '../data/' + dataset + '/'
adj, features, labels, idx_train_ori, idx_val_ori, idx_test_ori, idx_map = load_data(
    path=path, dataset=dataset)
N = len(adj)

input_adj_train, input_features_train, idx_out_train = adj, features, idx_train_ori
input_adj_val, input_features_val, idx_out_val = adj, features, idx_val_ori
input_adj_test, input_features_test, idx_out_test = adj, features, idx_test_ori
idx_train, idx_val, idx_test = idx_train_ori, idx_val_ori, idx_test_ori


if args.cuda:
    N = len(features)
    for i in range(N):
        if input_features_train[i] is not None:
            input_features_train[i] = input_features_train[i].cuda()
        if input_features_val[i] is not None:
            input_features_val[i] = input_features_val[i].cuda()
        if input_features_test[i] is not None:
            input_features_test[i] = input_features_test[i].cuda()
    for i in range(N):
        for j in range(N):
            if input_adj_train[i][j] is not None:
                input_adj_train[i][j] = input_adj_train[i][j].cuda()
            if input_adj_val[i][j] is not None:
                input_adj_val[i][j] = input_adj_val[i][j].cuda()
            if input_adj_test[i][j] is not None:
                input_adj_test[i][j] = input_adj_test[i][j].cuda()
    labels = labels.cuda()
    idx_train, idx_out_train = idx_train.cuda(), idx_out_train.cuda()
    idx_val, idx_out_val = idx_val.cuda(), idx_out_val.cuda()
    idx_test, idx_out_test = idx_test.cuda(), idx_out_test.cuda()


FINAL_RESULT = []
for i in range(args.repeat):
    # Model and optimizer
    print("\n\nNo. {} test.\n".format(i+1))
    if args.baseline:
        model, homo_features = gcn_model(input_adj_train, input_features_train,
                                         input_adj_val, input_features_val)
        input_features_train = homo_features
        input_features_test = homo_features
        input_features_val = homo_features
    else:
        model = HGAT(nfeat_list=[i.shape[1] for i in features],
                     type_attention=args.type,
                     node_attention=args.node,
                     nhid=args.hidden,
                     nclass=labels.shape[1],
                     dropout=args.dropout,
                     gamma=0.1,
                     orphan=True,
                     )
    optimizer = optim.Adam(model.parameters(), lr=args.lr,
                           weight_decay=args.weight_decay)

    if args.cuda:
        model.cuda()

    print(model)
    print(len(list(model.parameters())))
    print([i.size() for i in model.parameters()])

    # Train model
    t_total = time.time()
    vali_max = [0, [0, 0], -1]

    for epoch in range(args.epochs):
        if args.baseline:
            vali_acc, vali_f1 = baseline_train(
                epoch, input_features_train, idx_out_train, idx_train, input_features_val, idx_out_val, idx_val)
        else:
            vali_acc, vali_f1 = train(epoch,
                                      input_adj_train, input_features_train, idx_out_train, idx_train,
                                      input_adj_val, input_features_val, idx_out_val, idx_val)
        if args.baseline:
            test_acc, test_f1 = baseline_test(
                epoch,  input_features_test, idx_out_test, idx_test)
        else:
            test_acc, test_f1 = test(epoch,
                                     input_adj_test, input_features_test, idx_out_test, idx_test)

        if vali_acc > vali_max[0]:
            vali_max = [vali_acc, (test_acc, test_f1), epoch+1]
            with open(savedir + "{}.pkl".format(dataset), 'wb') as f:
                pkl.dump(model, f)

            if write_embeddings and not args.baseline:
                makedirs([embdir])
                with open(embdir + "{}.emb".format(dataset), 'w') as f:
                    for i in model.emb.tolist():
                        f.write("{}\n".format(i))
                with open(embdir + "{}.emb2".format(dataset), 'w') as f:
                    for i in model.emb2.tolist():
                        f.write("{}\n".format(i))

    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
    if 'multi' in dataset:
        print("The best result is ACC: {0:.4f}, where epoch is {2}\n{1}\n".format(
            vali_max[1][0],
            vali_max[1][1],
            vali_max[2]))
    else:
        print("The best result is: ACC: {0:.4f} F1: {1:.4f}, where epoch is {2}\n\n".format(
            vali_max[1][0],
            vali_max[1][1],
            vali_max[2]))
    FINAL_RESULT.append(list(vali_max))

print("\n")
for i in range(len(FINAL_RESULT)):
    if 'multi' in dataset:
        print("{0}:\tvali:  {1:.5f}\ttest:  ACC: {2:.4f}, epoch={4}.\n{3}".format(
            i,
            FINAL_RESULT[i][0],
            FINAL_RESULT[i][1][0],
            FINAL_RESULT[i][1][1],
            FINAL_RESULT[i][2]))
    else:
        print("{}:\tvali:  {:.5f}\ttest:  ACC: {:.4f} F1: {:.4f}, epoch={}".format(
            i,
            FINAL_RESULT[i][0],
            FINAL_RESULT[i][1][0],
            FINAL_RESULT[i][1][1],
            FINAL_RESULT[i][2]))
