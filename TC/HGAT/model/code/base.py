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
from scipy.sparse import coo_matrix, bmat
import dgl

from utils import load_data, accuracy, dense_tensor_to_sparse, resample, makedirs
import sys
from print_log import Logger
from baseline.GNN import myGAT, GAT

logdir = "log/"
savedir = 'model/'
embdir = 'embeddings/'
makedirs([logdir, savedir, embdir])


dataset = 'agnews'


# Training settings
# LR = 0.01 if dataset == 'snippets' else 0.005
LR = 0.01 if dataset == 'snippets' else 0.005
DP = 0.7
WD = 0 if dataset == 'snippets' else 5e-8
LR = 0.05 if 'multi' in dataset else LR
WD = 0 if 'multi' in dataset else WD
head_number = 2
parser = argparse.ArgumentParser()
parser.add_argument('--no_cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=100,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=LR,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=WD,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=50,
                    help='Number of hidden units.')
parser.add_argument('--layer', type=int, default=1,
                    help='Number of layer.')
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
parser.add_argument('--edge_feats', type=int, default=32)
parser.add_argument('--baseline', action='store_true', default=False,
                    help='Use baseline')
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


def cross_entropy(preds, y):
    y = y.max(1)[1].type_as(labels)
    return F.cross_entropy(preds, y)


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


LOSS = margin_loss if 'multi' in dataset else cross_entropy


def train(epoch, homo_features, e_feat, idx_out_train, idx_train, idx_out_val, idx_val):
    print('Epoch: {:04d}'.format(epoch+1), end='')
    t = time.time()
    model.train()
    optimizer.zero_grad()
    if args.baseline:
        output = model(homo_features, e_feat)
    else:
        output = model(homo_features)

    if isinstance(output, list):
        O, L = output[0][idx_out_train], labels[idx_train]
    else:
        O, L = output[idx_out_train], labels[idx_train]
    loss_train = LOSS(O, L)
    print(' | loss: {:.4f}'.format(loss_train.item()), end='')
    loss_train.backward()
    optimizer.step()

    model.eval()
    if args.baseline:
        output = model(homo_features, e_feat)
    else:
        output = model(homo_features)
    if isinstance(output, list):
        loss_val = LOSS(output[0][idx_out_val], labels[idx_val])
        print(' | loss: {:.4f}'.format(loss_val.item()), end='')
        output = F.softmax(output, 1)
        results = evaluate(output[0][idx_out_val], labels[idx_val])
    else:
        loss_val = LOSS(output[idx_out_val], labels[idx_val])
        print(' | loss: {:.4f}'.format(loss_val.item()), end='')
        output = F.softmax(output, 1)
        results = evaluate(output[idx_out_val], labels[idx_val])
    print(' | time: {:.4f}s'.format(time.time() - t))
    loss_list[epoch] = [loss_train.item()]

    if 'multi' in dataset:
        acc_val, res_line = results
        return float(acc_val.item()), res_line
    else:
        acc_val, f1_val = results
        return float(acc_val.item()), float(f1_val.item())


def test(epoch, homo_features, e_feat, idx_out_test, idx_test):
    print(' '*90 if 'multi' in dataset else ' '*65, end='')
    t = time.time()
    model.eval()
    if args.baseline:
        output = model(homo_features, e_feat)
    else:
        output = model(homo_features)

    if isinstance(output, list):
        loss_test = LOSS(output[0][idx_out_test], labels[idx_test])
        print(' | loss: {:.4f}'.format(loss_test.item()), end='')
        output = F.softmax(output, 1)
        results = evaluate(output[0][idx_out_test], labels[idx_test])
    else:
        loss_test = LOSS(output[idx_out_test], labels[idx_test])
        print(' | loss: {:.4f}'.format(loss_test.item()), end='')
        output = F.softmax(output, 1)
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
def change_to_homo(input_adj_train, input_features_train):
    sf = []  # scipy features
    for x in input_features_train:
        sf.append(coo_matrix(x.to_dense().numpy()))
    coo = bmat([[sf[0], None, None], [None, sf[1], None], [None, None, sf[2]]])

    values = coo.data
    indices = np.vstack((coo.row, coo.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = coo.shape
    homo_features = torch.sparse.FloatTensor(i, v, torch.Size(shape))

    new_adj = []
    for x in input_adj_train:
        new_adj.append([])
        for y in x:
            new_adj[-1].append(coo_matrix(y.to_dense().numpy()))
    homo_adj_sci = bmat(new_adj)

    values = homo_adj_sci.data
    indices = np.vstack((homo_adj_sci.row, homo_adj_sci.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = homo_adj_sci.shape
    homo_adj = torch.sparse.FloatTensor(i, v, torch.Size(shape))
    #print(homo_adj)
    #input()

    hg = dgl.from_scipy(homo_adj_sci, eweight_name='weight')
    #hg = dgl.remove_self_loop(hg)
    #hg = dgl.add_self_loop(hg)
    #hg.edata['weight'][hg.edata['weight'] == 0.] = 1.
    return hg, homo_features, coo.shape[1], homo_adj


def baseline_model(hg, args, edge_type_count, feature_dims):
    heads = [head_number]*args.layer + [head_number]
    if args.baseline:
        model = myGAT(
            g=hg,
            edge_dim=args.edge_feats,
            num_etypes=edge_type_count+1,
            in_dims=[feature_dims],
            num_hidden=args.hidden,
            num_classes=labels.shape[1],
            num_layers=args.layer,
            heads=heads,
            activation=F.elu,
            feat_drop=args.dropout,
            attn_drop=args.dropout,
            negative_slope=0.05,
            residual=True,
            alpha=0.4)
    else:
        model = GAT(
            g=hg,
            in_dims=[feature_dims],
            num_hidden=args.hidden,
            num_classes=labels.shape[1],
            num_layers=args.layer,
            heads=heads,
            activation=F.elu,
            feat_drop=args.dropout,
            attn_drop=args.dropout,
            negative_slope=0.05,
            residual=False)
        pass
    return model


path = '../data/' + dataset + '/'
adj, features, labels, idx_train_ori, idx_val_ori, idx_test_ori, idx_map = load_data(
    path=path, dataset=dataset)
N = len(adj)

input_adj_train, input_features_train, idx_out_train = adj, features, idx_train_ori
input_adj_val, input_features_val, idx_out_val = adj, features, idx_val_ori
input_adj_test, input_features_test, idx_out_test = adj, features, idx_test_ori
idx_train, idx_val, idx_test = idx_train_ori, idx_val_ori, idx_test_ori

hg, homo_features, feature_dims, homo_adj = change_to_homo(
    input_adj_train, input_features_train)

new_adj = []
edge_type_count = 0
edge2type = {}
e_feat = []
edge2id = {}
col_begin = 0
row_begin = 0
for x in input_adj_train:
    col_begin = 0
    for y in x:
        y = y.coalesce()
        for i, j, v in zip(y.indices()[0], y.indices()[1], y.values()):
            edge2type[(i+row_begin, j+col_begin)] = edge_type_count
            e_feat.append(edge_type_count)
            edge2id[(i+row_begin, j+col_begin)] = len(edge2id)
        edge_type_count += 1
        col_begin += y.size()[1]
    row_begin += x[0].size()[0]
for i in range(homo_adj.shape[0]):
    edge2type[(i, i)] = edge_type_count
    edge2id[(i, i)] = len(edge2id)
e_feat = torch.tensor(e_feat, dtype=torch.long)
homo_features = homo_features.unsqueeze(0)


if args.cuda:
    N = len(features)
    labels = labels.cuda()
    idx_train, idx_out_train = idx_train.cuda(), idx_out_train.cuda()
    idx_val, idx_out_val = idx_val.cuda(), idx_out_val.cuda()
    idx_test, idx_out_test = idx_test.cuda(), idx_out_test.cuda()
    e_feat = e_feat.cuda()
    homo_features = homo_features.cuda()
    hg = hg.to("cuda:0")


FINAL_RESULT = []

for i in range(args.repeat):
    # Model and optimizer
    print("\n\nNo. {} test.\n".format(i+1))

    model = baseline_model(hg, args, edge_type_count, feature_dims)
    optimizer = optim.Adam(model.parameters(), lr=args.lr,
                           weight_decay=args.weight_decay)
    if args.cuda:
        model = model.cuda()

    print(model)
    print(len(list(model.parameters())))
    print([i.size() for i in model.parameters()])

    # Train model
    t_total = time.time()
    vali_max = [0, [0, 0], -1]

    for epoch in range(args.epochs):
        vali_acc, vali_f1 = train(
            epoch, homo_features, e_feat, idx_out_train, idx_train, idx_out_val, idx_val)
        test_acc, test_f1 = test(epoch, homo_features,
                                 e_feat, idx_out_test, idx_test)
        if vali_acc > vali_max[0]:
            vali_max = [vali_acc, (test_acc, test_f1), epoch+1]
            with open(savedir + "{}.pkl".format(dataset), 'wb') as f:
                pkl.dump(model, f)

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
