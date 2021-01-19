"""
Modeling Relational Data with Graph Convolutional Networks
Paper: https://arxiv.org/abs/1703.06103
Code: https://github.com/MichSchli/RelationPrediction
Difference compared to MichSchli/RelationPrediction
* Report raw metrics instead of filtered metrics.
* By default, we use uniform edge sampling instead of neighbor-based edge
  sampling used in author's code. In practice, we find it achieves similar MRR
  probably because the model only uses one GNN layer so messages are propagated
  among immediate neighbors. User could specify "--edge-sampler=neighbor" to switch
  to neighbor-based edge sampling.
"""

import argparse
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import dgl
from dgl.nn.pytorch import RelGraphConv
import sys

from model import BaseRGCN

import utils

sys.path.append("../../")


class RGCN(BaseRGCN):
    def build_input_layer(self):
        return nn.ModuleList([nn.Linear(in_dim, self.h_dim, bias=True) for in_dim in self.in_dims])

    def build_hidden_layer(self, idx):
        act = F.relu if idx < self.num_hidden_layers - 1 else None
        return RelGraphConv(self.h_dim, self.h_dim, self.num_rels, "bdd",
                            self.num_bases, activation=act, self_loop=True,
                            dropout=self.dropout)


class LinkPredict(nn.Module):
    def __init__(self, in_dims, h_dim, num_rels, num_bases=-1,
                 num_hidden_layers=1, dropout=0, use_cuda=False, reg_param=0):
        super(LinkPredict, self).__init__()
        self.rgcn = RGCN(in_dims, h_dim, h_dim, num_rels * 2, num_bases,
                         num_hidden_layers, dropout, use_cuda)
        self.reg_param = reg_param
        self.w_relation = nn.Parameter(torch.Tensor(num_rels, h_dim))
        nn.init.xavier_uniform_(self.w_relation,
                                gain=nn.init.calculate_gain('relu'))

    def calc_score(self, embedding, triplets):
        # DistMult
        s = embedding[triplets[:, 0]]
        r = self.w_relation[triplets[:, 1]]
        o = embedding[triplets[:, 2]]
        score = torch.sum(s * r * o, dim=1)
        return score

    def forward(self, g, h, r, norm):
        return self.rgcn.forward(g, h, r, norm)

    def regularization_loss(self, embedding):
        return torch.mean(embedding.pow(2)) + torch.mean(self.w_relation.pow(2))

    def get_loss(self, embed, triplets, labels):
        # triplets is a list of data samples (positive and negative)
        # each row in the triplets is a 3-tuple of (source, relation, destination)
        score = self.calc_score(embed, triplets)
        logp = torch.sigmoid(score)
        predict_loss = F.binary_cross_entropy_with_logits(logp, labels)
        reg_loss = self.regularization_loss(embed)
        return predict_loss + self.reg_param * reg_loss


def run(args):
    feats_type = args.feats_type
    features_list, adjM, dl = utils.load_data(args.dataset)
    device = torch.device(
        'cuda:0' if torch.cuda.is_available() and args.gpu >= 0 else 'cpu')
    use_cuda = args.gpu >= 0 and torch.cuda.is_available()
    features_list = [utils.mat2tensor(features).to(device)
                     for features in features_list]
    in_dims = []
    if feats_type == 0:
        in_dims = [features.shape[1] for features in features_list]
    elif feats_type == 1 or feats_type == 5:
        save = 0 if feats_type == 1 else 2
        # [features_list[0].shape[1]] + [10] * (len(features_list) - 1)
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

    edge2type = {}
    for k in dl.links['data']:
        for u, v in zip(*dl.links['data'][k].nonzero()):
            edge2type[(u, v)] = k
    for i in range(dl.nodes['total']):
        if (i, i) not in edge2type:
            edge2type[(i, i)] = len(dl.links['count'])
    for k in dl.links['data']:
        for u, v in zip(*dl.links['data'][k].nonzero()):
            if (v, u) not in edge2type:
                edge2type[(v, u)] = k+1+len(dl.links['count'])

    g = dgl.DGLGraph(adjM+(adjM.T))
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)
    g = g.to(device)
    e_feat = []
    for u, v in zip(*g.edges()):
        u = u.cpu().item()
        v = v.cpu().item()
        e_feat.append(edge2type[(u, v)])
    e_feat = torch.tensor(e_feat, dtype=torch.long).to(device)

    test_norm = utils.comp_deg_norm(g.cpu())
    edge_norm = utils.node_norm_to_edge_norm(
        g.cpu(), torch.from_numpy(test_norm).view(-1, 1)).to(device)

    for test_edge_type in dl.links_test['data'].keys():
        train_pos, valid_pos = dl.get_train_valid_pos()
        train_pos = train_pos[test_edge_type]
        valid_pos = valid_pos[test_edge_type]

        net = LinkPredict(
            in_dims,
            args.hidden_dim,
            len(dl.links['count'])+1,
            num_bases=args.n_bases,
            num_hidden_layers=args.num_layers,
            dropout=args.dropout,
            use_cuda=use_cuda,
            reg_param=args.regularization)
        net.to(device)
        optimizer = torch.optim.Adam(
            net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        # training loop
        net.train()
        early_stopping = utils.EarlyStopping(patience=args.patience, verbose=True,
                                             save_path='checkpoint/checkpoint_{}_{}.pt'.format(args.dataset, args.num_layers))
        for epoch in range(args.epoch):
            t_start = time.time()

            train_neg = dl.get_train_neg(edge_types=[test_edge_type])[
                test_edge_type]
            train_pos_head = np.array(train_pos[0])
            train_pos_tail = np.array(train_pos[1])
            train_neg_head = np.array(train_neg[0])
            train_neg_tail = np.array(train_neg[1])

            # training
            net.train()

            left = np.concatenate([train_pos_head, train_neg_head])
            right = np.concatenate([train_pos_tail, train_neg_tail])
            mid = np.zeros(
                train_pos_head.shape[0]+train_neg_head.shape[0], dtype=np.int32)
            labels = torch.FloatTensor(np.concatenate(
                [np.ones(train_pos_head.shape[0]), np.zeros(train_neg_head.shape[0])])).to(device)

            embed = net(g, features_list, e_feat, edge_norm)
            triplets = [[i[0], i[1], i[2]]for i in zip(left, mid, right)]
            train_loss = net.get_loss(
                embed, torch.LongTensor(triplets), labels)

            # autograd
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            t_end = time.time()

            # print training info
            print('Epoch {:05d} | Train_Loss: {:.4f} | Time: {:.4f}'.format(
                epoch, train_loss.item(), t_end-t_start))

            t_start = time.time()
            # validation
            net.eval()
            with torch.no_grad():
                valid_neg = dl.get_valid_neg(edge_types=[test_edge_type])[
                    test_edge_type]
                valid_pos_head = np.array(valid_pos[0])
                valid_pos_tail = np.array(valid_pos[1])
                valid_neg_head = np.array(valid_neg[0])
                valid_neg_tail = np.array(valid_neg[1])
                left = np.concatenate([valid_pos_head, valid_neg_head])
                right = np.concatenate([valid_pos_tail, valid_neg_tail])
                mid = np.zeros(
                    valid_pos_head.shape[0]+valid_neg_head.shape[0], dtype=np.int32)
                labels = torch.FloatTensor(np.concatenate(
                    [np.ones(valid_pos_head.shape[0]), np.zeros(valid_neg_head.shape[0])])).to(device)
                embed = net(g, features_list, e_feat, edge_norm)
                triplets = [[i[0], i[1], i[2]]for i in zip(left, mid, right)]
                val_loss = net.get_loss(
                    embed, torch.LongTensor(triplets), labels)
            t_end = time.time()
            # print validation info
            print('Epoch {:05d} | Val_Loss {:.4f} | Time(s) {:.4f}'.format(
                epoch, val_loss.item(), t_end - t_start))
            # early stopping
            early_stopping(val_loss, net)
            if early_stopping.early_stop:
                print('Early stopping!')
                break

        # testing with evaluate_results_nc
        net.load_state_dict(torch.load(
            'checkpoint/checkpoint_{}_{}.pt'.format(args.dataset, args.num_layers)))
        net.eval()
        test_logits = []
        with torch.no_grad():
            test_neigh, test_label = dl.get_test_neigh(
                edge_types=[test_edge_type])
            test_neigh = test_neigh[test_edge_type]
            test_label = test_label[test_edge_type]
            left = np.array(test_neigh[0])
            right = np.array(test_neigh[1])
            mid = np.zeros(left.shape[0], dtype=np.int32)
            labels = torch.FloatTensor(test_label).to(device)
            embed = net(g, features_list, e_feat, edge_norm)
            triplets = [[i[0], i[1], i[2]]for i in zip(left, mid, right)]
            logits = net.calc_score(embed, torch.LongTensor(triplets))
            pred = torch.sigmoid(logits).cpu().numpy()
            edge_list = np.concatenate(
                [left.reshape((1, -1)), right.reshape((1, -1))], axis=0)
            labels = labels.cpu().numpy()
            print(dl.evaluate(edge_list, pred, labels))
        with torch.no_grad():
            test_neigh, test_label = dl.get_test_neigh_w_random(
                edge_types=[test_edge_type])
            test_neigh = test_neigh[test_edge_type]
            test_label = test_label[test_edge_type]
            left = np.array(test_neigh[0])
            right = np.array(test_neigh[1])
            mid = np.zeros(left.shape[0], dtype=np.int32)
            labels = torch.FloatTensor(test_label).to(device)
            embed = net(g, features_list, e_feat, edge_norm)
            triplets = [[i[0], i[1], i[2]]for i in zip(left, mid, right)]
            logits = net.calc_score(embed, torch.LongTensor(triplets))
            pred = torch.sigmoid(logits).cpu().numpy()
            edge_list = np.concatenate(
                [left.reshape((1, -1)), right.reshape((1, -1))], axis=0)
            labels = labels.cpu().numpy()
            print(dl.evaluate(edge_list, pred, labels))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='RGCN testing')
    ap.add_argument('--feats-type', type=int, default=3,
                    help='Type of the node features used. ' +
                         '0 - loaded features; ' +
                         '1 - only target node features (zero vec for others); ' +
                         '2 - only target node features (id vec for others); ' +
                         '3 - all id vec. Default is 2;' +
                    '4 - only term features (id vec for others);' +
                    '5 - only term features (zero vec for others).')
    ap.add_argument('--hidden-dim', type=int, default=64,
                    help='Dimension of the node hidden state. Default is 64.')
    ap.add_argument('--epoch', type=int, default=300, help='Number of epochs.')
    ap.add_argument('--patience', type=int, default=30, help='Patience.')
    ap.add_argument('--num-layers', type=int, default=2)
    ap.add_argument('--lr', type=float, default=5e-4)
    ap.add_argument('--dropout', type=float, default=0.5)
    ap.add_argument('--weight-decay', type=float, default=1e-4)
    ap.add_argument('--slope', type=float, default=0.05)
    ap.add_argument('--dataset', type=str)
    ap.add_argument('--edge-feats', type=int, default=64)
    ap.add_argument('--batch-size', type=int, default=1024)
    ap.add_argument('--gpu', type=int, default=-1)
    ap.add_argument('--n-bases', type=int, default=10)
    ap.add_argument("--regularization", type=float, default=0.01,
                    help="regularization weight")

    args = ap.parse_args()
    run(args)
