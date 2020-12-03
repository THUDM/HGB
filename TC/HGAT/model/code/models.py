import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import *
from torch.nn.parameter import Parameter
from dgl.nn.pytorch import GraphConv, GATConv
from functools import reduce
from utils import dense_tensor_to_sparse


class weighted_GCN(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout,
                 sparse_input=False):
        super(weighted_GCN, self).__init__()
        self.sparse_input = sparse_input
        self.activation = activation
        if self.sparse_input:
            self.linear = nn.Linear(in_feats, n_hidden)
            in_feats = n_hidden
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(
            MyGraphConvolution(in_feats, n_hidden))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(
                MyGraphConvolution(n_hidden, n_hidden))
        # output layer
        self.layers.append(MyGraphConvolution(n_hidden, n_classes))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, features, adj_metrix):
        if self.sparse_input:
            h = self.linear(features)
        else:
            h = features
        h = self.activation(h)
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(h, adj_metrix)
            h = self.activation(h)
        h = F.log_softmax(h, dim=1)
        return h


class GCN(nn.Module):
    def __init__(self,
                 g,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout,
                 sparse_input=False):
        super(GCN, self).__init__()
        self.g = g
        self.sparse_input = sparse_input
        if self.sparse_input:
            self.linear = nn.Linear(in_feats, n_hidden)
            in_feats = n_hidden
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(
            GraphConv(in_feats, n_hidden, activation=activation))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(
                GraphConv(n_hidden, n_hidden, activation=activation))
        # output layer
        self.layers.append(GraphConv(n_hidden, n_classes))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, features, adj_metrix):
        if self.sparse_input:
            h = self.linear(features)
        else:
            h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(self.g, h)
        h = F.log_softmax(h, dim=1)
        return h


class GAT(nn.Module):
    def __init__(self,
                 g,
                 in_dim,
                 num_hidden,
                 num_classes,
                 num_layers,
                 activation,
                 feat_drop,
                 attn_drop,
                 heads,
                 negative_slope,
                 residual=False,
                 sparse_input=False):
        super(GAT, self).__init__()
        self.g = g
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        self.sparse_input = sparse_input
        if self.sparse_input:
            self.linear = nn.Linear(in_dim, num_hidden)
            in_dim = num_hidden
        # input projection (no residual)
        self.gat_layers.append(GATConv(
            in_dim, num_hidden, heads[0],
            feat_drop, attn_drop, negative_slope, False, self.activation))
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(GATConv(
                num_hidden * heads[l-1], num_hidden, heads[l],
                feat_drop, attn_drop, negative_slope, residual, self.activation))
        # output projection
        self.gat_layers.append(GATConv(
            num_hidden * heads[-2], num_classes, heads[-1],
            feat_drop, attn_drop, negative_slope, residual, None))

    def forward(self, inputs, empty):
        if self.sparse_input:
            h = self.linear(inputs)
        else:
            h = inputs
        for l in range(self.num_layers):
            h = self.gat_layers[l](self.g, h).flatten(1)
        # output projection
        logits = self.gat_layers[-1](self.g, h).mean(1)
        logits = F.log_softmax(logits, dim=1)
        return logits


class HGAT(nn.Module):
    def __init__(self, nfeat_list, nhid, nclass, dropout, type_attention=True, node_attention=True, gamma=0.1, sigmoid=False, orphan=True):
        super(HGAT, self).__init__()
        self.sigmoid = sigmoid
        self.type_attention = type_attention
        self.node_attention = node_attention

        self.write_emb = True
        if self.write_emb:
            self.emb = None
            self.emb2 = None

        self.nonlinear = F.relu_

        self.nclass = nclass
        self.ntype = len(nfeat_list)

        dim_1st = nhid
        dim_2nd = nclass
        if orphan:
            dim_2nd += self.ntype - 1

        self.gc2 = nn.ModuleList()
        if not self.node_attention:
            self.gc1 = nn.ModuleList()
            for t in range(self.ntype):
                self.gc1.append(GraphConvolution(
                    nfeat_list[t], dim_1st, bias=False))
                self.bias1 = Parameter(torch.FloatTensor(dim_1st))
                stdv = 1. / math.sqrt(dim_1st)
                self.bias1.data.uniform_(-stdv, stdv)
        else:
            self.gc1 = GraphAttentionConvolution(
                nfeat_list, dim_1st, gamma=gamma)
        self.gc2.append(GraphConvolution(dim_1st, dim_2nd, bias=True))

        if self.type_attention:
            self.at1 = nn.ModuleList()
            self.at2 = nn.ModuleList()
            for t in range(self.ntype):
                self.at1.append(SelfAttention(dim_1st, t, 50))
                self.at2.append(SelfAttention(dim_2nd, t, 50))

        self.dropout = dropout

    def forward(self, x_list, adj_list, adj_all=None):
        x0 = x_list

        if not self.node_attention:
            x1 = [None for _ in range(self.ntype)]
            # First Layer
            for t1 in range(self.ntype):
                x_t1 = []
                for t2 in range(self.ntype):
                    idx = t2
                    x_t1.append(self.gc1[idx](
                        x0[t2], adj_list[t1][t2]) + self.bias1)
                if self.type_attention:
                    x_t1, weights = self.at1[t1](torch.stack(x_t1, dim=1))
                else:
                    x_t1 = reduce(torch.add, x_t1)

                x_t1 = self.nonlinear(x_t1)
                x_t1 = F.dropout(x_t1, self.dropout, training=self.training)
                x1[t1] = x_t1
        else:
            x1 = [None for _ in range(self.ntype)]
            x1_in = self.gc1(x0, adj_list)
            for t1 in range(len(x1_in)):
                x_t1 = x1_in[t1]
                if self.type_attention:
                    x_t1, weights = self.at1[t1](torch.stack(x_t1, dim=1))
                else:
                    x_t1 = reduce(torch.add, x_t1)
                x_t1 = self.nonlinear(x_t1)
                x_t1 = F.dropout(x_t1, self.dropout, training=self.training)
                x1[t1] = x_t1
        if self.write_emb:
            self.emb = x1[0]

        x2 = [None for _ in range(self.ntype)]
        # Second Layer
        for t1 in range(self.ntype):
            x_t1 = []
            for t2 in range(self.ntype):
                if adj_list[t1][t2] is None:
                    continue
                idx = 0
                x_t1.append(self.gc2[idx](x1[t2], adj_list[t1][t2]))
            if self.type_attention:
                x_t1, weights = self.at2[t1](torch.stack(x_t1, dim=1))
            else:
                x_t1 = reduce(torch.add, x_t1)

            x2[t1] = x_t1
            if self.write_emb and t1 == 0:
                self.emb2 = x2[t1]

            # output layer
            if self.sigmoid:
                x2[t1] = torch.sigmoid(x_t1)
            else:
                x2[t1] = F.log_softmax(x_t1, dim=1)

        return x2
