import torch
import torch.nn as nn
import dgl
from dgl.nn.pytorch import GraphConv

import dgl.function as fn
from dgl.nn.pytorch import edge_softmax, GATConv
from conv import myGATConv

class myGAT(nn.Module):
    def __init__(self,
                 g,
                 edge_dim,
                 num_etypes,
                 in_dims,
                 num_hidden,
                 num_classes,
                 num_layers,
                 heads,
                 activation,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 residual,
                 alpha):
        super(myGAT, self).__init__()
        self.g = g
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        self.fc_list = nn.ModuleList([nn.Linear(in_dim, num_hidden, bias=True) for in_dim in in_dims])
        for fc in self.fc_list:
            nn.init.xavier_normal_(fc.weight, gain=1.414)
        # input projection (no residual)
        self.gat_layers.append(myGATConv(edge_dim, num_etypes,
            num_hidden, num_hidden, heads[0],
            feat_drop, attn_drop, negative_slope, False, self.activation, alpha=alpha))
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(myGATConv(edge_dim, num_etypes,
                num_hidden * heads[l-1], num_hidden, heads[l],
                feat_drop, attn_drop, negative_slope, residual, self.activation, alpha=alpha))
        # output projection
        self.gat_layers.append(myGATConv(edge_dim, num_etypes,
            num_hidden * heads[-2], num_classes, heads[-1],
            feat_drop, attn_drop, negative_slope, residual, None, alpha=alpha))
        self.epsilon = torch.FloatTensor([1e-12]).cuda()

    def forward(self, features_list, e_feat):
        h = []
        for fc, feature in zip(self.fc_list, features_list):
            h.append(fc(feature))
        h = torch.cat(h, 0)
        res_attn = None
        for l in range(self.num_layers):
            h, res_attn = self.gat_layers[l](self.g, h, e_feat, res_attn=res_attn)
            h = h.flatten(1)
        # output projection
        logits, _ = self.gat_layers[-1](self.g, h, e_feat, res_attn=None)
        logits = logits.mean(1)
        # This is an equivalent replacement for tf.l2_normalize, see https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/math/l2_normalize for more information.
        logits = logits / (torch.max(torch.norm(logits, dim=1, keepdim=True), self.epsilon))
        return logits

class RGAT(nn.Module):
    def __init__(self,
                 gs,
                 in_dims,
                 num_hidden,
                 num_classes,
                 num_layers,
                 heads,
                 activation,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 residual):
        super(GAT, self).__init__()
        self.gs = gs
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList([nn.ModuleList() for i in range(len(gs))])
        self.activation = activation
        self.weights = nn.Parameter(torch.zeros((len(in_dims), num_layers+1, len(gs))))
        self.sm = nn.Softmax(2)
        self.fc_list = nn.ModuleList([nn.Linear(in_dim, num_hidden, bias=True) for in_dim in in_dims])
        for fc in self.fc_list:
            nn.init.xavier_normal_(fc.weight, gain=1.414)
        for i in range(len(gs)):
            # input projection (no residual)
            self.gat_layers[i].append(GATConv(
                num_hidden, num_hidden, heads[0],
                feat_drop, attn_drop, negative_slope, False, self.activation))
            # hidden layers
            for l in range(1, num_layers):
                # due to multi-head, the in_dim = num_hidden * num_heads
                self.gat_layers[i].append(GATConv(
                    num_hidden * heads[l-1], num_hidden, heads[l],
                    feat_drop, attn_drop, negative_slope, residual, self.activation))
            # output projection
            self.gat_layers[i].append(GATConv(
                num_hidden * heads[-2], num_classes, heads[-1],
                feat_drop, attn_drop, negative_slope, residual, None))

    def forward(self, features_list):
        nums = [feat.size(0) for feat in features_list]
        weights = self.sm(self.weights)
        h = []
        for fc, feature in zip(self.fc_list, features_list):
            h.append(fc(feature))
        h = torch.cat(h, 0)
        for l in range(self.num_layers):
            out = []
            for i in range(len(self.gs)):
                out.append(torch.split(self.gat_layers[i][l](self.gs[i], h).flatten(1), nums))
            h = []
            for k in range(len(nums)):
                tmp = []
                for i in range(len(self.gs)):
                    tmp.append(out[i][k]*weights[k,l,i])
                h.append(sum(tmp))
            h = torch.cat(h, 0)
        out = []
        for i in range(len(self.gs)):
            out.append(torch.split(self.gat_layers[i][-1](self.gs[i], h).mean(1), nums))
        logits = []
        for k in range(len(nums)):
            tmp = []
            for i in range(len(self.gs)):
                tmp.append(out[i][k]*weights[k,-1,i])
            logits.append(sum(tmp))
        logits = torch.cat(logits, 0)
        return logits

class GAT(nn.Module):
    def __init__(self,
                 g,
                 in_dims,
                 num_hidden,
                 num_classes,
                 num_layers,
                 heads,
                 activation,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 residual):
        super(GAT, self).__init__()
        self.g = g
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        self.fc_list = nn.ModuleList([nn.Linear(in_dim, num_hidden, bias=True) for in_dim in in_dims])
        for fc in self.fc_list:
            nn.init.xavier_normal_(fc.weight, gain=1.414)
        # input projection (no residual)
        self.gat_layers.append(GATConv(
            num_hidden, num_hidden, heads[0],
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

    def forward(self, features_list):
        h = []
        for fc, feature in zip(self.fc_list, features_list):
            h.append(fc(feature))
        h = torch.cat(h, 0)
        for l in range(self.num_layers):
            h = self.gat_layers[l](self.g, h).flatten(1)
        # output projection
        logits = self.gat_layers[-1](self.g, h).mean(1)
        return logits


class GCN(nn.Module):
    def __init__(self,
                 g,
                 in_dims,
                 num_hidden,
                 num_classes,
                 num_layers,
                 activation,
                 dropout):
        super(GCN, self).__init__()
        self.g = g
        self.layers = nn.ModuleList()
        self.fc_list = nn.ModuleList([nn.Linear(in_dim, num_hidden, bias=True) for in_dim in in_dims])
        for fc in self.fc_list:
            nn.init.xavier_normal_(fc.weight, gain=1.414)
        # input layer
        self.layers.append(GraphConv(num_hidden, num_hidden, activation=activation, weight=False))
        # hidden layers
        for i in range(num_layers - 1):
            self.layers.append(GraphConv(num_hidden, num_hidden, activation=activation))
        # output layer
        self.layers.append(GraphConv(num_hidden, num_classes))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, features_list):
        h = []
        for fc, feature in zip(self.fc_list, features_list):
            h.append(fc(feature))
        h = torch.cat(h, 0)
        for i, layer in enumerate(self.layers):
            h = self.dropout(h)
            h = layer(self.g, h)
        return h
