import torch as th
from torch import nn
from torch_geometric.nn import GCNConv, GATConv
import torch.nn.functional as F
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
