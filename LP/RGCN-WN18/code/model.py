import torch as th
import torch.nn as nn
from dgl.nn.pytorch import GraphConv, GATConv, RelGraphConv
import torch.nn.functional as F


class BaseRGCN(nn.Module):
    def __init__(self, in_dims, h_dim, out_dim, num_rels, num_bases,
                 num_hidden_layers=1, dropout=0,
                 use_self_loop=False):
        super(BaseRGCN, self).__init__()
        self.in_dims = in_dims
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.num_rels = num_rels
        self.num_bases = None if num_bases < 0 else num_bases
        self.num_hidden_layers = num_hidden_layers
        self.dropout = dropout
        self.use_self_loop = use_self_loop

        # create rgcn layers
        self.build_model()

    def build_model(self):
        self.layers = nn.ModuleList()
        # i2h
        self.i2h = self.build_input_layer()
        # h2h
        for idx in range(self.num_hidden_layers):
            h2h = self.build_hidden_layer(idx)
            self.layers.append(h2h)
        # h2o
        h2o = self.build_output_layer()
        if h2o is not None:
            self.layers.append(h2o)

    def build_input_layer(self):
        return None

    def build_hidden_layer(self, idx):
        raise NotImplementedError

    def build_output_layer(self):
        return None

    def forward(self, g, features_list, r, norm):
        h = []
        for i2h, feature in zip(self.i2h, features_list):
            h.append(i2h(feature))
        h = th.cat(h, 0)
        for layer in self.layers:
            h = layer(g, h, r, norm)
        return h


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
        self.w_relation = nn.Parameter(th.Tensor(num_rels, h_dim))
        nn.init.xavier_uniform_(self.w_relation,
                                gain=nn.init.calculate_gain('relu'))

    def calc_score(self, embedding, triplets):
        # DistMult
        s = embedding[triplets[:, 0]]
        r = self.w_relation[triplets[:, 1]]
        o = embedding[triplets[:, 2]]
        score = th.sum(s * r * o, dim=1)
        return score

    def forward(self, g, h, r, norm):
        return self.rgcn.forward(g, h, r, norm)

    def regularization_loss(self, embedding):
        return th.mean(embedding.pow(2)) + th.mean(self.w_relation.pow(2))

    def get_loss(self, embed, triplets, labels):
        # triplets is a list of data samples (positive and negative)
        # each row in the triplets is a 3-tuple of (source, relation, destination)
        score = self.calc_score(embed, triplets)
        logp = th.sigmoid(score)
        predict_loss = F.binary_cross_entropy_with_logits(logp, labels)
        reg_loss = self.regularization_loss(embed)
        return predict_loss + self.reg_param * reg_loss


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
        self.fc_list = nn.ModuleList(
            [nn.Linear(feats_dim, hid_feats, bias=True) for feats_dim in in_feats])
        for fc in self.fc_list:
            nn.init.xavier_normal_(fc.weight, gain=1.414)
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(GraphConv(hid_feats, hid_feats))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(GraphConv(hid_feats, hid_feats))
        self.dropout = nn.Dropout(p=dropout)
        for layer in self.layers:
            nn.init.xavier_normal_(layer.weight, gain=1.4)
        if decoder == 'dismult':
            self.decode = DisMult(rel_num=rel_num, dim=hid_feats)
        elif decoder == 'dot':
            self.decode = Dot()
        else:
            print(decoder)
            exit(0)

    def encode(self, g, x):
        h = []
        for fc, feature in zip(self.fc_list, x):
            h.append(fc(feature))
        x = th.cat(h, 0)
        for i, layer in enumerate(self.layers):
            if i != 0:
                x = self.dropout(x)
            x = layer(g, x)
            if i < len(self.layers) - 1:
                x = F.elu(x)
        return x


class GAT(th.nn.Module):
    def __init__(self, in_feats, hid_feats, n_layers=2, dropout=0.5, heads=[1], decoder="dot", rel_num=1):
        super(GAT, self).__init__()
        self.fc_list = nn.ModuleList(
            [nn.Linear(feats_dim, hid_feats, bias=True) for feats_dim in in_feats])
        for fc in self.fc_list:
            nn.init.xavier_normal_(fc.weight, gain=1.414)
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(GATConv(hid_feats, hid_feats, heads[0]))
        # hidden layers
        for l in range(1, n_layers - 1):
            self.layers.append(
                GATConv(hid_feats * heads[l - 1], hid_feats, heads[l]))
        # output layer
        self.layers.append(
            GATConv(hid_feats * heads[-2], hid_feats, heads[-1]))
        self.dropout = nn.Dropout(p=dropout)
        for layer in self.layers:
            nn.init.xavier_normal_(layer.lin_l.weight, gain=1.4)
        if decoder == 'dismult':
            self.decode = DisMult(rel_num=rel_num, dim=hid_feats)
        elif decoder == 'dot':
            self.decode = Dot()

    def encode(self, g, x):
        h = []
        for fc, feature in zip(self.fc_list, x):
            h.append(fc(feature))
        x = th.cat(h, 0)
        for i, layer in enumerate(self.layers):
            if i != 0:
                x = self.dropout(x)
            x = layer(g, x)
            if i < len(self.layers) - 1:
                x = F.elu(x)
        return x
