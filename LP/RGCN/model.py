import torch as th
from torch._C import Graph
import torch.nn as nn
import torch as th
from torch import nn
import torch.nn.functional as F
from dgl.nn.pytorch import GATConv, GraphConv
from torch.nn.modules import activation


class DisMult(th.nn.Module):
    def __init__(self, rel_num, dim):
        super(DisMult, self).__init__()
        self.dim = dim
        self.weights = nn.Parameter(th.FloatTensor(size=(rel_num, dim)))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(
            self.weights, gain=nn.init.calculate_gain('relu'))

    def forward(self, r_list, input1, input2):
        r = self.weights[r_list]
        return th.sum(input1 * r * input2, dim=1)


class GCN(th.nn.Module):
    def __init__(self, in_feats, hid_feats, n_layers=2, dropout=0.5,  rel_num=1):
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
        self.decode = DisMult(rel_num=rel_num, dim=hid_feats)

    def encode(self, data):
        g, feats_list = data
        h = []
        for fc, feature in zip(self.fc_list, feats_list):
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
    def __init__(self, in_feats, hid_feats, n_layers=2, dropout=0.5, heads=[1], rel_num=1):
        super(GAT, self).__init__()
        self.fc_list = nn.ModuleList(
            [nn.Linear(feats_dim, hid_feats, bias=True) for feats_dim in in_feats])
        for fc in self.fc_list:
            nn.init.xavier_normal_(fc.weight, gain=1.414)
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(GATConv(hid_feats, hid_feats,
                                   heads[0], activation=F.elu))
        # hidden layers
        for l in range(1, n_layers - 1):
            self.layers.append(
                GATConv(hid_feats * heads[l - 1], hid_feats, heads[l], activation=F.elu))
        # output layer
        self.layers.append(
            GATConv(hid_feats * heads[-2], hid_feats, heads[-1], activation=None))
        self.dropout = nn.Dropout(p=dropout)
        for layer in self.layers:
            layer.reset_parameters()
        self.decode = DisMult(rel_num=rel_num, dim=hid_feats)

    def encode(self, data):
        g, feats_list = data
        h = []
        for fc, feature in zip(self.fc_list, feats_list):
            h.append(fc(feature))
        x = th.cat(h, 0)
        for i, layer in enumerate(self.layers):
            if i != 0:
                x = self.dropout(x)
            x = layer(g, x).flatten(1)
        return x


class BaseRGCN(nn.Module):
    def __init__(self, num_nodes, h_dim, out_dim, num_rels, num_bases,
                 num_hidden_layers=1, dropout=0,
                 use_self_loop=False, use_cuda=False):
        super(BaseRGCN, self).__init__()
        self.num_nodes = num_nodes
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.num_rels = num_rels
        self.num_bases = None if num_bases < 0 else num_bases
        self.num_hidden_layers = num_hidden_layers
        self.dropout = dropout
        self.use_self_loop = use_self_loop
        self.use_cuda = use_cuda

        # create rgcn layers
        self.build_model()

    def build_model(self):
        self.layers = nn.ModuleList()
        # i2h
        i2h = self.build_input_layer()
        if i2h is not None:
            self.layers.append(i2h)
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

    def forward(self, g, h, r, norm):
        for layer in self.layers:
            h = layer(g, h, r, norm)
        return h


class RelGraphEmbedLayer(nn.Module):
    r"""Embedding layer for featureless heterograph.
    Parameters
    ----------
    dev_id : int
        Device to run the layer.
    num_nodes : int
        Number of nodes.
    node_tides : tensor
        Storing the node type id for each node starting from 0
    num_of_ntype : int
        Number of node types
    input_size : list of int
        A list of input feature size for each node type. If None, we then
        treat certain input feature as an one-hot encoding feature.
    embed_size : int
        Output embed size
    embed_name : str, optional
        Embed name
    """

    def __init__(self,
                 dev_id,
                 num_nodes,
                 node_tids,
                 num_of_ntype,
                 input_size,
                 embed_size,
                 sparse_emb=False,
                 embed_name='embed'):
        super(RelGraphEmbedLayer, self).__init__()
        self.dev_id = th.device(dev_id if dev_id >= 0 else 'cpu')
        self.embed_size = embed_size
        self.embed_name = embed_name
        self.num_nodes = num_nodes
        self.sparse_emb = sparse_emb

        # create weight embeddings for each node for each relation
        self.embeds = nn.ParameterDict()
        self.num_of_ntype = num_of_ntype
        self.idmap = th.empty(num_nodes).long()

        for ntype in range(num_of_ntype):
            if input_size[ntype] is not None:
                input_emb_size = input_size[ntype].shape[1]
                embed = nn.Parameter(
                    th.Tensor(input_emb_size, self.embed_size))
                nn.init.xavier_uniform_(embed)
                self.embeds[str(ntype)] = embed

        self.node_embeds = th.nn.Embedding(
            node_tids.shape[0], self.embed_size, sparse=self.sparse_emb)
        nn.init.uniform_(self.node_embeds.weight, -1.0, 1.0)

    def forward(self, node_ids, node_tids, type_ids, features):
        """Forward computation
        Parameters
        ----------
        node_ids : tensor
            node ids to generate embedding for.
        node_ids : tensor
            node type ids
        features : list of features
            list of initial features for nodes belong to different node type.
            If None, the corresponding features is an one-hot encoding feature,
            else use the features directly as input feature and matmul a
            projection matrix.
        Returns
        -------
        tensor
            embeddings as the input of the next layer
        """
        tsd_ids = node_ids.to(self.node_embeds.weight.device)
        embeds = th.empty(node_ids.shape[0],
                          self.embed_size, device=self.dev_id)
        for ntype in range(self.num_of_ntype):
            if features[ntype] is not None:
                loc = node_tids == ntype
                embeds[loc] = features[ntype][type_ids[loc]].to(
                    self.dev_id) @ self.embeds[str(ntype)].to(self.dev_id)
            else:
                loc = node_tids == ntype
                embeds[loc] = self.node_embeds(tsd_ids[loc]).to(self.dev_id)

        return embeds
