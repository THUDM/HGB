import numpy as np
import torch
import torch.nn as nn
import dgl
from dgl.nn.pytorch import GraphConv

import dgl.function as fn
from dgl.nn.pytorch import edge_softmax, GATConv
from conv import myGATConv

class myGAT(nn.Module):
    def __init__(self,
                 num_entity,
                 edge_dim,
                 num_etypes,
                 in_dim,
                 num_hidden,
                 num_classes,
                 num_layers,
                 heads,
                 activation,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 residual,
                 pretrain=None,
                 alpha=0.):
        super(myGAT, self).__init__()
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        self.embed = nn.Parameter(torch.zeros((num_entity, in_dim)))
        nn.init.xavier_normal_(self.embed, gain=1.414)
        if pretrain is not None:
            user_embed = pretrain['user_embed']
            item_embed = pretrain['item_embed']
            user = user_embed.shape[0]
            item = item_embed.shape[0]
            self.ret_num = user+item
            self.ini = torch.FloatTensor(np.concatenate([user_embed, item_embed], axis=0)).cuda()
            """self.embed.data[:user] = nn.Parameter(torch.tensor(user_embed))
            self.embed.data[user:user+item] = nn.Parameter(torch.tensor(item_embed))
            left = nn.Parameter(torch.zeros((num_entity-item-user, in_dim)))
            nn.init.xavier_normal_(left, gain=1.414)
            self.embed.data[user+item:] = left"""
        # input projection (no residual)
        self.gat_layers.append(myGATConv(edge_dim, num_etypes,
            in_dim, num_hidden, heads[0],
            feat_drop, attn_drop, negative_slope, False, self.activation, bias=True, alpha=alpha))
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(myGATConv(edge_dim, num_etypes,
                num_hidden * heads[l-1],
                 num_hidden, heads[l],
                feat_drop, attn_drop, negative_slope, residual, self.activation, bias=True, alpha=alpha))
        # output projection
        self.gat_layers.append(myGATConv(edge_dim, num_etypes,
            num_hidden * heads[-2],
             num_classes, heads[-1],
            feat_drop, attn_drop, negative_slope, residual, None, bias=True, alpha=alpha))
        self.epsilon = torch.FloatTensor([1e-12]).cuda()

    def forward(self, g, e_feat):
        all_embed = []
        h = self.embed
        tmp = (h / (torch.max(torch.norm(h, dim=1, keepdim=True),self.epsilon)))
        all_embed.append(tmp)
        res_attn = None
        for l in range(self.num_layers):
            h, res_attn = self.gat_layers[l](g, h, e_feat, res_attn=res_attn)
            h = h.flatten(1)
            tmp = (h / (torch.max(torch.norm(h, dim=1, keepdim=True),self.epsilon)))
            all_embed.append(tmp)
        # output projection
        logits, _ = self.gat_layers[-1](g, h, e_feat, res_attn=res_attn)
        logits = logits.mean(1)
        all_embed.append(logits / (torch.max(torch.norm(logits, dim=1, keepdim=True),self.epsilon)))
        all_embed = torch.cat(all_embed, 1)
        return torch.cat([all_embed[:self.ret_num], self.ini], 1)

