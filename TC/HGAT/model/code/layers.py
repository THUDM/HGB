import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F
from torch import nn

class GraphConvolution(Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, inputs, adj, global_W = None):
        if len(adj._values()) == 0:
            return torch.zeros(adj.shape[0], self.out_features, device=inputs.device)

        support = torch.spmm(inputs, self.weight)
        if global_W is not None:
            support = torch.spmm(support, global_W)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class SelfAttention_ori(Module):
    def __init__(self, in_features):
        super(SelfAttention, self).__init__()
        self.a = Parameter(torch.FloatTensor(2 * in_features, 1))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.a.size(1))
        self.a.data.uniform_(-stdv, stdv)

    def forward(self, inputs):
        x = inputs.transpose(0, 1)
        self.n = x.size()[0]
        x = torch.cat([x, torch.stack([x] * self.n, dim=0)], dim=2)
        U = torch.matmul(x, self.a).transpose(0, 1)
        U = F.leaky_relu(U)
        weights = F.softmax(U, dim=1)
        outputs = torch.matmul(weights.transpose(1, 2), inputs).squeeze(1)
        return outputs, weights


class SelfAttention(Module):
    def __init__(self, in_features, idx, hidden_dim):
        super(SelfAttention, self).__init__()
        self.idx = idx
        self.linear = torch.nn.Linear(in_features, hidden_dim)
        self.a = Parameter(torch.FloatTensor(2 * hidden_dim, 1))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.a.size(1))
        self.a.data.uniform_(-stdv, stdv)

    def forward(self, inputs):
        # inputs size:  node_num * 3 * in_features
        x = self.linear(inputs).transpose(0, 1)
        self.n = x.size()[0]
        x = torch.cat([x, torch.stack([x[self.idx]] * self.n, dim=0)], dim=2)
        U = torch.matmul(x, self.a).transpose(0, 1)
        U = F.leaky_relu_(U)
        weights = F.softmax(U, dim=1)
        outputs = torch.matmul(weights.transpose(1, 2), inputs).squeeze(1) * 3
        return outputs, weights


class GraphAttentionConvolution(Module):
    def __init__(self, in_features_list, out_features, bias=True, gamma = 0.1):
        super(GraphAttentionConvolution, self).__init__()
        self.ntype = len(in_features_list)
        self.in_features_list = in_features_list
        self.out_features = out_features
        self.weights = nn.ParameterList()
        for i in range(self.ntype):
            cache = Parameter(torch.FloatTensor(in_features_list[i], out_features))
            nn.init.xavier_normal_(cache.data, gain=1.414)
            self.weights.append( cache )
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
            stdv = 1. / math.sqrt(out_features)
            self.bias.data.uniform_(-stdv, stdv)
        else:
            self.register_parameter('bias', None)
        
        self.att_list = nn.ModuleList()
        for i in range(self.ntype):
            self.att_list.append( Attention_NodeLevel(out_features, gamma) )


    def forward(self, inputs_list, adj_list, global_W = None):
        h = []
        for i in range(self.ntype):
            h.append( torch.spmm(inputs_list[i], self.weights[i]) )
        if global_W is not None:
            for i in range(self.ntype):
                h[i] = ( torch.spmm(h[i], global_W) )
        outputs = []
        for t1 in range(self.ntype):
            x_t1 = []
            for t2 in range(self.ntype):
                # adj has no non-zeros
                if len(adj_list[t1][t2]._values()) == 0:
                    x_t1.append( torch.zeros(adj_list[t1][t2].shape[0], self.out_features, device=self.bias.device) )
                    continue

                if self.bias is not None:
                    x_t1.append( self.att_list[t1](h[t1], h[t2], adj_list[t1][t2]) + self.bias )
                else:
                    x_t1.append( self.att_list[t1](h[t1], h[t2], adj_list[t1][t2]) )
            outputs.append(x_t1)
            
        return outputs


        
     

class Attention_NodeLevel(nn.Module):
    def __init__(self, dim_features, gamma = 0.1):
        super(Attention_NodeLevel, self).__init__()

        self.dim_features = dim_features

        self.a1 = nn.Parameter(torch.zeros(size=(dim_features, 1)))
        self.a2 = nn.Parameter(torch.zeros(size=(dim_features, 1)))
        nn.init.xavier_normal_(self.a1.data, gain=1.414)
        nn.init.xavier_normal_(self.a2.data, gain=1.414)        

        self.leakyrelu = nn.LeakyReLU(0.2)
        self.gamma = gamma

    def forward(self, input1, input2, adj):
        h = input1
        g = input2
        N = h.size()[0]
        M = g.size()[0]

        e1 = torch.matmul(h, self.a1).repeat(1, M)
        e2 = torch.matmul(g, self.a2).repeat(1, N).t()
        e = e1 + e2  
        e = self.leakyrelu(e)
        
        zero_vec = -9e15*torch.ones_like(e)
        if 'sparse' in adj.type():
            adj_dense = adj.to_dense()
            attention = torch.where(adj_dense > 0, e, zero_vec)
            attention = F.softmax(attention, dim=1)
            attention = torch.mul(attention, adj_dense.sum(1).repeat(M, 1).t())
            attention = torch.add(attention * self.gamma, adj_dense * (1 - self.gamma))
            del(adj_dense)
        else:
            attention = torch.where(adj > 0, e, zero_vec)
            attention = F.softmax(attention, dim=1)
            attention = torch.mul(attention, adj.sum(1).repeat(M, 1).t())
            attention = torch.add(attention * self.gamma, adj.to_dense() * (1 - self.gamma))
        del(zero_vec)

        h_prime = torch.matmul(attention, g)

        return h_prime
    
        
