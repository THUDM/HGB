import torch
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing

from ..inits import reset, uniform


class NNConv(MessagePassing):
    def __init__(self,
                 in_channels,
                 out_channels,
                 nn,
                 aggr="add",
                 root_weight=False,
                 bias=False):
        super(NNConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.nn = nn
        self.aggr = aggr
        self.weight = Parameter(torch.Tensor(in_channels, out_channels))

        if root_weight:
            self.root = Parameter(torch.Tensor(in_channels, out_channels))
        else:
            self.register_parameter('root', None)

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        reset(self.nn)
        size = self.in_channels
        uniform(size, self.weight)
        uniform(size, self.root)
        uniform(size, self.bias)

    def forward(self, x, edge_index, pseudo):
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        edge_weight = pseudo.unsqueeze(-1) if pseudo.dim() == 1 else pseudo
        edge_weight = self.nn(edge_weight).view(-1, self.out_channels)

        x = torch.matmul(x, self.weight)
        return self.propagate(self.aggr, edge_index, x=x, edge_weight=edge_weight)


    def message(self, x_j, edge_weight):
        message = x_j - edge_weight
        return message

    def update(self, aggr_out, x):
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out + x

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)
