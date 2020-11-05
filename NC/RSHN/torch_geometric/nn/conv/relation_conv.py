import torch
from torch.nn import Parameter
import torch.nn.functional as F
from torch_sparse import spmm
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax, add_self_edge_attr_loops


class RelationConv(torch.nn.Module):

    def __init__(self, eps=0, train_eps=False, requires_grad=True):
        super(RelationConv, self).__init__()

        self.initial_eps = eps

        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))

        '''beta'''
        self.requires_grad = requires_grad
        if requires_grad:
            self.beta = Parameter(torch.Tensor(1))
        else:
            self.register_buffer('beta', torch.ones(1))

        self.reset_parameters()

    def reset_parameters(self):
        self.eps.data.fill_(self.initial_eps)
        if self.requires_grad:
            self.beta.data.fill_(1)

    def forward(self, x, edge_index, edge_attr):
        """"""
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
        row, col = edge_index

        '''co-occurrence rate'''
        for i in range(len(x)):
            mask = torch.eq(row, i)
            edge_attr[mask] = F.normalize(edge_attr[mask], p=2, dim=0)

        '''add-self-loops'''
        edge_index = add_self_loops(edge_index, x.size(0))
        row, col = edge_index
        edge_attr = add_self_edge_attr_loops(edge_attr, x.size(0))

        x = F.normalize(x, p=2, dim=-1)
        beta = self.beta if self.requires_grad else self._buffers['beta']
        alpha = beta * edge_attr
        alpha = softmax(alpha, row, num_nodes=x.size(0))

        '''Perform the propagation.'''
        out = spmm(edge_index, alpha, x.size(0), x.size(1), x)
        out = (1 + self.eps) * x + out
        return out

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)
