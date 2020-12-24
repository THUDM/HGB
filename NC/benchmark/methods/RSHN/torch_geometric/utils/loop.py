import torch

from .num_nodes import maybe_num_nodes


def contains_self_loops(edge_index):
    row, col = edge_index
    mask = row == col
    return mask.sum().item() > 0


def remove_self_loops(edge_index, edge_attr=None):
    row, col = edge_index
    mask = row != col
    edge_attr = edge_attr if edge_attr is None else edge_attr[mask]
    mask = mask.unsqueeze(0).expand_as(edge_index)
    edge_index = edge_index[mask].view(2, -1)

    return edge_index, edge_attr


def add_self_loops(edge_index, num_nodes=None):
    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    dtype, device = edge_index.dtype, edge_index.device
    loop = torch.arange(0, num_nodes, dtype=dtype, device=device)
    loop = loop.unsqueeze(0).repeat(2, 1)
    edge_index = torch.cat([edge_index, loop], dim=1)

    return edge_index

def add_self_edge_attr_loops(edge_attr, num_nodes=None):
    dtype, device = edge_attr.dtype, edge_attr.device
    loop = torch.ones(num_nodes, dtype=dtype, device=device)
    edge_attr = torch.cat([edge_attr, loop], dim=0)

    return edge_attr