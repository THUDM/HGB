import sys
sys.path.insert(0, '../')
import os.path as osp

import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear
from torch_geometric.datasets import Entities
from torch_geometric.nn import NNConv, RelationConv
from torch_geometric.utils import remove_self_loops
from build_coarsened_line_graph import relation_graph
import argparse

parser = argparse.ArgumentParser(description='RSHN')
parser.add_argument('--dataset', type=str, default='AIFB')
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--dim', type=int, default=16)
parser.add_argument('--num_node_layer', type=int, default=2)
parser.add_argument('--num_edge_layer', type=int, default=1)
parser.add_argument('--dropout', type=float, default=0.6)
parser.add_argument('--epoch', type=int, default=50)
parser.add_argument('--has_CLG', type=bool, default=True)
parser.add_argument('--seed', type=int, default=1233)
args = parser.parse_args()

# Set random seed
torch.manual_seed(args.seed)
name = args.dataset
path = osp.join(
    osp.dirname(osp.realpath(__file__)), '..', 'data', 'Entities', name)
dataset = Entities(path, name)
data = dataset[0]
data.edge_index, data.edge_type = remove_self_loops(data.edge_index, data.edge_type)


def build_x():
    num_nodes = data.num_nodes
    if name == 'AM' or name == 'BGS':
        data.x = torch.sparse_coo_tensor(torch.arange(0, num_nodes).repeat(2, 1), torch.ones(num_nodes))
    else:
        data.x = torch.eye(num_nodes)


if args.has_CLG is not True:
    relation_graph.build_coarsened_line_graph(dataset, rw_len=4, batch_size=5000, name=name)
rel_data = relation_graph.load_rel_graph(name)

build_x()
dim = args.dim
p = args.dropout


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        nn = Sequential(Linear(rel_data.num_features, dim))
        self.fc1 = torch.nn.Parameter(torch.FloatTensor(data.num_features, dim))
        self.NNConv1 = NNConv(dim, dim, nn, root_weight=False)
        if args.num_node_layer == 2:
            self.NNConv2 = NNConv(dim, dim, nn, root_weight=False)
        self.fc2 = Linear(dim, dataset.num_classes)

        self.RConv1 = RelationConv(train_eps=False)
        if args.num_edge_layer == 2:
            self.RConv2 = RelationConv(train_eps=False)

    def forward(self, data, rel_data):
        x = torch.spmm(data.x, self.fc1)
        x = F.dropout(x, p=p, training=False)
        rel_x = rel_data.x

        # learn relation embedding
        rel_x = torch.relu(self.RConv1(rel_x, rel_data.edge_index, rel_data.edge_attr))
        rel_x = F.dropout(rel_x, p=p, training=False)
        if args.num_edge_layer == 2:
            rel_x = torch.relu(self.RConv2(rel_x, rel_data.edge_index, rel_data.edge_attr))
            rel_x = F.dropout(rel_x, p=p, training=False)
        edge_attr = F.embedding(data.edge_type, rel_x)

        # learn node embedding
        x = torch.tanh(self.NNConv1(x, data.edge_index, edge_attr))
        x = F.dropout(x, p=p, training=False)
        if args.num_node_layer == 2:
            x = torch.tanh(self.NNConv2(x, data.edge_index, edge_attr))
            x = F.dropout(x, p=p, training=False)

        x = self.fc2(x)
        x = F.dropout(x, p=p, training=False)

        return F.log_softmax(x, dim=1)


if name == 'AM':
    device = 'cpu'
else:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, data = Net().to(device), data.to(device)
rel_data = rel_data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


def train():
    model.train()
    optimizer.zero_grad()
    out = model(data, rel_data)
    loss = F.nll_loss(out[data.train_idx], data.train_y)
    loss.backward()
    optimizer.step()
    return loss


def test():
    model.eval()
    out = model(data, rel_data)
    accs = []
    pred = out[data.train_idx].max(1)[1]
    acc = pred.eq(data.train_y).sum().item() / data.train_y.size(0)
    accs.append(acc)
    pred = out[data.test_idx].max(1)[1]
    acc = pred.eq(data.test_y).sum().item() / data.test_y.size(0)
    accs.append(acc)
    return accs


for epoch in range(1, args.epoch):
    l = train()
    train_acc, test_acc = test()
    log = 'Epoch: {:03d}, loss: {:.4f}, Train: {:.4f}, Test: {:.4f}'
    print(log.format(epoch, l.item(), train_acc, test_acc))

print(args)
