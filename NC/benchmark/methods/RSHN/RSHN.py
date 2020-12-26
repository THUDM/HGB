import sys

import numpy
import time
from numpy.core.shape_base import stack
sys.path.insert(0, '../')
import os.path as osp
from sklearn.metrics import f1_score

import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear
from torch_geometric.datasets import Entities
from torch_geometric.nn import NNConv, RelationConv
from torch_geometric.utils import remove_self_loops
from build_coarsened_line_graph import relation_graph
from torch_geometric.data import  Data
from scipy.sparse import coo, find
import argparse

import sys
sys.path.append('../../')
from scripts.data_loader import data_loader

def evaluate(model_pred, labels):
    pred_result = model_pred.argmax(dim=1)
    labels = labels.cpu()
    pred_result = pred_result.cpu()

    micro = f1_score(labels, pred_result, average='micro')
    macro = f1_score(labels, pred_result, average='macro')

    return micro, macro


def multi_evaluate(model_pred, labels):
    # 注意这里的model_pred是经过sigmoid处理的，sigmoid处理后可以视为预测是这一类的概率
    # 预测结果，大于这个阈值则视为预测正确
    model_pred = F.sigmoid(model_pred)
    accuracy_th = 0.5
    pred_result = model_pred > accuracy_th
    pred_result = pred_result.float()
    labels = labels.cpu()
    pred_result = pred_result.cpu()

    micro = f1_score(labels, pred_result, average='micro')
    macro = f1_score(labels, pred_result, average='macro')

    return micro, macro

parser = argparse.ArgumentParser(description='RSHN')
parser.add_argument('--dataset', type=str, default='IMDB')
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--dim', type=int, default=16)
parser.add_argument('--num_node_layer', type=int, default=2)
parser.add_argument('--num_edge_layer', type=int, default=1)
parser.add_argument('--dropout', type=float, default=0.6)
parser.add_argument('--epoch', type=int, default=50)
parser.add_argument('--has_CLG', action='store_false', default=True)
parser.add_argument('--seed', type=int, default=1233)
parser.add_argument('--testing', action='store_true', default=False)
args = parser.parse_args()

# Set random seed
torch.manual_seed(args.seed)
name = args.dataset
dl = data_loader("./data/"+name)
start = numpy.zeros(shape=0)
end = numpy.zeros(shape=0)
edge_type = numpy.zeros(shape=0)
for i in dl.links['meta']:
    coo_metrix = find(dl.links['data'][i])
    start = numpy.concatenate((start,coo_metrix[0]))
    end = numpy.concatenate((end,coo_metrix[1]))
    edge_type = numpy.concatenate((edge_type,i*numpy.ones(len(coo_metrix[0]))))

data = Data(edge_index=torch.LongTensor(numpy.stack((start,end))))
data.edge_type = torch.LongTensor(edge_type)
train_idx = torch.LongTensor(numpy.nonzero(dl.labels_train['mask'])[0])
test_idx = torch.LongTensor(numpy.nonzero(dl.labels_test['mask'])[0])
if args.testing :
    val_idx = train_idx
else:
    val_idx = train_idx[:len(train_idx) // 5]
    train_idx = train_idx[len(train_idx) // 5:]
if args.dataset == "IMDB":
    labels = torch.FloatTensor(dl.labels_train['data']+dl.labels_test['data'])
else:
    labels = torch.LongTensor(dl.labels_train['data']+dl.labels_test['data']).argmax(dim=1)
train_y = labels[train_idx]
test_y = labels[test_idx]
val_y = labels[val_idx]

data.train_idx = train_idx
# data.valid_idx = val_idx 
data.test_idx = test_idx
data.train_y = train_y
data.test_y = test_y
# data.val_y = val_y 

num_nodes = dl.nodes['total']
num_rel = len(dl.links['meta'])

if args.has_CLG is not True:
    relation_graph.build_coarsened_line_graph(data,num_rel, rw_len=4, batch_size=5000, name=name)
rel_data = relation_graph.load_rel_graph(name)

num_classes = dl.labels_test['num_classes']

data.x = torch.sparse_coo_tensor(torch.arange(0, num_nodes).repeat(2, 1), torch.ones(num_nodes))

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
        self.fc2 = Linear(dim, num_classes)

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

        return x


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, data = Net().to(device), data.to(device)
rel_data = rel_data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

if args.dataset == 'IMDB':
    LOSS = F.binary_cross_entropy_with_logits
else:
    LOSS = lambda x,y: F.nll_loss(F.log_softmax(x),y)

if args.dataset == "IMDB":
    EVALUATE = multi_evaluate
else:
    EVALUATE = evaluate 

print("start training...")
forward_time = []
backward_time = []
save_dict_micro = model.state_dict()
save_dict_macro = model.state_dict()
best_result_micro = 0
best_result_macro = 0
best_epoch_micro = 0
best_epoch_macro = 0

model.train()
for epoch in range(1, args.epoch):
    # model.eval()
    # out = model(data, rel_data)
    # pred = out[data.train_idx]
    # train_micro,train_macro = EVALUATE(pred,data.train_y)
    # pred = out[val_idx]
    # micro,macro = EVALUATE(pred,val_y)
    # pred = out[data.test_idx]
    # micro,macro = EVALUATE(pred,data.test_y)

    # log = 'Epoch: {:03d}, loss: {:.4f}, Train: {:.4f}, Validation: {:.4f}, Test F1 micro: {:.4f}, Test F1 macro: {:.4f}'
    # print(log.format(epoch, l.item(), train_acc, test_acc, micro,macro))
    optimizer.zero_grad()
    t0 = time.time()
    logits = model(data,rel_data)
    loss = LOSS(logits[train_idx], labels[train_idx])
    t1 = time.time()
    loss.backward()
    optimizer.step()
    t2 = time.time()

    forward_time.append(t1 - t0)
    backward_time.append(t2 - t1)
    print("Epoch {:05d} | Train Forward Time(s) {:.4f} | Backward Time(s) {:.4f}".
            format(epoch, forward_time[-1], backward_time[-1]))
    val_loss = LOSS(logits[val_idx], labels[val_idx])
    train_micro, train_macro = EVALUATE(
        logits[train_idx], labels[train_idx])
    valid_micro, valid_macro = EVALUATE(
        logits[val_idx], labels[val_idx])
    if valid_micro > best_result_micro:
        save_dict_micro = model.state_dict()
        best_result_micro = valid_micro
        best_epoch_micro = epoch 
    if valid_macro > best_result_macro:
        save_dict_macro = model.state_dict()
        best_result_macro = valid_macro
        best_epoch_macro = epoch 

    print("Train micro: {:.4f} | Train macro: {:.4f} | Train Loss: {:.4f} | Validation micro: {:.4f} | Validation macro: {:.4f} | Validation loss: {:.4f}".
            format(train_micro, train_macro, loss.item(), valid_micro, valid_macro, val_loss.item()))
print()

model.eval()
result = [save_dict_micro,save_dict_macro]
for i in range(2):
    if i == 0:
        print("Best Micro At:"+str(best_epoch_micro))
    else:
        print("Best Macro At:"+str(best_epoch_macro))
    model.load_state_dict(result[i])
    logits = model(data,rel_data)
    test_loss = LOSS(logits[test_idx], labels[test_idx])
    test_micro, test_macro = EVALUATE(
        logits[test_idx], labels[test_idx])
    print("Test micro: {:.4f} | Test macro: {:.4f} | Test loss: {:.4f}".format(
        test_micro, test_macro, test_loss.item()))
    print()

print("Mean forward time: {:4f}".format(
    numpy.mean(forward_time[len(forward_time) // 4:])))
print("Mean backward time: {:4f}".format(
    numpy.mean(backward_time[len(backward_time) // 4:])))

print(args)
