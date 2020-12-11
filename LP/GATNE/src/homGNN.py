import sys
import random
import torch as th
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torch_geometric.nn import GCNConv, SAGEConv, TopKPooling, GATConv
import torch.nn.functional as F
from sklearn.metrics import (auc, f1_score, precision_recall_curve,
                             roc_auc_score)

node_num = {'amazon': 10166, 'youtube': 2000, 'twitter': 10000}
dataset_eval_type = {'amazon': ['1','2'], 'youtube': ['1', '2', '3', '4', '5'], 'twitter': ['1']}
class_num = {'amazon': 3, 'youtube': 6, 'twitter': 2}
device = th.device('cuda:0' if th.cuda.is_available() else 'cpu')
patience = 20

def gen_edge_list(train_file):
    edge_list = [[], []]
    with open(train_file) as train_file_handle:
        for line in train_file_handle:
            _, left, right, label = line[:-1].split('\t')
            if label != '0':
                edge_list[0].append(int(left))
                edge_list[1].append(int(right))
    print(f'generate graph with {len(edge_list[0])} edges')
    return th.LongTensor(edge_list)


def gen_feat(node_num, feat_dim=200, feat_type=0):
    feat = None
    if feat_type == 0:
        indices = np.vstack((np.arange(node_num), np.arange(node_num)))
        indices = th.LongTensor(indices)

        values = th.FloatTensor(np.ones(node_num))
        feat = th.sparse.FloatTensor(indices, values, th.Size([node_num, node_num])).to(device)

    elif feat_type == 1:
        feat = th.FloatTensor(np.eye(node_num))
    return feat


class hom_data(Dataset):
    def __init__(self, train_file, eval_type):

        data_list = [[], [], []]
        with open(train_file) as train_file_handle:
            for line in train_file_handle:
                edge_type, left, right, label = line[:-1].split('\t')
                if edge_type != eval_type:
                    continue
                data_list[0].append(int(left))
                data_list[1].append(int(right))
                data_list[2].append(float(label))
        self.data_list = data_list
        print(f'train size: {len(data_list[0])}')

    def __len__(self):
        return len(self.data_list[0])

    def __getitem__(self, item):
        return [self.data_list[0][item], self.data_list[1][item], self.data_list[2][item]]


class EarlyStop:
    def __init__(self, save_path='./best.pt', patience=20):
        self.patience = patience
        self.best_auc = 0
        self.model_auc = 0
        self.patience_now = 0
        self.stop = False
        self.save_path = save_path

    def step(self, auc, model):
        if auc >= self.best_auc:
            self.best_auc = auc
            self.patience_now = 0
            # update model
            if self.best_auc - self.model_auc > 0.001:
                self.model_auc = self.best_auc
                self.save_checkpoint(model)
        else:
            self.patience_now += 1
            if self.patience_now >= self.patience:
                self.stop = True

    def get_best(self):
        return self.best_auc, self.pr, self.f1

    def save_checkpoint(self, model):
        print('Saving model')
        th.save(model.state_dict(), self.save_path)


class GCN(th.nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, n_layers=2, dropout=0.5):
        super(GCN, self).__init__()
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(GCNConv(in_feats, hid_feats))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(GCNConv(hid_feats, hid_feats))
        self.fc = nn.Linear(hid_feats, out_feats)  # only used in decode
        self.dropout = nn.Dropout(p=dropout)
        nn.init.xavier_normal_(self.fc.weight, gain=1)
        for layer in self.layers:
            nn.init.xavier_normal_(layer.weight, gain=1)
    def encode(self, data):
        x, edge_list = data
        for i, layer in enumerate(self.layers):
            if i != 0:
                x = self.dropout(x)
            x = layer(x, edge_list)
            if i < len(self.layers) - 1:
                x = F.relu(x)
        return x

    def decode(self, x, edge_index):
        x = (x[edge_index[0]] * x[edge_index[1]])
        x = self.fc(x)
        x = F.relu(x)
        return x

class GAT(th.nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats,n_layers=2,dropout=0.5,heads=[1]):
        super(GAT, self).__init__()
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(GATConv(in_feats, hid_feats,heads[0]))
        # hidden layers
        for l in range(1,n_layers-1):
            self.layers.append(GATConv(hid_feats* heads[l-1], hid_feats, heads[l]))
        # output layer
        self.layers.append(GATConv(hid_feats*heads[-2], hid_feats,heads[-1]))
        self.fc = nn.Linear(hid_feats,out_feats)
        self.dropout = nn.Dropout(p=dropout)
        nn.init.xavier_normal_(self.fc.weight,gain=1)
        for layer in self.layers:
            nn.init.xavier_normal_(layer.weight, gain=1)

    def encode(self, data):
        x, edge_list= data.x,data.edge_list
        for i, layer in enumerate(self.layers):
            if i != 0:
                pass
                x = self.dropout(x)
            x = layer(x,edge_list)
            x = F.relu(x)
        return x

    def decode(self, x, edge_index):
        x = (x[edge_index[0]] * x[edge_index[1]])
        x = self.fc(x)
        x = F.relu(x)
        return x

def evaluate(out_feat, true_edges, false_edges):
    true_list = list()
    prediction_list = list()
    true_num = int(out_feat.size()[0] / 2)

    true_list = np.ones(true_num, dtype=int).tolist() + np.zeros(true_num, dtype=int).tolist()
    prediction_list = out_feat.cpu().detach().numpy().tolist()

    sorted_pred = prediction_list[:]
    sorted_pred.sort()
    threshold = sorted_pred[-true_num]

    y_pred = np.zeros(len(prediction_list), dtype=np.int32)
    for i in range(len(prediction_list)):
        if prediction_list[i] >= threshold:
            y_pred[i] = 1

    y_true = np.array(true_list)
    y_scores = np.array(prediction_list)
    ps, rs, _ = precision_recall_curve(y_true, y_scores)
    return roc_auc_score(y_true, y_scores), auc(rs, ps), f1_score(y_true, y_pred)


def file2list(file_name, eval_type):
    file_list = [[], [], []]
    with open(file_name) as file_handle:
        for line in file_handle:
            edge_type, left, right, label = line[:-1].split('\t')
            if edge_type != eval_type:
                continue
            file_list[0].append(int(left))
            file_list[1].append(int(right))
            file_list[2].append(int(label))
    return file_list


def main(data_name, eval_type, model):
    train_file = f'../data/{data_name}/hom_train.txt'
    valid_file = f'../data/{data_name}/hom_valid.txt'
    test_file = f'../data/{data_name}/hom_test.txt'
    model_save_path = f'../data/{data_name}/best_eval_{eval_type}.pt'
    edge_list = gen_edge_list(train_file).to(device)
    feat = gen_feat(node_num[data_name], feat_type=0).to(device)
    train_data_loader = DataLoader(hom_data(train_file, eval_type), batch_size=10000, shuffle=True, num_workers=0)

    epochs = 50
    early_stopping = EarlyStop(save_path=model_save_path,patience=20)

    lossFun = nn.MSELoss()
    optimizer = th.optim.Adam([{'params': model.parameters()}], lr=0.01, weight_decay=0)
    valid_list = file2list(valid_file, eval_type)
    print(f'valid size: {len(valid_list[0])}')
    for epoch in range(epochs):
        for index, train_data_bach in enumerate(train_data_loader):
            model.train()
            hid_feat = model.encode([feat, edge_list])
            out_feat = model.decode(hid_feat, [train_data_bach[0], train_data_bach[1]])
            target = th.DoubleTensor(train_data_bach[2]).float().view(-1, 1).to(device)
            loss = lossFun(out_feat, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if index % 10 == 0:
                print(f'epoch {epoch} | batch {index} | loss {loss.item()}')

            # valid
            model.eval()
            hid_feat = model.encode([feat, edge_list])
            out_feat = model.decode(hid_feat, [valid_list[0], valid_list[1]])
            target = th.FloatTensor(valid_list[2]).view(-1, 1).to(device)
            loss = lossFun(out_feat, target)
            valid_half_len = int(len(valid_list[0]) / 2)
            roc_auc, fpr_auc, f1 = evaluate(out_feat, [valid_list[0][:valid_half_len], valid_list[1][:valid_half_len]],
                                            [valid_list[0][valid_half_len:], valid_list[1][valid_half_len:]])
            if index % 10 == 0:
                print(
                    f"-------------------------------------------------------------valid loss {loss}, auc {roc_auc}, pr {fpr_auc}, f1 {f1}")
                early_stopping.step(roc_auc, model)
                if early_stopping.stop:
                    break
        if early_stopping.stop:
            break
        # test
        test_list = file2list(test_file, eval_type)
        model.eval()
        hid_feat = model.encode([feat, edge_list])
        out_feat = model.decode(hid_feat, [test_list[0], test_list[1]])
        target = th.FloatTensor(test_list[2]).view(-1, 1).to(device)
        loss = lossFun(out_feat, target)
        valid_half_len = int(len(valid_list[0]) / 2)
        roc_auc, fpr_auc, f1 = evaluate(out_feat, [test_list[0][:valid_half_len], test_list[1][:valid_half_len]],
                                        [test_list[0][valid_half_len:], test_list[1][valid_half_len:]])
        print(
            f"-------------------------------------------------------------test loss {loss}, auc {roc_auc}, pr {fpr_auc}, f1 {f1}")
    # test
    model.load_state_dict(th.load(model_save_path))
    model.eval()
    test_list = file2list(test_file, eval_type)
    hid_feat = model.encode([feat, edge_list])
    out_feat = model.decode(hid_feat, [test_list[0], test_list[1]])
    target = th.FloatTensor(test_list[2]).view(-1, 1).to(device)
    loss = lossFun(out_feat, target)
    valid_half_len = int(len(valid_list[0]) / 2)
    roc_auc, fpr_auc, f1 = evaluate(out_feat, [test_list[0][:valid_half_len], test_list[1][:valid_half_len]],
                                    [test_list[0][valid_half_len:], test_list[1][valid_half_len:]])
    print(
        f"-------------------------------------------------------------test loss {loss}, auc {roc_auc}, pr {fpr_auc}, f1 {f1}")
    return roc_auc, fpr_auc, f1


if __name__ == '__main__':
    if len(sys.argv) <= 2:
        exit('need dataset_name para and model_type para')
    data_name = sys.argv[1]
    model_type = sys.argv[2]
    print(data_name)
    if model_type=='GCN' or model_type=='GAT':
        model = globals()[model_type](in_feats=node_num[data_name], hid_feats=200, out_feats=1).to(device)
    auc_list, pr_list, f1_list = list(), list(), list()
    for eval_type in dataset_eval_type[data_name]:
        print(f'dataset: {data_name}, eval_type: {eval_type}')
        roc_auc, pr_auc, f1 = main(data_name, eval_type, model)
        auc_list.append(roc_auc)
        pr_list.append(pr_auc)
        f1_list.append(f1)
        print(f'auc {roc_auc}, pr {pr_auc}, f1 {f1}')
    print(auc_list, pr_list, f1_list)
    print(f'avg_auc:{np.mean(auc_list)}, avg_pr:{np.mean(pr_list)}, avg_f1:{np.mean(f1_list)}')
