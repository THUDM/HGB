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
import os
import math

os.environ['CUDA_VISIBLE_DEVICES'] = ''

node_num = {'amazon': 10166, 'youtube': 2000, 'twitter': 10000}
dataset_eval_type = {'amazon': ['1', '2'], 'youtube': ['1', '2', '3', '4', '5'], 'twitter': ['1']}
class_num = {'amazon': 3, 'youtube': 6, 'twitter': 2}
device = th.device('cuda:0' if th.cuda.is_available() else 'cpu')
print(f'use device: {device}')
patience = 20


def gen_edge_list(train_file, data_name, relation_type):
    edge_list = [[], []]
    with open(train_file) as train_file_handle:
        for line in train_file_handle:
            r_id, left, right, label = line[:-1].split('\t')
            if label != '0':
                edge_list[0].append(int(left))
                edge_list[1].append(int(right))
                edge_list[0].append(int(right))
                edge_list[1].append(int(left))
    print(f'generate graph with {len(edge_list[0])} edges')
    return th.LongTensor(edge_list)


def gen_feat(node_num, feat_dim=200, feat_type=0):
    feat = None
    if feat_type == 0:
        # sparse
        indices = np.vstack((np.arange(node_num), np.arange(node_num)))
        indices = th.LongTensor(indices)
        values = th.FloatTensor(np.ones(node_num))
        feat = th.sparse.FloatTensor(indices, values, th.Size([node_num, node_num])).to(device)

    elif feat_type == 1:
        # dense
        feat = th.FloatTensor(np.eye(node_num)).to(device)
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
    def __init__(self, save_path='', patience=5):
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
                if self.save_path != '':
                    self.save_checkpoint(model)
        else:
            self.patience_now += 1

            if self.patience_now >= self.patience:
                self.stop = True
            else:
                print(f'EarlyStopping counter: {self.patience_now} out of {self.patience}')

    def get_best(self):
        return self.best_auc, self.pr, self.f1

    def save_checkpoint(self, model):
        print('Saving model')
        th.save(model.state_dict(), self.save_path)


class DisMult(th.nn.Module):
    def __init__(self, emb_size):
        super(DisMult, self).__init__()
        self.emb_size = emb_size
        self.weights = nn.Parameter(th.Tensor(emb_size, emb_size), requires_grad=True)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.weights)
        # stdv = 1. / math.sqrt(self.weights.size(0))
        # self.weights.data.uniform_(-stdv, stdv)

    def forward(self, input1, input2):
        batch_size = input1.size(0)
        input1 = input1.view(batch_size, 1, -1)
        input2 = input2.view(batch_size, -1, 1)
        tmp = input1.matmul(self.weights)
        tmp = tmp.matmul(input2)

        # result = []
        # for i in range(batch_size):
        #     result_i = input1[i] @ self.weights @ (input2[i].t())
        #     result.append(result_i)
        # result = th.FloatTensor(result).to(device)
        result = th.squeeze(tmp, 1)
        return result


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
        self.dismult = DisMult(emb_size=hid_feats)  # only used in decode
        self.dropout = nn.Dropout(p=dropout)
        nn.init.xavier_normal_(self.fc.weight, gain=1.4)
        for layer in self.layers:
            nn.init.xavier_normal_(layer.weight, gain=1.4)

    def encode(self, data):
        x, edge_list = data
        for i, layer in enumerate(self.layers):
            if i != 0:
                x = self.dropout(x)
            x = layer(x, edge_list)
            x = F.leaky_relu(x)
        return x

    def decode(self, x, edge_index):
        return self.de_dismult(x, edge_index)

    def de_dot(self, x, edge_index):
        x = (x[edge_index[0]] * x[edge_index[1]])
        x = self.fc(x)
        x = F.leaky_relu(x)
        return x

    def de_dismult(self, x, edge_index):
        x = self.dismult(x[edge_index[0]], x[edge_index[1]])
        return x

    def de_cosine(self, x, edge_index):
        embed_size = x[edge_index[0]].shape[0]
        feature1 = x[edge_index[0]].view(embed_size, -1)  # 将特征转换为N*(C*W*H)，即两维
        feature2 = x[edge_index[1]].view(embed_size, -1)
        feature1 = F.normalize(feature1)  # F.normalize只能处理两维的数据，L2归一化
        feature2 = F.normalize(feature2)
        distance = feature1.mm(feature2.t())  # 计算余弦相似度
        distance = th.diag(distance).view(-1, 1)
        return distance


class GAT(th.nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, n_layers=2, dropout=0.5, heads=[1]):
        super(GAT, self).__init__()
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(GATConv(in_feats, hid_feats, heads[0]))
        # hidden layers
        for l in range(1, n_layers - 1):
            self.layers.append(GATConv(hid_feats * heads[l - 1], hid_feats, heads[l]))
        # output layer
        self.layers.append(GATConv(hid_feats * heads[-2], hid_feats, heads[-1]))
        self.fc = nn.Linear(hid_feats, out_feats) # only used in decode
        self.dismult = DisMult(emb_size=hid_feats)  # only used in decode
        self.dropout = nn.Dropout(p=dropout)
        nn.init.xavier_normal_(self.fc.weight, gain=1.4)
        for layer in self.layers:
            nn.init.xavier_normal_(layer.lin_l.weight, gain=1.4)

    def encode(self, data):
        x, edge_list = data
        for i, layer in enumerate(self.layers):
            if i != 0:
                x = self.dropout(x)
            x = layer(x, edge_list)
            x = F.leaky_relu(x, inplace=True)
        return x

    def decode(self, x, edge_index):
        return self.de_dismult(x, edge_index)

    def de_dot(self, x, edge_index):
        x = (x[edge_index[0]] * x[edge_index[1]])
        x = self.fc(x)
        x = F.leaky_relu(x)
        return x

    def de_dismult(self, x, edge_index):
        x = self.dismult(x[edge_index[0]], x[edge_index[1]])
        return x

    def de_cosine(self, x, edge_index):
        embed_size = x[edge_index[0]].shape[0]
        feature1 = x[edge_index[0]].view(embed_size, -1)  # 将特征转换为N*(C*W*H)，即两维
        feature2 = x[edge_index[1]].view(embed_size, -1)
        feature1 = F.normalize(feature1)  # F.normalize只能处理两维的数据，L2归一化
        feature2 = F.normalize(feature2)
        distance = feature1.mm(feature2.t())  # 计算余弦相似度
        distance = th.diag(distance).view(-1, 1)
        return distance


def evaluate(out_feat, pos_num):
    neg_num = out_feat.size()[0] - pos_num
    true_list = np.ones(pos_num, dtype=int).tolist() + np.zeros(neg_num, dtype=int).tolist()
    prediction_list = out_feat.cpu().detach().numpy().tolist()

    sorted_pred = prediction_list[:]
    sorted_pred.sort()
    threshold = sorted_pred[-pos_num]

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
    pos_num = 0
    with open(file_name) as file_handle:
        for line in file_handle:
            edge_type, left, right, label = line[:-1].split('\t')
            if edge_type != eval_type:
                continue
            file_list[0].append(int(left))
            file_list[1].append(int(right))
            file_list[2].append(int(label))
            if int(label) == 1:
                pos_num += 1
    return file_list, pos_num


def main(data_name, eval_type, model, test_with_CPU=False):
    train_file = f'../data/{data_name}/hom_train.txt'
    valid_file = f'../data/{data_name}/hom_valid.txt'
    test_file = f'../data/{data_name}/hom_test.txt'
    model_save_path = f'../data/{data_name}/best_eval_{eval_type}.pt'
    edge_list = gen_edge_list(train_file, data_name=data_name, relation_type=eval_type).to(device)
    feat = gen_feat(node_num[data_name], feat_type=0).to(device)
    train_data_loader = DataLoader(hom_data(train_file, eval_type), batch_size=10000, shuffle=True, num_workers=0)

    epochs = 30
    early_stopping = EarlyStop(save_path=model_save_path, patience=3)
    edge_sample_ratio = 1
    if data_name == 'youtube' and test_with_CPU:
        edge_sample_ratio = 0.5
    edge_num = edge_list.shape[1]

    lossFun = nn.MSELoss()
    optimizer = th.optim.Adam([{'params': model.parameters()}], lr=0.01, weight_decay=0)
    valid_list, valid_pos_num = file2list(valid_file, eval_type)
    print(f'valid size: {len(valid_list[0])}')
    for epoch in range(epochs):
        for index, train_data_bach in enumerate(train_data_loader):
            model.train()
            if edge_sample_ratio < 1:
                edge_sample_index = np.random.choice(edge_num, int(edge_num * edge_sample_ratio), replace=False)
                edge_list_part = edge_list[:, edge_sample_index].to(device)
                hid_feat = model.encode([feat, edge_list_part])
            else:
                hid_feat = model.encode([feat, edge_list])
            out_feat = model.decode(hid_feat, [train_data_bach[0], train_data_bach[1]]).to(device)
            target = train_data_bach[2].float().view(-1, 1).to(device)
            loss = lossFun(out_feat, target)
            # loss.requires_grad = True
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if index % 10 == 0:
                print(f'epoch {epoch} | batch {index} |train loss {loss.item()}')

        # valid
        model.eval()
        if edge_sample_ratio < 1:
            edge_sample_index = np.random.choice(edge_num, int(edge_num * edge_sample_ratio), replace=False)
            edge_list_part = edge_list[:, edge_sample_index].to(device)
            hid_feat = model.encode([feat, edge_list_part])
        else:
            hid_feat = model.encode([feat, edge_list])
        out_feat = model.decode(hid_feat, [valid_list[0], valid_list[1]]).to(device)
        target = th.FloatTensor(valid_list[2]).view(-1, 1).to(device)
        loss = lossFun(out_feat, target)
        roc_auc, fpr_auc, f1 = evaluate(out_feat, valid_pos_num)
        print(
            f"------------------------------------------------valid loss {loss}, auc {roc_auc}, pr {fpr_auc}, f1 {f1}")
        early_stopping.step(roc_auc, model)
        if early_stopping.stop:
            break

    th.cuda.empty_cache()
    # test
    model.load_state_dict(th.load(model_save_path))
    model.eval()
    test_list, train_pos_num = file2list(test_file, eval_type)
    target = th.FloatTensor(test_list[2]).view(-1, 1)
    '''use CPU to avoid CUDA OOM'''
    if test_with_CPU:
        model = model.cpu()
        feat = feat.cpu()
        edge_list = edge_list.cpu()
        target = target.cpu()
    else:
        model = model.to(device)
        feat = feat.to(device)
        edge_list = edge_list.to(device)
        target = target.to(device)
    hid_feat = model.encode([feat, edge_list])
    out_feat = model.decode(hid_feat, [test_list[0], test_list[1]])

    loss = lossFun(out_feat, target)
    roc_auc, fpr_auc, f1 = evaluate(out_feat, train_pos_num)
    print(
        f"test loss {loss}, auc {roc_auc}, pr {fpr_auc}, f1 {f1}")
    return roc_auc, fpr_auc, f1


if __name__ == '__main__':
    if len(sys.argv) <= 2:
        exit('need dataset_name para and model_type para')
    data_name = sys.argv[1]
    model_type = sys.argv[2]
    model = None
    test_with_CPU = False
    if model_type == 'GCN':
        model = GCN(in_feats=node_num[data_name], hid_feats=200, out_feats=1)
    elif model_type == "GAT":
        test_with_CPU = True
        n_heads = [2]
        n_layers = 2
        heads = n_heads * (n_layers - 1) + [1]
        model = GAT(in_feats=node_num[data_name], hid_feats=200, out_feats=1, n_layers=n_layers, heads=heads)
    else:
        exit('please input true model_type within [GCN, GAT]')
    auc_list, pr_list, f1_list = list(), list(), list()
    for eval_type in dataset_eval_type[data_name]:
        print(f'dataset: {data_name}, eval_type: {eval_type} of {len(dataset_eval_type[data_name])}')
        model = model.to(device)
        roc_auc, pr_auc, f1 = main(data_name, eval_type, model, test_with_CPU=test_with_CPU)
        auc_list.append(roc_auc)
        pr_list.append(pr_auc)
        f1_list.append(f1)
    print(f'auc_list:{auc_list}, pr_list:{pr_list}, f1_list:{f1_list}')
    print(f'avg_auc:{np.mean(auc_list)}, avg_pr:{np.mean(pr_list)}, avg_f1:{np.mean(f1_list)}')
