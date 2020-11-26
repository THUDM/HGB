import numpy as np
import torch as th
from torch_geometric.data import Data
from torch import nn
from torch_geometric.nn import GCNConv, SAGEConv,TopKPooling,GATConv
import torch.nn.functional as F
import re
import random
import csv
import argparse
import datetime
from sklearn.metrics import f1_score, roc_auc_score
import math
from itertools import *
from torch_geometric.utils import train_test_split_edges
from torch_geometric.utils import to_undirected

class EarlyStopping(object):
    def __init__(self, patience=10):
        dt = datetime.datetime.now()
        self.filename = 'early_stop_{}_{:02d}-{:02d}-{:02d}.pth'.format(
            dt.date(), dt.hour, dt.minute, dt.second)
        self.patience = patience
        self.counter = 0
        self.best_auc = None
        self.best_loss = None
        self.early_stop = False

    def step(self, loss, auc, model):
        if self.best_loss is None:
            self.best_auc = auc
            self.best_loss = loss
            # self.save_checkpoint(model)
        elif (loss >= self.best_loss) and (auc <= self.best_auc):
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            if (loss < self.best_loss) and (auc > self.best_auc):
                pass
                # self.save_checkpoint(model)
            self.best_loss = np.min((loss, self.best_loss))
            self.best_auc = np.max((auc, self.best_auc))
            self.counter = 0
        return self.early_stop

    def save_checkpoint(self, model):
        """Saves model when validation loss decreases."""
        th.save(model.state_dict(), self.filename)

    def load_checkpoint(self, model):
        """Load the latest checkpoint."""
        model.load_state_dict(th.load(self.filename))

class GCN(th.nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats,n_layers=2,dropout=0.5):
        super(GCN, self).__init__()
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(GCNConv(in_feats, hid_feats))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(GCNConv(hid_feats, hid_feats))
        self.fc = nn.Linear(hid_feats, out_feats)
        self.dropout = nn.Dropout(p=dropout)

    def encode(self, data):
        x, edge_list= data.x,data.edge_list
        for i, layer in enumerate(self.layers):
            if i != 0:
                x = self.dropout(x)
            x = layer(x,edge_list)
            if i<len(self.layers)-1:
                x = F.relu(x)
        return x

    def decode(self,x,edge_index):
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
        for l in range(1,n_layers):
            self.layers.append(GATConv(hid_feats* heads[l-1], hid_feats, heads[l]))
        # output layer
        self.layers.append(GATConv(hid_feats*heads[-2], hid_feats,heads[-1]))
        self.fc = nn.Linear(hid_feats,out_feats)
        self.dropout = nn.Dropout(p=dropout)
    def encode(self, data):
        x, edge_list= data.x,data.edge_list
        for i, layer in enumerate(self.layers):
            if i != 0:
                x = self.dropout(x)
            x = layer(x,edge_list)
            x = F.relu(x)
        return x

    def decode(self, x, edge_index):
        x = (x[edge_index[0]] * x[edge_index[1]])
        x = self.fc(x)
        x = F.relu(x)
        return x

class GSAGE(th.nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, n_layers=2, dropout=0.5):
        super(GSAGE, self).__init__()
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(SAGEConv(in_feats, hid_feats))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(SAGEConv(hid_feats, hid_feats))
        # output layer
        self.layers.append(SAGEConv(hid_feats, out_feats))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, data):
        x, edge_list = data.x, data.edge_list
        for i, layer in enumerate(self.layers):
            if i != 0:
                x = self.dropout(x)
            x = layer(x, edge_list)
            x = F.relu(x)
        return x

class edge_data():
    def __init__(self,args):
        if args.data_name == 'cite':
            self.train_file_pth = args.data_path + "a_p_cite_list_train.txt"
            self.test_file_pth = args.data_path + "a_p_cite_list_test.txt"
        elif args.data_name=='colab':
            self.train_file_pth = args.data_path+"a_a_list_train.txt"
            self.test_file_pth = args.data_path + "a_a_list_test.txt"
        else:
            exit('data_name errors.')
        train_edge_index = [[], []]
        train_label = []
        test_edge_index = [[], []]
        test_label = []
        with open(self.train_file_pth) as f:
            data_file = csv.reader(f)
            for i, d in enumerate(data_file):
                train_edge_index[0].append(int(d[0]))
                second_edge = int(d[1]) if args.data_name == 'colab' else int(d[1]) + args.A_n
                train_edge_index[1].append(second_edge)
                train_label.append(int(d[2]))
            f.close()
        with open(self.test_file_pth) as f:
            data_file = csv.reader(f)
            for i, d in enumerate(data_file):
                test_edge_index[0].append(int(d[0]))
                second_edge = int(d[1]) if args.data_name == 'colab' else int(d[1]) + args.A_n
                test_edge_index[1].append(second_edge)
                test_label.append(int(d[2]))
            f.close()
        row, col = test_edge_index
        row, col = th.LongTensor(row), th.LongTensor(col)
        test_edge_index = th.stack([row, col], dim=0)
        test_label = th.LongTensor(test_label)

        #random train_data and transform to tensor
        row, col = train_edge_index
        ran_seed = random.random()
        random.seed(ran_seed)
        random.shuffle(row)
        random.seed(ran_seed)
        random.shuffle(col)
        random.seed(ran_seed)
        random.shuffle(train_label)


        row, col = th.LongTensor(row), th.LongTensor(col)
        train_label = th.LongTensor(train_label)
        # Return upper triangular portion.
        # mask = row < col
        # row, col = row[mask], col[mask]

        n_v = int(math.floor(args.val_ratio_of_train * row.size(0)))

        # split edges.
        # perm = th.randperm(row.size(0))
        # row, col = row[perm], col[perm]


        r, c = row[:n_v], col[:n_v]
        val_edge_index = th.stack([r, c], dim=0)
        val_label = train_label[:n_v]

        r, c = row[n_v :], col[n_v :]
        train_edge_index = th.stack([r, c], dim=0)
        # train_edge_index = to_undirected(train_edge_index)
        train_label = train_label[n_v:]
        self.train_edge_index,self.val_edge_index,self.test_edge_index = train_edge_index, val_edge_index, test_edge_index
        self.train_label,self.val_label,self.test_label = train_label,val_label, test_label
    def split_train(self, batch_size=1000):

        row,col = self.train_edge_index
        row,col = row.numpy().tolist(),col.numpy().tolist()
        label_list = self.train_label.numpy().tolist()
        #random
        ran_seed = random.random()
        random.seed(ran_seed)
        random.shuffle(row)
        random.seed(ran_seed)
        random.shuffle(col)
        random.seed(ran_seed)
        random.shuffle(label_list)
        #to tensor
        row = [th.LongTensor(row[i:i + batch_size]) for i in range(len(row)) if i % batch_size == 0]
        col = [th.LongTensor(col[i:i + batch_size]) for i in range(len(col)) if i % batch_size == 0]

        label = [th.LongTensor(label_list[i:i + batch_size]) for i in range(len(label_list)) if i % batch_size == 0]
        return [row,col],label

class random_edge_data():
    def __init__(self,args):
        if args.data_name == 'cite':
            self.train_file_pth = args.data_path + "a_p_cite_list_train.txt"
            self.test_file_pth = args.data_path + "a_p_cite_list_test.txt"
        elif args.data_name=='colab':
            self.train_file_pth = args.data_path+"a_a_list_train.txt"
            self.test_file_pth = args.data_path + "a_a_list_test.txt"
        else:
            exit('data_name errors.')
        all_edge_index = [[],[]]
        all_label = []
        with open(self.train_file_pth) as f:
            data_file = csv.reader(f)
            for i, d in enumerate(data_file):
                all_edge_index[0].append(int(d[0]))
                second_edge = int(d[1]) if args.data_name == 'colab' else int(d[1]) + args.A_n
                all_edge_index[1].append(second_edge)
                all_label.append(int(d[2]))
            f.close()
        with open(self.test_file_pth) as f:
            data_file = csv.reader(f)
            for i, d in enumerate(data_file):
                all_edge_index[0].append(int(d[0]))
                second_edge = int(d[1]) if args.data_name == 'colab' else int(d[1]) + args.A_n
                all_edge_index[1].append(second_edge)
                all_label.append(int(d[2]))
            f.close()
        #random all edge and label
        row, col = all_edge_index
        ran_seed = random.random()
        random.seed(ran_seed)
        random.shuffle(row)
        random.seed(ran_seed)
        random.shuffle(col)
        random.seed(ran_seed)
        random.shuffle(all_label)

        n_val = int(math.floor(args.val_ratio * len(row)))
        n_test = int(math.floor(args.test_ratio * len(row)))

        # valid data
        row, col = all_edge_index
        row, col = row[:n_val], col[:n_val]
        row, col = th.LongTensor(row), th.LongTensor(col)
        val_label = all_label[:n_val]
        val_edge_index = th.stack([row, col], dim=0)
        val_label = th.FloatTensor(val_label)
        # test data
        row, col = all_edge_index
        row, col = row[n_val:n_val+n_test], col[n_val:n_val+n_test]
        row, col = th.LongTensor(row), th.LongTensor(col)
        test_label = all_label[n_val:n_val+n_test]
        test_edge_index = th.stack([row, col], dim=0)
        test_label = th.FloatTensor(test_label)
        # train data
        row, col = all_edge_index
        row, col = row[n_val + n_test:], col[n_val + n_test:]
        row, col = th.LongTensor(row), th.LongTensor(col)
        train_label = all_label[n_val + n_test:]
        train_edge_index = th.stack([row, col], dim=0)
        train_label = th.FloatTensor(train_label)

        self.train_edge_index,self.val_edge_index,self.test_edge_index = train_edge_index, val_edge_index, test_edge_index
        self.train_label,self.val_label,self.test_label = train_label,val_label, test_label
        print('get data size (train,val,test): ', self.train_label.size()[0], self.val_label.size()[0],
              self.test_label.size()[0])
    def split_train(self, batch_size=1000):

        row,col = self.train_edge_index
        row,col = row.numpy().tolist(),col.numpy().tolist()
        label_list = self.train_label.numpy().tolist()
        #random
        ran_seed = random.random()
        random.seed(ran_seed)
        random.shuffle(row)
        random.seed(ran_seed)
        random.shuffle(col)
        random.seed(ran_seed)
        random.shuffle(label_list)
        #to tensor
        row = [th.LongTensor(row[i:i + batch_size]) for i in range(len(row)) if i % batch_size == 0]
        col = [th.LongTensor(col[i:i + batch_size]) for i in range(len(col)) if i % batch_size == 0]
        label = [th.FloatTensor(label_list[i:i + batch_size]) for i in range(len(label_list)) if i % batch_size == 0]

        return [row,col],label

def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type = str, default = '../data/academic_test/',
                   help='path to data')
    parser.add_argument('--model_path', type = str, default = '../model_save/',
                   help='path to save model')
    parser.add_argument('--A_n', type = int, default = 28646,
                   help = 'number of author node')
    parser.add_argument('--P_n', type = int, default = 21044,
                   help = 'number of paper node')
    parser.add_argument('--V_n', type = int, default = 18,
                   help = 'number of venue node')
    parser.add_argument('--in_f_d', type = int, default = 128,
                   help = 'input feature dimension')
    parser.add_argument('--embed_d', type = int, default = 128,
                   help = 'embedding dimension')
    parser.add_argument('--train_iter_n', type = int, default = 50,
                   help = 'max number of training iteration')
    parser.add_argument("--cuda", default = 0, type = int)
    parser.add_argument("--checkpoint", default = '', type=str)
    parser.add_argument("--epochs", default=500, type=str)
    parser.add_argument("--patience", default=10, type=str)
    parser.add_argument("--n_layers", default=3, type=int)
    parser.add_argument("--n_heads", default=[4], type=list)
    parser.add_argument("--dropout", default=0.0, type=float)
    parser.add_argument("--model", default='GAT', type=str)
    parser.add_argument('--lr', type=float, default=0.005, )
    parser.add_argument('--weight_decay', type=float, default=0.000)
    parser.add_argument('--val_ratio_of_train', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=100000)
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--test_ratio', type=float, default=0.2)
    parser.add_argument('--data_name', type=str, default='cite')

    args = parser.parse_args()
    return args

def gen_embed(args, with_paper=True):
    if with_paper:
        p_abstract_embed = np.zeros((args.P_n, args.in_f_d))
        p_a_e_f = open(args.data_path + "p_abstract_embed.txt", "r")
        for line in islice(p_a_e_f, 1, None):
            values = line.split()
            index = int(values[0])
            embeds = np.asarray(values[1:], dtype='float32')
            p_abstract_embed[index] = embeds
        p_a_e_f.close()

        p_title_embed = np.zeros((args.P_n, args.in_f_d))
        p_t_e_f = open(args.data_path + "p_title_embed.txt", "r")
        for line in islice(p_t_e_f, 1, None):
            values = line.split()
            index = int(values[0])
            embeds = np.asarray(values[1:], dtype='float32')
            p_title_embed[index] = embeds
        p_t_e_f.close()
        p_embed = np.hstack((p_title_embed, p_abstract_embed))
        a_embed = np.zeros((args.A_n, args.in_f_d * 2))
        v_embed = np.zeros((args.V_n, args.in_f_d * 2))
        embed = np.vstack((a_embed, p_embed, v_embed))
        embed=th.FloatTensor(embed)
    else:  #sparse one-hot
        node_all = args.A_n+args.P_n+args.V_n
        embed_index_list=[]
        for i in range(args.A_n):
            embed_index_list.append([i,i])
        embed_value_list = [1] * args.A_n

        embed_index_tensor= th.tensor(embed_index_list,dtype=th.long)
        embed_value_tensor = th.FloatTensor(embed_value_list)

        embed = th.sparse.FloatTensor(embed_index_tensor.t(),embed_value_tensor,th.Size([node_all,args.A_n]))
    return embed

def gen_edge_list(args):
    a_p_list = [[], []]
    p_a_list = [[], []]
    p_p_list = [[], []]
    v_p_list = [[], []]
    p_v_list = [[], []]

    relation_f = ["a_p_list_train.txt", "p_a_list_train.txt", "p_p_cite_list_train.txt",  "v_p_list_train.txt"]
    for i in range(len(relation_f)):
        f_name = relation_f[i]
        neigh_f = open(args.data_path + f_name, "r")
        for line in neigh_f:
            line = line.strip()
            node_id = int(re.split(':', line)[0])
            neigh_list = re.split(':', line)[1]
            neigh_list_id = re.split(',', neigh_list)
            if f_name == 'a_p_list_train.txt':
                for j in range(len(neigh_list_id)):
                    a_p_list[0].append(node_id)
                    a_p_list[1].append(int(neigh_list_id[j]))
            elif f_name == 'p_a_list_train.txt':
                for j in range(len(neigh_list_id)):
                    p_a_list[0].append(node_id)
                    p_a_list[1].append(int(neigh_list_id[j]))
            elif f_name == 'p_p_cite_list_train.txt':
                for j in range(len(neigh_list_id)):
                    p_p_list[0].append(node_id)
                    p_p_list[1].append(int(neigh_list_id[j]))
            elif f_name == 'v_p_list_train.txt':
                for j in range(len(neigh_list_id)):
                    v_p_list[0].append(node_id)
                    v_p_list[1].append(int(neigh_list_id[j]))
            else:
                print('Some errors occur.')
        neigh_f.close()
    #get paper-venue edge
    p_v = [0] * args.P_n

    p_v_f = open(args.data_path + 'p_v.txt', "r")
    for line in p_v_f:
        line = line.strip()
        p_id = int(re.split(',', line)[0])
        v_id = int(re.split(',', line)[1])
        p_v[p_id] = v_id
        p_v_list[0].append(p_id)
        p_v_list[1].append(v_id)
    p_v_f.close()
    #set range of a,p,v (0-28645,0-21043,0-17) -> (0-28645,28646-49689,49690-49707)
    for i in range(len(a_p_list[1])):
        a_p_list[1][i] += args.A_n
    for i in range(len(p_a_list[0])):
        p_a_list[0][i] += args.A_n
    for i in range(len(p_p_list[0])):
        p_p_list[0][i] += args.A_n
        p_p_list[1][i] += args.A_n
    for i in range(len(v_p_list[0])):
        v_p_list[0][i] += args.A_n+args.P_n
        v_p_list[1][i] += args.A_n
    for i in range(len(p_v_list[0])):
        p_v_list[0][i] += args.A_n
        p_v_list[1][i] += args.A_n+args.P_n
    #42379 42379 ...
    start_list = a_p_list[0] + p_a_list[0] + p_p_list[0] + v_p_list[0] + p_v_list[0]
    end_list = a_p_list[1] + p_a_list[1] + p_p_list[1] + v_p_list[1] + p_v_list[1]
    return [start_list,end_list]

def gen_data(embed,edge_list):
    return Data(x=embed,
                edge_list=th.LongTensor(edge_list),
                )

# This function modified from author's code.
def score_AUC_f1(test_predict, test_target):

    _, test_predict = th.max(test_predict,1)
    test_predict, test_target = test_predict.cpu().detach().numpy(), test_target.cpu().detach().numpy()
    AUC_score = roc_auc_score(test_target, test_predict)

    total_count = 0
    correct_count = 0
    true_p_count = 0
    false_p_count = 0
    false_n_count = 0

    for i in range(len(test_predict)):
        total_count += 1
        if (int(test_predict[i]) == int(test_target[i])):
            correct_count += 1
        if (int(test_predict[i]) == 1 and int(test_target[i]) == 1):
            true_p_count += 1
        if (int(test_predict[i]) == 1 and int(test_target[i]) == 0):
            false_p_count += 1
        if (int(test_predict[i]) == 0 and int(test_target[i]) == 1):
            false_n_count += 1

    # print("accuracy: " + str(float(correct_count) / total_count))
    precision = float(true_p_count) / (true_p_count + false_p_count) if true_p_count + false_p_count>0 else 0
    # print("precision: " + str(precision))
    recall = float(true_p_count) / (true_p_count + false_n_count) if true_p_count + false_n_count>0 else 0
    # print("recall: " + str(recall))
    F1 = float(2 * precision * recall) / (precision + recall) if precision + recall> 0 else 0
    return AUC_score, F1

def train(model,data,args):
    device = th.device('cuda:0' if th.cuda.is_available() else 'cpu')
    # train
    stopper = EarlyStopping(patience=args.patience)
    optimizer = th.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    edge_data_ = edge_data(args)
    loss_func = th.nn.CrossEntropyLoss()
    for epoch in range(args.epochs):
        model.train()
        [row,col], train_label_ = edge_data_.split_train(batch_size=args.batch_size)
        for i in range(len(train_label_)):
            z = model.encode(data)
            row_, col_ = row[i].to(device), col[i].to(device)
            out = model.decode(z, th.stack([row_, col_], dim=0))
            link_labels = train_label_[i].to(device)
            loss = loss_func(out, link_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            auc,f1 = score_AUC_f1(out, link_labels)
            if i%50==0:
                print('batch {:d} | train loss {:.4f} | train auc {:.4f} | train f1 {:.4f}'.format(i,loss,auc,f1))

                # test_loss, test_auc, test_f1 = evaluate(model, data, edge_data_.test_edge_index.to(device),
                #                                     edge_data_.test_label.to(device))
                # print('---------------------------------------------------------------------------'
                #       'Score of test_data  Loss {:.4f} | auc {:.4f} | f1 {:.4f} '
                #     .format(
                #     test_loss, test_auc, test_f1
                # ))
        val_loss, val_auc, val_f1 = evaluate(model, data, edge_data_.val_edge_index.to(device), edge_data_.val_label.to(device))
        print('---------------------------------------------------------------------------'
              'Score of valid_data Epoch {:d}  Loss {:.4f} | auc {:.4f} | f1 {:.4f} '
            .format(
            epoch + 1, val_loss, val_auc, val_f1
        ))
        # early_stop = stopper.step(val_loss, val_auc, model)

        test_loss, test_auc, test_f1 = evaluate(model, data, edge_data_.test_edge_index.to(device), edge_data_.test_label.to(device))
        print('---------------------------------------------------------------------------'
              '---------------------------------------------------------------------------'
             'Score of test_data  Loss {:.4f} | auc {:.4f} | f1 {:.4f} '
            .format(
             test_loss, test_auc, test_f1
        ))

        # if early_stop:
        #     break
    # stopper.load_checkpoint(model)
    # print('Score of test_data(accuracy, micro_f1, macro_f1):',evaluate(model, data, edge_data_.test_edge_index.to(device), edge_data_.test_label.to(device) ))

def evaluate(model,data,edge_index,labels):
    model.eval()
    with th.no_grad():
        z = model.encode(data)
        out = model.decode(z, edge_index)
    loss_func = th.nn.CrossEntropyLoss()
    loss = loss_func(out, labels)
    auc,f1 = score_AUC_f1(out, labels)
    # print('Test loss {:.4f} | Test Micro f1 {:.4f} | Test Macro f1 {:.4f}'.format(
    #     loss.item(), micro_f1, macro_f1))
    return loss, auc, f1

def main(args):
    device = th.device('cuda:0' if th.cuda.is_available() else 'cpu')
    if th.cuda.is_available():
        print("Use GPU(%s) success." % th.cuda.get_device_name())

    edge_list = gen_edge_list(args)
    embed = gen_embed(args,with_paper=True).to(device) #sparse
    feat_size = embed.size()[1]
    data = gen_data(embed, edge_list).to(device)

    if args.model=='GCN':
        model= GCN(in_feats=feat_size,hid_feats=128,out_feats=2,n_layers=args.n_layers,dropout=args.dropout).to(device)
    elif args.model=='GSAGE':
        model= GSAGE(in_feats=feat_size,hid_feats=128,out_feats=2,n_layers=args.n_layers,dropout=args.dropout).to(device)
    else :
        heads = args.n_heads * args.n_layers + [1]
        model = GAT(in_feats=feat_size, hid_feats=128,out_feats=2,n_layers=args.n_layers,dropout=args.dropout,heads=heads).to(device)
    train(model,data,args)
    # evaluate(model,data,,labels)

def test_model(args):
    device = th.device('cuda:0' if th.cuda.is_available() else 'cpu')
    if args.model == 'GCN':
        model = GCN(in_feats=args.A_n, hid_feats=128, out_feats=64, n_layers=args.n_layers, dropout=args.dropout).to(
            device)
    elif args.model == 'GSAGE':
        model = GSAGE(in_feats=256, hid_feats=128, out_feats=4, n_layers=args.n_layers, dropout=args.dropout).to(device)
    else:
        heads = args.n_heads * args.n_layers + [1]
        model = GAT(in_feats=256, hid_feats=128, out_feats=4, n_layers=args.n_layers, dropout=args.dropout,
                    heads=heads).to(device)
    model.eval()
    edge_list = gen_edge_list(args)
    embed = gen_embed(args).to(device)  # sparse
    data = gen_data(embed, edge_list).to(device)

    edge_data_ = edge_data(args)
    filename = 'early_stop_2020-11-24_13-08-59.pth'

    model.load_state_dict(th.load(filename))
    print('Score of test_data(accuracy, micro_f1, macro_f1):',
          evaluate(model, data, edge_data_.test_edge_index.to(device), edge_data_.test_label.to(device)))

if __name__=='__main__':
    args = read_args()
    print(args)
    # test_model(args)
    main(args)