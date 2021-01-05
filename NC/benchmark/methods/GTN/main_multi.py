import sys
sys.path.append('../../')
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from model import GTN
import pdb
import pickle
import argparse
from utils import f1_score
from scripts.data_loader import data_loader
import scipy.sparse as sp

def load_data(args):
    dataset, full, feats_type = args.dataset, args.full, args.feats_type
    drop_feat = []
    if dataset == 'DBLP':
        if feats_type == 1:
            drop_feat = [0,1,3]
        elif feats_type == 2:
            drop_feat = [0]
    if dataset == 'IMDB':
        if feats_type == 1:
            drop_feat = [0]
    dl = data_loader('../../data/'+dataset)
    if dataset == 'DBLP' and not full:
        dl.get_sub_graph([0,1,3])
    if dataset == 'ACM' and not full:
        dl.get_sub_graph([0,1,2])
    if dataset == 'IMDB' and not full:
        dl.get_sub_graph([0,1,2])
    edges = list(dl.links['data'].values())
    node_features = []
    w_ins = []
    for k in dl.nodes['attr']:
        th = dl.nodes['attr'][k]
        if th is None or k in drop_feat:
            cnt = dl.nodes['count'][k]
            node_features.append(sp.eye(cnt))
            w_ins.append(cnt)
        else:
            node_features.append(th)
            w_ins.append(th.shape[1])
    val_ratio = 0.2
    train_idx = np.nonzero(dl.labels_train['mask'])[0]
    np.random.shuffle(train_idx)
    split = int(train_idx.shape[0]*val_ratio)
    val_idx = train_idx[:split]
    train_idx = train_idx[split:]
    train_idx = np.sort(train_idx)
    val_idx = np.sort(val_idx)
    test_idx = np.nonzero(dl.labels_test['mask'])[0]
    labels = np.zeros((dl.nodes['count'][0], dl.labels_train['num_classes']), dtype=int)
    labels[train_idx] = dl.labels_train['data'][train_idx]
    labels[val_idx] = dl.labels_train['data'][val_idx]
    #labels = labels.argmax(axis=1)
    train_label = labels[train_idx]
    val_label = labels[val_idx]
    train_label = (train_idx.tolist(), train_label) #list(zip(train_idx.tolist(), train_label.tolist()))
    val_label = (val_idx.tolist(), val_label) #list(zip(val_idx.tolist(), val_label.tolist()))
    labels = [train_label, val_label, test_idx.tolist()]
    return node_features, edges, labels, dl, w_ins

def sp_to_spt(mat):
    coo = mat.tocoo()
    values = coo.data
    indices = np.vstack((coo.row, coo.col))

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = coo.shape

    return torch.sparse.FloatTensor(i, v, torch.Size(shape))

def mat2tensor(mat):
    if type(mat) is np.ndarray:
        return torch.from_numpy(mat).type(torch.FloatTensor)
    return sp_to_spt(mat)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str,
                        help='Dataset')
    parser.add_argument('--epoch', type=int, default=40,
                        help='Training Epochs')
    parser.add_argument('--node_dim', type=int, default=64,
                        help='Node dimension')
    parser.add_argument('--num_channels', type=int, default=2,
                        help='number of channels')
    parser.add_argument('--lr', type=float, default=0.005,
                        help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.001,
                        help='l2 reg')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='number of layer')
    parser.add_argument('--norm', type=str, default='true',
                        help='normalization')
    parser.add_argument('--adaptive_lr', type=str, default='false',
                        help='adaptive learning rate')
    parser.add_argument('--full', type=bool, default=False)
    parser.add_argument('--feats-type', type=int, default=0)
    args = parser.parse_args()
    print(args)
    epochs = args.epoch
    node_dim = args.node_dim
    num_channels = args.num_channels
    lr = args.lr
    weight_decay = args.weight_decay
    num_layers = args.num_layers
    norm = args.norm
    adaptive_lr = args.adaptive_lr

    """with open('data/'+args.dataset+'/node_features.pkl','rb') as f:
        node_features = pickle.load(f)
    with open('data/'+args.dataset+'/edges.pkl','rb') as f:
        edges = pickle.load(f)
    with open('data/'+args.dataset+'/labels.pkl','rb') as f:
        labels = pickle.load(f)"""
    node_features, edges, labels, dl, w_ins = load_data(args)
    num_nodes = edges[0].shape[0]

    for i,edge in enumerate(edges):
        if i ==0:
            A = torch.from_numpy(edge.todense()).type(torch.FloatTensor).unsqueeze(-1)
        else:
            A = torch.cat([A,torch.from_numpy(edge.todense()).type(torch.FloatTensor).unsqueeze(-1)], dim=-1)
    A = torch.cat([A,torch.eye(num_nodes).type(torch.FloatTensor).unsqueeze(-1)], dim=-1)
    
    node_features = [mat2tensor(x) for x in node_features] # torch.from_numpy(node_features).type(torch.FloatTensor)
    train_node = torch.from_numpy(np.array(labels[0][0])).type(torch.LongTensor)
    train_target = torch.from_numpy(np.array(labels[0][1])).type(torch.FloatTensor)
    valid_node = torch.from_numpy(np.array(labels[1][0])).type(torch.LongTensor)
    valid_target = torch.from_numpy(np.array(labels[1][1])).type(torch.FloatTensor)
    test_node = torch.from_numpy(np.array(labels[2])).type(torch.LongTensor)
    #test_target = torch.from_numpy(np.array(labels[2])[:,1]).type(torch.LongTensor)
    
    num_classes = dl.labels_train['num_classes']#torch.max(train_target).item()+1
    final_f1 = 0
    for l in range(1):
        model = GTN(num_edge=A.shape[-1],
                            num_channels=num_channels,
                            w_ins = w_ins, #node_features.shape[1],
                            w_out = node_dim,
                            num_class=num_classes,
                            num_layers=num_layers,
                            norm=norm)
        if adaptive_lr == 'false':
            optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0.001)
        else:
            optimizer = torch.optim.Adam([{'params':model.weights},
                                        {'params':model.linear1.parameters()},
                                        {'params':model.linear2.parameters()},
                                        {"params":model.layers.parameters(), "lr":0.5}
                                        ], lr=0.005, weight_decay=0.001)
        loss_func = nn.BCELoss() #nn.CrossEntropyLoss()
        # Train & Valid & Test
        best_val_loss = 10000
        best_test_loss = 10000
        best_train_loss = 10000
        best_train_f1 = 0
        best_val_f1 = 0
        best_test_f1 = 0
        
        for i in range(epochs):
            for param_group in optimizer.param_groups:
                if param_group['lr'] > 0.005:
                    param_group['lr'] = param_group['lr'] * 0.9
            print('Epoch:  ',i+1)
            model.zero_grad()
            model.train()
            loss,y_train,Ws = model(A, node_features, train_node, None)#, train_target)
            y_train = F.sigmoid(y_train)
            loss = loss_func(y_train, train_target)
            train_f1 = torch.mean(f1_score(y_train.detach()>0.5, train_target, num_classes=num_classes)).cpu().numpy()
            print('Train - Loss: {}, Macro_F1: {}'.format(loss.detach().cpu().numpy(), train_f1))
            loss.backward()
            optimizer.step()
            model.eval()
            # Valid
            with torch.no_grad():
                val_loss, y_valid,_ = model.forward(A, node_features, valid_node, None)#, valid_target)
                y_valid = F.sigmoid(y_valid)
                val_loss = loss_func(y_valid, valid_target)
                val_f1 = torch.mean(f1_score(y_valid>0.5, valid_target, num_classes=num_classes)).cpu().numpy()
                print('Valid - Loss: {}, Macro_F1: {}'.format(val_loss.detach().cpu().numpy(), val_f1))
                test_loss, y_test,W = model.forward(A, node_features, test_node, None)
                pred = F.sigmoid(y_test).cpu().numpy()>0.5
                test_f1 = dl.evaluate(pred)
                print(test_f1)
                #test_f1 = torch.mean(f1_score(torch.argmax(y_test,dim=1), test_target, num_classes=num_classes)).cpu().numpy()
                #print('Test - Loss: {}, Macro_F1: {}\n'.format(test_loss.detach().cpu().numpy(), test_f1))
            if val_f1 > best_val_f1:
                best_val_loss = val_loss.detach().cpu().numpy()
                #best_test_loss = test_loss.detach().cpu().numpy()
                best_train_loss = loss.detach().cpu().numpy()
                best_train_f1 = train_f1
                best_val_f1 = val_f1
                best_test_f1 = test_f1 
        print('---------------Best Results--------------------')
        print('Train - Loss: {}, Macro_F1: {}'.format(best_train_loss, best_train_f1))
        print('Valid - Loss: {}, Macro_F1: {}'.format(best_val_loss, best_val_f1))
        print(best_test_f1)
        #print('Test - Loss: {}, Macro_F1: {}'.format(best_test_loss, best_test_f1))
        final_f1 += best_test_f1
