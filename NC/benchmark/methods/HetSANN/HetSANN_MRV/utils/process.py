import numpy as np
import networkx as nx
import scipy.sparse as sp
import scipy.io as sio
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys
import random
import os
from scripts.data_loader import data_loader
import json
from collections import OrderedDict

import pdb;

def mkdir(file_path):
    if not os.path.exists(file_path):
        os.makedirs(file_path)
"""
 Prepare adjacency matrix by expanding up to a given neighbourhood.
 This will insert loops on every node.
 Finally, the matrix is converted to bias vectors.
 Expected shape: [graph, nodes, nodes]
"""
def adj_to_bias(adj, sizes, nhood=1):
    nb_graphs = adj.shape[0]
    mt = np.empty(adj.shape)
    for g in range(nb_graphs):
        mt[g] = np.eye(adj.shape[1])
        for _ in range(nhood):
            mt[g] = np.matmul(mt[g], (adj[g] + np.eye(adj.shape[1])))
        for i in range(sizes[g]):
            for j in range(sizes[g]):
                if mt[g][i][j] > 0.0:
                    mt[g][i][j] = 1.0
    return -1e9 * (1.0 - mt)


###############################################
# This section of code adapted from tkipf/gcn #
###############################################

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx

def standardize_data(f, train_mask):
    """Standardize feature matrix and convert to tuple representation"""
    # standardize data
    f = f.todense()
    mu = f[train_mask == True, :].mean(axis=0)
    sigma = f[train_mask == True, :].std(axis=0)
    f = f[:, np.squeeze(np.array(sigma > 0))]
    mu = f[train_mask == True, :].mean(axis=0)
    sigma = f[train_mask == True, :].std(axis=0)
    f = (f - mu) / sigma
    return f

def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense(), sparse_to_tuple(features)

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)

def preprocess_adj_bias(adj):
    num_nodes = adj.shape[0]
    adj = adj + sp.eye(num_nodes)  # self-loop
    adj[adj > 0.0] = 1.0
    if not sp.isspmatrix_coo(adj):
        adj = adj.tocoo()
    adj = adj.astype(np.float32)
    indices = np.vstack((adj.col, adj.row)).transpose()
    return indices, adj.data, adj.shape

def preprocess_adj_hete(adj):
    if not sp.isspmatrix_coo(adj):
        adj = adj.tocoo()
    adj = adj.astype(np.float32)
    adj = adj.T # transpose the adjacency matrix here
    indices = np.vstack((adj.row, adj.col)).transpose()
    return indices, adj.data, adj.shape

def load_imdb(dataset=r'../data/imdb/IMDB_processed.mat', target_node=[0]):
    data = sio.loadmat(dataset)
    label = [data['label']]
    MvsA = data['MvsA']
    MvsD = data['MvsD']
    M_features = data['MvsP'].tocsr()
    A_features = sp.csr_matrix((MvsA.shape[1], 1))
    D_features = sp.csr_matrix((MvsD.shape[1], 1))
    M_loop = sp.eye(MvsA.shape[0])
    A_loop = sp.eye(MvsA.shape[1])
    D_loop = sp.eye(MvsD.shape[1])
    adj_list = [M_loop, A_loop, D_loop, MvsA, MvsA.T, MvsD, MvsD.T]
    adj_type = [(0, 0), (1, 1), (2, 2), (0, 1), (1, 0), (0, 2), (2, 0)]
    edge_list = [(3, 4), (5, 6)]
    features = [M_features, A_features, D_features]
    return adj_list, adj_type, edge_list, features, label

def load_dblp(dataset=r'../data/DBLP_four_area/DBLP_processed.mat', target_node=[0]):
    data = sio.loadmat(dataset)
    node_label_list = ['paper_label', 'author_label']
    PvsA = data['PvsA']
    PvsC = data['PvsC']
    P_features = data['PvsT'].tocsr()
    A_features = sp.csr_matrix((PvsA.shape[1], 1))
    C_features = sp.csr_matrix((PvsC.shape[1], 1))
    P_loop = sp.eye(PvsA.shape[0])
    A_loop = sp.eye(PvsA.shape[1])
    C_loop = sp.eye(PvsC.shape[1])
    adj_list = [P_loop, A_loop, C_loop, PvsA, PvsA.T, PvsC, PvsC.T]
    adj_type = [(0, 0), (1, 1), (2, 2), (0, 1), (1, 0), (0, 2), (2, 0)]
    edge_list = [(3, 4), (5, 6)]
    features = [P_features, A_features, C_features]
    label = []
    for t in target_node:
        if t < len(node_label_list):
            label.append(data[node_label_list[t]].toarray())
        else:
            print("type %s node have not label" %(t))
            exit(0)
    return adj_list, adj_type, edge_list, features, label

def load_aminer(dataset=r'../data/Aminer/Aminer_processed.mat', target_node=[0]):
    data = sio.loadmat(dataset)
    node_label_list = ['PvsC', 'AvsC']
    PvsA = data['PvsA']
    PvsP = data['PvsP']
    AvsA = data['AvsA']
    P_features = sp.csr_matrix(data['PvsF'])
    A_features = sp.csr_matrix(data['AvsF'])
    P_loop = sp.eye(PvsA.shape[0])
    A_loop = sp.eye(PvsA.shape[1])
    adj_list = [P_loop, A_loop, PvsA, PvsA.T, PvsP, PvsP.T, AvsA]
    adj_type = [(0, 0), (1, 1), (0, 1), (1, 0), (0, 0), (0, 0), (1, 1)]
    edge_list = [(2, 3), (4, 5)]
    features = [P_features, A_features]
    label = []
    for t in target_node:
        if t < len(node_label_list):
            label.append(data[node_label_list[t]].toarray())
        else:
            print("type %s node have not label" %(t))
            exit(0)
    return adj_list, adj_type, edge_list, features, label


def load_data(dataset_str, target_node=[0], feats_type=0):
    dataset = ['ACM', 'DBLP', 'IMDB', 'FREEBASE']
    if dataset_str.upper() in dataset:
        dataset = None
    else:
        raise ValueError

    path = '../../../data/' + dataset_str
    loader = data_loader(path)
    if dataset_str.upper() == "DBLP":
        loader.get_sub_graph([0,1,2])

    '''
    with open(path+"/info.dat") as info:
        info_dict = json.load(info)
    with open(path+"/meta.dat") as meta:
        meta_dict = json.load(meta)
    '''
    adj_type = list(loader.links['meta'].values())

    edge_list = []
    for (a, b) in adj_type:
        if (b, a) in adj_type and b != a:
            index1 = adj_type.index((a, b))
            index2 = adj_type.index((b, a))
            if (index1, index2) not in edge_list and (index2, index1) not in edge_list:
                edge_list.append((index1, index2))

    for k in loader.nodes['attr'].keys():
        adj_type.append((k, k))

    diction = loader.links['data']
    adj = list(OrderedDict(sorted(diction.items())).values())
    
    adj_list = []

    for l, (a, b) in zip(adj, adj_type):
        adj_list.append(sp.csr_matrix(l[loader.nodes['shift'][a] : loader.nodes['shift'][a] + loader.nodes['count'][a], loader.nodes['shift'][b] : loader.nodes['shift'][b] + loader.nodes['count'][b]]))
    
    for k in loader.nodes['attr'].keys():
        adj_list.append(sp.eye(loader.nodes['count'].get(k)))
    #pdb.set_trace()

    #features = [[] if f is None else f for f in list(loader.nodes['attr'].values())]

    features = []

    for f, c in zip(list(loader.nodes['attr'].values()), loader.nodes['count'].values()):
        if f is None:
            features.append(sp.eye(c))
        else:
            features.append(sp.csr_matrix(f))

    if feats_type == 2:
        for i in range(len(features)):
            if i not in target_node:
                # spdb.set_trace()
                features[i] = sp.eye(features[i].shape[0])
    elif feats_type == 3:
        for i in range(len(features)):
            features[i] = sp.eye(features[i].shape[0])

    labels_train = []

    for t in target_node:
        if t < len(loader.nodes['attr']):
            labels_train.append(loader.labels_train['data'][loader.nodes['shift'][t] : loader.nodes['shift'][t] + loader.nodes['count'][t], : ])
        else:
            print("type %s node have not label" %(t))
            exit(0)

    labels_test = []
    for t in target_node:
        if t < len(loader.nodes['attr']):
            labels_test.append(loader.labels_test['data'][loader.nodes['shift'][t] : loader.nodes['shift'][t] + loader.nodes['count'][t], : ])
        else:
            print("type %s node have not label" %(t))
            exit(0)
    #pdb.set_trace()
    return adj_list, adj_type, edge_list, features, labels_train, labels_test

def load_heterogeneous_data(dataset_str,
                            train_rate=0.1,
                            val_rate=0.1,
                            test_rate=0.1,
                            feats_type=0,
                            target_node=[0]): # {'imdb', 'dblp', 'aminer'}

    adj, adj_type, edge_list, features, labels_train, labels_test = load_data(dataset_str, target_node=target_node, feats_type=feats_type)
    
    if len(labels_train) + len(labels_test) < 1:
        print("Error: nodes have not labels!")
        exit(0)
    
    ''' training and validation '''
    sum_training_labels = [np.sum(l, axis=1) for l in labels_train]
    all_training_sample_with_label = [np.where(l>0)[0] for l in sum_training_labels]

    val_ratio = 0.2
    num_targets = [len(l) for l in all_training_sample_with_label]
    num_train = [int(num * (1 - val_ratio)) for num in num_targets]
    num_val = [int(num * val_ratio) for num in num_targets]
    
    idx_random = all_training_sample_with_label#list(range(num_targets))
    for i in range(len(all_training_sample_with_label)):
        random.shuffle(idx_random[i])
    idx_train = [idx_random[i][0 : num_train[i]] for i in range(len(idx_random))]
    idx_val = [idx_random[i][num_train[i] : num_train[i] + num_val[i]] for i in range(len(idx_random))]
    
    train_mask = [sample_mask(idx_train[i], labels_train[i].shape[0]) for i in range(len(labels_train))]
    val_mask = [sample_mask(idx_val[i], labels_train[i].shape[0]) for i in range(len(labels_train))]

    y_train = [np.zeros(labels_train[i].shape) for i in range(len(labels_train))]
    y_val = [np.zeros(labels_train[i].shape) for i in range(len(labels_train))]
    for i in range(len(labels_train)):
        y_train[i][train_mask[i], :] = labels_train[i][train_mask[i], :]
        y_val[i][val_mask[i], :] = labels_train[i][val_mask[i], :]

    ''' test '''
    sum_labels_test = [np.sum(l, axis=1) for l in labels_test]
    all_sample_with_label_test = [np.where(l>0)[0] for l in sum_labels_test]
    num_test_targets = [len(l) for l in all_sample_with_label_test]
    num_test = [num_test_targets]
    
    test_idx_random = all_sample_with_label_test#list(range(num_targets))
    for i in range(len(all_sample_with_label_test)):
        random.shuffle(test_idx_random[i])
    idx_test = [test_idx_random[i] for i in range(len(test_idx_random))]
    
    test_mask = [sample_mask(idx_test[i], labels_test[i].shape[0]) for i in range(len(labels_test))]

    y_test = [np.zeros(labels_test[i].shape) for i in range(len(labels_test))]
    for i in range(len(labels_test)):
        y_test[i][test_mask[i], :] = labels_test[i][test_mask[i], :]
    
    #pdb.set_trace()
    print("all adj shape:", [a.shape for a in adj])
    print("responding adj type:", [a for a in adj_type])
    print("all features of nodes shape:", [f.shape for f in features])
    print("all y_train num:", [len(y) for y in idx_train])
    print("all y_val num:", [len(y) for y in idx_val])
    print("all y_test num:", [len(y) for y in idx_test])
    return adj, adj_type, edge_list, features, y_train, y_val, y_test, train_mask, val_mask, test_mask
