import networkx as nx
import numpy as np
import scipy
import pickle
import scipy.sparse as sp

def load_data(prefix='DBLP'):
    from scripts.data_loader import data_loader
    dl = data_loader('../../data/'+prefix)
    """new_feat = [[] for _ in range(len(dl.nodes['count']))]
    for i in range(len(dl.links['count'])):
        th = dl.links['data'][i].sum(axis=1)
        for j in range(len(dl.nodes['count'])):
            beg = dl.nodes['shift'][j]
            end = beg + dl.nodes['count'][j]
            new_feat[j].append(sp.csr_matrix(th[beg:end]).reshape([-1, 1]))
    new_feat = [sp.hstack(x) for x in new_feat]
    for i in range(len(new_feat)):
        norm = new_feat[i].sum(axis=1).reshape([-1, 1])
        new_feat[i] = sp.hstack([new_feat[i]/norm, sp.eye(new_feat[i].shape[0])])"""
    features = []
    for i in range(len(dl.nodes['count'])):
        th = dl.nodes['attr'][i]
        if th is None:
            features.append(sp.eye(dl.nodes['count'][i]))
        else:
            features.append(th)
    adjs = list(dl.links['data'].values()) # # # # # # # # # + [sp.eye(dl.nodes['total']).tocsr()]
    labels = np.zeros((dl.nodes['count'][0], dl.labels_train['num_classes']), dtype=int)
    val_ratio = 0.2
    train_idx = np.nonzero(dl.labels_train['mask'])[0]
    np.random.shuffle(train_idx)
    split = int(train_idx.shape[0]*val_ratio)
    val_idx = train_idx[:split]
    train_idx = train_idx[split:]
    train_idx = np.sort(train_idx)
    val_idx = np.sort(val_idx)
    test_idx = np.nonzero(dl.labels_test['mask'])[0]
    labels[train_idx] = dl.labels_train['data'][train_idx]
    labels[val_idx] = dl.labels_train['data'][val_idx]
    labels = labels.argmax(axis=1)
    train_val_test_idx = {}
    train_val_test_idx['train_idx'] = train_idx
    train_val_test_idx['val_idx'] = val_idx
    train_val_test_idx['test_idx'] = test_idx
    return features,\
           adjs, \
           labels,\
           train_val_test_idx,\
            dl
