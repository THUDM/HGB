import scipy.io
import urllib.request
import dgl
import math
import numpy as np
from model import *
import torch
from data_loader import data_loader
from utils.data import load_data
from utils.pytorchtools import EarlyStopping
import argparse
import time
from collections import defaultdict

ap = argparse.ArgumentParser(description='MRGNN testing for the DBLP dataset')
ap.add_argument('--feats-type', type=int, default=3,
                help='Type of the node features used. ' +
                        '0 - loaded features; ' +
                        '1 - only target node features (zero vec for others); ' +
                        '2 - only target node features (id vec for others); ' +
                        '3 - all id vec. Default is 2;' +
                    '4 - only term features (id vec for others);' + 
                    '5 - only term features (zero vec for others).')
ap.add_argument('--hidden_dim', type=int, default=64, help='Dimension of the node hidden state. Default is 64.')
ap.add_argument('--num_heads', type=int, default=8, help='Number of the attention heads. Default is 8.')
ap.add_argument('--epoch', type=int, default=300, help='Number of epochs.')
ap.add_argument('--batch_size', type=int, default=1024)
ap.add_argument('--patience', type=int, default=30, help='Patience.')
ap.add_argument('--repeat', type=int, default=1, help='Repeat the training and testing for N times. Default is 1.')
ap.add_argument('--num_layers', type=int, default=2)
ap.add_argument('--lr', type=float, default=5e-3)
ap.add_argument('--dropout', type=float, default=0.5)
ap.add_argument('--weight-decay', type=float, default=1e-4)
ap.add_argument('--slope', type=float, default=0.05)
ap.add_argument('--dataset', type=str, default='LastFM_magnn')
ap.add_argument('--decode', type=str, default='distmult')
ap.add_argument('--edge-feats', type=int, default=64)
ap.add_argument('--device', type=int, default=0)
ap.add_argument('--use_norm', type=bool, default=False)

args = ap.parse_args()
device = torch.device("cuda:"+str(args.device))

def meta2str(meta_tuple):
    return str(meta_tuple[0]) + '_' + str(meta_tuple[1])

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

# dataset = data_loader('../'+args.dataset)
features_list, _, dl = load_data(args.dataset)
edge_dict = {}
# edge_type2meta = {}
for i, meta_path in dl.links['meta'].items():
    edge_dict[(str(meta_path[0]), str(i), str(meta_path[1]))] = (torch.tensor(dl.links['data'][i].tocoo().row - dl.nodes['shift'][meta_path[0]]), torch.tensor(dl.links['data'][i].tocoo().col - dl.nodes['shift'][meta_path[1]]))
    # edge_type2meta[i] = str(i) #str(meta_path[0]) + '_' + str(meta_path[1])


node_count = {}
for i, count in dl.nodes['count'].items():
    print(i, node_count)
    node_count[str(i)] = count

G = dgl.heterograph(edge_dict, num_nodes_dict = node_count, device=device)
"""
for ntype in G.ntypes:
    G.nodes[ntype].data['inp'] = dataset.nodes['attr'][ntype]
    # print(G.nodes['attr'][ntype].shape)
"""

G.node_dict = {}
G.edge_dict = {}
for ntype in G.ntypes:
    print(ntype, dl.nodes['shift'][int(ntype)])
for ntype in G.ntypes:
    G.node_dict[ntype] = len(G.node_dict)
for etype in G.etypes:
    G.edge_dict[etype] = len(G.edge_dict)
    G.edges[etype].data['id'] = torch.ones(G.number_of_edges(etype), dtype=torch.long).to(device) * G.edge_dict[etype] 


feats_type = args.feats_type
# features_list, adjM, labels, train_val_test_idx, dl = load_data(args.dataset)
features_list = [mat2tensor(features).to(device) for features in features_list]

if feats_type == 0:
    in_dims = [features.shape[1] for features in features_list]
elif feats_type == 1 or feats_type == 5:
    save = 0 if feats_type == 1 else 2
    in_dims = []#[features_list[0].shape[1]] + [10] * (len(features_list) - 1)
    for i in range(0, len(features_list)):
        if i == save:
            in_dims.append(features_list[i].shape[1])
        else:
            in_dims.append(10)
            features_list[i] = torch.zeros((features_list[i].shape[0], 10)).to(device)
elif feats_type == 2 or feats_type == 4:
    save = feats_type - 2
    in_dims = [features.shape[0] for features in features_list]
    for i in range(0, len(features_list)):
        if i == save:
            in_dims[i] = features_list[i].shape[1]
            continue
        dim = features_list[i].shape[0]
        indices = np.vstack((np.arange(dim), np.arange(dim)))
        indices = torch.LongTensor(indices)
        values = torch.FloatTensor(np.ones(dim))
        features_list[i] = torch.sparse.FloatTensor(indices, values, torch.Size([dim, dim])).to(device)
elif feats_type == 3:
    in_dims = [features.shape[0] for features in features_list]
    for i in range(len(features_list)):
        dim = features_list[i].shape[0]
        indices = np.vstack((np.arange(dim), np.arange(dim)))
        indices = torch.LongTensor(indices)
        values = torch.FloatTensor(np.ones(dim))
        features_list[i] = torch.sparse.FloatTensor(indices, values, torch.Size([dim, dim])).to(device)


for ntype in G.ntypes:
    G.nodes[ntype].data['inp'] = features_list[int(ntype)]#.to(device)

train_pos, valid_pos = dl.get_train_valid_pos()


# labels = torch.LongTensor(labels).to(device)
# train_idx = train_val_test_idx['train_idx']
# train_idx = np.sort(train_idx)
# val_idx = train_val_test_idx['val_idx']
# val_idx = np.sort(val_idx)
# test_idx = train_val_test_idx['test_idx']
# test_idx = np.sort(test_idx)

model = HGT(G, n_inps=in_dims, n_hid=args.hidden_dim, n_layers=args.num_layers, n_heads=args.num_heads, use_norm = args.use_norm, decode=args.decode).to(device)

optimizer = torch.optim.AdamW(model.parameters())#, lr=args.lr, weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, total_steps=300, max_lr = 1e-2, pct_start=0.05)

early_stopping = EarlyStopping(patience=args.patience, verbose=True, save_path='checkpoint/checkpoint_{}_{}.pt'.format(args.dataset, args.num_layers))
train_step = 0
loss_func = torch.nn.BCELoss()
train_pos_head_full = np.array([])
train_pos_tail_full = np.array([])
train_neg_head_full = np.array([])
train_neg_tail_full = np.array([])
r_id_full = []
for test_edge_type in dl.links_test['data'].keys():
    print(test_edge_type)
    train_neg = dl.get_train_neg()[test_edge_type]
    train_pos_head_full = np.concatenate([train_pos_head_full, np.array(train_pos[test_edge_type][0])])
    train_pos_tail_full = np.concatenate([train_pos_tail_full, np.array(train_pos[test_edge_type][1])])
    train_neg_head_full = np.concatenate([train_neg_head_full, np.array(train_neg[0])])
    train_neg_tail_full = np.concatenate([train_neg_tail_full, np.array(train_neg[1])])
    # r_id_full = np.concatenate([r_id_full, np.array([test_edge_type]*len(train_pos[test_edge_type][0]))])
    r_id_full.extend([int(test_edge_type)]* len(train_pos[test_edge_type][0]))

res_2hop = defaultdict(float)
res_random = defaultdict(float)
total = len(list(dl.links_test['data'].keys()))
# print(r_id_full)
r_id_full = np.asarray(r_id_full)
train_step = 0 

for epoch in range(args.epoch):
    train_idx = np.arange(len(train_pos_head_full))
    np.random.shuffle(train_idx)
    batch_size = len(train_pos_head_full)#len(train_pos_head_full)#args.batch_size
    for step, start in enumerate(range(0, len(train_pos_head_full), batch_size)):
        t_start = time.time()
        model.train()
        train_pos_head = train_pos_head_full[train_idx[start:start+batch_size]]
        train_neg_head = train_neg_head_full[train_idx[start:start+batch_size]]
        train_pos_tail = train_pos_tail_full[train_idx[start:start+batch_size]]
        train_neg_tail = train_neg_tail_full[train_idx[start:start+batch_size]]
        r_id = r_id_full[train_idx[start:start+batch_size]]       
        left = np.concatenate([train_pos_head, train_neg_head])
        right = np.concatenate([train_pos_tail, train_neg_tail])
        mid = np.concatenate([r_id, r_id])
        # left_shift = np.asarray([dl.nodes['shift'][x] for x in mid[:,0]])
        # right_shift = np.asarray([dl.nodes['shift'][x] for x in mid[:,1]])

        labels = torch.FloatTensor(np.concatenate([np.ones(train_pos_head.shape[0]), np.zeros(train_neg_head.shape[0])])).to(device)

        logits = model(G, left, right, mid)
        logp = F.sigmoid(logits)
        train_loss = loss_func(logp, labels)
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        train_step += 1
        scheduler.step(train_step)
        t_end = time.time()
        print('Epoch {:05d}, Step{:05d} | Train_Loss: {:.4f} | Time: {:.4f}'.format(epoch, step, train_loss.item(), t_end-t_start))
        t_start = time.time()
            # validation
        model.eval()
        with torch.no_grad():
            valid_pos_head = np.array([])
            valid_pos_tail = np.array([])
            valid_neg_head = np.array([])
            valid_neg_tail = np.array([])
            valid_r_id = []
            for test_edge_type in dl.links_test['data'].keys():
                valid_neg = dl.get_valid_neg()[test_edge_type]
                valid_pos_head = np.concatenate([valid_pos_head, np.array(valid_pos[test_edge_type][0])])
                valid_pos_tail = np.concatenate([valid_pos_tail, np.array(valid_pos[test_edge_type][1])])
                valid_neg_head = np.concatenate([valid_neg_head, np.array(valid_neg[0])])
                valid_neg_tail = np.concatenate([valid_neg_tail, np.array(valid_neg[1])])
                valid_r_id.extend([int(test_edge_type)]*len(valid_pos[test_edge_type][0]))
                # print(valid_r_id, valid_neg_tail.shape, len(valid_pos[test_edge_type][0]), [edge_type2meta[test_edge_type]]*len(valid_pos[test_edge_type][0]))
            left = np.concatenate([valid_pos_head, valid_neg_head])
            right = np.concatenate([valid_pos_tail, valid_neg_tail])
            mid = np.concatenate([valid_r_id, valid_r_id])
            labels = torch.FloatTensor(np.concatenate([np.ones(valid_pos_head.shape[0]), np.zeros(valid_neg_head.shape[0])])).to(device)
            logits = model(G, left, right, mid)
            logp = F.sigmoid(logits)
            val_loss = loss_func(logp, labels)
        t_end = time.time()
        # print validation info
        print('Epoch {:05d} | Val_Loss {:.4f} | Time(s) {:.4f}'.format(
            epoch, val_loss.item(), t_end - t_start))
        # early stopping
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print('Early stopping!')
            break
    if early_stopping.early_stop:
        print('Early stopping!')
        break

for test_edge_type in dl.links_test['data'].keys():
    # testing with evaluate_results_nc
    model.load_state_dict(torch.load('checkpoint/checkpoint_{}_{}.pt'.format(args.dataset, args.num_layers)))
    model.eval()
    test_logits = []
    with torch.no_grad():
        test_neigh, test_label = dl.get_test_neigh()
        test_neigh = test_neigh[test_edge_type]
        test_label = test_label[test_edge_type]
        left = np.array(test_neigh[0])
        right = np.array(test_neigh[1])
        # mid = np.zeros(left.shape[0], dtype=np.int32)
        mid = [int(test_edge_type)] * left.shape[0]
        mid = np.asarray(mid)
        labels = torch.FloatTensor(test_label).to(device)
        logits = model(G, left, right, mid)
        pred = F.sigmoid(logits).cpu().numpy()
        edge_list = np.concatenate([left.reshape((1,-1)), right.reshape((1,-1))], axis=0)
        labels = labels.cpu().numpy()
        res = dl.evaluate(edge_list, pred, labels)
        # print(res)
        for k in res:
            res_2hop[k] += res[k]
    with torch.no_grad():
        test_neigh, test_label = dl.get_test_neigh_w_random()
        test_neigh = test_neigh[test_edge_type]
        test_label = test_label[test_edge_type]
        left = np.array(test_neigh[0])
        right = np.array(test_neigh[1])
        # mid = np.zeros(left.shape[0], dtype=np.int32)
        # mid[:] = test_edge_type
        mid = [int(test_edge_type)] * left.shape[0]
        labels = torch.FloatTensor(test_label).to(device)
        logits = model(G, left, right, mid)
        pred = F.sigmoid(logits).cpu().numpy()
        edge_list = np.concatenate([left.reshape((1,-1)), right.reshape((1,-1))], axis=0)
        labels = labels.cpu().numpy()
        res = dl.evaluate(edge_list, pred, labels)
        # print(res)
        for k in res:
            res_random[k] += res[k]
for k in res_2hop:
    res_2hop[k] /= total
for k in res_random:
    res_random[k] /= total
print(res_2hop)
print(res_random)
