import sys
sys.path.append('../../')
import time
import argparse

from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from utils.pytorchtools import EarlyStopping
from utils.data import load_data
from GNN import myGAT
import dgl
import os

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

def run_model_DBLP(args):
    feats_type = args.feats_type
    features_list, adjM, dl = load_data(args.dataset)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
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
    
    edge2type = {}
    for k in dl.links['data']:
        for u,v in zip(*dl.links['data'][k].nonzero()):
            edge2type[(u,v)] = k
    for i in range(dl.nodes['total']):
        if (i,i) not in edge2type:
            edge2type[(i,i)] = len(dl.links['count'])
    for k in dl.links['data']:
        for u,v in zip(*dl.links['data'][k].nonzero()):
            if (v,u) not in edge2type:
                edge2type[(v,u)] = k+1+len(dl.links['count'])

    g = dgl.DGLGraph(adjM+(adjM.T))
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)
    g = g.to(device)
    e_feat = []
    for u, v in zip(*g.edges()):
        u = u.cpu().item()
        v = v.cpu().item()
        e_feat.append(edge2type[(u,v)])
    e_feat = torch.tensor(e_feat, dtype=torch.long).to(device)
    res_2hop = defaultdict(float)
    res_random = defaultdict(float)
    total = len(list(dl.links_test['data'].keys()))

    if True:
        train_pos, valid_pos = dl.get_train_valid_pos()#edge_types=[test_edge_type])
        num_classes = args.hidden_dim
        heads = [args.num_heads] * args.num_layers + [args.num_heads]
        net = myGAT(g, args.edge_feats, len(dl.links['count'])*2+1, in_dims, args.hidden_dim, num_classes, args.num_layers, heads, F.elu, args.dropout, args.dropout, args.slope, False, 0., decode=args.decoder)
        net.to(device)
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        # training loop
        net.train()
        early_stopping = EarlyStopping(patience=args.patience, verbose=True, save_path='checkpoint/checkpoint_{}_{}.pt'.format(args.dataset, args.num_layers))
        loss_func = nn.BCELoss()
    for epoch in range(args.epoch):
        train_pos_head_full = np.array([])
        train_pos_tail_full = np.array([])
        train_neg_head_full = np.array([])
        train_neg_tail_full = np.array([])
        r_id_full = np.array([])
        for test_edge_type in dl.links_test['data'].keys():
          train_neg = dl.get_train_neg(edge_types=[test_edge_type])[test_edge_type]
          train_pos_head_full = np.concatenate([train_pos_head_full, np.array(train_pos[test_edge_type][0])])
          train_pos_tail_full = np.concatenate([train_pos_tail_full, np.array(train_pos[test_edge_type][1])])
          train_neg_head_full = np.concatenate([train_neg_head_full, np.array(train_neg[0])])
          train_neg_tail_full = np.concatenate([train_neg_tail_full, np.array(train_neg[1])])
          r_id_full = np.concatenate([r_id_full, np.array([test_edge_type]*len(train_pos[test_edge_type][0]))])
        train_idx = np.arange(len(train_pos_head_full))
        np.random.shuffle(train_idx)
        batch_size = args.batch_size
        for step, start in enumerate(range(0, len(train_pos_head_full), args.batch_size)):
            t_start = time.time()
            # training
            net.train()
            train_pos_head = train_pos_head_full[train_idx[start:start+batch_size]]
            train_neg_head = train_neg_head_full[train_idx[start:start+batch_size]]
            train_pos_tail = train_pos_tail_full[train_idx[start:start+batch_size]]
            train_neg_tail = train_neg_tail_full[train_idx[start:start+batch_size]]
            r_id = r_id_full[train_idx[start:start+batch_size]]
            left = np.concatenate([train_pos_head, train_neg_head])
            right = np.concatenate([train_pos_tail, train_neg_tail])
            mid = np.concatenate([r_id, r_id])
            labels = torch.FloatTensor(np.concatenate([np.ones(train_pos_head.shape[0]), np.zeros(train_neg_head.shape[0])])).to(device)

            logits = net(features_list, e_feat, left, right, mid)
            logp = F.sigmoid(logits)
            train_loss = loss_func(logp, labels)

            # autograd
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            t_end = time.time()

            # print training info
            print('Epoch {:05d}, Step{:05d} | Train_Loss: {:.4f} | Time: {:.4f}'.format(epoch, step, train_loss.item(), t_end-t_start))

            t_start = time.time()
            # validation
            net.eval()
            with torch.no_grad():
                valid_pos_head = np.array([])
                valid_pos_tail = np.array([])
                valid_neg_head = np.array([])
                valid_neg_tail = np.array([])
                valid_r_id = np.array([])
                for test_edge_type in dl.links_test['data'].keys():
                    valid_neg = dl.get_valid_neg(edge_types=[test_edge_type])[test_edge_type]
                    valid_pos_head = np.concatenate([valid_pos_head, np.array(valid_pos[test_edge_type][0])])
                    valid_pos_tail = np.concatenate([valid_pos_tail, np.array(valid_pos[test_edge_type][1])])
                    valid_neg_head = np.concatenate([valid_neg_head, np.array(valid_neg[0])])
                    valid_neg_tail = np.concatenate([valid_neg_tail, np.array(valid_neg[1])])
                    valid_r_id = np.concatenate([valid_r_id, np.array([test_edge_type]*len(valid_pos[test_edge_type][0]))])
                left = np.concatenate([valid_pos_head, valid_neg_head])
                right = np.concatenate([valid_pos_tail, valid_neg_tail])
                mid = np.concatenate([valid_r_id, valid_r_id])
                labels = torch.FloatTensor(np.concatenate([np.ones(valid_pos_head.shape[0]), np.zeros(valid_neg_head.shape[0])])).to(device)
                logits = net(features_list, e_feat, left, right, mid)
                logp = F.sigmoid(logits)
                val_loss = loss_func(logp, labels)
            t_end = time.time()
            # print validation info
            print('Epoch {:05d} | Val_Loss {:.4f} | Time(s) {:.4f}'.format(
                epoch, val_loss.item(), t_end - t_start))
            # early stopping
            early_stopping(val_loss, net)
            if early_stopping.early_stop:
                print('Early stopping!')
                break
        if early_stopping.early_stop:
            print('Early stopping!')
            break

    first_flag = True
    for test_edge_type in dl.links_test['data'].keys():
        # testing with evaluate_results_nc
        net.load_state_dict(torch.load('checkpoint/checkpoint_{}_{}.pt'.format(args.dataset, args.num_layers)))
        net.eval()
        test_logits = []
        with torch.no_grad():
            test_neigh, test_label = dl.get_test_neigh()
            test_neigh = test_neigh[test_edge_type]
            test_label = test_label[test_edge_type]
            # save = np.array([test_neigh[0], test_neigh[1], test_label])
            # print(save)
            # np.savetxt(f"{args.dataset}_{test_edge_type}_label.txt", save, fmt="%i")
            save = np.loadtxt(os.path.join(dl.path, f"{args.dataset}_ini_{test_edge_type}_label.txt"), dtype=int)
            test_neigh = [save[0], save[1]]
            test_label = np.random.randint(2, size=save[0].shape[0])
            # test_label = save[2]
            left = np.array(test_neigh[0])
            right = np.array(test_neigh[1])
            mid = np.zeros(left.shape[0], dtype=np.int32)
            mid[:] = test_edge_type
            labels = torch.FloatTensor(test_label).to(device)
            logits = net(features_list, e_feat, left, right, mid)
            pred = F.sigmoid(logits).cpu().numpy()
            edge_list = np.concatenate([left.reshape((1,-1)), right.reshape((1,-1))], axis=0)
            labels = labels.cpu().numpy()
            dl.gen_file_for_evaluate(test_neigh, pred, test_edge_type, file_path=f"{args.dataset}_{args.run}.txt", flag=first_flag)
            first_flag = False
            res = dl.evaluate(edge_list, pred, labels)
            print(res)
            for k in res:
                res_2hop[k] += res[k]
        with torch.no_grad():
            test_neigh, test_label = dl.get_test_neigh_w_random()
            test_neigh = test_neigh[test_edge_type]
            test_label = test_label[test_edge_type]
            left = np.array(test_neigh[0])
            right = np.array(test_neigh[1])
            mid = np.zeros(left.shape[0], dtype=np.int32)
            mid[:] = test_edge_type
            labels = torch.FloatTensor(test_label).to(device)
            logits = net(features_list, e_feat, left, right, mid)
            pred = F.sigmoid(logits).cpu().numpy()
            edge_list = np.concatenate([left.reshape((1,-1)), right.reshape((1,-1))], axis=0)
            labels = labels.cpu().numpy()
            res = dl.evaluate(edge_list, pred, labels)
            print(res)
            for k in res:
                res_random[k] += res[k]
    for k in res_2hop:
        res_2hop[k] /= total
    for k in res_random:
        res_random[k] /= total
    print(res_2hop)
    print(res_random)

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='MRGNN testing for the DBLP dataset')
    ap.add_argument('--feats-type', type=int, default=3,
                    help='Type of the node features used. ' +
                         '0 - loaded features; ' +
                         '1 - only target node features (zero vec for others); ' +
                         '2 - only target node features (id vec for others); ' +
                         '3 - all id vec. Default is 2;' +
                        '4 - only term features (id vec for others);' + 
                        '5 - only term features (zero vec for others).')
    ap.add_argument('--hidden-dim', type=int, default=64, help='Dimension of the node hidden state. Default is 64.')
    ap.add_argument('--num-heads', type=int, default=2, help='Number of the attention heads. Default is 8.')
    ap.add_argument('--epoch', type=int, default=300, help='Number of epochs.')
    ap.add_argument('--patience', type=int, default=40, help='Patience.')
    ap.add_argument('--num-layers', type=int, default=2)
    ap.add_argument('--lr', type=float, default=5e-4)
    ap.add_argument('--dropout', type=float, default=0.5)
    ap.add_argument('--weight-decay', type=float, default=1e-4)
    ap.add_argument('--slope', type=float, default=0.01)
    ap.add_argument('--dataset', type=str)
    ap.add_argument('--edge-feats', type=int, default=32)
    ap.add_argument('--batch-size', type=int, default=1024)
    ap.add_argument('--decoder', type=str, default='distmult')
    ap.add_argument('--run', type=int, default=1)

    args = ap.parse_args()
    run_model_DBLP(args)
