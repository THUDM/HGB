import dgl
from model import GCN, GAT
from utils import load_data
from utils import EarlyStopping
import utils
import numpy as np
import torch.nn as nn
import torch
from collections import defaultdict
import argparse
import time
import sys

sys.path.append('../../')


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
    device = torch.device(
        'cuda:'+str(args.gpu) if torch.cuda.is_available() and args.gpu > 0 else 'cpu')
    features_list = [mat2tensor(features).to(device)
                     for features in features_list]
    in_dims = []
    if feats_type == 0:
        in_dims = [features.shape[1] for features in features_list]
    elif feats_type == 1 or feats_type == 5:
        save = 0 if feats_type == 1 else 2
        # [features_list[0].shape[1]] + [10] * (len(features_list) - 1)
        in_dims = []
        for i in range(0, len(features_list)):
            if i == save:
                in_dims.append(features_list[i].shape[1])
            else:
                in_dims.append(10)
                features_list[i] = torch.zeros(
                    (features_list[i].shape[0], 10)).to(device)
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
            features_list[i] = torch.sparse.FloatTensor(
                indices, values, torch.Size([dim, dim])).to(device)
    elif feats_type == 3:
        in_dims = [features.shape[0] for features in features_list]
        for i in range(len(features_list)):
            dim = features_list[i].shape[0]
            indices = np.vstack((np.arange(dim), np.arange(dim)))
            indices = torch.LongTensor(indices)
            values = torch.FloatTensor(np.ones(dim))
            features_list[i] = torch.sparse.FloatTensor(
                indices, values, torch.Size([dim, dim])).to(device)
    rel_num = len(dl.links['count'])

    edge2type = {}
    for k in dl.links['data']:
        for u, v in zip(*dl.links['data'][k].nonzero()):
            edge2type[(u, v)] = k
    for i in range(dl.nodes['total']):
        if (i, i) not in edge2type:
            edge2type[(i, i)] = len(dl.links['count'])
    for k in dl.links['data']:
        for u, v in zip(*dl.links['data'][k].nonzero()):
            if (v, u) not in edge2type:
                edge2type[(v, u)] = k+1+len(dl.links['count'])

    g = dgl.DGLGraph(adjM+(adjM.T))
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)
    g = g.to(device)
    e_feat = []

    test_norm = utils.comp_deg_norm(g.cpu())
    edge_norm = utils.node_norm_to_edge_norm(
        g.cpu(), torch.from_numpy(test_norm).view(-1, 1)).to(device)

    for u, v in zip(*g.edges()):
        u = u.cpu().item()
        v = v.cpu().item()
        e_feat.append(edge2type[(u, v)])
    e_feat = torch.tensor(e_feat, dtype=torch.long).to(device)
    res_2hop = defaultdict(float)
    res_random = defaultdict(float)
    total = len(list(dl.links_test['data'].keys()))

    if True:
        # edge_types=[test_edge_type])
        train_pos, valid_pos = dl.train_pos, dl.valid_pos
        heads = [args.num_heads] * args.num_layers + [args.num_heads]
        if args.model == 'GCN':
            net = GCN(in_feats=in_dims, hid_feats=args.hidden_dim, n_layers=args.num_layers, dropout=args.dropout,
                      decoder=args.decoder, rel_num=rel_num)
        elif args.model == "GAT":
            heads = args.n_heads * args.n_layers + [1]
            net = GAT(in_feats=in_dims, hid_feats=args.hidden_dim, n_layers=args.num_layers, heads=heads,
                      dropout=args.dropout, decoder=args.decoder, rel_num=rel_num)
        else:
            exit('please input true model_type within [GCN, GAT]')
        net.to(device)
        optimizer = torch.optim.Adam(
            net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        # training loop
        net.train()
        early_stopping = EarlyStopping(patience=args.patience, verbose=True,
                                       save_path='checkpoint/'+args.model+'checkpoint_{}_{}.pt'.format(args.dataset, args.num_layers))
        loss_func = nn.BCELoss()
    for epoch in range(args.epoch):
        train_pos_head_full = np.array([])
        train_pos_tail_full = np.array([])
        train_neg_head_full = np.array([])
        train_neg_tail_full = np.array([])
        r_id_full = np.array([])
        for test_edge_type in dl.links_test['data'].keys():
            train_neg = dl.get_train_neg(edge_types=[test_edge_type])[
                test_edge_type]
            train_pos_head_full = np.concatenate(
                [train_pos_head_full, np.array(train_pos[test_edge_type][0])])
            train_pos_tail_full = np.concatenate(
                [train_pos_tail_full, np.array(train_pos[test_edge_type][1])])
            train_neg_head_full = np.concatenate(
                [train_neg_head_full, np.array(train_neg[0])])
            train_neg_tail_full = np.concatenate(
                [train_neg_tail_full, np.array(train_neg[1])])
            r_id_full = np.concatenate([r_id_full, np.array(
                [test_edge_type]*len(train_pos[test_edge_type][0]))])
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
            labels = torch.FloatTensor(np.concatenate(
                [np.ones(train_pos_head.shape[0]), np.zeros(train_neg_head.shape[0])])).to(device)

            embed = net.encode(g, features_list)
            logits = net.decode(mid, embed[left], embed[right])
            pred = torch.sigmoid(logits)
            train_loss = loss_func(pred, labels)

            # autograd
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            t_end = time.time()

            # print training info
            # print('Epoch {:05d}, Step{:05d} | Train_Loss: {:.4f} | Time: {:.4f}'.format(
            #     epoch, step, train_loss.item(), t_end-t_start))

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
                    valid_neg = dl.get_valid_neg(edge_types=[test_edge_type])[
                        test_edge_type]
                    valid_pos_head = np.concatenate(
                        [valid_pos_head, np.array(valid_pos[test_edge_type][0])])
                    valid_pos_tail = np.concatenate(
                        [valid_pos_tail, np.array(valid_pos[test_edge_type][1])])
                    valid_neg_head = np.concatenate(
                        [valid_neg_head, np.array(valid_neg[0])])
                    valid_neg_tail = np.concatenate(
                        [valid_neg_tail, np.array(valid_neg[1])])
                    valid_r_id = np.concatenate([valid_r_id, np.array(
                        [test_edge_type]*len(valid_pos[test_edge_type][0]))])
                left = np.concatenate([valid_pos_head, valid_neg_head])
                right = np.concatenate([valid_pos_tail, valid_neg_tail])
                mid = np.concatenate([valid_r_id, valid_r_id])
                labels = torch.FloatTensor(np.concatenate(
                    [np.ones(valid_pos_head.shape[0]), np.zeros(valid_neg_head.shape[0])])).to(device)
                embed = net.encode(g, features_list)
                logits = net.decode(mid, embed[left], embed[right])
                pred = torch.sigmoid(logits)
                val_loss = loss_func(pred, labels)
            t_end = time.time()
            # print validation info
            # print('Epoch {:05d} | Val_Loss {:.4f} | Time(s) {:.4f}'.format(
            #     epoch, val_loss.item(), t_end - t_start))
            # early stopping
            early_stopping(val_loss, net)
            if early_stopping.early_stop:
                # print('Early stopping!')
                break
        if early_stopping.early_stop:
            # print('Early stopping!')
            break

    for test_edge_type in dl.links_test['data'].keys():
        # testing with evaluate_results_nc
        net.load_state_dict(torch.load(
            'checkpoint/'+args.model+'checkpoint_{}_{}.pt'.format(args.dataset, args.num_layers)))
        net.eval()
        with torch.no_grad():
            test_neigh, test_label = dl.get_test_neigh()
            test_neigh = test_neigh[test_edge_type]
            test_label = test_label[test_edge_type]
            left = np.array(test_neigh[0])
            right = np.array(test_neigh[1])
            mid = np.zeros(left.shape[0], dtype=np.int32)
            mid[:] = test_edge_type
            labels = torch.FloatTensor(test_label).to(device)
            embed = net.encode(g, features_list)
            logits = net.decode(mid, embed[left], embed[right])
            pred = torch.sigmoid(logits).cpu().numpy()
            edge_list = np.concatenate(
                [left.reshape((1, -1)), right.reshape((1, -1))], axis=0)
            labels = labels.cpu().numpy()
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
            embed = net.encode(g, features_list)
            logits = net.decode(mid, embed[left], embed[right])
            pred = torch.sigmoid(logits).cpu().numpy()
            edge_list = np.concatenate(
                [left.reshape((1, -1)), right.reshape((1, -1))], axis=0)
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
    ap = argparse.ArgumentParser(
        description='MRGNN testing for the DBLP dataset')
    ap.add_argument('--feats-type', type=int, default=3,
                    help='Type of the node features used. ' +
                         '0 - loaded features; ' +
                         '1 - only target node features (zero vec for others); ' +
                         '2 - only target node features (id vec for others); ' +
                         '3 - all id vec. Default is 2;' +
                    '4 - only term features (id vec for others);' +
                    '5 - only term features (zero vec for others).')
    ap.add_argument('--hidden-dim', type=int, default=64,
                    help='Dimension of the node hidden state. Default is 64.')
    ap.add_argument('--num-heads', type=int, default=2,
                    help='Number of the attention heads. Default is 8.')
    ap.add_argument('--epoch', type=int, default=300, help='Number of epochs.')
    ap.add_argument('--patience', type=int, default=40, help='Patience.')
    ap.add_argument('--num-layers', type=int, default=2)
    ap.add_argument('--lr', type=float, default=5e-4)
    ap.add_argument('--dropout', type=float, default=0.5)
    ap.add_argument('--weight-decay', type=float, default=1e-4)
    ap.add_argument('--slope', type=float, default=0.01)
    ap.add_argument('--dataset', type=str)
    ap.add_argument('--model', type=str)
    ap.add_argument('--edge-feats', type=int, default=32)
    ap.add_argument('--batch-size', type=int, default=1024)
    ap.add_argument('--decoder', type=str, default='dismult')
    ap.add_argument('--gpu', type=int, default=-1)

    args = ap.parse_args()
    run_model_DBLP(args)
