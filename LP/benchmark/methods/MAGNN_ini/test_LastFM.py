import time
import argparse

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

from utils.pytorchtools import EarlyStopping
from utils.data import load_LastFM_data
from utils.tools import index_generator, parse_minibatch_LastFM
from model import MAGNN_lp
from scripts.data_loader import data_loader

# Params
num_ntype = 3
dropout_rate = 0.5
lr = 0.005
weight_decay = 0.001
etypes_lists = [[[0, 1], [0, 2, 3, 1], [None]],
                [[1, 0], [2, 3], [1, None, 0]]]
use_masks = [[True, True, False],
             [True, False, True]]
no_masks = [[False] * 3, [False] * 3]
num_user = 1892
num_artist = 17632
expected_metapaths = [
    [(0, 1, 0), (0, 1, 2, 1, 0), (0, 0)],
    [(1, 0, 1), (1, 2, 1), (1, 0, 0, 1)]
]


def run_model_LastFM(feats_type, hidden_dim, num_heads, attn_vec_dim, rnn_type,
                     num_epochs, patience, batch_size, neighbor_samples, repeat, save_postfix):
    adjlists_ua, edge_metapath_indices_list_ua, _, type_mask, train_val_test_pos_user_artist, train_val_test_neg_user_artist = load_LastFM_data(prefix='data/preprocessed/'+save_postfix+'_processed')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    features_list = []
    in_dims = []
    if feats_type == 0:
        for i in range(num_ntype):
            dim = (type_mask == i).sum()
            in_dims.append(dim)
            indices = np.vstack((np.arange(dim), np.arange(dim)))
            indices = torch.LongTensor(indices)
            values = torch.FloatTensor(np.ones(dim))
            features_list.append(torch.sparse.FloatTensor(indices, values, torch.Size([dim, dim])).to(device))
    elif feats_type == 1:
        for i in range(num_ntype):
            dim = 10
            num_nodes = (type_mask == i).sum()
            in_dims.append(dim)
            features_list.append(torch.zeros((num_nodes, 10)).to(device))

    def test_func(test_user_artist):
        test_idx_generator = index_generator(batch_size=batch_size, num_data=len(test_user_artist), shuffle=False)
        net.eval()
        proba_list = []
        with torch.no_grad():
            for iteration in range(test_idx_generator.num_iterations()):
                # forward
                test_idx_batch = test_idx_generator.next()
                test_user_artist_batch = test_user_artist[test_idx_batch].tolist()
                test_g_lists, test_indices_lists, test_idx_batch_mapped_lists = parse_minibatch_LastFM(
                    adjlists_ua, edge_metapath_indices_list_ua, test_user_artist_batch, device, neighbor_samples, no_masks, num_user)
                [embedding_user, embedding_artist], _ = net(
                    (test_g_lists, features_list, type_mask, test_indices_lists, test_idx_batch_mapped_lists))
                embedding_user = embedding_user.view(-1, 1, embedding_user.shape[1])
                embedding_artist = embedding_artist.view(-1, embedding_artist.shape[1], 1)

                out = torch.bmm(embedding_user, embedding_artist).flatten()
                proba_list.append(torch.sigmoid(out))
            y_proba_test = torch.cat(proba_list)
            y_proba_test = y_proba_test.cpu().numpy()
        return y_proba_test

    dl = data_loader('data/'+save_postfix)

    net = MAGNN_lp([3, 3], 4, etypes_lists, in_dims, hidden_dim, hidden_dim, num_heads, attn_vec_dim, rnn_type, dropout_rate)
    net.to(device)
    net.load_state_dict(torch.load('checkpoint/checkpoint_{}.pt'.format(save_postfix)))

    # Here, we use three pairs of experiemnts to verify the label leakage of MAGNN.

    def shuffle(neigh, label):
        new_idx = np.arange(len(label))
        np.random.shuffle(new_idx)
        new_neigh = []
        new_label = []
        for i in range(len(label)):
            new_neigh.append((neigh[0][new_idx[i]], neigh[1][new_idx[i]]))
            new_label.append(label[new_idx[i]])
        new_neigh = np.array(new_neigh).T
        new_label = np.array(new_label)
        return new_neigh, new_label

    test_pos = train_val_test_pos_user_artist['test_pos_user_artist']
    test_neg = train_val_test_neg_user_artist['test_neg_user_artist']
    y_true = np.array([1]*len(test_pos)+[0]*len(test_neg))
    test_pair = np.concatenate([test_pos, test_neg], axis=0).T
    y_pred = test_func(test_pair.T)
    test_pair[1,:] += dl.nodes['shift'][1]
    print("MAGNN result no shuffle:", dl.evaluate(test_pair, y_pred, y_true))

    test_pos = train_val_test_pos_user_artist['test_pos_user_artist']
    test_neg = train_val_test_neg_user_artist['test_neg_user_artist']
    y_true = np.array([1]*len(test_pos)+[0]*len(test_neg))
    test_pair = np.concatenate([test_pos, test_neg], axis=0).T
    test_pair, y_true = shuffle(test_pair, y_true)
    y_pred = test_func(test_pair.T)
    test_pair[1,:] += dl.nodes['shift'][1]
    print("MAGNN result shuffled:", dl.evaluate(test_pair, y_pred, y_true))

    def reorder(neigh, label):
        new_neigh = [[], []]
        new_label = [[], []]
        for i in range(len(label)):
            new_neigh[label[i]].append((neigh[0][i], neigh[1][i]))
            new_label[label[i]].append(label[i])
        new_neigh = np.array(new_neigh[1] + new_neigh[0]).T
        new_label = np.array(new_label[1] + new_label[0])
        return new_neigh, new_label

    test_neigh, test_label = dl.get_test_neigh()
    test_neigh = np.array(test_neigh[0])
    test_label = np.array(test_label[0])
    test_neigh, test_label = reorder(test_neigh, test_label)
    tmp = test_neigh.T.copy()
    tmp[:,1] -= dl.nodes['shift'][1]
    y_pred_2hop = test_func(tmp)
    print("2-hop result with data leakage:", dl.evaluate(test_neigh, y_pred_2hop, test_label))

    test_neigh, test_label = dl.get_test_neigh()
    test_neigh = np.array(test_neigh[0])
    test_label = np.array(test_label[0])
    tmp = test_neigh.T.copy()
    tmp[:,1] -= dl.nodes['shift'][1]
    y_pred_2hop = test_func(tmp)
    print("2-hop result without data leakage:", dl.evaluate(test_neigh, y_pred_2hop, test_label))
    
    test_neigh, test_label = dl.get_test_neigh_w_random()
    test_neigh = np.array(test_neigh[0])
    test_label = np.array(test_label[0])
    test_neigh, test_label = reorder(test_neigh, test_label)
    tmp = test_neigh.T.copy()
    tmp[:,1] -= dl.nodes['shift'][1]
    y_pred_random = test_func(tmp)
    print("random result with data leakage:", dl.evaluate(test_neigh, y_pred_random, test_label))

    test_neigh, test_label = dl.get_test_neigh_w_random()
    test_neigh = np.array(test_neigh[0])
    test_label = np.array(test_label[0])
    tmp = test_neigh.T.copy()
    tmp[:,1] -= dl.nodes['shift'][1]
    y_pred_random = test_func(tmp)
    print("random result without data leakage:", dl.evaluate(test_neigh, y_pred_random, test_label))



if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='MRGNN testing for the recommendation dataset')
    ap.add_argument('--feats-type', type=int, default=0,
                    help='Type of the node features used. ' +
                         '0 - all id vectors; ' +
                         '1 - all zero vector. Default is 0.')
    ap.add_argument('--hidden-dim', type=int, default=64, help='Dimension of the node hidden state. Default is 64.')
    ap.add_argument('--num-heads', type=int, default=8, help='Number of the attention heads. Default is 8.')
    ap.add_argument('--attn-vec-dim', type=int, default=128, help='Dimension of the attention vector. Default is 128.')
    ap.add_argument('--rnn-type', default='RotatE0', help='Type of the aggregator. Default is RotatE0.')
    ap.add_argument('--epoch', type=int, default=100, help='Number of epochs. Default is 100.')
    ap.add_argument('--patience', type=int, default=5, help='Patience. Default is 5.')
    ap.add_argument('--batch-size', type=int, default=8, help='Batch size. Default is 8.')
    ap.add_argument('--samples', type=int, default=100, help='Number of neighbors sampled. Default is 100.')
    ap.add_argument('--repeat', type=int, default=1, help='Repeat the training and testing for N times. Default is 1.')
    ap.add_argument('--save-postfix', default='LastFM', help='Postfix for the saved model and result. Default is LastFM.')

    args = ap.parse_args()
    run_model_LastFM(args.feats_type, args.hidden_dim, args.num_heads, args.attn_vec_dim, args.rnn_type, args.epoch,
                     args.patience, args.batch_size, args.samples, args.repeat, args.save_postfix)
