import time
import argparse

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

from utils.pytorchtools import EarlyStopping
from utils.data import load_LastFM_data
from utils.tools import index_generator, parse_minibatch_LastFM
from scipy import sparse
import scipy
# from model import MAGNN_lp
import dgl
from GNN import GCN, GAT, GCN_dense

# Hyper Params
num_ntype = 3
dropout_rate = 0.5
lr = 0.01
weight_decay = 0.000
num_layers = 1
# etypes_lists = [[[0, 1], [0, 2, 3, 1], [None]],
#                 [[1, 0], [2, 3], [1, None, 0]]]
# use_masks = [[True, True, False],
#              [True, False, True]]
# no_masks = [[False] * 3, [False] * 3]
num_user = 1892
num_artist = 17632
num_tag = 1088
# expected_metapaths = [
#     [(0, 1, 0), (0, 1, 2, 1, 0), (0, 0)],
#     [(1, 0, 1), (1, 2, 1), (1, 0, 0, 1)]
# ]
# out_dim = 2
# etypes_list = [[0, 1], [0, 2, 3, 1], [0, 4, 5, 1]]



def run_model_LastFM(feats_type, hidden_dim, num_heads, attn_vec_dim, rnn_type,
                     num_epochs, patience, batch_size, neighbor_samples, repeat, save_postfix, model_):
    prefix = 'data/preprocessed/LastFM_processed'

    adjM = scipy.sparse.load_npz(prefix + '/adjM.npz')
    type_mask = np.load(prefix + '/node_types.npy')
    train_val_test_pos_user_artist = np.load(prefix + '/train_val_test_pos_user_artist.npz') #[-1, 2]  [[1,2],[2,3]]
    train_val_test_neg_user_artist = np.load(prefix + '/train_val_test_neg_user_artist.npz')
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
    elif feats_type == 2:
        num_all_node = num_user + num_artist + num_tag
        features_list = torch.eye(num_all_node).to(device)
    train_pos_user_artist = train_val_test_pos_user_artist['train_pos_user_artist'] #[64984,2]
    val_pos_user_artist = train_val_test_pos_user_artist['val_pos_user_artist']# [9283,2]
    test_pos_user_artist = train_val_test_pos_user_artist['test_pos_user_artist'] #[18567,2]
    train_neg_user_artist = train_val_test_neg_user_artist['train_neg_user_artist']
    val_neg_user_artist = train_val_test_neg_user_artist['val_neg_user_artist']
    test_neg_user_artist = train_val_test_neg_user_artist['test_neg_user_artist']
    y_true_test = np.array([1] * len(test_pos_user_artist) + [0] * len(test_neg_user_artist))
    # transform artist's range from [0,17631] to [1892, 1892+17631]
    train_pos_user_artist[:, 1] += 1892
    val_pos_user_artist[:, 1] += 1892
    test_pos_user_artist[:, 1] += 1892
    train_neg_user_artist[:, 1] += 1892
    val_neg_user_artist[:, 1] += 1892
    test_neg_user_artist[:, 1] += 1892

    g = dgl.DGLGraph(adjM)
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)
    g = g.to(device)
    print(type(g))

    auc_list = []
    ap_list = []
    heads = ([num_heads] * num_layers) + [1]
    for _ in range(repeat):
        if model_ == 'gcn':
            if feats_type == 2:
                net = GCN_dense(g, features_list.size()[1], hidden_dim, num_layers, F.relu, dropout_rate).cuda()
            else:
                net = GCN(g, hidden_dim, hidden_dim, num_layers, F.relu, dropout_rate, in_dims).cuda()
        elif model_=='gat':
            net = GAT(g, num_layers, hidden_dim, hidden_dim, heads, F.elu, dropout_rate, dropout_rate, 0.01,
                      False, in_dims).cuda()
        net.to(device)
        optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)

        # training loop
        net.train()
        early_stopping = EarlyStopping(patience=patience, verbose=True, save_path='checkpoint/checkpoint_{}.pt'.format(save_postfix))
        dur1 = []
        dur2 = []
        dur3 = []
        train_pos_idx_generator = index_generator(batch_size=batch_size, num_data=len(train_pos_user_artist))
        val_idx_generator = index_generator(batch_size=batch_size, num_data=len(val_pos_user_artist), shuffle=False)
        print('train iteration count of each epoch :%d ,val iteration count of each epoch: %d' % (train_pos_idx_generator.num_iterations(),val_idx_generator.num_iterations()))
        for epoch in range(num_epochs):
            t_start = time.time()
            # training
            net.train()
            for iteration in range(train_pos_idx_generator.num_iterations()):
                # forward
                t0 = time.time()

                train_pos_idx_batch = train_pos_idx_generator.next()
                train_pos_idx_batch.sort()
                train_pos_user_artist_batch = train_pos_user_artist[train_pos_idx_batch].tolist()
                train_neg_idx_batch = np.random.choice(len(train_neg_user_artist), len(train_pos_idx_batch))
                train_neg_idx_batch.sort()
                train_neg_user_artist_batch = train_neg_user_artist[train_neg_idx_batch].tolist()

                # train_pos_g_lists, train_pos_indices_lists, train_pos_idx_batch_mapped_lists = parse_minibatch_LastFM(
                #     adjlists_ua, edge_metapath_indices_list_ua, train_pos_user_artist_batch, device, neighbor_samples, use_masks, num_user)
                # train_neg_g_lists, train_neg_indices_lists, train_neg_idx_batch_mapped_lists = parse_minibatch_LastFM(
                #     adjlists_ua, edge_metapath_indices_list_ua, train_neg_user_artist_batch, device, neighbor_samples, no_masks, num_user)

                t1 = time.time()
                dur1.append(t1 - t0)
                hid_feature = net(features_list)
                # list transposition
                train_pos_user_artist_batch = list(map(list, zip(*train_pos_user_artist_batch)))
                train_neg_user_artist_batch = list(map(list, zip(*train_neg_user_artist_batch)))
                pos_embedding_user=hid_feature[train_pos_user_artist_batch[0]]
                pos_embedding_artist = hid_feature[train_pos_user_artist_batch[1]]
                neg_embedding_user = hid_feature[train_neg_user_artist_batch[0]]
                neg_embedding_artist = hid_feature[train_neg_user_artist_batch[1]]
                # [neg_embedding_user, neg_embedding_artist], _ = net(
                #     (train_neg_g_lists, features_list, type_mask, train_neg_indices_lists, train_neg_idx_batch_mapped_lists))

                pos_embedding_user = pos_embedding_user.view(-1, 1, pos_embedding_user.shape[1])
                pos_embedding_artist = pos_embedding_artist.view(-1, pos_embedding_artist.shape[1], 1)
                neg_embedding_user = neg_embedding_user.view(-1, 1, neg_embedding_user.shape[1])
                neg_embedding_artist = neg_embedding_artist.view(-1, neg_embedding_artist.shape[1], 1)
                pos_out = torch.bmm(pos_embedding_user, pos_embedding_artist)
                neg_out = -torch.bmm(neg_embedding_user, neg_embedding_artist)
                train_loss = -torch.mean(F.logsigmoid(pos_out) + F.logsigmoid(neg_out))

                t2 = time.time()
                dur2.append(t2 - t1)

                # autograd
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()

                t3 = time.time()
                dur3.append(t3 - t2)

                # print training info
                if iteration % 100 == 0:
                    print(
                        'Epoch {:05d} | Iteration {:05d} | Train_Loss {:.4f} | Time1(s) {:.4f} | Time2(s) {:.4f} | Time3(s) {:.4f}'.format(
                            epoch, iteration, train_loss.item(), np.mean(dur1), np.mean(dur2), np.mean(dur3)))
            # validation
            net.eval()
            val_loss = []
            with torch.no_grad():
                for iteration in range(val_idx_generator.num_iterations()):
                    # forward
                    val_idx_batch = val_idx_generator.next()
                    val_pos_user_artist_batch = val_pos_user_artist[val_idx_batch].tolist()
                    val_neg_user_artist_batch = val_neg_user_artist[val_idx_batch].tolist()
                    # list transposition
                    val_pos_user_artist_batch = list(map(list, zip(*val_pos_user_artist_batch)))
                    val_neg_user_artist_batch = list(map(list, zip(*val_neg_user_artist_batch)))
                    # val_pos_g_lists, val_pos_indices_lists, val_pos_idx_batch_mapped_lists = parse_minibatch_LastFM(
                    #     adjlists_ua, edge_metapath_indices_list_ua, val_pos_user_artist_batch, device, neighbor_samples, no_masks, num_user)
                    # val_neg_g_lists, val_neg_indices_lists, val_neg_idx_batch_mapped_lists = parse_minibatch_LastFM(
                    #     adjlists_ua, edge_metapath_indices_list_ua, val_neg_user_artist_batch, device, neighbor_samples, no_masks, num_user)

                    # [pos_embedding_user, pos_embedding_artist], _ = net(
                    #     (val_pos_g_lists, features_list, type_mask, val_pos_indices_lists, val_pos_idx_batch_mapped_lists))
                    # [neg_embedding_user, neg_embedding_artist], _ = net(
                    #     (val_neg_g_lists, features_list, type_mask, val_neg_indices_lists, val_neg_idx_batch_mapped_lists))
                    hid_feature = net(features_list)
                    pos_embedding_user = hid_feature[val_pos_user_artist_batch[0]]
                    pos_embedding_artist = hid_feature[val_pos_user_artist_batch[1]]
                    neg_embedding_user = hid_feature[val_neg_user_artist_batch[0]]
                    neg_embedding_artist = hid_feature[val_neg_user_artist_batch[1]]

                    pos_embedding_user = pos_embedding_user.view(-1, 1, pos_embedding_user.shape[1])
                    pos_embedding_artist = pos_embedding_artist.view(-1, pos_embedding_artist.shape[1], 1)
                    neg_embedding_user = neg_embedding_user.view(-1, 1, neg_embedding_user.shape[1])
                    neg_embedding_artist = neg_embedding_artist.view(-1, neg_embedding_artist.shape[1], 1)

                    pos_out = torch.bmm(pos_embedding_user, pos_embedding_artist)
                    neg_out = -torch.bmm(neg_embedding_user, neg_embedding_artist)
                    val_loss.append(-torch.mean(F.logsigmoid(pos_out) + F.logsigmoid(neg_out)))
                val_loss = torch.mean(torch.tensor(val_loss))
            t_end = time.time()
            # print validation info
            print('Epoch {:05d} | Val_Loss {:.4f} | Time(s) {:.4f}'.format(
                epoch, val_loss.item(), t_end - t_start))
            # early stopping
            early_stopping(val_loss, net)
            if early_stopping.early_stop:
                print('Early stopping!')
                break

        test_idx_generator = index_generator(batch_size=batch_size, num_data=len(test_pos_user_artist), shuffle=False)
        net.load_state_dict(torch.load('checkpoint/checkpoint_{}.pt'.format(save_postfix)))
        net.eval()
        pos_proba_list = []
        neg_proba_list = []
        with torch.no_grad():
            for iteration in range(test_idx_generator.num_iterations()):
                # forward
                test_idx_batch = test_idx_generator.next()
                test_pos_user_artist_batch = test_pos_user_artist[test_idx_batch].tolist()
                test_neg_user_artist_batch = test_neg_user_artist[test_idx_batch].tolist()
                test_pos_user_artist_batch = list(map(list, zip(*test_pos_user_artist_batch)))
                test_neg_user_artist_batch = list(map(list, zip(*test_neg_user_artist_batch)))
                # test_pos_g_lists, test_pos_indices_lists, test_pos_idx_batch_mapped_lists = parse_minibatch_LastFM(
                #     adjlists_ua, edge_metapath_indices_list_ua, test_pos_user_artist_batch, device, neighbor_samples, no_masks, num_user)
                # test_neg_g_lists, test_neg_indices_lists, test_neg_idx_batch_mapped_lists = parse_minibatch_LastFM(
                #     adjlists_ua, edge_metapath_indices_list_ua, test_neg_user_artist_batch, device, neighbor_samples, no_masks, num_user)

                # [pos_embedding_user, pos_embedding_artist], _ = net(
                #     (test_pos_g_lists, features_list, type_mask, test_pos_indices_lists, test_pos_idx_batch_mapped_lists))
                # [neg_embedding_user, neg_embedding_artist], _ = net(
                #     (test_neg_g_lists, features_list, type_mask, test_neg_indices_lists, test_neg_idx_batch_mapped_lists))
                hid_feature=net(features_list)
                pos_embedding_user = hid_feature[test_pos_user_artist_batch[0]]
                pos_embedding_artist = hid_feature[test_pos_user_artist_batch[1]]
                neg_embedding_user = hid_feature[test_neg_user_artist_batch[0]]
                neg_embedding_artist = hid_feature[test_neg_user_artist_batch[1]]

                pos_embedding_user = pos_embedding_user.view(-1, 1, pos_embedding_user.shape[1])
                pos_embedding_artist = pos_embedding_artist.view(-1, pos_embedding_artist.shape[1], 1)
                neg_embedding_user = neg_embedding_user.view(-1, 1, neg_embedding_user.shape[1])
                neg_embedding_artist = neg_embedding_artist.view(-1, neg_embedding_artist.shape[1], 1)

                pos_out = torch.bmm(pos_embedding_user, pos_embedding_artist).flatten()
                neg_out = torch.bmm(neg_embedding_user, neg_embedding_artist).flatten()
                pos_proba_list.append(torch.sigmoid(pos_out))
                neg_proba_list.append(torch.sigmoid(neg_out))
            y_proba_test = torch.cat(pos_proba_list + neg_proba_list)
            y_proba_test = y_proba_test.cpu().numpy()
        auc = roc_auc_score(y_true_test, y_proba_test)
        ap = average_precision_score(y_true_test, y_proba_test)
        print('Link Prediction Test')
        print('AUC = {}'.format(auc))
        print('AP = {}'.format(ap))
        auc_list.append(auc)
        ap_list.append(ap)

    print('----------------------------------------------------------------')
    print('Link Prediction Tests Summary')
    print('AUC_mean = {}, AUC_std = {}'.format(np.mean(auc_list), np.std(auc_list)))
    print('AP_mean = {}, AP_std = {}'.format(np.mean(ap_list), np.std(ap_list)))


if __name__ == '__main__':
    print('program start at ', time.localtime())
    ap = argparse.ArgumentParser(description='MRGNN testing for the recommendation dataset')
    ap.add_argument('--feats-type', type=int, default=0,
                    help='Type of the node features used. ' +
                         '0 - all id vectors; ' +
                         '1 - all zero vector.' +
                         '2 - all eye vector. Default is 2.')
    ap.add_argument('--hidden-dim', type=int, default=64, help='Dimension of the node hidden state. Default is 64.')
    ap.add_argument('--num-heads', type=int, default=8, help='Number of the attention heads. Default is 8.')
    ap.add_argument('--attn-vec-dim', type=int, default=128, help='Dimension of the attention vector. Default is 128.')
    ap.add_argument('--rnn-type', default='RotatE0', help='Type of the aggregator. Default is RotatE0.')
    ap.add_argument('--epoch', type=int, default=100, help='Number of epochs. Default is 100.')
    ap.add_argument('--patience', type=int, default=5, help='Patience. Default is 5.')
    ap.add_argument('--batch-size', type=int, default=100000, help='Batch size. Default is 100000.')
    ap.add_argument('--samples', type=int, default=100, help='Number of neighbors sampled. Default is 100.')
    ap.add_argument('--repeat', type=int, default=1, help='Repeat the training and testing for N times. Default is 1.')
    ap.add_argument('--save-postfix', default='LastFM', help='Postfix for the saved model and result. Default is LastFM.')
    ap.add_argument('--model', default='gcn')
    args = ap.parse_args()
    run_model_LastFM(args.feats_type, args.hidden_dim, args.num_heads, args.attn_vec_dim, args.rnn_type, args.epoch,
                     args.patience, args.batch_size, args.samples, args.repeat, args.save_postfix, args.model)
    print('program finished at ', time.localtime())