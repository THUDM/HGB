"""
Modeling Relational Data with Graph Convolutional Networks
Paper: https://arxiv.org/abs/1703.06103
Code: https://github.com/MichSchli/RelationPrediction
Difference compared to MichSchli/RelationPrediction
* Report raw metrics instead of filtered metrics.
* By default, we use uniform edge sampling instead of neighbor-based edge
  sampling used in author's code. In practice, we find it achieves similar MRR
  probably because the model only uses one GNN layer so messages are propagated
  among immediate neighbors. User could specify "--edge-sampler=neighbor" to switch
  to neighbor-based edge sampling.
"""
import random
import argparse
import numpy as np
import time
import torch as th
import torch.nn.functional as F
from dgl.data.knowledge_graph import load_data
import dgl

from GNN import GCN, GAT

import utils


class HOM_data():
    def __init__(self, args):
        self.args = args
        self.neg_times = args.negative_sample
        self.heter_data = load_data(args.dataset)
        self.node_total = self.heter_data.num_nodes
        self.edge_list = self.gen_edge_list()
        self.feat = self.gen_feat()

    def gen_edge_list(self):
        row, col = self.heter_data.train.T[[0, 2], :]
        edge_list = th.LongTensor([row, col])
        return edge_list

    def gen_feat(self):
        dim = self.node_total
        indices = np.vstack((np.arange(dim), np.arange(dim)))
        indices = th.LongTensor(indices)
        values = th.FloatTensor(np.ones(dim))
        feat = th.sparse.FloatTensor(
            indices, values, th.Size([dim, dim]))
        return feat

    def get_data(self, name='train'):
        data = None
        neg_times = 1 if name == 'train' else self.neg_times
        if name == 'train':
            data = self.heter_data.train
        elif name == 'test':
            data = self.heter_data.test
        elif name == 'valid':
            data = self.heter_data.valid
        link_num = len(data)
        pos_links = data.T[[0, 2], :]
        pos_labels = np.array([1] * link_num)

        neg_links_src = np.vstack([pos_links[0]] * neg_times).flatten('f')
        nodes_for_choice = np.arange(self.node_total)
        neg_samples = random.choices(nodes_for_choice, k=link_num * neg_times)
        neg_links_tgt = np.array(neg_samples)
        neg_links = np.array([neg_links_src, neg_links_tgt])
        neg_labels = np.array([0] * link_num * neg_times)

        all_src = np.hstack((pos_links[0].reshape(-1, 1), neg_links.reshape(2, -1, neg_times)[0])).flatten()
        all_tgt = np.hstack((pos_links[1].reshape(-1, 1), neg_links.reshape(2, -1, neg_times)[1])).flatten()
        all_labels = np.hstack((pos_labels.reshape(-1, 1), neg_labels.reshape(-1, neg_times))).flatten()
        all_links = np.vstack((all_src, all_tgt))
        return all_links, all_labels


def sort_and_rank(out_feat, args):
    neg_times = args.negative_sample
    out_feat = out_feat.reshape((-1,neg_times + 1))
    sort_index = th.argsort(-out_feat)
    ranks = th.argmin(sort_index, 1)
    return ranks


def main(args):
    # load graph data
    hom_data = HOM_data(args)
    train_data, train_label = hom_data.get_data(name='train')
    valid_data, valid_label = hom_data.get_data(name='valid')
    test_data, test_label = hom_data.get_data(name='test')
    # create model
    model = None
    if args.model == 'gcn':
        model = GCN(in_feats=[hom_data.node_total], hid_feats=args.n_hidden, n_layers=args.n_layers,
                    dropout=args.dropout,
                    decoder='dismult')
    elif args.model == 'gat':
        heads = [args.n_heads] * args.n_layers + [1]
        model = GAT(in_feats=[num_nodes], hid_feats=args.n_hidden,
                    n_layers=args.n_layers, heads=heads, dropout=args.dropout, rel_num=num_rels)
    else:
        exit(0)

    # check cuda
    use_cuda = args.gpu >= 0 and th.cuda.is_available()
    if use_cuda:
        th.cuda.set_device(args.gpu)
        hom_data.feat = hom_data.feat.cuda()
        hom_data.edge_list = hom_data.edge_list.cuda()
        model.cuda()
    """train setting"""
    lossFun = th.nn.BCELoss()
    best_mrr = 0
    optimizer = th.optim.Adam(model.parameters(), lr=args.lr)
    model_state_file = args.model + '_model_state.pth'
    forward_time = []
    backward_time = []

    print("start training...")
    for epoch in range(0, args.n_epochs):
        model.train()
        train_batch_size = 8192
        train_epoch_size = len(train_data[0])
        unsort_index = np.arange(train_epoch_size)
        random.shuffle(unsort_index)

        shift = 0
        while True:
            if shift > train_epoch_size:
                break
            index_batch = unsort_index[shift:shift+train_batch_size]
            target_batch = th.FloatTensor(train_label[index_batch])
            if use_cuda:
                target_batch = target_batch.cuda()
            train_data_batch = [train_data[0][index_batch], train_data[1][index_batch]]
            hid_feat = model.encode([[hom_data.feat], hom_data.edge_list])
            r_list=[0]*len(train_data_batch[0])
            out_feat = model.decode(r_list, hid_feat[train_data_batch[0]], hid_feat[train_data_batch[1]])
            pred = th.sigmoid(out_feat)
            loss = lossFun(pred, target_batch)
            print(f'epoch {epoch} | batch {shift//train_batch_size}/{train_epoch_size//train_batch_size} |train loss {loss.item()}')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            shift += train_batch_size

        # validation
        if True:
            # perform validation on CPU because full graph is too large
            model.eval()
            print("start evaling")
            with th.no_grad():
                hid_feat = model.encode([[hom_data.feat], hom_data.edge_list])
                valid_batch_size = 1024 * (args.negative_sample+1)
                valid_epoch_size = len(valid_data[0])
                shift = 0
                out_feats=None
                while True:
                    if shift > valid_epoch_size:
                        break
                    valid_data_batch = np.array([valid_data[0][shift:shift+valid_batch_size], valid_data[1][shift:shift+valid_batch_size]])
                    r_list = [0] * len(valid_data_batch[0])
                    out_feat = model.decode(r_list, hid_feat[valid_data_batch[0]], hid_feat[valid_data_batch[1]])
                    out_feat = th.sigmoid(out_feat)
                    out_feats=out_feat if out_feats==None else th.cat((out_feats,out_feat))
                    shift += valid_batch_size
            ranks = sort_and_rank(out_feats, args)
            ranks += 1  # change to 1-indexed

            mrr = th.mean(1.0 / ranks.float())
            print("MRR (raw): {:.6f}".format(mrr.item()))
            hits = [1, 3, 10]
            for hit in hits:
                avg_count = th.mean((ranks <= hit).float())
                print("Hits (raw) @ {}: {:.6f}".format(hit, avg_count.item()))

            # save best model
            if mrr > best_mrr:
                best_mrr = mrr
                th.save({'state_dict': model.state_dict(), 'epoch': epoch},
                        model_state_file)

    print("training done")

    print("\nstart testing:")
    # use best model checkpoint
    checkpoint = th.load(model_state_file)
    model.load_state_dict(checkpoint['state_dict'])
    with th.no_grad():
        hid_feat = model.encode([[hom_data.feat], hom_data.edge_list])
        test_batch_size = 1024 * (args.negative_sample + 1)
        test_epoch_size = len(test_data[0])
        shift = 0
        out_feats = None
        while True:
            if shift > test_epoch_size:
                break
            test_data_batch = np.array(
                [test_data[0][shift:shift + test_batch_size], test_data[1][shift:shift + test_batch_size]])
            r_list = [0] * len(test_data_batch[0])
            out_feat = model.decode(r_list, hid_feat[test_data_batch[0]], hid_feat[test_data_batch[1]])
            out_feat = th.sigmoid(out_feat)
            out_feats = out_feat if out_feats == None else th.cat((out_feats, out_feat))
            shift += test_batch_size
    ranks = sort_and_rank(out_feats, args)
    ranks += 1  # change to 1-indexed
    mrr = th.mean(1.0 / ranks.float())
    print("Using best epoch: {}".format(checkpoint['epoch']))
    print("MRR (raw): {:.6f}".format(mrr.item()))
    hits = [1, 3, 10]
    for hit in hits:
        avg_count = th.mean((ranks <= hit).float())
        print("Hits (raw) @ {}: {:.6f}".format(hit, avg_count.item()))
    print('testing down')



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RGCN')
    parser.add_argument("--dropout", type=float, default=0.2,
                        help="dropout probability")
    parser.add_argument("--n-hidden", type=int, default=500,
                        help="number of hidden units")
    parser.add_argument("--gpu", type=int, default=-1,
                        help="gpu")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="learning rate")
    parser.add_argument("--n-bases", type=int, default=100,
                        help="number of weight blocks for each relation")
    parser.add_argument("--n-layers", type=int, default=2,
                        help="number of propagation rounds")
    parser.add_argument("--n-epochs", type=int, default=100,
                        help="number of minimum training epochs")
    parser.add_argument("-d", "--dataset", type=str, required=True,
                        help="dataset to use")
    parser.add_argument("--eval-batch-size", type=int, default=500,
                        help="batch size when evaluating")
    parser.add_argument("--eval-protocol", type=str, default="raw",
                        help="type of evaluation protocol: 'raw' or 'filtered' mrr")
    parser.add_argument("--regularization", type=float, default=0.01,
                        help="regularization weight")
    parser.add_argument("--grad-norm", type=float, default=1.0,
                        help="norm to clip gradient to")
    parser.add_argument("--graph-batch-size", type=int, default=30000,
                        help="number of edges to sample in each iteration")
    parser.add_argument("--graph-split-size", type=float, default=0.5,
                        help="portion of edges used as positive sample")
    parser.add_argument("--negative-sample", type=int, default=10,
                        help="number of negative samples per positive sample")
    parser.add_argument("--evaluate-every", type=int, default=1,
                        help="perform evaluation every n epochs")
    parser.add_argument("--n-heads", type=int, default=4,
                        help="heads for gat")
    parser.add_argument("--edge-sampler", type=str, default="uniform",
                        help="type of edge sampler: 'uniform' or 'neighbor'")
    parser.add_argument("--model", type=str, default="gcn",
                        help="type of model")
    parser.add_argument('--decoder', type=str, default="dismult", choices=['dot', 'dismult'])

    args = parser.parse_args()
    print(args)
    main(args)
