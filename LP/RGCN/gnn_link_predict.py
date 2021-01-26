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

import argparse
import numpy as np
import time
import torch
import torch.nn.functional as F
from dgl.data.knowledge_graph import load_data
import dgl

from model import GCN, GAT

import utils


def main(args):
    # load graph data
    data = load_data(args.dataset)
    num_nodes = data.num_nodes
    train_data = data.train
    valid_data = data.valid
    test_data = data.test
    num_rels = data.num_rels

    # check cuda
    use_cuda = args.gpu >= 0 and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(args.gpu)

    # create model
    if args.model == 'gcn':
        model = GCN(in_feats=[num_nodes], hid_feats=args.n_hidden,
                    n_layers=args.n_layers, dropout=args.dropout, rel_num=num_rels)
    elif args.model == 'gat':
        heads = [args.n_heads] * args.n_layers + [1]
        model = GAT(in_feats=[num_nodes], hid_feats=args.n_hidden,
                    n_layers=args.n_layers, heads=heads, dropout=args.dropout, rel_num=num_rels)
    else:
        exit(0)
    # validation and testing triplets
    valid_data = torch.LongTensor(valid_data)
    test_data = torch.LongTensor(test_data)

    # build test graph
    test_graph, test_rel, test_norm = utils.build_test_graph(
        num_nodes, num_rels, train_data)
    test_graph = dgl.add_self_loop(test_graph)

    indices = np.vstack((np.arange(num_nodes), np.arange(num_nodes)))
    indices = torch.LongTensor(indices)
    values = torch.FloatTensor(np.ones(num_nodes))
    test_feat = torch.sparse.FloatTensor(
        indices, values, torch.Size([num_nodes, num_nodes]))
    if use_cuda:
        test_feat = test_feat.cuda()

    if use_cuda:
        model.cuda()

    # build adj list and calculate degrees for sampling
    adj_list, degrees = utils.get_adj_and_degrees(num_nodes, train_data)
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    model_state_file = args.model + 'model_state.pth'
    forward_time = []
    backward_time = []

    # training loop
    print("start training...")

    epoch = 0
    best_mrr = 0
    while True:
        model.train()
        epoch += 1

        # perform edge neighborhood sampling to generate training graph and data
        g, node_id, edge_type, node_norm, data, labels = \
            utils.generate_sampled_graph_and_labels(train_data, args.graph_batch_size, args.graph_split_size,
                                                    num_rels, adj_list, degrees, args.negative_sample,
                                                    args.edge_sampler)
        print("Done edge sampling")
        if not model == "rgcn":
            g = dgl.add_self_loop(g)
        sample_num = g.num_nodes()

        # set node/edge feature
        node_id = torch.from_numpy(node_id).view(-1, 1).long()
        edge_type = torch.from_numpy(edge_type)
        data, labels = torch.from_numpy(data), torch.from_numpy(labels)
        deg = g.in_degrees(range(g.number_of_nodes())).float().view(-1, 1)
        if use_cuda:
            node_id, deg = node_id.cuda(), deg.cuda()
            edge_type = edge_type.cuda()
            data, labels = data.cuda(), labels.cuda()
            g = g.to(args.gpu)

        indices = np.vstack((np.arange(sample_num), np.arange(sample_num)))
        indices = torch.LongTensor(indices)
        values = torch.FloatTensor(np.ones(sample_num))
        feat = torch.sparse.FloatTensor(
            indices, values, torch.Size([sample_num, num_nodes]))
        if use_cuda:
            feat = feat.cuda()

        t0 = time.time()
        hid_feat = model.encode([g, [feat]])
        reg_loss = torch.mean(hid_feat.pow(2)) + \
            torch.mean(model.decode.weights.pow(2))
        r_list = data[:, 1]
        out_feat = model.decode(
            r_list, hid_feat[data[:, 0]], hid_feat[data[:, 2]])
        print(out_feat)
        loss = F.binary_cross_entropy_with_logits(
            out_feat, labels) + reg_loss * args.regularization
        t1 = time.time()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), args.grad_norm)  # clip gradients
        optimizer.step()
        t2 = time.time()

        forward_time.append(t1 - t0)
        backward_time.append(t2 - t1)
        print("Epoch {:04d} | Loss {:f} | Best MRR {:.4f} | Forward {:.4f}s | Backward {:.4f}s".
              format(epoch, loss.item(), best_mrr, forward_time[-1], backward_time[-1]))

        optimizer.zero_grad()

        # validation
        if epoch % args.evaluate_every == 0:
            # perform validation on CPU because full graph is too large
            model.eval()
            if use_cuda:
                model.cpu()  # test on CPU
                test_feat = test_feat.cpu()
            print("start eval")
            embed = model.encode([test_graph, [test_feat]])
            mrr = utils.calc_mrr(embed, model.decode.weights, torch.LongTensor(train_data),
                                 valid_data, test_data, hits=[1, 3, 10], eval_bz=args.eval_batch_size,
                                 eval_p=args.eval_protocol)
            # save best model
            if mrr < best_mrr:
                if epoch >= args.n_epochs:
                    break
            else:
                best_mrr = mrr
                torch.save({'state_dict': model.state_dict(), 'epoch': epoch},
                           model_state_file)
            if use_cuda:
                model.cuda()

    print("training done")
    print("Mean forward time: {:4f}s".format(np.mean(forward_time)))
    print("Mean Backward time: {:4f}s".format(np.mean(backward_time)))

    print("\nstart testing:")
    # use best model checkpoint
    checkpoint = torch.load(model_state_file)
    if use_cuda:
        model.cpu()  # test on CPU
        test_feat = test_feat.cpu()
    model.eval()
    model.load_state_dict(checkpoint['state_dict'])
    print("Using best epoch: {}".format(checkpoint['epoch']))
    embed = model.encode([test_graph, [test_feat]])
    utils.calc_mrr(embed, model.decode.weights, torch.LongTensor(train_data), valid_data,
                   test_data, hits=[1, 3, 10], eval_bz=args.eval_batch_size, eval_p=args.eval_protocol)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RGCN')
    parser.add_argument("--dropout", type=float, default=0.2,
                        help="dropout probability")
    parser.add_argument("--n-hidden", type=int, default=500,
                        help="number of hidden units")
    parser.add_argument("--gpu", type=int, default=-1,
                        help="gpu")
    parser.add_argument("--lr", type=float, default=1e-2,
                        help="learning rate")
    parser.add_argument("--n-bases", type=int, default=100,
                        help="number of weight blocks for each relation")
    parser.add_argument("--n-layers", type=int, default=2,
                        help="number of propagation rounds")
    parser.add_argument("--n-epochs", type=int, default=6000,
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
    parser.add_argument("--evaluate-every", type=int, default=500,
                        help="perform evaluation every n epochs")
    parser.add_argument("--n-heads", type=int, default=4,
                        help="heads for gat")
    parser.add_argument("--edge-sampler", type=str, default="uniform",
                        help="type of edge sampler: 'uniform' or 'neighbor'")
    parser.add_argument("--model", type=str, default="gcn",
                        help="type of model")

    args = parser.parse_args()
    print(args)
    main(args)
