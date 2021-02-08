import argparse
import multiprocessing
from collections import defaultdict
from operator import index
import numpy as np
from six import iteritems
from sklearn.metrics import (auc, f1_score, precision_recall_curve,
                             roc_auc_score)
from tqdm import tqdm
import os
import random
import pickle

from walk import RWGraph


class Vocab(object):

    def __init__(self, count, index):
        self.count = count
        self.index = index


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data', type=str, default='amazon',
                        help='Input dataset path')

    parser.add_argument('--features', type=str, default=None,
                        help='Input node features')

    parser.add_argument('--walk-file', type=str, default=None,
                        help='Input random walks')

    parser.add_argument('--epoch', type=int, default=100,
                        help='Number of epoch. Default is 100.')

    parser.add_argument('--batch-size', type=int, default=1000,
                        help='Number of batch_size. Default is 1000.')

    parser.add_argument('--eval-type', type=str, default='all',
                        help='The edge type(s) for evaluation.')

    parser.add_argument('--schema', type=str, default=None,
                        help='The metapath schema (e.g., U-I-U,I-U-I).')

    parser.add_argument('--dimensions', type=int, default=200,
                        help='Number of dimensions. Default is 200.')

    parser.add_argument('--edge-dim', type=int, default=10,
                        help='Number of edge embedding dimensions. Default is 10.')

    parser.add_argument('--att-dim', type=int, default=20,
                        help='Number of attention dimensions. Default is 20.')

    parser.add_argument('--walk-length', type=int, default=10,
                        help='Length of walk per source. Default is 10.')

    parser.add_argument('--num-walks', type=int, default=20,
                        help='Number of walks per source. Default is 20.')

    parser.add_argument('--window-size', type=int, default=5,
                        help='Context size for optimization. Default is 5.')

    parser.add_argument('--negative-samples', type=int, default=5,
                        help='Negative samples for optimization. Default is 5.')

    parser.add_argument('--neighbor-samples', type=int, default=10,
                        help='Neighbor samples for aggregation. Default is 10.')

    parser.add_argument('--patience', type=int, default=5,
                        help='Early stopping patience. Default is 5.')
    '''Set num-workers=1  in windows to avoid error.'''
    parser.add_argument('--num-workers', type=int, default=1,
                        help='Number of workers for generating random walks. Default is 1.')

    return parser.parse_args()


def get_G_from_edges(edges):
    edge_dict = defaultdict(set)
    for edge in edges:
        u, v = str(edge[0]), str(edge[1])
        edge_dict[u].add(v)
        edge_dict[v].add(u)
    return edge_dict


def load_train_data(dl):
    print('Data: Load train data from data_loader')
    edge_data_by_type = defaultdict(list)
    all_nodes = list()
    train_links = dl.links['data']
    for r_id in train_links.keys():
        row, col = train_links[r_id].nonzero()
        for (h, t) in zip(row, col):
            h, t, r_id = str(h), str(t), str(r_id)
            edge_data_by_type[r_id].append((h, t))
            all_nodes.append(h)
            all_nodes.append(t)
    all_nodes = list(set(all_nodes))
    print('Total training nodes: ' + str(len(all_nodes)))
    return edge_data_by_type


def load_valid_data(dl):
    print('Data: Load valid data from data_loader')
    valid_true_data_by_edge, valid_false_data_by_edge = defaultdict(list), defaultdict(list)
    for r_id in dl.valid_pos.keys():
        for h_id, t_id in zip(dl.valid_pos[r_id][0], dl.valid_pos[r_id][1]):
            valid_true_data_by_edge[str(r_id)].append((str(h_id), str(t_id)))
    for r_id in dl.valid_neg.keys():
        for h_id, t_id in zip(dl.valid_neg[r_id][0], dl.valid_neg[r_id][1]):
            valid_false_data_by_edge[str(r_id)].append((str(h_id), str(t_id)))
    return valid_true_data_by_edge, valid_false_data_by_edge


def load_test_data(dl, random_test=True):
    print('Data: Load test data from data_loader')
    test_neigh, test_label = dl.get_test_neigh_w_random() if random_test else dl.get_test_neigh()
    test_true_data_by_edge, test_false_data_by_edge = defaultdict(list), defaultdict(list)
    for r_id in test_neigh.keys():
        for i in range(len(test_neigh[r_id][0])):
            h_id, t_id = test_neigh[r_id][0][i], test_neigh[r_id][1][i]
            if test_label[r_id][i] == 1:
                test_true_data_by_edge[str(r_id)].append((str(h_id), str(t_id)))
            else:
                test_false_data_by_edge[str(r_id)].append((str(h_id), str(t_id)))
    return test_true_data_by_edge, test_false_data_by_edge


def load_node_type(node_type):
    print('Info: We are loading node type')
    node_type_str = {}
    for k in node_type.keys():
        node_type_str[str(k)] = str(node_type[k])
    return node_type_str


def load_feature_data(f_name):
    feature_dic = {}
    with open(f_name, 'r') as f:
        first = True
        for line in f:
            if first:
                first = False
                continue
            items = line.strip().split()
            feature_dic[items[0]] = items[1:]
    return feature_dic


def generate_walks(network_data, num_walks, walk_length, schema, node_type, num_workers):
    if schema is not None:
        node_type = load_node_type(node_type)
        schema = schema.split(';')
    else:
        node_type = None

    all_walks = []
    for layer_id, layer_name in enumerate(network_data):
        tmp_data = network_data[layer_name]
        # start to do the random walk on a layer

        layer_walker = RWGraph(get_G_from_edges(tmp_data), node_type, num_workers)
        print('Generating random walks for layer', layer_id)
        layer_walks = layer_walker.simulate_walks(num_walks, walk_length,
                                                  schema=schema[layer_id] if schema != None else schema)

        all_walks.append(layer_walks)

    print('Finish generating the walks')

    return all_walks


def generate_pairs(all_walks, vocab, window_size, num_workers):
    pairs = []
    skip_window = window_size // 2
    for layer_id, walks in enumerate(all_walks):
        print('Generating training pairs for layer', layer_id)
        for walk in tqdm(walks):
            for i in range(len(walk)):
                for j in range(1, skip_window + 1):
                    if i - j >= 0:
                        pairs.append((vocab[walk[i]].index, vocab[walk[i - j]].index, layer_id))
                    if i + j < len(walk):
                        pairs.append((vocab[walk[i]].index, vocab[walk[i + j]].index, layer_id))
    return pairs


def generate_vocab(all_walks):
    index2word = []
    raw_vocab = defaultdict(int)

    for layer_id, walks in enumerate(all_walks):
        print('Counting vocab for layer', layer_id)
        for walk in tqdm(walks):
            for word in walk:
                raw_vocab[word] += 1

    vocab = {}
    for word, v in iteritems(raw_vocab):
        vocab[word] = Vocab(count=v, index=len(index2word))
        index2word.append(word)

    index2word.sort(key=lambda word: vocab[word].count, reverse=True)
    for i, word in enumerate(index2word):
        vocab[word].index = i

    return vocab, index2word


def generate(network_data, num_walks, walk_length, schema, window_size, num_workers, walk_file, pair_file,
             node_type=None):
    if os.path.exists(walk_file):
        print(f'Data: Load walks from {walk_file}')
        all_walks = pickle.load(open(walk_file, 'rb'))
    else:
        print(f'Data: Generate walks and write into {walk_file}')
        all_walks = generate_walks(network_data, num_walks, walk_length, schema, node_type, num_workers)
        pickle.dump(all_walks, open(walk_file, 'wb'))

    vocab, index2word = generate_vocab(all_walks)
    if os.path.exists(pair_file):
        print(f'Data: Load pairs from {pair_file}')
        train_pairs = pickle.load(open(pair_file, 'rb'))
    else:
        print(f'Data: Generate pairs and write into {pair_file}')
        train_pairs = generate_pairs(all_walks, vocab, window_size, num_workers)
        pickle.dump(train_pairs, open(pair_file, 'wb'))

    return vocab, index2word, train_pairs


def generate_neighbors(network_data, vocab, num_nodes, edge_types, neighbor_samples):
    edge_type_count = len(edge_types)
    neighbors = [[[] for __ in range(edge_type_count)] for _ in range(num_nodes)]
    for r in range(edge_type_count):
        print('Generating neighbors for layer', r)
        g = network_data[edge_types[r]]
        for (x, y) in tqdm(g):
            tmp = set(vocab.keys())
            if x in tmp and y in tmp:
                ix = vocab[x].index
                iy = vocab[y].index
                neighbors[ix][r].append(iy)
                neighbors[iy][r].append(ix)
        for i in range(num_nodes):
            if len(neighbors[i][r]) == 0:
                neighbors[i][r] = [i] * neighbor_samples
            elif len(neighbors[i][r]) < neighbor_samples:
                neighbors[i][r].extend(
                    list(np.random.choice(neighbors[i][r], size=neighbor_samples - len(neighbors[i][r]))))
            elif len(neighbors[i][r]) > neighbor_samples:
                neighbors[i][r] = list(np.random.choice(neighbors[i][r], size=neighbor_samples))
    return neighbors


def get_score(local_model, node1, node2):
    try:
        vector1 = local_model[node1]
        vector2 = local_model[node2]
        return np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
    except Exception as e:
        pass


def evaluate(model, true_edges, false_edges, dl):
    true_list = list()
    prediction_list = list()
    true_num = 0
    edge_list = [[], []]
    for edge in true_edges:
        tmp_score = get_score(model, str(edge[0]), str(edge[1]))
        if tmp_score is not None:
            true_list.append(1)
            prediction_list.append(tmp_score)
            true_num += 1
            edge_list[0].append(int(edge[0]))
            edge_list[1].append(int(edge[1]))

    for edge in false_edges:
        tmp_score = get_score(model, str(edge[0]), str(edge[1]))
        if tmp_score is not None:
            true_list.append(0)
            prediction_list.append(tmp_score)
            edge_list[0].append(int(edge[0]))
            edge_list[1].append(int(edge[1]))

    y_true = np.array(true_list)
    dl_scores = dl.evaluate(edge_list=edge_list, confidence=prediction_list, labels=y_true)
    return dl_scores
