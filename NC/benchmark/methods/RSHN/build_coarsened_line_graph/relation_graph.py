from build_coarsened_line_graph import utils
import numpy as np
import scipy.sparse as sp
from torch_geometric.data import Data
import torch


def build_relation_adj(org_graph, num_relations, rw_len=3, batch_size=3):
    '''build a AIFB-relation graph adjacency based on random walk from the original graph'''
    # graph = sp.csc_matrix(([1, 2, 2, 3, 3], ([0, 0, 1, 1, 2], [1, 2, 2, 3, 0])), shape=(4, 4))
    print('generating random walks, rw_len={:03d}, batch_size={:03d}.'.format(
        rw_len, batch_size))
    walker = utils.RandomWalker(org_graph, rw_len, batch_size=batch_size)
    rws = walker.walk().__next__()
    rws_edge = utils.edge_path_from_random_walk(org_graph, torch.LongTensor(rws))
    rws_edge = rws_edge - 1  # edge count from zero

    # Assemble score matrix from the random walks
    scores_matrix = utils.score_matrix_from_random_walks(
        rws_edge, num_relations).tocsr().tocoo()
    # Assemble a graph from the score matrix
    # AIFB-relation = utils.graph_from_scores(scores_matrix, n_edges=2)
    print('Random walk done!')
    return scores_matrix


def build_coarsened_line_graph(data,num_rel, rw_len, batch_size, name):
    num_relations = num_rel
    org_graph = data
    edge_index = org_graph.edge_index.numpy()
    edge_type = org_graph.edge_type.numpy() + 1
    row, col = edge_index
    org_adj = sp.coo_matrix((edge_type, (row, col)), shape=(
        org_graph.num_nodes, org_graph.num_nodes))

    scores_matrix = build_relation_adj(org_adj, num_relations, rw_len=rw_len,
                                       batch_size=batch_size)

    rel_graph = Data()
    rel_graph.x = np.eye(num_relations)
    rel_graph.edge_attr = scores_matrix.data
    rel_graph.edge_index = [scores_matrix.row, scores_matrix.col]

    # rel_graph.x = torch.FloatTensor(rel_graph.x)
    # rel_graph.edge_attr = torch.FloatTensor(rel_graph.edge_attr)
    # rel_graph.edge_index = torch.LongTensor(rel_graph.edge_index)

    save_rel_graph(rel_graph, name)
    return


def save_rel_graph(rel_graph, name):
    name = name
    x_path = './data/' + name + '-relation/rel_graph_x.npy'
    edge_index_path = './data/' + name + '-relation/rel_graph_edge_index.npy'
    edge_attr_path = './data/' + name + '-relation/rel_graph_edge_attr.npy'
    np.save(x_path, rel_graph.x)
    np.save(edge_index_path, rel_graph.edge_index)
    np.save(edge_attr_path, rel_graph.edge_attr)
    print('save ' + name + '-coarsened line graph via random walk successfully!')
    return


def load_rel_graph(name):
    name = name
    x_path = './data/' + name + '-relation/rel_graph_x.npy'
    edge_index_path = './data/' + name + '-relation/rel_graph_edge_index.npy'
    edge_attr_path = './data/' + name + '-relation/rel_graph_edge_attr.npy'
    rel_graph = Data()
    rel_graph.x = torch.FloatTensor(np.load(x_path))
    rel_graph.edge_index = torch.LongTensor(np.load(edge_index_path))
    rel_graph.edge_attr = torch.FloatTensor(np.load(edge_attr_path))
    print('load ' + name + '-coarsened line graph via random walk successfully!')
    return rel_graph
