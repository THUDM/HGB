from tokenize import Comment
import scipy.sparse as sp
import numpy as np
from numba import jit
import torch
import random

random.seed(1233)


def score_matrix_from_random_walks(random_walks, N, symmetric=True):
    """
    Compute the transition scores, i.e. how often a transition occurs, for all node pairs from
    the random walks provided.
    Parameters
    ----------
    random_walks: np.array of shape (n_walks, rw_len, N)
                  The input random walks to count the transitions in.
    N: int
       The number of nodes
    symmetric: bool, default: True
               Whether to symmetrize the resulting scores matrix.

    Returns
    -------
    scores_matrix: sparse matrix, shape (N, N)
                   Matrix whose entries (i,j) correspond to the number of times a transition from node i to j was
                   observed in the input random walks.

    """

    random_walks = np.array(random_walks)
    bigrams = np.array(list(zip(random_walks[:, :-1], random_walks[:, 1:])))
    bigrams = np.transpose(bigrams, [0, 2, 1])
    bigrams = bigrams.reshape([-1, 2])
    if symmetric:
        bigrams = np.row_stack((bigrams, bigrams[:, ::-1]))

    mat = sp.coo_matrix((np.ones(bigrams.shape[0]), (bigrams[:, 0], bigrams[:, 1])),
                        shape=[N, N])
    return mat


@jit(nopython=True)
def random_walk(edges, node_v, node_ixs, rwlen, n_walks=1):
    N = len(node_ixs)
    walk = []
    for w in range(n_walks):
        source_node_ixs = np.random.choice(N)
        source_node = node_v[source_node_ixs]
        walk.append(source_node)
        for it in range(rwlen - 1):
            source_node = walk[-1]
            source_node_ixs = np.where(node_v == source_node)[0][0]

            if source_node_ixs == N - 1:  # last_walk_node=last_node_v
                des_nodes = edges[node_ixs[source_node_ixs]::, 1]
            else:
                des_nodes = edges[node_ixs[source_node_ixs]:node_ixs[source_node_ixs + 1], 1]

            des_node = np.random.choice(des_nodes)
            walk.append(des_node)
    return np.array(walk)


class RandomWalker:
    def __init__(self, adj, rw_len, batch_size=128):
        self.adj = adj
        if not "lil" in str(type(adj)):
            self.adj = self.adj.tolil()

        self.rw_len = rw_len
        self.edges = np.array(self.adj.nonzero()).T  # all edges
        self.node_v, self.node_ixs = np.unique(self.edges[:, 0], return_index=True)  # get unique start node_index

        self.batch_size = batch_size

    def walk(self):
        while True:
            yield random_walk(self.edges, self.node_v, self.node_ixs, self.rw_len, self.batch_size).reshape(
                [-1, self.rw_len])


def edge_path_from_random_walk(graph, rws):
    '''build random walk edge_path'''
    edge_path = []
    # graph = graph.todense()
    row = torch.Tensor(graph.row)
    col = torch.Tensor(graph.col)
    data = torch.Tensor(graph.data)
    for rw in rws:
        edge = []
        node_src = rw[0]
        node_des = rw[1]
        # there could be several edges between two nodes
        row_index = row.eq(node_src)
        col_index = col.eq(node_des)


        common_index = (row_index + col_index).nonzero()
        common_edge = data[common_index]
        id = random.randint(1, len(common_edge))
        edge.append(int(common_edge[id - 1]))

        for i in range(2, len(rw)):
            node_src = node_des
            node_des = rw[i]
            row_index = row.eq(node_src)
            col_index = col.eq(node_des)
            common_index = (row_index + col_index).nonzero()
            common_edge = data[common_index]
            id = random.randint(1, len(common_edge))
            edge.append(int(common_edge[id - 1]))

        edge_path.append(edge)

    return np.array(edge_path)


def scores_graph_normalize(scores):
    max, min = scores.max(), scores.min()
    scores = float((scores - min)) / (max - min)
    return scores


def graph_from_scores(scores, n_edges):
    if len(scores.nonzero()[0]) < n_edges:
        return symmetric(scores) > 0

    target_g = np.zeros(scores.shape)  # initialize target graph
    scores_int = scores.toarray().copy()  # internal copy of the scores matrix
    scores_int[np.diag_indices_from(scores_int)] = 0  # set diagonal to zero

    N = scores.shape[0]

    for n in np.random.choice(N, replace=False, size=N):  # Iterate the nodes in random order

        row = scores_int[n, :].copy()
        if row.sum() == 0:
            continue

        probs = row / row.sum()

        target = np.random.choice(N, p=probs)
        target_g[n, target] = 1
        target_g[target, n] = 1

    diff = np.round((n_edges - target_g.sum()) / 2)
    if diff > 0:
        triu = np.triu(scores_int)
        triu[target_g > 0] = 0
        triu[np.diag_indices_from(scores_int)] = 0
        triu = triu / triu.sum()

        triu_ixs = np.triu_indices_from(scores_int)
        extra_edges = np.random.choice(triu_ixs[0].shape[0], replace=False, p=triu[triu_ixs], size=int(diff))

        target_g[(triu_ixs[0][extra_edges], triu_ixs[1][extra_edges])] = 1
        target_g[(triu_ixs[1][extra_edges], triu_ixs[0][extra_edges])] = 1

    target_g = symmetric(target_g)
    return target_g


def symmetric(directed_adjacency, clip_to_one=True):
    A_symmetric = directed_adjacency + directed_adjacency.T
    if clip_to_one:
        A_symmetric[A_symmetric > 1] = 1
    return A_symmetric
