import os
import numpy as np
import scipy.sparse as sp
from collections import Counter, defaultdict, OrderedDict
from sklearn.metrics import f1_score, auc, roc_auc_score, precision_recall_curve
import random
import copy


class data_loader:
    def __init__(self, path, edge_types=[]):
        self.path = path
        self.splited = False
        self.nodes = self.load_nodes()
        self.links = self.load_links('link.dat')
        self.links_test = self.load_links('link.dat.test')
        self.test_types = list(self.links_test['data'].keys()) if edge_types == [] else edge_types
        self.types = self.load_types('node.dat')
        self.train_pos, self.valid_pos = self.get_train_valid_pos()
        self.train_neg, self.valid_neg = self.get_train_neg(), self.get_valid_neg()
        self.gen_transpose_links()
        self.nonzero = False

    def get_train_valid_pos(self, train_ratio=0.9):
        if self.splited:
            return self.train_pos, self.valid_pos
        else:
            edge_types = self.links['data'].keys()
            train_pos, valid_pos = dict(), dict()
            for r_id in edge_types:
                train_pos[r_id] = [[], []]
                valid_pos[r_id] = [[], []]
                row, col = self.links['data'][r_id].nonzero()
                last_h_id = -1
                for (h_id, t_id) in zip(row, col):
                    if h_id != last_h_id:
                        train_pos[r_id][0].append(h_id)
                        train_pos[r_id][1].append(t_id)
                        last_h_id = h_id

                    else:
                        if random.random() < train_ratio:
                            train_pos[r_id][0].append(h_id)
                            train_pos[r_id][1].append(t_id)
                        else:
                            valid_pos[r_id][0].append(h_id)
                            valid_pos[r_id][1].append(t_id)
                            self.links['data'][r_id][h_id, t_id] = 0
                            self.links['count'][r_id] -= 1
                            self.links['total'] -= 1
                self.links['data'][r_id].eliminate_zeros()
            self.splited = True
            return train_pos, valid_pos

    def get_sub_graph(self, node_types_tokeep):
        """
        node_types_tokeep is a list or set of node types that you want to keep in the sub-graph
        We only support whole type sub-graph for now.
        This is an in-place update function!
        return: old node type id to new node type id dict, old edge type id to new edge type id dict
        """
        keep = set(node_types_tokeep)
        new_node_type = 0
        new_node_id = 0
        new_nodes = {'total': 0, 'count': Counter(), 'attr': {}, 'shift': {}}
        new_links = {'total': 0, 'count': Counter(), 'meta': {}, 'data': defaultdict(list)}
        new_labels_train = {'num_classes': 0, 'total': 0, 'count': Counter(), 'data': None, 'mask': None}
        new_labels_test = {'num_classes': 0, 'total': 0, 'count': Counter(), 'data': None, 'mask': None}
        old_nt2new_nt = {}
        old_idx = []
        for node_type in self.nodes['count']:
            if node_type in keep:
                nt = node_type
                nnt = new_node_type
                old_nt2new_nt[nt] = nnt
                cnt = self.nodes['count'][nt]
                new_nodes['total'] += cnt
                new_nodes['count'][nnt] = cnt
                new_nodes['attr'][nnt] = self.nodes['attr'][nt]
                new_nodes['shift'][nnt] = new_node_id
                beg = self.nodes['shift'][nt]
                old_idx.extend(range(beg, beg + cnt))

                cnt_label_train = self.labels_train['count'][nt]
                new_labels_train['count'][nnt] = cnt_label_train
                new_labels_train['total'] += cnt_label_train
                cnt_label_test = self.labels_test['count'][nt]
                new_labels_test['count'][nnt] = cnt_label_test
                new_labels_test['total'] += cnt_label_test

                new_node_type += 1
                new_node_id += cnt

        new_labels_train['num_classes'] = self.labels_train['num_classes']
        new_labels_test['num_classes'] = self.labels_test['num_classes']
        for k in ['data', 'mask']:
            new_labels_train[k] = self.labels_train[k][old_idx]
            new_labels_test[k] = self.labels_test[k][old_idx]

        old_et2new_et = {}
        new_edge_type = 0
        for edge_type in self.links['count']:
            h, t = self.links['meta'][edge_type]
            if h in keep and t in keep:
                et = edge_type
                net = new_edge_type
                old_et2new_et[et] = net
                new_links['total'] += self.links['count'][et]
                new_links['count'][net] = self.links['count'][et]
                new_links['meta'][net] = tuple(map(lambda x: old_nt2new_nt[x], self.links['meta'][et]))
                new_links['data'][net] = self.links['data'][et][old_idx][:, old_idx]
                new_edge_type += 1

        self.nodes = new_nodes
        self.links = new_links
        self.labels_train = new_labels_train
        self.labels_test = new_labels_test
        return old_nt2new_nt, old_et2new_et

    def get_meta_path(self, meta=[]):
        """
        Get meta path matrix
            meta is a list of edge types (also can be denoted by a pair of node types)
            return a sparse matrix with shape [node_num, node_num]
        """
        ini = sp.eye(self.nodes['total'])
        meta = [self.get_edge_type(x) for x in meta]
        for x in meta:
            ini = ini.dot(self.links['data'][x]) if x >= 0 else ini.dot(self.links['data_trans'][-x - 1])
        return ini

    def get_nonzero(self):
        self.nonzero = True
        self.re_cache = defaultdict(dict)
        for k in self.links['data']:
            th_mat = self.links['data'][k]
            for i in range(th_mat.shape[0]):
                th = th_mat[i].nonzero()[1]
                self.re_cache[k][i] = th
        for k in self.links['data_trans']:
            th_mat = self.links['data_trans'][k]
            for i in range(th_mat.shape[0]):
                th = th_mat[i].nonzero()[1]
                self.re_cache[-k - 1][i] = th

    def dfs(self, now, meta, meta_dict):
        if len(meta) == 0:
            meta_dict[now[0]].append(now)
            return
        # th_mat = self.links['data'][meta[0]] if meta[0] >= 0 else self.links['data_trans'][-meta[0] - 1]
        th_node = now[-1]
        for col in self.re_cache[meta[0]][th_node]:  # th_mat[th_node].nonzero()[1]:
            self.dfs(now + [col], meta[1:], meta_dict)

    def get_full_meta_path(self, meta=[], symmetric=False):
        """
        Get full meta path for each node
            meta is a list of edge types (also can be denoted by a pair of node types)
            return a dict of list[list] (key is node_id)
        """
        if not self.nonzero:
            self.get_nonzero()
        meta = [self.get_edge_type(x) for x in meta]
        if len(meta) == 1:
            meta_dict = {}
            start_node_type = self.links['meta'][meta[0]][0] if meta[0] >= 0 else self.links['meta'][-meta[0] - 1][1]
            trav = range(self.nodes['shift'][start_node_type],
                         self.nodes['shift'][start_node_type] + self.nodes['count'][start_node_type])
            for i in trav:
                meta_dict[i] = []
                self.dfs([i], meta, meta_dict)
        else:
            meta_dict1 = {}
            meta_dict2 = {}
            mid = len(meta) // 2
            meta1 = meta[:mid]
            meta2 = meta[mid:]
            start_node_type = self.links['meta'][meta1[0]][0] if meta1[0] >= 0 else self.links['meta'][-meta1[0] - 1][1]
            trav = range(self.nodes['shift'][start_node_type],
                         self.nodes['shift'][start_node_type] + self.nodes['count'][start_node_type])
            for i in trav:
                meta_dict1[i] = []
                self.dfs([i], meta1, meta_dict1)
            start_node_type = self.links['meta'][meta2[0]][0] if meta2[0] >= 0 else self.links['meta'][-meta2[0] - 1][1]
            trav = range(self.nodes['shift'][start_node_type],
                         self.nodes['shift'][start_node_type] + self.nodes['count'][start_node_type])
            for i in trav:
                meta_dict2[i] = []
            if symmetric:
                for k in meta_dict1:
                    paths = meta_dict1[k]
                    for x in paths:
                        meta_dict2[x[-1]].append(list(reversed(x)))
            else:
                for i in trav:
                    self.dfs([i], meta2, meta_dict2)
            meta_dict = {}
            start_node_type = self.links['meta'][meta1[0]][0] if meta1[0] >= 0 else self.links['meta'][-meta1[0] - 1][1]
            for i in range(self.nodes['shift'][start_node_type],
                           self.nodes['shift'][start_node_type] + self.nodes['count'][start_node_type]):
                meta_dict[i] = []
                for beg in meta_dict1[i]:
                    for end in meta_dict2[beg[-1]]:
                        meta_dict[i].append(beg + end[1:])
        return meta_dict

    @staticmethod
    def evaluate(edge_list, confidence, labels):
        """
        :param edge_list: shape(2, edge_num)
        :param confidence: shape(edge_num,)
        :param labels: shape(edge_num,)
        :return: dict with all scores we need
        """
        confidence = np.array(confidence)
        labels = np.array(labels)
        roc_auc = roc_auc_score(labels, confidence)
        mrr_list, cur_mrr = [], 0
        t_dict, labels_dict, conf_dict = defaultdict(list), defaultdict(list), defaultdict(list)
        for i, h_id in enumerate(edge_list[0]):
            t_dict[h_id].append(edge_list[1][i])
            labels_dict[h_id].append(labels[i])
            conf_dict[h_id].append(confidence[i])
        for h_id in t_dict.keys():
            conf_array = np.array(conf_dict[h_id])
            rank = np.argsort(-conf_array)
            sorted_label_array = np.array(labels_dict[h_id])[rank]
            pos_index = np.where(sorted_label_array == 1)[0]
            if len(pos_index) == 0:
                continue
            pos_min_rank = np.min(pos_index)
            cur_mrr = 1 / (1 + pos_min_rank)
            mrr_list.append(cur_mrr)
        mrr = np.mean(mrr_list)

        return {'roc_auc': roc_auc, 'MRR': mrr}

    def get_node_type(self, node_id):
        for i in range(len(self.nodes['shift'])):
            if node_id < self.nodes['shift'][i] + self.nodes['count'][i]:
                return i

    def get_edge_type(self, info):
        if type(info) is int or len(info) == 1:
            return info
        for i in range(len(self.links['meta'])):
            if self.links['meta'][i] == info:
                return i
        info = (info[1], info[0])
        for i in range(len(self.links['meta'])):
            if self.links['meta'][i] == info:
                return -i - 1
        raise Exception('No available edge type')

    def get_edge_info(self, edge_id):
        return self.links['meta'][edge_id]

    def list_to_sp_mat(self, li):
        data = [x[2] for x in li]
        i = [x[0] for x in li]
        j = [x[1] for x in li]
        return sp.coo_matrix((data, (i, j)), shape=(self.nodes['total'], self.nodes['total'])).tocsr()

    def load_types(self, name):
        """
        return types dict
            types: list of types
            total: total number of nodes
            data: a dictionary of type of all nodes)
        """
        types = {'types': list(), 'total': 0, 'data': dict()}
        with open(os.path.join(self.path, name), 'r', encoding='utf-8') as f:
            for line in f:
                th = line.strip().split('\t')
                node_id, node_name, node_type = int(th[0]), th[1], int(th[2])
                types['data'][node_id] = node_type
                types['types'].append(node_type)
                types['total'] += 1
        types['types'] = list(set(types['types']))
        return types

    def get_train_neg(self, edge_types=[]):
        edge_types = self.test_types if edge_types == [] else edge_types
        train_neg = dict()
        for r_id in edge_types:
            h_type, t_type = self.links['meta'][r_id]
            t_range = (self.nodes['shift'][t_type], self.nodes['shift'][t_type] + self.nodes['count'][t_type])
            '''get neg_neigh'''
            train_neg[r_id] = [[], []]
            for h_id in self.train_pos[r_id][0]:
                train_neg[r_id][0].append(h_id)
                neg_t = int(random.random() * (t_range[1] - t_range[0])) + t_range[0]
                train_neg[r_id][1].append(neg_t)
        return train_neg

    def get_valid_neg(self, edge_types=[]):
        edge_types = self.test_types if edge_types == [] else edge_types
        valid_neg = dict()
        for r_id in edge_types:
            h_type, t_type = self.links['meta'][r_id]
            t_range = (self.nodes['shift'][t_type], self.nodes['shift'][t_type] + self.nodes['count'][t_type])
            '''get neg_neigh'''
            valid_neg[r_id] = [[], []]
            for h_id in self.valid_pos[r_id][0]:
                valid_neg[r_id][0].append(h_id)
                neg_t = int(random.random() * (t_range[1] - t_range[0])) + t_range[0]
                valid_neg[r_id][1].append(neg_t)
        return valid_neg

    def get_test_neigh_2hop(self):
        return self.get_test_neigh()

    def get_test_neigh(self):
        random.seed(1)
        neg_neigh, pos_neigh, test_neigh, test_label = dict(), dict(), dict(), dict()
        edge_types = self.test_types
        '''get sec_neigh'''
        pos_links = 0
        for r_id in self.links['data'].keys():
            pos_links += self.links['data'][r_id] + self.links['data'][r_id].T
        for r_id in self.links_test['data'].keys():
            pos_links += self.links_test['data'][r_id] + self.links_test['data'][r_id].T
        for r_id in self.valid_pos.keys():
            values = [1] * len(self.valid_pos[r_id][0])
            valid_of_rel = sp.coo_matrix((values, self.valid_pos[r_id]), shape=pos_links.shape)
            pos_links += valid_of_rel

        r_double_neighs = np.dot(pos_links, pos_links)
        data = r_double_neighs.data
        data[:] = 1
        r_double_neighs = \
            sp.coo_matrix((data, r_double_neighs.nonzero()), shape=np.shape(pos_links), dtype=int) \
            - sp.coo_matrix(pos_links, dtype=int) \
            - sp.lil_matrix(np.eye(np.shape(pos_links)[0], dtype=int))
        data = r_double_neighs.data
        pos_count_index = np.where(data > 0)
        row, col = r_double_neighs.nonzero()
        r_double_neighs = sp.coo_matrix((data[pos_count_index], (row[pos_count_index], col[pos_count_index])),
                                        shape=np.shape(pos_links))

        row, col = r_double_neighs.nonzero()
        data = r_double_neighs.data
        sec_index = np.where(data > 0)
        row, col = row[sec_index], col[sec_index]

        relation_range = [self.nodes['shift'][k] for k in range(len(self.nodes['shift']))] + [self.nodes['total']]
        for r_id in self.links_test['data'].keys():
            neg_neigh[r_id] = defaultdict(list)
            h_type, t_type = self.links_test['meta'][r_id]
            r_id_index = np.where((row >= relation_range[h_type]) & (row < relation_range[h_type + 1])
                                  & (col >= relation_range[t_type]) & (col < relation_range[t_type + 1]))[0]
            # r_num = np.zeros((3, 3))
            # for h_id, t_id in zip(row, col):
            #     r_num[self.get_node_type(h_id)][self.get_node_type(t_id)] += 1
            r_row, r_col = row[r_id_index], col[r_id_index]
            for h_id, t_id in zip(r_row, r_col):
                neg_neigh[r_id][h_id].append(t_id)

        for r_id in edge_types:
            '''get pos_neigh'''
            pos_neigh[r_id] = defaultdict(list)
            (row, col), data = self.links_test['data'][r_id].nonzero(), self.links_test['data'][r_id].data
            for h_id, t_id in zip(row, col):
                pos_neigh[r_id][h_id].append(t_id)

            '''sample neg as same number as pos for each head node'''
            test_neigh[r_id] = [[], []]
            pos_list = [[], []]
            test_label[r_id] = []
            for h_id in sorted(list(pos_neigh[r_id].keys())):
                pos_list[0] = [h_id] * len(pos_neigh[r_id][h_id])
                pos_list[1] = pos_neigh[r_id][h_id]
                test_neigh[r_id][0].extend(pos_list[0])
                test_neigh[r_id][1].extend(pos_list[1])
                test_label[r_id].extend([1] * len(pos_list[0]))

                neg_list = random.choices(neg_neigh[r_id][h_id], k=len(pos_list[0])) if len(
                    neg_neigh[r_id][h_id]) != 0 else []
                test_neigh[r_id][0].extend([h_id] * len(neg_list))
                test_neigh[r_id][1].extend(neg_list)
                test_label[r_id].extend([0] * len(neg_list))
        return test_neigh, test_label

    def get_test_neigh_w_random(self):
        random.seed(1)
        all_had_neigh = defaultdict(list)
        neg_neigh, pos_neigh, test_neigh, test_label = dict(), dict(), dict(), dict()
        edge_types = self.test_types
        '''get pos_links of train and test data'''
        pos_links = 0
        for r_id in self.links['data'].keys():
            pos_links += self.links['data'][r_id] + self.links['data'][r_id].T
        for r_id in self.links_test['data'].keys():
            pos_links += self.links_test['data'][r_id] + self.links_test['data'][r_id].T
        for r_id in self.valid_pos.keys():
            values = [1] * len(self.valid_pos[r_id][0])
            valid_of_rel = sp.coo_matrix((values, self.valid_pos[r_id]), shape=pos_links.shape)
            pos_links += valid_of_rel

        row, col = pos_links.nonzero()
        for h_id, t_id in zip(row, col):
            all_had_neigh[h_id].append(t_id)
        for h_id in all_had_neigh.keys():
            all_had_neigh[h_id] = set(all_had_neigh[h_id])
        for r_id in edge_types:
            h_type, t_type = self.links_test['meta'][r_id]
            t_range = (self.nodes['shift'][t_type], self.nodes['shift'][t_type] + self.nodes['count'][t_type])
            '''get pos_neigh and neg_neigh'''
            pos_neigh[r_id], neg_neigh[r_id] = defaultdict(list), defaultdict(list)
            (row, col), data = self.links_test['data'][r_id].nonzero(), self.links_test['data'][r_id].data
            for h_id, t_id in zip(row, col):
                pos_neigh[r_id][h_id].append(t_id)
                neg_t = int(random.random() * (t_range[1] - t_range[0])) + t_range[0]
                while neg_t in all_had_neigh[h_id]:
                    neg_t = int(random.random() * (t_range[1] - t_range[0])) + t_range[0]
                neg_neigh[r_id][h_id].append(neg_t)
            '''get the test_neigh'''
            test_neigh[r_id] = [[], []]
            pos_list = [[], []]
            neg_list = [[], []]
            test_label[r_id] = []
            for h_id in sorted(list(pos_neigh[r_id].keys())):
                pos_list[0] = [h_id] * len(pos_neigh[r_id][h_id])
                pos_list[1] = pos_neigh[r_id][h_id]
                test_neigh[r_id][0].extend(pos_list[0])
                test_neigh[r_id][1].extend(pos_list[1])
                test_label[r_id].extend([1] * len(pos_neigh[r_id][h_id]))
                neg_list[0] = [h_id] * len(neg_neigh[r_id][h_id])
                neg_list[1] = neg_neigh[r_id][h_id]
                test_neigh[r_id][0].extend(neg_list[0])
                test_neigh[r_id][1].extend(neg_list[1])
                test_label[r_id].extend([0] * len(neg_neigh[r_id][h_id]))
        return test_neigh, test_label

    def get_test_neigh_full_random(self):
        edge_types = self.test_types
        random.seed(1)
        '''get pos_links of train and test data'''
        all_had_neigh = defaultdict(list)
        pos_links = 0
        for r_id in self.links['data'].keys():
            pos_links += self.links['data'][r_id] + self.links['data'][r_id].T
        for r_id in self.links_test['data'].keys():
            pos_links += self.links_test['data'][r_id] + self.links_test['data'][r_id].T
        for r_id in self.valid_pos.keys():
            values = [1] * len(self.valid_pos[r_id][0])
            valid_of_rel = sp.coo_matrix((values, self.valid_pos[r_id]), shape=pos_links.shape)
            pos_links += valid_of_rel

        row, col = pos_links.nonzero()
        for h_id, t_id in zip(row, col):
            all_had_neigh[h_id].append(t_id)
        for h_id in all_had_neigh.keys():
            all_had_neigh[h_id] = set(all_had_neigh[h_id])
        test_neigh, test_label = dict(), dict()
        for r_id in edge_types:
            test_neigh[r_id] = [[], []]
            test_label[r_id] = []
            h_type, t_type = self.links_test['meta'][r_id]
            h_range = (self.nodes['shift'][h_type], self.nodes['shift'][h_type] + self.nodes['count'][h_type])
            t_range = (self.nodes['shift'][t_type], self.nodes['shift'][t_type] + self.nodes['count'][t_type])
            (row, col), data = self.links_test['data'][r_id].nonzero(), self.links_test['data'][r_id].data
            for h_id, t_id in zip(row, col):
                test_neigh[r_id][0].append(h_id)
                test_neigh[r_id][1].append(t_id)
                test_label[r_id].append(1)
                neg_h = int(random.random() * (h_range[1] - h_range[0])) + h_range[0]
                neg_t = int(random.random() * (t_range[1] - t_range[0])) + t_range[0]
                while neg_t in all_had_neigh[neg_h]:
                    neg_h = int(random.random() * (h_range[1] - h_range[0])) + h_range[0]
                    neg_t = int(random.random() * (t_range[1] - t_range[0])) + t_range[0]
                test_neigh[r_id][0].append(neg_h)
                test_neigh[r_id][1].append(neg_t)
                test_label[r_id].append(0)
        return test_neigh, test_label

    def gen_transpose_links(self):
        self.links['data_trans'] = defaultdict()
        for r_id in self.links['data'].keys():
            self.links['data_trans'][r_id] = self.links['data'][r_id].T

    def load_links(self, name):
        """
        return links dict
            total: total number of links
            count: a dict of int, number of links for each type
            meta: a dict of tuple, explaining the link type is from what type of node to what type of node
            data: a dict of sparse matrices, each link type with one matrix. Shapes are all (nodes['total', nodes['total'])
        """
        links = {'total': 0, 'count': Counter(), 'meta': {}, 'data': defaultdict(list)}
        with open(os.path.join(self.path, name), 'r', encoding='utf-8') as f:
            for line in f:
                th = line.split('\t')
                h_id, t_id, r_id, link_weight = int(th[0]), int(th[1]), int(th[2]), float(th[3])
                if r_id not in links['meta']:
                    h_type = self.get_node_type(h_id)
                    t_type = self.get_node_type(t_id)
                    links['meta'][r_id] = (h_type, t_type)
                links['data'][r_id].append((h_id, t_id, link_weight))
                links['count'][r_id] += 1
                links['total'] += 1
        new_data = {}
        for r_id in links['data']:
            new_data[r_id] = self.list_to_sp_mat(links['data'][r_id])
        links['data'] = new_data
        return links

    def load_nodes(self):
        """
        return nodes dict
        total: total number of nodes
        count: a dict of int, number of nodes for each type
        attr: a dict of np.array (or None), attribute matrices for each type of nodes
        shift: node_id shift for each type. You can get the id range of a type by
                    [ shift[node_type], shift[node_type]+count[node_type] )
        """
        nodes = {'total': 0, 'count': Counter(), 'attr': {}, 'shift': {}}
        with open(os.path.join(self.path, 'node.dat'), 'r', encoding='utf-8') as f:
            for line in f:
                th = line.split('\t')
                if len(th) == 4:
                    # Then this line of node has attribute
                    node_id, node_name, node_type, node_attr = th
                    node_id = int(node_id)
                    node_type = int(node_type)
                    node_attr = list(map(float, node_attr.split(',')))
                    nodes['count'][node_type] += 1
                    nodes['attr'][node_id] = node_attr
                    nodes['total'] += 1
                elif len(th) == 3:
                    # Then this line of node doesn't have attribute
                    node_id, node_name, node_type = th
                    node_id = int(node_id)
                    node_type = int(node_type)
                    nodes['count'][node_type] += 1
                    nodes['total'] += 1
                else:
                    raise Exception("Too few information to parse!")
        shift = 0
        attr = {}
        for i in range(len(nodes['count'])):
            nodes['shift'][i] = shift
            if shift in nodes['attr']:
                mat = []
                for j in range(shift, shift + nodes['count'][i]):
                    mat.append(nodes['attr'][j])
                attr[i] = np.array(mat)
            else:
                attr[i] = None
            shift += nodes['count'][i]
        nodes['attr'] = attr
        return nodes
