import os
import numpy as np
import scipy.sparse as sp
from collections import Counter, defaultdict
from sklearn.metrics import f1_score

class data_loader:
    def __init__(self, path):
        self.path = path
        self.nodes = self.load_nodes()
        self.links = self.load_links()
        self.labels_train = self.load_labels('label.dat')
        self.labels_test = self.load_labels('label.dat.test')

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
        new_nodes = {'total':0, 'count':Counter(), 'attr':{}, 'shift':{}}
        new_links = {'total':0, 'count':Counter(), 'meta':{}, 'data':defaultdict(list)}
        new_labels_train = {'num_classes':0, 'total':0, 'count':Counter(), 'data':None, 'mask':None}
        new_labels_test = {'num_classes':0, 'total':0, 'count':Counter(), 'data':None, 'mask':None}
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
                old_idx.extend(range(beg, beg+cnt))
                
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
                new_links['meta'][net] = tuple(map(lambda x:old_nt2new_nt[x], self.links['meta'][et]))
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
            ini = ini.dot(self.links['data'][x])
        return ini

    def dfs(self, now, meta, meta_dict):
        if len(meta) == 0:
            meta_dict[now[0]].append(now)
            return
        th_mat = self.links['data'][meta[0]]
        th_node = now[-1]
        for col in th_mat[th_node].nonzero()[1]:
            self.dfs(now+[col], meta[1:], meta_dict)

    def get_full_meta_path(self, meta=[]):
        """
        Get full meta path for each node
            meta is a list of edge types (also can be denoted by a pair of node types)
            return a dict of list[list] (key is node_id)
        """
        meta = [self.get_edge_type(x) for x in meta]
        if len(meta) == 1:
            meta_dict = {}
            for i in range(self.nodes['total']):
                meta_dict[i] = []
                self.dfs([i], meta, meta_dict)
        else:
            meta_dict1 = {}
            meta_dict2 = {}
            mid = len(meta) // 2
            meta1 = meta[:mid]
            meta2 = meta[mid:]
            for i in range(self.nodes['total']):
                meta_dict1[i] = []
                self.dfs([i], meta1, meta_dict1)
            for i in range(self.nodes['total']):
                meta_dict2[i] = []
                self.dfs([i], meta2, meta_dict2)
            meta_dict = {}
            for i in range(self.nodes['total']):
                meta_dict[i] = []
                for beg in meta_dict1[i]:
                    for end in meta_dict2[beg[-1]]:
                        meta_dict[i].append(beg+end[1:])
        return meta_dict

    def evaluate(self, pred):
        y_true = self.labels_test['data'][self.labels_test['mask']]
        print('eval', y_true, pred)
        # print('y_sum', y_true.sum(0))
        micro = f1_score(y_true, pred, average='micro')
        macro = f1_score(y_true, pred, average='macro')
        result = {
            'micro-f1': micro,
            'macro-f1': macro
        }
        return result

    def load_labels(self, name):
        """
        return labels dict
            num_classes: total number of labels
            total: total number of labeled data
            count: number of labeled data for each node type
            data: a numpy matrix with shape (self.nodes['total'], self.labels['num_classes'])
            mask: to indicate if that node is labeled, if False, that line of data is masked
        """
        labels = {'num_classes':0, 'total':0, 'count':Counter(), 'data':None, 'mask':None}
        nc = 0
        mask = np.zeros(self.nodes['total'], dtype=bool)
        data = [None for i in range(self.nodes['total'])]
        with open(os.path.join(self.path, name), 'r', encoding='utf-8') as f:
            for line in f:
                th = line.split('\t')
                node_id, node_name, node_type, node_label = int(th[0]), th[1], int(th[2]), list(map(int, th[3].split(',')))
                for label in node_label:
                    nc = max(nc, label+1)
                mask[node_id] = True
                data[node_id] = node_label
                labels['count'][node_type] += 1
                labels['total'] += 1
        labels['num_classes'] = nc
        new_data = np.zeros((self.nodes['total'], labels['num_classes']), dtype=int)
        for i,x in enumerate(data):
            if x is not None:
                for j in x:
                    new_data[i, j] = 1
        labels['data'] = new_data
        labels['mask'] = mask
        return labels

    def get_node_type(self, node_id):
        for i in range(len(self.nodes['shift'])):
            if node_id < self.nodes['shift'][i]+self.nodes['count'][i]:
                return i

    def get_edge_type(self, info):
        if type(info) is int or len(info) == 1:
            return info
        for i in range(len(self.links['meta'])):
            if self.links['meta'][i] == info:
                return i
        raise Exception('No available edge type')

    def get_edge_info(self, edge_id):
        return self.links['meta'][edge_id]
    
    def list_to_sp_mat(self, li):
        data = [x[2] for x in li]
        i = [x[0] for x in li]
        j = [x[1] for x in li]
        return sp.coo_matrix((data, (i,j)), shape=(self.nodes['total'], self.nodes['total'])).tocsr()
    
    def load_links(self):
        """
        return links dict
            total: total number of links
            count: a dict of int, number of links for each type
            meta: a dict of tuple, explaining the link type is from what type of node to what type of node
            data: a dict of sparse matrices, each link type with one matrix. Shapes are all (nodes['total'], nodes['total'])
        """
        links = {'total':0, 'count':Counter(), 'meta':{}, 'data':defaultdict(list)}
        with open(os.path.join(self.path, 'link.dat'), 'r', encoding='utf-8') as f:
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
        nodes = {'total':0, 'count':Counter(), 'attr':{}, 'shift':{}}
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
                for j in range(shift, shift+nodes['count'][i]):
                    mat.append(nodes['attr'][j])
                attr[i] = np.array(mat)
            else:
                attr[i] = None
            shift += nodes['count'][i]
        nodes['attr'] = attr
        return nodes