import six.moves.cPickle as pickle
import numpy as np
import string
import re
import random
import math
from collections import Counter
from itertools import *
import sys
import os
import json
from collections import defaultdict
import torch as th

class input_data(object):
    def __init__(self, args, dl):
        self.args = args
        self.dl = dl
        node_n, node_shift = self.dl.nodes['count'], self.dl.nodes['shift']
        self.node_n, self.node_shift = node_n, node_shift
        data_info = json.load(open(f"../../data/{args.data}/info.dat", 'r'))
        node_type2name = data_info['node.dat']
        self.node_type2name, self.node_name2type = dict(), dict()
        node_types = [int(k) for k in node_type2name.keys()]
        # todo all new setting of new dataset for random_walk_length and top k neighs
        self.while_max_count = 1e5
        if args.data=='amazon':
            self.standand_node_L = [100]
            self.top_k = {0:10}
            self.neigh_L = 100
        elif args.data=='LastFM' or args.data=='LastFM_magnn':
            self.standand_node_L = [20,70,10]
            self.top_k = {0:10,1:10,2:5}
            self.neigh_L = 80
        elif args.data=='PubMed':
            self.standand_node_L = [20,30,40,10]
            self.top_k = {0:10,1:10,2:10,3:5}
            self.neigh_L = 70
            self.while_max_count = 1e3

        for k in node_types:
            name = node_type2name[str(k)][0]
            self.node_type2name[k] = name
            self.node_name2type[name] = k
        edge_list = dict()
        for edge_type in sorted(dl.links['meta'].keys()):
            h_type, t_type = dl.links['meta'][edge_type]
            h_node_count = dl.nodes['count'][h_type]
            edge_list[edge_type] = [[] for k in range(h_node_count)]
        neigh_list = dict()
        for node_type in dl.nodes['count'].keys():
            node_count = dl.nodes['count'][node_type]
            neigh_list[node_type] = [[] for k in range(node_count)]

        """ get neigh of each edge type"""
        for edge_type in sorted(dl.links['meta'].keys()):
            h_type, t_type = dl.links['meta'][edge_type]
            row,col = self.dl.links['data'][edge_type].nonzero()
            for r,c in zip(row,col):
                h_id,t_id = r-node_shift[h_type],c-node_shift[t_type]
                edge_list[edge_type][h_id].append(f"{self.node_type2name[t_type]}{t_id}")
        """ get neigh of each node type"""
        for node_type in dl.nodes['count'].keys():
            for edge_type in edge_list.keys():
                h_type, t_type = dl.links['meta'][edge_type]
                if node_type == h_type:
                    for n_id in range(node_n[node_type]):
                        neigh_list[node_type][n_id] += edge_list[edge_type][n_id]

        self.edge_list = edge_list
        self.neigh_list = neigh_list

    """genarate het_neigh_train.txt with walk_restart"""

    def gen_het_w_walk_restart(self, file_):
        print(f'Info: generate {file_} start.')
        node_n = self.dl.nodes['count']
        neigh_list_train = dict()
        node_types = list(self.dl.nodes['count'].keys())
        for node_type in node_types:
            neigh_list_train[node_type] = [[] for k in range(node_n[node_type])]

        # generate neighbor set via random walk with restart
        for node_type in node_types:
            for n_id in range(node_n[node_type]):
                neigh_temp = self.neigh_list[node_type][n_id]
                neigh_train = neigh_list_train[node_type][n_id]
                curNode = self.node_type2name[node_type] + str(n_id)
                curNodeType = node_type
                if len(neigh_temp):
                    neigh_L = 0
                    node_L = np.array([0 for node_type in node_types])
                    while_count = 0
                    while neigh_L < self.neigh_L and while_count<self.while_max_count:  # maximum neighbor size = 100
                        while_count+=1
                        rand_p = random.random()  # return p
                        if rand_p > 0.5:
                            if self.neigh_list[curNodeType][int(curNode[1:])]==[]:
                                break
                            curNode = random.choice(self.neigh_list[curNodeType][int(curNode[1:])])
                            curNodeType = self.node_name2type[curNode[0]]
                            if node_L[curNodeType] < self.standand_node_L[curNodeType]:
                                neigh_train.append(curNode)
                                neigh_L += 1
                                node_L[curNodeType] += 1
                        else:
                            curNode = self.node_type2name[node_type] + str(n_id)
                            curNodeType = self.node_name2type[curNode[0]]

        for node_type in node_types:
            for n_id in range(node_n[node_type]):
                neigh_list_train[node_type][n_id] = list(neigh_list_train[node_type][n_id])

        neigh_f = open(file_, "w")
        for node_type in node_types:
            for n_id in range(node_n[node_type]):
                neigh_train = neigh_list_train[node_type][n_id]
                curNode = self.node_type2name[node_type] + str(n_id)
                if len(neigh_train):
                    neigh_f.write(curNode + ":")
                    for k in range(len(neigh_train) - 1):
                        neigh_f.write(neigh_train[k] + ",")
                    neigh_f.write(neigh_train[-1] + "\n")
        neigh_f.close()
        print(f'Info: generate {file_} done.')

    def gen_het_w_walk(self, file_):
        node_type = 0
        edge_type = 2
        print(f'Info: generate {file_} start.')
        het_walk_f = open(file_, "w")
        for i in range(self.args.walk_n):
            for n_id in range(self.node_n[node_type]):
                if len(self.neigh_list[node_type][n_id]):
                    curNode = self.node_type2name[node_type] + str(n_id)
                    curNodeType = self.node_name2type[curNode[0]]
                    het_walk_f.write(curNode + " ")
                    for l in range(self.args.walk_L - 1):
                        curNode = int(curNode[1:])
                        curNode = random.choice(self.neigh_list[curNodeType][curNode])
                        curNodeType = self.node_name2type[curNode[0]]
                        het_walk_f.write(curNode + " ")
                    het_walk_f.write("\n")
        het_walk_f.close()
        print(f'Info: generate {file_} done.')

    def compute_sample_p(self):
        print("Info :compute sampling ratio for each kind of triple start.")
        window = self.args.window
        walk_L = self.args.walk_L
        node_n, node_shift = self.dl.nodes['count'], self.dl.nodes['shift']
        total_triple_n = np.zeros((len(self.node_n), len(self.node_n)), dtype=np.float)  # sixteen kinds of triples
        temp_dir = os.path.join(sys.path[0], f'{self.args.data}-temp')
        het_walk_f = open(os.path.join(temp_dir, "het_random_walk.txt"), "r")
        centerNode = ''
        neighNode = ''
        for line in het_walk_f:
            line = line.strip()
            path = re.split(' ', line)
            for j in range(walk_L):
                centerNode = path[j]
                if len(centerNode) > 1:
                    centerType = self.node_name2type[centerNode[0]]
                    for k in range(j - window, j + window + 1):
                        if k >= 0 and k < walk_L and k != j:
                            neighNode = path[k]
                            neighType = self.node_name2type[neighNode[0]]
                            total_triple_n[centerType][neighType] += 1
        het_walk_f.close()
        total_triple_n = self.args.batch_s / (total_triple_n * 10)
        # 10 is equal to window x 2
        print("Info: compute sampling ratio for each kind of triple done.")

        return total_triple_n

    def gen_embeds_w_neigh(self):
        self.triple_sample_p = self.compute_sample_p()
        node_types = list(self.dl.nodes['attr'].keys())
        self.feature_list = dict()
        for node_type in node_types:
            if self.args.feat_type == 1 and isinstance(self.dl.nodes['attr'][node_type], np.ndarray):
                self.feature_list[node_type] = th.FloatTensor(self.dl.nodes['attr'][node_type])
            elif self.args.feat_type == 0 or self.args.feat_type == 1:
                dim = self.node_n[node_type]
                indices = np.vstack((np.arange(dim), np.arange(dim)))
                indices = th.LongTensor(indices)
                values = th.FloatTensor(np.ones(dim))
                self.feature_list[node_type] = th.sparse.FloatTensor(indices, values, th.Size([dim, dim]))
            elif self.args.feat_type == 2:
                # dense
                dim = self.node_n[node_type]
                self.feature_list[node_type] = th.FloatTensor(np.eye(dim))
            else:
                raise Exception('Error: Invalid feat_type')
        # store neighbor set from random walk sequence
        neigh_list_train = dict()
        for node_type in node_types:
            neigh_list_train[node_type] = [[[] for i in range(self.node_n[node_type])] for j in node_types]
        temp_dir = os.path.join(sys.path[0], f'{self.args.data}-temp')
        het_neigh_train_f = open(os.path.join(temp_dir, "het_neigh_train.txt"), "r")
        for line in het_neigh_train_f:
            line = line.strip()
            node_id, neigh = re.split(':', line)
            node_type = self.node_name2type[node_id[0]]
            neigh_list = re.split(',', neigh)
            for neigh in neigh_list:
                if len(node_id) > 1:
                    neigh_type = self.node_name2type[neigh[0]]
                    neigh_list_train[node_type][neigh_type][int(node_id[1:])].append(int(neigh[1:]))
        het_neigh_train_f.close()
        # print a_neigh_list_train[0][1]

        # store top neighbor set (based on frequency) from random walk sequence
        neigh_list_train_top = dict()
        for node_type in node_types:
            neigh_list_train_top[node_type] = [[[] for i in range(self.node_n[node_type])] for j in node_types]
        for node_type in node_types:
            for n_id in range(self.node_n[node_type]):
                for neigh_type in node_types:
                    neigh_list_train_temp = Counter(neigh_list_train[node_type][neigh_type][n_id])
                    top_list = neigh_list_train_temp.most_common(self.top_k[neigh_type])
                    neigh_size = self.top_k[neigh_type]
                    for k in range(len(top_list)):
                        neigh_list_train_top[node_type][neigh_type][n_id].append(int(top_list[k][0]))
                    if len(neigh_list_train_top[node_type][neigh_type][n_id]) and \
                            len(neigh_list_train_top[node_type][neigh_type][n_id]) < neigh_size:
                        for l in range(len(neigh_list_train_top[node_type][neigh_type][n_id]), neigh_size):
                            neigh_list_train_top[node_type][neigh_type][n_id]. \
                                append(random.choice(neigh_list_train_top[node_type][neigh_type][n_id]))
        for h_type in neigh_list_train_top.keys():
            for t_type in range(len(neigh_list_train_top[h_type])):
                for n_list in neigh_list_train_top[h_type][t_type]:
                    num_lack = self.top_k[t_type]-len(n_list)
                    if num_lack > 0:
                        random_list = [0]*num_lack
                        n_list.extend(random_list)
        self.neigh_list_train = neigh_list_train_top
        '''至少有一个邻居的点成为可用的训练点'''
        # store ids of author/paper/venue used in training
        self.train_id_list = dict()
        train_id_list = [[] for i in node_types]
        for node_type in node_types:
            for n_id in range(self.node_n[node_type]):
                if len(neigh_list_train_top[node_type][node_type][n_id]):
                    train_id_list[node_type].append(n_id)
            self.train_id_list[node_type] = np.array(train_id_list[node_type])


    # print (len(self.v_train_id_list))

    def sample_het_walk_triple(self):
        print("Info: sampling triple relations start.")
        triple_list = defaultdict(list)
        window = self.args.window
        walk_L = self.args.walk_L
        temp_dir = os.path.join(sys.path[0], f'{self.args.data}-temp')
        het_walk_f = open(os.path.join(temp_dir, "het_random_walk.txt"), "r")
        centerNode = ''
        neighNode = ''
        for line in het_walk_f:
            line = line.strip()
            path = re.split(' ', line)
            for j in range(walk_L):
                centerNode = path[j]
                centerType = self.node_name2type[centerNode[0]]
                if len(centerNode) > 1:
                    for k in range(j - window, j + window + 1):
                        if k >= 0 and k < walk_L and k != j:
                            neighNode = path[k]
                            neighType = self.node_name2type[neighNode[0]]
                            if random.random() < self.triple_sample_p[centerType][neighType]:
                                negNode = random.randint(0, self.node_n[neighType] - 1)
                                while len(self.neigh_list[neighType][negNode]) == 0:
                                    negNode = random.randint(0, self.node_n[neighType] - 1)
                                # random negative sampling get similar performance as noise distribution sampling
                                triple = [int(centerNode[1:]), int(neighNode[1:]), int(negNode)]
                                triple_list[(centerType, neighType)].append(triple)
        het_walk_f.close()
        print("Info: sampling triple relations done.")
        return triple_list
