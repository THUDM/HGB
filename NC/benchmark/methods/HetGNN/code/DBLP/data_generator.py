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

class input_data(object):
    def __init__(self, args, dl):
        self.args = args
        self.dl = dl
        node_n, node_shift = self.dl.nodes['count'], self.dl.nodes['shift']
        self.node_n, self.node_shift = node_n, node_shift
        a_p_list = [[] for k in range(node_n[0])]
        p_a_list = [[] for k in range(node_n[1])]
        p_t_list = [[] for k in range(node_n[1])]
        p_v_list = [[] for k in range(node_n[1])]
        t_p_list = [[] for k in range(node_n[2])]
        v_p_list = [[] for k in range(node_n[3])]
        p_neigh_list = [[] for k in range(node_n[1])]
        # a-p
        for row, row_array in enumerate(
                np.split(self.dl.links['data'][0].indices, self.dl.links['data'][0].indptr)[1:-1]):
            for col in row_array:
                row_, col_ = row - node_shift[0], col - node_shift[1]
                a_p_list[row_].append('p' + str(col_))
        # p-t
        for row, row_array in enumerate(
                np.split(self.dl.links['data'][1].indices, self.dl.links['data'][1].indptr)[1:-1]):
            for col in row_array:
                row_, col_ = row - node_shift[1], col - node_shift[2]
                p_t_list[row_].append('t' + str(col_))
        # p-v
        for row, row_array in enumerate(
                np.split(self.dl.links['data'][2].indices, self.dl.links['data'][2].indptr)[1:-1]):
            for col in row_array:
                row_, col_ = row - node_shift[1], col - node_shift[3]
                p_v_list[row_].append('v' + str(col_))
        # p-a
        for row, row_array in enumerate(
                np.split(self.dl.links['data'][3].indices, self.dl.links['data'][3].indptr)[1:-1]):
            for col in row_array:
                row_, col_ = row - node_shift[1], col - node_shift[0]
                p_a_list[row_].append('a' + str(col_))
        # t-p
        for row, row_array in enumerate(
                np.split(self.dl.links['data'][4].indices, self.dl.links['data'][4].indptr)[1:-1]):
            for col in row_array:
                row_, col_ = row - node_shift[2], col - node_shift[1]
                t_p_list[row_].append('p' + str(col_))
        # v-p
        for row, row_array in enumerate(
                np.split(self.dl.links['data'][5].indices, self.dl.links['data'][5].indptr)[1:-1]):
            for col in row_array:
                row_, col_ = row - node_shift[3], col - node_shift[1]
                v_p_list[row_].append('p' + str(col_))
        # p-neigh
        for p_id in range(node_n[1]):
            p_neigh_list[p_id] = p_a_list[p_id] + p_t_list[p_id] + p_v_list[p_id]

        self.a_p_list = a_p_list
        self.p_a_list = p_a_list
        self.p_t_list = p_t_list
        self.p_v_list = p_v_list
        self.t_p_list = t_p_list
        self.v_p_list = v_p_list
        self.p_neigh_list = p_neigh_list

    '''genarate het_neigh_train.txt with walk_restart'''

    def gen_het_w_walk_restart(self,file_):
        print(f'Info: generate {file_} start.')
        node_n = self.dl.nodes['count']
        a_neigh_list_train = [[] for k in range(node_n[0])]
        p_neigh_list_train = [[] for k in range(node_n[1])]
        t_neigh_list_train = [[] for k in range(node_n[2])]
        v_neigh_list_train = [[] for k in range(node_n[3])]

        # generate neighbor set via random walk with restart
        for i in range(4):
            for j in range(node_n[i]):
                if i == 0:
                    neigh_temp = self.a_p_list[j]
                    neigh_train = a_neigh_list_train[j]
                    curNode = "a" + str(j)
                elif i == 1:
                    neigh_temp = self.p_neigh_list[j]
                    neigh_train = p_neigh_list_train[j]
                    curNode = "p" + str(j)
                elif i == 2:
                    neigh_temp = self.t_p_list[j]
                    neigh_train = t_neigh_list_train[j]
                    curNode = "t" + str(j)
                else:
                    neigh_temp = self.v_p_list[j]
                    neigh_train = v_neigh_list_train[j]
                    curNode = "v" + str(j)
                if len(neigh_temp):
                    neigh_L = 0
                    a_L = 0
                    p_L = 0
                    t_L = 0
                    v_L = 0
                    while neigh_L < 100 or a_L<2 or p_L<2 or t_L<2 or v_L<2:  # maximum neighbor size = 100
                        rand_p = random.random()  # return p
                        if rand_p > 0.5:
                            if curNode[0] == "a":
                                curNode = random.choice(self.a_p_list[int(curNode[1:])])
                                if p_L < 41:  # size constraint (make sure each type of neighobr is sampled)
                                    neigh_train.append(curNode)
                                    neigh_L += 1
                                    p_L += 1
                            elif curNode[0] == "p":
                                curNode = random.choice(self.p_neigh_list[int(curNode[1:])])
                                if curNode != ('a' + str(j)) and curNode[0] == 'a' and a_L < 41:
                                    neigh_train.append(curNode)
                                    neigh_L += 1
                                    a_L += 1
                                elif curNode[0] == 't' and t_L < 41:
                                    neigh_train.append(curNode)
                                    neigh_L += 1
                                    t_L += 1
                                elif curNode[0] == 'v' and v_L < 11:
                                    neigh_train.append(curNode)
                                    neigh_L += 1
                                    v_L += 1
                            elif curNode[0] == "t":
                                curNode = random.choice(self.t_p_list[int(curNode[1:])])
                                if p_L < 41:
                                    neigh_train.append(curNode)
                                    neigh_L += 1
                                    p_L += 1
                            elif curNode[0] == "v":
                                curNode = random.choice(self.v_p_list[int(curNode[1:])])
                                if p_L < 41:
                                    neigh_train.append(curNode)
                                    neigh_L += 1
                                    p_L += 1
                        else:
                            if i == 0:
                                curNode = ('a' + str(j))
                            elif i == 1:
                                curNode = ('p' + str(j))
                            elif i == 2:
                                curNode = ('t' + str(j))
                            else:
                                curNode = ('v' + str(j))


        for i in range(4):
            for j in range(node_n[i]):
                if i == 0:
                    a_neigh_list_train[j] = list(a_neigh_list_train[j])
                elif i == 1:
                    p_neigh_list_train[j] = list(p_neigh_list_train[j])
                elif i == 2:
                    t_neigh_list_train[j] = list(t_neigh_list_train[j])
                else:
                    v_neigh_list_train[j] = list(v_neigh_list_train[j])
        t = t_neigh_list_train[6868]
        neigh_f = open(file_ , "w")
        for i in range(4):
            for j in range(node_n[i]):
                if i == 0:
                    neigh_train = a_neigh_list_train[j]
                    curNode = "a" + str(j)
                elif i == 1:
                    neigh_train = p_neigh_list_train[j]
                    curNode = "p" + str(j)
                elif i == 2:
                    neigh_train = t_neigh_list_train[j]
                    curNode = "t" + str(j)
                else:
                    neigh_train = v_neigh_list_train[j]
                    curNode = "v" + str(j)
                if len(neigh_train):
                    neigh_f.write(curNode + ":")
                    for k in range(len(neigh_train) - 1):
                        neigh_f.write(neigh_train[k] + ",")
                    neigh_f.write(neigh_train[-1] + "\n")
        neigh_f.close()
        print(f'Info: generate {file_} done.')

    def gen_het_w_walk(self, file_):
        print(f'Info: generate {file_} start.')
        het_walk_f = open(file_, "w")
        for i in range(self.args.walk_n):
            for j in range(self.node_n[0]):
                if len(self.a_p_list[j]):
                    curNode = "a" + str(j)
                    het_walk_f.write(curNode + " ")
                    for l in range(self.args.walk_L - 1):
                        if curNode[0] == "a":
                            curNode = int(curNode[1:])
                            curNode = random.choice(self.a_p_list[curNode])
                            het_walk_f.write(curNode + " ")
                        elif curNode[0] == "p":
                            curNode = int(curNode[1:])
                            curNode = random.choice(self.p_neigh_list[curNode])
                            het_walk_f.write(curNode + " ")
                        elif curNode[0] == "t":
                            curNode = int(curNode[1:])
                            curNode = random.choice(self.t_p_list[curNode])
                            het_walk_f.write(curNode + " ")
                        elif curNode[0] == "v":
                            curNode = int(curNode[1:])
                            curNode = random.choice(self.v_p_list[curNode])
                            het_walk_f.write(curNode + " ")
                    het_walk_f.write("\n")
        het_walk_f.close()
        print(f'Info: generate {file_} done.')

    def compute_sample_p(self):
        print("Info :compute sampling ratio for each kind of triple start.")
        window = self.args.window
        walk_L = self.args.walk_L
        node_n, node_shift = self.dl.nodes['count'], self.dl.nodes['shift']
        total_triple_n = [0.0] * 16  # sixteen kinds of triples
        het_walk_f = open(os.path.join(sys.path[0], 'temp', "het_random_walk.txt"), "r")
        centerNode = ''
        neighNode = ''
        for line in het_walk_f:
            line = line.strip()
            path = re.split(' ', line)
            for j in range(walk_L):
                centerNode = path[j]
                if len(centerNode) > 1:
                    if centerNode[0] == 'a':
                        for k in range(j - window, j + window + 1):
                            if k >= 0 and k < walk_L and k != j:
                                neighNode = path[k]
                                if neighNode[0] == 'a':
                                    total_triple_n[0] += 1
                                elif neighNode[0] == 'p':
                                    total_triple_n[1] += 1
                                elif neighNode[0] == 't':
                                    total_triple_n[2] += 1
                                elif neighNode[0] == 'v':
                                    total_triple_n[3] += 1
                    elif centerNode[0] == 'p':
                        for k in range(j - window, j + window + 1):
                            if k >= 0 and k < walk_L and k != j:
                                neighNode = path[k]
                                if neighNode[0] == 'a':
                                    total_triple_n[4] += 1
                                elif neighNode[0] == 'p':
                                    total_triple_n[5] += 1
                                elif neighNode[0] == 't':
                                    total_triple_n[6] += 1
                                elif neighNode[0] == 'v':
                                    total_triple_n[7] += 1
                    elif centerNode[0] == 't':
                        for k in range(j - window, j + window + 1):
                            if k >= 0 and k < walk_L and k != j:
                                neighNode = path[k]
                                if neighNode[0] == 'a':
                                    total_triple_n[8] += 1
                                elif neighNode[0] == 'p':
                                    total_triple_n[9] += 1
                                elif neighNode[0] == 't':
                                    total_triple_n[10] += 1
                                elif neighNode[0] == 'v':
                                    total_triple_n[11] += 1
                    elif centerNode[0] == 'v':
                        for k in range(j - window, j + window + 1):
                            if k >= 0 and k < walk_L and k != j:
                                neighNode = path[k]
                                if neighNode[0] == 'a':
                                    total_triple_n[12] += 1
                                elif neighNode[0] == 'p':
                                    total_triple_n[13] += 1
                                elif neighNode[0] == 't':
                                    total_triple_n[14] += 1
                                elif neighNode[0] == 'v':
                                    total_triple_n[15] += 1
        het_walk_f.close()
        for i in range(len(total_triple_n)):
            total_triple_n[i] = self.args.batch_s / (total_triple_n[i] * 10)
        # 10 = window x 2
        print("Info: compute sampling ratio for each kind of triple done.")

        return total_triple_n

    def gen_embeds_w_neigh(self):

        self.triple_sample_p = self.compute_sample_p()
        self.dl.nodes['attr'][3] = np.eye(self.node_n[3])

        # todo store pre-trained network/content embedding
        # todo store early aggregation embeds

        self.feature_list = []
        for node_type in range(4):
            self.feature_list.append(self.dl.nodes['attr'][node_type])

        # store neighbor set from random walk sequence
        a_neigh_list_train = [[[] for i in range(self.node_n[0])] for j in range(4)]
        p_neigh_list_train = [[[] for i in range(self.node_n[1])] for j in range(4)]
        t_neigh_list_train = [[[] for i in range(self.node_n[2])] for j in range(4)]
        v_neigh_list_train = [[[] for i in range(self.node_n[3])] for j in range(4)]
        het_neigh_train_f = open(os.path.join(sys.path[0], 'temp', "het_neigh_train.txt"), "r")
        for line in het_neigh_train_f:
            line = line.strip()
            node_id, neigh = re.split(':', line)
            neigh_list = re.split(',', neigh)
            if node_id[0] == 'a' and len(node_id) > 1:
                for j in range(len(neigh_list)):
                    if neigh_list[j][0] == 'a':
                        a_neigh_list_train[0][int(node_id[1:])].append(int(neigh_list[j][1:]))
                    elif neigh_list[j][0] == 'p':
                        a_neigh_list_train[1][int(node_id[1:])].append(int(neigh_list[j][1:]))
                    elif neigh_list[j][0] == 't':
                        a_neigh_list_train[2][int(node_id[1:])].append(int(neigh_list[j][1:]))
                    elif neigh_list[j][0] == 'v':
                        a_neigh_list_train[3][int(node_id[1:])].append(int(neigh_list[j][1:]))
            elif node_id[0] == 'p' and len(node_id) > 1:
                for j in range(len(neigh_list)):
                    if neigh_list[j][0] == 'a':
                        p_neigh_list_train[0][int(node_id[1:])].append(int(neigh_list[j][1:]))
                    if neigh_list[j][0] == 'p':
                        p_neigh_list_train[1][int(node_id[1:])].append(int(neigh_list[j][1:]))
                    if neigh_list[j][0] == 't':
                        p_neigh_list_train[2][int(node_id[1:])].append(int(neigh_list[j][1:]))
                    if neigh_list[j][0] == 'v':
                        p_neigh_list_train[3][int(node_id[1:])].append(int(neigh_list[j][1:]))
            elif node_id[0] == 't' and len(node_id) > 1:
                for j in range(len(neigh_list)):
                    if neigh_list[j][0] == 'a':
                        t_neigh_list_train[0][int(node_id[1:])].append(int(neigh_list[j][1:]))
                    if neigh_list[j][0] == 'p':
                        t_neigh_list_train[1][int(node_id[1:])].append(int(neigh_list[j][1:]))
                    if neigh_list[j][0] == 't':
                        t_neigh_list_train[2][int(node_id[1:])].append(int(neigh_list[j][1:]))
                    if neigh_list[j][0] == 'v':
                        t_neigh_list_train[3][int(node_id[1:])].append(int(neigh_list[j][1:]))
            elif node_id[0] == 'v' and len(node_id) > 1:
                for j in range(len(neigh_list)):
                    if neigh_list[j][0] == 'a':
                        v_neigh_list_train[0][int(node_id[1:])].append(int(neigh_list[j][1:]))
                    if neigh_list[j][0] == 'p':
                        v_neigh_list_train[1][int(node_id[1:])].append(int(neigh_list[j][1:]))
                    if neigh_list[j][0] == 't':
                        v_neigh_list_train[2][int(node_id[1:])].append(int(neigh_list[j][1:]))
                    if neigh_list[j][0] == 'v':
                        v_neigh_list_train[3][int(node_id[1:])].append(int(neigh_list[j][1:]))
        het_neigh_train_f.close()
        # print a_neigh_list_train[0][1]

        # store top neighbor set (based on frequency) from random walk sequence
        a_neigh_list_train_top = [[[] for i in range(self.node_n[0])] for j in range(4)]
        p_neigh_list_train_top = [[[] for i in range(self.node_n[1])] for j in range(4)]
        t_neigh_list_train_top = [[[] for i in range(self.node_n[2])] for j in range(4)]
        v_neigh_list_train_top = [[[] for i in range(self.node_n[3])] for j in range(4)]
        top_k = [10, 10, 10, 3]  # fix each neighor type size
        for i in range(self.node_n[0]):
            for j in range(4):
                a_neigh_list_train_temp = Counter(a_neigh_list_train[j][i])
                top_list = a_neigh_list_train_temp.most_common(top_k[j])
                neigh_size = 0
                if j == 0 or j == 1 or j == 2:
                    neigh_size = 10
                else:
                    neigh_size = 3
                for k in range(len(top_list)):
                    a_neigh_list_train_top[j][i].append(int(top_list[k][0]))
                if len(a_neigh_list_train_top[j][i]) and len(a_neigh_list_train_top[j][i]) < neigh_size:
                    for l in range(len(a_neigh_list_train_top[j][i]), neigh_size):
                        a_neigh_list_train_top[j][i].append(random.choice(a_neigh_list_train_top[j][i]))

        for i in range(self.node_n[1]):
            for j in range(4):
                p_neigh_list_train_temp = Counter(p_neigh_list_train[j][i])
                top_list = p_neigh_list_train_temp.most_common(top_k[j])
                neigh_size = 0
                if j == 0 or j == 1 or j == 2:
                    neigh_size = 10
                else:
                    neigh_size = 3
                for k in range(len(top_list)):
                    p_neigh_list_train_top[j][i].append(int(top_list[k][0]))
                if len(p_neigh_list_train_top[j][i]) and len(p_neigh_list_train_top[j][i]) < neigh_size:
                    for l in range(len(p_neigh_list_train_top[j][i]), neigh_size):
                        p_neigh_list_train_top[j][i].append(random.choice(p_neigh_list_train_top[j][i]))
        for i in range(self.node_n[2]):
            for j in range(4):
                t_neigh_list_train_temp = Counter(t_neigh_list_train[j][i])
                top_list = t_neigh_list_train_temp.most_common(top_k[j])
                neigh_size = 0
                if j == 0 or j == 1 or j == 2:
                    neigh_size = 10
                else:
                    neigh_size = 3
                for k in range(len(top_list)):
                    t_neigh_list_train_top[j][i].append(int(top_list[k][0]))
                if len(t_neigh_list_train_top[j][i]) and len(t_neigh_list_train_top[j][i]) < neigh_size:
                    for l in range(len(t_neigh_list_train_top[j][i]), neigh_size):
                        t_neigh_list_train_top[j][i].append(random.choice(t_neigh_list_train_top[j][i]))

        for i in range(self.node_n[3]):
            for j in range(4):
                v_neigh_list_train_temp = Counter(v_neigh_list_train[j][i])
                top_list = v_neigh_list_train_temp.most_common(top_k[j])
                neigh_size = 0
                if j == 0 or j == 1 or j == 2:
                    neigh_size = 10
                else:
                    neigh_size = 3
                for k in range(len(top_list)):
                    v_neigh_list_train_top[j][i].append(int(top_list[k][0]))
                if len(v_neigh_list_train_top[j][i]) and len(v_neigh_list_train_top[j][i]) < neigh_size:
                    for l in range(len(v_neigh_list_train_top[j][i]), neigh_size):
                        v_neigh_list_train_top[j][i].append(random.choice(v_neigh_list_train_top[j][i]))

        a_neigh_list_train[:] = []
        p_neigh_list_train[:] = []
        t_neigh_list_train[:] = []
        v_neigh_list_train[:] = []

        self.a_neigh_list_train = a_neigh_list_train_top
        self.p_neigh_list_train = p_neigh_list_train_top
        self.t_neigh_list_train = t_neigh_list_train_top
        self.v_neigh_list_train = v_neigh_list_train_top
        '''至少有一个邻居的点成为可用的训练点'''
        # store ids of author/paper/venue used in training
        train_id_list = [[] for i in range(4)]
        for i in range(4):
            if i == 0:
                for l in range(self.node_n[0]):
                    if len(a_neigh_list_train_top[i][l]):
                        train_id_list[i].append(l)
                self.a_train_id_list = np.array(train_id_list[i])
            elif i == 1:
                for l in range(self.node_n[1]):
                    if len(p_neigh_list_train_top[i][l]):
                        train_id_list[i].append(l)
                self.p_train_id_list = np.array(train_id_list[i])
            elif i == 2:
                for l in range(self.node_n[2]):
                    if len(t_neigh_list_train_top[i][l]):
                        train_id_list[i].append(l)
                self.t_train_id_list = np.array(train_id_list[i])
            else:
                for l in range(self.node_n[3]):
                    if len(v_neigh_list_train_top[i][l]):
                        train_id_list[i].append(l)
                self.v_train_id_list = np.array(train_id_list[i])

    # print (len(self.v_train_id_list))

    def sample_het_walk_triple(self):
        print("Info: sampling triple relations start.")
        triple_list = [[] for k in range(16)]
        window = self.args.window
        walk_L = self.args.walk_L
        A_n, P_n, T_n, V_n = self.node_n[0], self.node_n[1], self.node_n[2], self.node_n[3]
        triple_sample_p = self.triple_sample_p  # use sampling to avoid memory explosion

        het_walk_f = open(os.path.join(sys.path[0], 'temp', "het_random_walk.txt"), "r")
        centerNode = ''
        neighNode = ''
        for line in het_walk_f:
            line = line.strip()
            path = re.split(' ', line)
            for j in range(walk_L):
                centerNode = path[j]
                if len(centerNode) > 1:
                    if centerNode[0] == 'a':
                        for k in range(j - window, j + window + 1):
                            if k>=0 and k < walk_L and k != j:
                                neighNode = path[k]
                                if neighNode[0] == 'a' and random.random() < triple_sample_p[0]:
                                    negNode = random.randint(0, A_n - 1)
                                    while len(self.a_p_list[negNode]) == 0:
                                        negNode = random.randint(0, A_n - 1)
                                    # random negative sampling get similar performance as noise distribution sampling
                                    triple = [int(centerNode[1:]), int(neighNode[1:]), int(negNode)]
                                    triple_list[0].append(triple)
                                elif neighNode[0] == 'p' and random.random() < triple_sample_p[1]:
                                    negNode = random.randint(0, P_n - 1)
                                    while len(self.p_a_list[negNode]) == 0:
                                        negNode = random.randint(0, P_n - 1)
                                    triple = [int(centerNode[1:]), int(neighNode[1:]), int(negNode)]
                                    triple_list[1].append(triple)
                                elif neighNode[0] == 't' and random.random() < triple_sample_p[2]:
                                    negNode = random.randint(0, T_n - 1)
                                    while len(self.t_p_list[negNode]) == 0:
                                        negNode = random.randint(0, T_n - 1)
                                    triple = [int(centerNode[1:]), int(neighNode[1:]), int(negNode)]
                                    triple_list[2].append(triple)
                                elif neighNode[0] == 'v' and random.random() < triple_sample_p[3]:
                                    negNode = random.randint(0, V_n - 1)
                                    while len(self.v_p_list[negNode]) == 0:
                                        negNode = random.randint(0, V_n - 1)
                                    triple = [int(centerNode[1:]), int(neighNode[1:]), int(negNode)]
                                    triple_list[3].append(triple)
                    elif centerNode[0] == 'p':
                        for k in range(j - window, j + window + 1):
                            if k>=0 and k < walk_L and k != j:
                                neighNode = path[k]
                                if neighNode[0] == 'a' and random.random() < triple_sample_p[4]:
                                    negNode = random.randint(0, A_n - 1)
                                    while len(self.a_p_list[negNode]) == 0:
                                        negNode = random.randint(0, A_n - 1)
                                    triple = [int(centerNode[1:]), int(neighNode[1:]), int(negNode)]
                                    triple_list[4].append(triple)
                                elif neighNode[0] == 'p' and random.random() < triple_sample_p[5]:
                                    negNode = random.randint(0, P_n - 1)
                                    while len(self.p_a_list[negNode]) == 0:
                                        negNode = random.randint(0, P_n - 1)
                                    triple = [int(centerNode[1:]), int(neighNode[1:]), int(negNode)]
                                    triple_list[5].append(triple)
                                elif neighNode[0] == 't' and random.random() < triple_sample_p[6]:
                                    negNode = random.randint(0, T_n - 1)
                                    while len(self.t_p_list[negNode]) == 0:
                                        negNode = random.randint(0, T_n - 1)
                                    triple = [int(centerNode[1:]), int(neighNode[1:]), int(negNode)]
                                    triple_list[6].append(triple)
                                elif neighNode[0] == 'v' and random.random() < triple_sample_p[7]:
                                    negNode = random.randint(0, V_n - 1)
                                    while len(self.v_p_list[negNode]) == 0:
                                        negNode = random.randint(0, V_n - 1)
                                    triple = [int(centerNode[1:]), int(neighNode[1:]), int(negNode)]
                                    triple_list[7].append(triple)
                    elif centerNode[0] == 't':
                        for k in range(j - window, j + window + 1):
                            if k>=0 and k < walk_L and k != j:
                                neighNode = path[k]
                                if neighNode[0] == 'a' and random.random() < triple_sample_p[8]:
                                    negNode = random.randint(0, A_n - 1)
                                    while len(self.a_p_list[negNode]) == 0:
                                        negNode = random.randint(0, A_n - 1)
                                    triple = [int(centerNode[1:]), int(neighNode[1:]), int(negNode)]
                                    triple_list[8].append(triple)
                                elif neighNode[0] == 'p' and random.random() < triple_sample_p[9]:
                                    negNode = random.randint(0, P_n - 1)
                                    while len(self.p_a_list[negNode]) == 0:
                                        negNode = random.randint(0, P_n - 1)
                                    triple = [int(centerNode[1:]), int(neighNode[1:]), int(negNode)]
                                    triple_list[9].append(triple)
                                elif neighNode[0] == 't' and random.random() < triple_sample_p[10]:
                                    negNode = random.randint(0, T_n - 1)
                                    while len(self.t_p_list[negNode]) == 0:
                                        negNode = random.randint(0, T_n - 1)
                                    triple = [int(centerNode[1:]), int(neighNode[1:]), int(negNode)]
                                    triple_list[10].append(triple)
                                elif neighNode[0] == 'v' and random.random() < triple_sample_p[11]:
                                    negNode = random.randint(0, V_n - 1)
                                    while len(self.v_p_list[negNode]) == 0:
                                        negNode = random.randint(0, V_n - 1)
                                    triple = [int(centerNode[1:]), int(neighNode[1:]), int(negNode)]
                                    triple_list[11].append(triple)
                    elif centerNode[0] == 'v':
                        for k in range(j - window, j + window + 1):
                            if k>=0 and k < walk_L and k != j:
                                neighNode = path[k]
                                if neighNode[0] == 'a' and random.random() < triple_sample_p[12]:
                                    negNode = random.randint(0, A_n - 1)
                                    while len(self.a_p_list[negNode]) == 0:
                                        negNode = random.randint(0, A_n - 1)
                                    triple = [int(centerNode[1:]), int(neighNode[1:]), int(negNode)]
                                    triple_list[12].append(triple)
                                elif neighNode[0] == 'p' and random.random() < triple_sample_p[13]:
                                    negNode = random.randint(0, P_n - 1)
                                    while len(self.p_a_list[negNode]) == 0:
                                        negNode = random.randint(0, P_n - 1)
                                    triple = [int(centerNode[1:]), int(neighNode[1:]), int(negNode)]
                                    triple_list[13].append(triple)
                                elif neighNode[0] == 't' and random.random() < triple_sample_p[14]:
                                    negNode = random.randint(0, T_n - 1)
                                    while len(self.t_p_list[negNode]) == 0:
                                        negNode = random.randint(0, T_n - 1)
                                    triple = [int(centerNode[1:]), int(neighNode[1:]), int(negNode)]
                                    triple_list[14].append(triple)
                                elif neighNode[0] == 'v' and random.random() < triple_sample_p[15]:
                                    negNode = random.randint(0, V_n - 1)
                                    while len(self.v_p_list[negNode]) == 0:
                                        negNode = random.randint(0, V_n - 1)
                                    triple = [int(centerNode[1:]), int(neighNode[1:]), int(negNode)]
                                    triple_list[15].append(triple)
        het_walk_f.close()
        print("Info: sampling triple relations done.")
        return triple_list
