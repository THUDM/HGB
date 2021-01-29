import string
import re
import numpy as np
import os
import random
from itertools import *
import argparse
import pickle
import sklearn
from sklearn import linear_model
import sklearn.metrics as Metric
import json
import sys
from collections import defaultdict

sys.path.append('../../')

from scripts.data_loader import data_loader



parser = argparse.ArgumentParser(description='application data process')
parser.add_argument('--embed_d', type=int, default=128,
                    help='embedding dimension')
parser.add_argument('--data', type=str, default='amazon',
                    help='select dataset')
args = parser.parse_args()
print(args)

data_name = args.data
temp_dir = os.path.join(sys.path[0], f'{data_name}-temp')
data_path = f'../../data/{data_name}'
dl_pickle_f = os.path.join(data_path, 'dl_pickle')
if os.path.exists(dl_pickle_f):
    dl = pickle.load(open(dl_pickle_f, 'rb'))
    print(f'Info: load {data_name} from {dl_pickle_f}')
else:
    dl = data_loader(data_path)
    pickle.dump(dl, open(dl_pickle_f, 'wb'))
    print(f'Info: load {data_name} from original data and generate {dl_pickle_f}')

node_n = dl.nodes['total']
node_shift = dl.nodes['shift']
data_info = json.load(open(f"../../data/{data_name}/info.dat", 'r'))
f_node_type2name = data_info['node.dat']
node_types = [int(k) for k in f_node_type2name.keys()]
node_type2name, node_name2type = dict(), dict()
for k in node_types:
    name = f_node_type2name[str(k)][0]
    node_type2name[k] = name
    node_name2type[name] = k
iter=300
while True:
    # read node embed
    decimal_keep = 4
    p_embed = np.around(np.random.normal(0, 0.01, [node_n, args.embed_d]), decimal_keep)
    embed_f = os.path.join(temp_dir,f"node_embedding{iter}.txt")
    if not os.path.exists(embed_f):
        break
    embed_f = open(embed_f, "r")
    for line in islice(embed_f, 0, None):
        line = line.strip()
        node_id = re.split(' ', line)[0]
        if len(node_id):
            node_type = node_name2type[node_id[0]]
            index = int(node_id[1:]) + node_shift[node_type]
            embed = np.asarray(re.split(' ', line)[1:], dtype='float32')
            p_embed[index] = embed
    embed_f.close()

    train_feature, test_feature_2hop,test_feature_random = dict(), dict(), dict()
    train_label = dict()
    for r_id in dl.links_test['data'].keys():
        train_feature[r_id] = p_embed[dl.train_pos[r_id][0]] * p_embed[dl.train_pos[r_id][1]]
        train_label[r_id] = [1] * len(dl.train_pos[r_id][0])
        train_feature[r_id] = np.vstack((train_feature[r_id], (p_embed[dl.train_neg[r_id][0]] * p_embed[dl.train_neg[r_id][1]])))
        train_label[r_id].extend([0] * len(dl.train_neg[r_id][0]))
    test_data_2hop,test_label_2hop = dl.get_test_neigh()
    test_data_random,test_label_random = dl.get_test_neigh_w_random()
    for r_id in test_data_2hop.keys():
        test_feature_2hop[r_id] = p_embed[test_data_2hop[r_id][0]] * p_embed[test_data_2hop[r_id][1]]
    for r_id in test_data_random.keys():
        test_feature_random[r_id] = p_embed[test_data_random[r_id][0]] * p_embed[test_data_random[r_id][1]]


    print(f'Info: iter: {iter}')
    scores=defaultdict(list)
    for r_id in test_feature_2hop.keys():
        learner = linear_model.LogisticRegression(max_iter=200)
        learner.fit(train_feature[r_id], train_label[r_id])
        test_predict = learner.predict_proba(test_feature_2hop[r_id]).T[1]
        score=dl.evaluate(test_data_2hop[r_id], test_predict, test_label_2hop[r_id])
        for s in score.keys():
            scores[s].append(score[s])
    for s in scores.keys():
        scores[s] = np.round(np.mean(scores[s]),decimal_keep)
    print(f'2hot score:{scores}')
    scores=defaultdict(list)
    for r_id in test_feature_random.keys():
        learner = linear_model.LogisticRegression(max_iter=200)
        learner.fit(train_feature[r_id], train_label[r_id])
        test_predict = learner.predict_proba(test_feature_random[r_id]).T[1]
        score=dl.evaluate(test_data_random[r_id], test_predict, test_label_random[r_id])
        for s in score.keys():
            scores[s].append(score[s])
    for s in scores.keys():
        scores[s] = np.round(np.mean(scores[s]),decimal_keep)
    print(f'random score:{scores}')
    iter += 10
