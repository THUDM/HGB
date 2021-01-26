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
import sys
from collections import defaultdict
sys.path.append('../../')

from scripts.data_loader import data_loader

# todo  set data_name for new dataset
data_name = 'Freebase'

parser = argparse.ArgumentParser(description='application data process')
parser.add_argument('--embed_d', type=int, default=128,
                    help='embedding dimension')
temp_dir = os.path.join(sys.path[0], 'temp')
if not os.path.exists(temp_dir):
    os.makedirs(temp_dir)
dl_pickle_f = os.path.join(temp_dir, 'dl_pickle')
if os.path.exists(dl_pickle_f):
    dl = pickle.load(open(dl_pickle_f, 'rb'))
    print(f'Info: load {data_name} from {dl_pickle_f}')
else:
    dl = data_loader(f'../../data/{data_name}')
    pickle.dump(dl, open(dl_pickle_f, 'wb'))
    print(f'Info: load {data_name} from original data and generate {dl_pickle_f}')
args = parser.parse_args()
print(args)
node_n = dl.nodes['count']
temp_dir = os.path.join(sys.path[0], 'temp')
embed_f = os.path.join(temp_dir, 'node_embedding45.txt')

p_embed = np.around(np.random.normal(0, 0.01, [node_n[0], args.embed_d]), 4)
embed_f = open(embed_f, "r")
for line in islice(embed_f, 0, None):
    line = line.strip()
    node_id = re.split(' ', line)[0]
    if len(node_id) and (node_id[0] in ('p')):
        type_label = node_id[0]
        index = int(node_id[1:])
        embed = np.asarray(re.split(' ', line)[1:], dtype='float32')
        if type_label == 'p':
            p_embed[index] = embed
embed_f.close()

train_feature, test_feature_2hop,test_feature_random = dict(), dict(), dict()
train_label = dict()
for r_id in dl.train_pos.keys():
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



scores=defaultdict(list)
for r_id in test_feature_2hop.keys():
    learner = linear_model.LinearRegression()
    learner.fit(train_feature[r_id], train_label[r_id])
    test_predict = learner.predict(test_feature_2hop[r_id])
    score=dl.evaluate(test_data_2hop[r_id], test_predict, test_label_2hop[r_id])
    for s in score.keys():
        scores[s].append(score[s])
for s in scores.keys():
    scores[s] = np.mean(scores[s])
print(f'2hot score:{scores}')
scores=defaultdict(list)
for r_id in test_feature_random.keys():
    learner = linear_model.LinearRegression()
    learner.fit(train_feature[r_id], train_label[r_id])
    test_predict = learner.predict(test_feature_random[r_id])
    score=dl.evaluate(test_data_random[r_id], test_predict, test_label_random[r_id])
    for s in score.keys():
        scores[s].append(score[s])
for s in scores.keys():
    scores[s] = np.mean(scores[s])
print(f'random score:{scores}')
print("test prediction finish!")



