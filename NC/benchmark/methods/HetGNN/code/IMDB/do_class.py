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
from sklearn.tree import DecisionTreeClassifier
import sys

sys.path.append('../../')

from scripts.data_loader import data_loader

# todo  set data_name/node_2type_class/type_labels/type_class for new dataset
data_name = 'IMDB'
node_type_2class = 0
type_labels = ('m', 'd', 'a', 'k')
type_class = 'm'

parser = argparse.ArgumentParser(description='application data process')
parser.add_argument('--C_n', type=int, default=4,
                    help='number of node class label')
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


def get_class_embed(iter_i):
    decimal_keep = 4
    class_embed = np.around(np.random.normal(0, 0.01, [node_n[node_type_2class], args.embed_d]), decimal_keep)
    embed_f = open(os.path.join(temp_dir, f"node_embedding-{iter_i}.txt"), "r")
    for line in islice(embed_f, 0, None):
        line = line.strip()
        node_id = re.split(' ', line)[0]
        if len(node_id) and (node_id[0] in type_labels):
            type_label = node_id[0]
            index = int(node_id[1:])
            embed = np.asarray(re.split(' ', line)[1:], dtype='float32')
            if type_label == type_class:
                class_embed[index] = embed
    embed_f.close()
    return class_embed


def model():
    iter_i=200
    print(f'Info: Use embed of {iter_i}')
    class_embed = get_class_embed(iter_i)
    train_id = np.where(dl.labels_train['mask'])
    train_features = class_embed[train_id]
    train_target = dl.labels_train['data'][train_id]
    train_target = np.array(train_target)

    learner =DecisionTreeClassifier()
    learner.fit(train_features, train_target)
    print("training finish!")

    test_id = np.where(dl.labels_test['mask'])
    test_features = class_embed[test_id]
    test_predict = learner.predict(test_features)
    test_target = dl.labels_test['data'][test_id]
    test_target = np.array(test_target)
    print("test prediction finish!")

    print("MicroF1: ")
    print(sklearn.metrics.f1_score(test_target, test_predict, average='micro'))
    print("MacroF1: ")
    print(sklearn.metrics.f1_score(test_target, test_predict, average='macro'))


print(f"------{data_name} classification------")
model()
print(f"------{data_name} classification end------")
