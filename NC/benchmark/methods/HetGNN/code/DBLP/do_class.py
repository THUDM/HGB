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
import sys
sys.path.append('../../')

from scripts.data_loader import data_loader
parser = argparse.ArgumentParser(description = 'application data process')
parser.add_argument('--A_n', type = int, default = 4057,
			   help = 'number of author node')

parser.add_argument('--embed_d', type = int, default = 128,
			   help = 'embedding dimension')
temp_dir = os.path.join(sys.path[0], 'temp')
if not os.path.exists(temp_dir):
	os.makedirs(temp_dir)
dl_pickle_f=os.path.join(temp_dir, 'dl_pickle')
if os.path.exists(dl_pickle_f):
	dl = pickle.load(open(dl_pickle_f, 'rb'))
	print(f'Info: load DBLP from {dl_pickle_f}')
else:
	dl = data_loader('../../data/DBLP')
	pickle.dump(dl, open(dl_pickle_f, 'wb'))
	print(f'Info: load DBLP from original data and generate {dl_pickle_f}')
args = parser.parse_args()
print(args)

def get_author_embed():
	a_embed = np.around(np.random.normal(0, 0.01, [args.A_n, args.embed_d]), 10)
	embed_f = open(os.path.join(temp_dir, "node_embedding-200.txt"), "r")
	for line in islice(embed_f, 0, None):
		line = line.strip()
		node_id = re.split(' ', line)[0]
		if len(node_id) and (node_id[0] in ('a', 'p', 't', 'v')):
			type_label = node_id[0]
			index = int(node_id[1:])
			embed = np.asarray(re.split(' ',line)[1:], dtype='float32')
			if type_label == 'a':
				a_embed[index] = embed
	embed_f.close()
	return a_embed

def model():
	a_embed = get_author_embed()
	train_id = np.where(dl.labels_train['mask'])
	train_features = a_embed[train_id]
	train_target = dl.labels_train['data'][train_id]
	train_target = [np.argmax(l)for l in train_target]
	train_target = np.array(train_target)

	learner = linear_model.LogisticRegression()
	learner.fit(train_features, train_target)
	print("training finish!")

	test_id = np.where(dl.labels_test['mask'])
	test_features = a_embed[test_id]
	test_target = dl.labels_test['data'][test_id]
	test_target = [np.argmax(l) for l in test_target]
	test_target = np.array(test_target)

	test_predict = learner.predict(test_features)
	print("test prediction finish!")


	print ("MicroF1: ")
	print (sklearn.metrics.f1_score(test_target,test_predict,average='micro'))
	print("MacroF1: ")
	print(sklearn.metrics.f1_score(test_target, test_predict, average='macro'))


print("------author classification------")
model()
print("------author classification end------")



