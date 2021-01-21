import numpy as np
import pickle
import os
from collections import defaultdict
from scripts.data_loader import data_loader
import scipy.sparse as sp

data_name = 'LastFM'

data_dir = f'./data/{data_name}'
dl_pickle_f = os.path.join(data_dir, 'dl_pickle')
if os.path.exists(dl_pickle_f):
    dl = pickle.load(open(dl_pickle_f, 'rb'))
    print(f'Info: load {data_name} from {dl_pickle_f}')
else:
    dl = data_loader(data_dir)
    pickle.dump(dl, open(dl_pickle_f, 'wb'))
    print(f'Info: load {data_name} from original data and generate {dl_pickle_f}')

with open('data/LasfFM-magnn-test/magnn.pkl', 'rb') as f:
    mag = pickle.load(f)
with open('data/LasfFM-magnn-test/our.pkl', 'rb') as f:
    our = pickle.load(f)

# test neg sample for edge type
edge_type = 0
edge_types = [r_id for r_id in dl.links['meta'].keys()]
dim = dl.nodes['total']
# train
# indices = dl.train_pos[edge_type]
# values = [1] * len(indices[0])
# train_pos_a = sp.coo_matrix((values, indices), shape=(dim, dim)).tocsr()
# train_pos_a_reverse = sp.coo_matrix((values, (indices[1], indices[0])), shape=(dim, dim)).tocsr()
pos_links = 0
for r_id in dl.links['data'].keys():
    pos_links += dl.links['data'][r_id] + dl.links['data'][r_id].T
train_pos_a = pos_links
indices = dl.train_neg[edge_type]
values = [1] * len(indices[0])
train_neg_a = sp.coo_matrix((values, indices), shape=(dim, dim)).tocsr()
# valid
pos_links=0
for r_id in dl.valid_pos.keys():
    values = [1] * len(dl.valid_pos[r_id][0])
    valid_of_rel = sp.coo_matrix((values, dl.valid_pos[r_id]), shape=(dim, dim))
    pos_links += valid_of_rel
valid_pos_a = pos_links
indices = dl.valid_neg[edge_type]
values = [1] * len(indices[0])
valid_neg_a = sp.coo_matrix((values, indices), shape=(dim, dim)).tocsr()
# test
test_neigh, test_label = dl.get_test_neigh_full_random()
# # test_neg_index = np.where(np.array(test_label[edge_type]) == 0)[0]
# # test_neg = np.array(test_neigh[edge_type][1])
# # test_neg = test_neg[[test_neg_index]]
# #me = np.mean(test_neg)
test_pos_index = np.where(np.array(test_label[edge_type])==1)[0]
indices = [np.array(test_neigh[edge_type][0])[test_pos_index], np.array(test_neigh[edge_type][1])[test_pos_index]]
values = [1] * len(indices[0])
test_pos_a = sp.coo_matrix((values, indices), shape=(dim, dim)).tocsr()
test_neg_index = np.where(np.array(test_label[edge_type])==0)[0]
indices = [np.array(test_neigh[edge_type][0])[test_neg_index], np.array(test_neigh[edge_type][1])[test_neg_index]]
values = [1] * len(indices[0])
test_neg_a = sp.coo_matrix((values, indices), shape=(dim, dim)).tocsr()
# test_neigh, test_label = our
# test_pos_index = np.where(test_label == 1)[0]
# indices = [test_neigh[0][test_pos_index], test_neigh[1][test_pos_index]]
# values = [1] * len(indices[0])
# test_pos_a = sp.coo_matrix((values, indices), shape=(dim, dim)).tocsr()
# test_neg_index = np.where(test_label == 0)[0]
# indices = [test_neigh[0][test_neg_index], test_neigh[1][test_neg_index]]
# values = [1] * len(indices[0])
# test_neg_a = sp.coo_matrix((values, indices), shape=(dim, dim)).tocsr()

test_neigh, test_label = mag
test_pos_index = np.where(test_label == 1)[0]
indices = [test_neigh[0][test_pos_index], test_neigh[1][test_pos_index]]
values = [1] * len(indices[0])
test_pos_a_mag = sp.coo_matrix((values, indices), shape=(dim, dim)).tocsr()
test_neg_index = np.where(test_label == 0)[0]
indices = [test_neigh[0][test_neg_index], test_neigh[1][test_neg_index]]
values = [1] * len(indices[0])
test_neg_a_mag = sp.coo_matrix((values, indices), shape=(dim, dim)).tocsr()
a,b = test_pos_a.multiply(test_pos_a_mag), test_neg_a.multiply(test_neg_a_mag)
# statistic
repeat = np.zeros((6, 6))
array_list = [train_pos_a, train_neg_a, valid_pos_a, valid_neg_a, test_pos_a, test_neg_a]
for i, row in enumerate(array_list):
    for j, col in enumerate(array_list):
        re = row.multiply(col)
        repeat[i][j] = np.sum(re)

print(dl.nodes)
print(dl.links)
print(dl.links_test)
train_neg = dl.train_neg
edge_type = 2
train_data = [dl.train_pos[edge_type][0] + dl.train_neg[edge_type][0],
              dl.train_pos[edge_type][1] + dl.train_neg[edge_type][1]]
train_label = [1] * len(dl.train_pos[edge_type][0]) + [0] * len(dl.train_neg[edge_type][0])
pred = [1] * len(dl.train_pos[edge_type][0]) + [0] * len(dl.train_neg[edge_type][0])
print(dl.evaluate(train_data, pred, train_label))

edge_list, label = dl.get_test_neigh()
print(data_loader.evaluate(edge_list[0], label[0], label[0]))
edge_list, label = dl.get_test_neigh_w_random()
print(dl.evaluate(edge_list[0], label[0], label[0]))

meta = [0, -1]
meta = [(0, 1), (1, 0)]
print(dl.get_meta_path(meta))
print(dl.get_full_meta_path(meta)[0])
