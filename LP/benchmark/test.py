import numpy as np
import pickle
import os
from collections import defaultdict
from scripts.data_loader import data_loader

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

print(dl.nodes)
print(dl.links)
print(dl.links_test)
train_neg = dl.train_neg
valid_neg = dl.valid_neg
edge_type=0
train_data = [dl.train_pos[edge_type][0] + dl.train_neg[edge_type][0],
                  dl.train_pos[edge_type][1] + dl.train_neg[edge_type][1]]
train_label = [1] * len(dl.train_pos[edge_type][0]) + [0] * len(dl.train_neg[edge_type][0])
print(dl.evaluate(train_data,train_label,train_label))

edge_list, label = dl.get_test_neigh()
print(dl.evaluate(edge_list[0],label[0],label[0]))
meta = [0, -1]
meta = [(0, 1), (1, 0)]
print(dl.get_meta_path(meta))
print(dl.get_full_meta_path(meta)[0])
