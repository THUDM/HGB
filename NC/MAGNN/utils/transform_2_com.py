import numpy as np
from scipy import sparse
from tqdm import tqdm
import random

''' run the code in / dir '''
prefix='data/preprocessed/DBLP_processed'
new_prefix='../Data/DBLP'

features_0 = sparse.load_npz(prefix + '/features_0.npz').toarray()
features_1 = sparse.load_npz(prefix + '/features_1.npz').toarray()
features_2 = np.load(prefix + '/features_2.npy')
features_3 = np.eye(20, dtype=np.float32)
feat_list = [features_0, features_1, features_2, features_3]
node_num_list = [np.shape(feat_)[0] for feat_ in feat_list]
s=sum(node_num_list)
adjM = sparse.load_npz(prefix + '/adjM.npz')

labels = np.load(prefix + '/labels.npy')

def split(full_list,shuffle=False,ratio=0.8):
    n_total = len(full_list)
    offset = int(n_total * ratio)
    if n_total==0 or offset<1:
        return [],full_list
    if shuffle:
        random.shuffle(full_list)
    sublist_1 = full_list[:offset]
    sublist_2 = full_list[offset:]
    return sublist_1,sublist_2

''' node_id node_type node_attributes '''
def gen_node_dat():
    node_dat_file = new_prefix + '/node.dat'
    print(f'Generating {node_dat_file}')
    node_dat_file_handle = open(node_dat_file, 'w')
    node_id = 0
    for index_, feat_ in enumerate(feat_list):
        print(f'\tFor node type {index_}')
        for i in tqdm(range(np.shape(feat_)[0])):
            node_dat_file_handle.write(str(node_id) + ' ' + str(index_) + ' ')
            node_dat_file_handle.write(",".join(str(i) for i in list(feat_[i])))
            node_dat_file_handle.write('\n')
            node_id += 1
    node_dat_file_handle.close()

''' node_id node_id link_type '''
def gen_link_dat():
    link_dat_file = new_prefix + '/link.dat'
    print(f'Generating {link_dat_file}')
    link_dat_file_handle = open(link_dat_file, 'w')
    nnz = adjM.nonzero()  # indices of nonzero values
    link_num_list=[0,0,0]
    for left, right in tqdm(zip(nnz[0], nnz[1])):
        link_type = 0
        # link_type=adjM[left, right]
        if left>right:
            left, right = right, left
        if left<sum(node_num_list[:1]) and right<sum(node_num_list[:2]) and right>=sum(node_num_list[:1]):
            link_type=0 # A-P
        elif left < sum(node_num_list[:2]) and left >= sum(node_num_list[:1]) and right<sum(node_num_list[:3]) and  right>=sum(node_num_list[:2]):
            link_type=1 # P-T
        elif left < sum(node_num_list[:2]) and left >= sum(node_num_list[:1]) and  right>=sum(node_num_list[:3]):
            link_type=2 # P-V
        else:
            exit('Link error occurs')
        link_num_list[link_type] += 1
        link_dat_file_handle.write(str(left) + ' ' + str(right) + ' ' + str(link_type) + '\n')
    link_dat_file_handle.close()

''' node_id node_type node_label '''
def gen_label_dat_w_test():
    label_list = list(labels)
    label_dic_list=[]
    for node_id in range(len(label_list)):
        label_dic_list.append((node_id, label_list[node_id]))
    [train_list, test_list] = split(label_dic_list, shuffle=True, ratio=0.8)

    label_dat_file = new_prefix + '/label.dat'
    label_dat_file_handle = open(label_dat_file, 'w')
    print(f'Generating {label_dat_file}')
    for node_w_label in tqdm(train_list):
        label_dat_file_handle.write(str(node_w_label[0]) + ' 0 ' + str(node_w_label[1]) + '\n')
    label_dat_file_handle.close()

    label_dat_test_file = new_prefix + '/label_test.dat'
    print((f'Generating {label_dat_test_file}'))
    label_dat_test_file_handle = open(label_dat_test_file, 'w')
    for node_w_label in tqdm(test_list):
        label_dat_test_file_handle.write(str(node_w_label[0]) + ' 0 ' + str(node_w_label[1]) + '\n')
    label_dat_test_file_handle.close()

# gen_node_dat()
# gen_link_dat()
gen_label_dat_w_test()