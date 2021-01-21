import numpy as np
from scipy import sparse
from tqdm import tqdm
import random

''' run the code in / dir '''
prefix='data/preprocessed/DBLP_processed'
new_prefix='../benchmark/data/DBLP'

features_0 = sparse.load_npz(prefix + '/features_0.npz').toarray()
# features_0 = np.eye(4057, dtype=np.float32)
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

''' node_id, node_name, node_type, node_attr '''
def gen_node_dat():
    # author name
    author_name_list=[]
    author_label_file='data/raw/DBLP/author_label.txt'
    with open(author_label_file) as author_label_file_handle:
        for line in author_label_file_handle:
            author_name = line.strip()[line.rfind('\t')+1:]
            author_name_list.append(author_name)
    # conf name
    conf_name_list = ['AAAI', 'CIKM', 'CVPR', 'ECIR', 'ECML', 'EDBT', 'ICDE', 'ICDM', 'ICML', 'IJCAI', 'KDD']+\
                     ['PAKDD', 'PKDD', 'PODS', 'SDM', 'SIGIR', 'SIGMOD', 'VLDB', 'WWW', 'WSDM']

    node_dat_file = new_prefix + '/node.dat'
    print(f'Generating {node_dat_file}')
    node_dat_file_handle = open(node_dat_file, 'w')
    node_id = 0
    for index_, feat_ in enumerate(feat_list):
        print(f'\tFor node type {index_}')
        for i in tqdm(range(np.shape(feat_)[0])):
            node_type = str(index_)
            node_name=''
            if node_type=='0':
                node_name=author_name_list[i]
            elif node_type=='3':
                node_name = conf_name_list[i]
            node_dat_file_handle.write(f'{str(node_id)}\t{node_name}\t{node_type}')
            if node_type=='0' or node_type=='1' or node_type=='2':
                node_dat_file_handle.write('\t')
                node_dat_file_handle.write(",".join(str(i) for i in list(feat_[i])))
            node_dat_file_handle.write('\n')
            node_id += 1
    node_dat_file_handle.close()

''' h_id, t_id, r_id, link_weight '''
def gen_link_dat():
    link_dat_file = new_prefix + '/link.dat'
    print(f'Generating {link_dat_file}')
    link_dat_file_handle = open(link_dat_file, 'w')
    nnz = adjM.nonzero()  # indices of nonzero values
    link_num_list=[0,0,0,0,0,0]
    sum_list=[sum(node_num_list[:1]), sum(node_num_list[:2]), sum(node_num_list[:3])]
    for left, right in tqdm(zip(nnz[0], nnz[1])):
        link_type = 0
        # link_type=adjM[left, right]
        if left < sum_list[0] and right >= sum_list[0] and right < sum_list[1]:
            link_type = 0 # A-P
        elif left >= sum_list[0] and left < sum_list[1]  and right >= sum_list[1] and  right < sum_list[2]:
            link_type = 1 # P-T
        elif left >= sum_list[0] and left < sum_list[1] and  right>=sum_list[2]:
            link_type = 2 # P-V
        elif right < sum_list[0] and left >= sum_list[0] and left < sum_list[1]:
            link_type = 3  # P-A
        elif right >= sum_list[0] and right < sum_list[1]  and left >= sum_list[1] and left < sum_list[2]:
            link_type = 4  # T-P
        elif right >= sum_list[0] and right < sum_list[1] and left>=sum_list[2]:
            link_type = 5 # V-P
        else:
            exit('Link error occurs')
        link_num_list[link_type] += 1
        link_dat_file_handle.write(str(left) + '\t' + str(right) + '\t' + str(link_type) + '\t1.0' + '\n')
    link_dat_file_handle.close()

'''  node_id, node_name, node_type, node_label'''
def gen_label_dat_w_test():
    # author name
    author_name_list = []
    author_label_file = 'data/raw/DBLP/author_label.txt'
    with open(author_label_file) as author_label_file_handle:
        for line in author_label_file_handle:
            author_name = line.strip()[line.rfind('\t') + 1:]
            author_name_list.append(author_name)

    label_num_list=[0,0,0,0]
    label_list = list(labels)
    label_dic_list=[]
    for node_id in range(len(label_list)):
        label_dic_list.append((node_id, label_list[node_id]))
    [train_list, test_list] = split(label_dic_list, shuffle=True, ratio=0.3)

    label_dat_file = new_prefix + '/label.dat'
    label_dat_file_handle = open(label_dat_file, 'w')
    print(f'Generating {label_dat_file}')
    for node_w_label in tqdm(train_list):
        label_dat_file_handle.write(str(node_w_label[0]) + f'\t{author_name_list[node_w_label[0]]}\t0\t' + str(node_w_label[1]) + '\n')
        label_num_list[node_w_label[1]]+=1
    label_dat_file_handle.close()

    label_dat_test_file = new_prefix + '/label.dat.test'
    print((f'Generating {label_dat_test_file}'))
    label_dat_test_file_handle = open(label_dat_test_file, 'w')
    for node_w_label in tqdm(test_list):
        label_dat_test_file_handle.write(str(node_w_label[0]) + f'\t{author_name_list[node_w_label[0]]}\t0\t' + str(node_w_label[1]) + '\n')
        label_num_list[node_w_label[1]] += 1
    label_dat_test_file_handle.close()

# gen_node_dat()
gen_link_dat()
# gen_label_dat_w_test()