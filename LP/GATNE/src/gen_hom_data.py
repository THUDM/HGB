import sys
import random
import numpy as np

if len(sys.argv) <= 1:
    exit('need least one para')
data_name = sys.argv[1]
node_num={'amazon':10166, 'youtube':2000, 'twitter':10000}
ori_train_file=f'../data/{data_name}/train.txt'
ori_valid_file=f'../data/{data_name}/valid.txt'
ori_test_file=f'../data/{data_name}/test.txt'
reindex_dic_file = f'../data/{data_name}/reindex.txt'

new_train_file=f'../data/{data_name}/hom_train.txt'
new_valid_file=f'../data/{data_name}/hom_valid.txt'
new_test_file=f'../data/{data_name}/hom_test.txt'


def gen_hom_data(data_name, dataset_type='train'):
    ori_data_file = globals()[f'ori_{dataset_type}_file']
    new_data_file=globals()[f'new_{dataset_type}_file']
    reindex_dic={}
    with open(reindex_dic_file, 'r') as reindex_dic_file_handle:
        for line in reindex_dic_file_handle:
            ori_index, new_index = line[:-1].split('\t')
            reindex_dic[ori_index] = new_index

    new_data_file_handle = open(new_data_file,'w')
    with open(ori_data_file) as ori_data_file_handle:
        for line in ori_data_file_handle:
            if dataset_type=='train':
                edge_type, left, right = line[:-1].split(' ')
            else:
                edge_type, left, right, label = line[:-1].split(' ')
            if dataset_type=='train':
                new_data_file_handle.write(f'{edge_type}\t{reindex_dic[left]}\t{reindex_dic[right]}\t{edge_type}\n')
                neg_left, neg_right = int(random.random()*node_num[data_name]), int(random.random()*node_num[data_name])
                new_data_file_handle.write(f'{edge_type}\t{neg_left}\t{neg_right}\t{0}\n')
            else:
                new_data_file_handle.write(f'{edge_type}\t{reindex_dic[left]}\t{reindex_dic[right]}\t{edge_type if label=="1" else label }\n')

    new_data_file_handle.close()

def gen_reindex_dic():

    ori_index_list = []

    with open(ori_train_file) as ori_train_file_handle:
        for line in ori_train_file_handle:
            edge_type, left, right = line[:-1].split(' ')
            ori_index_list.append(left)
            ori_index_list.append(right)
    with open(ori_valid_file) as ori_valid_file_handle:
        for line in ori_valid_file_handle:
            edge_type, left, right, label = line[:-1].split(' ')
            ori_index_list.append(left)
            ori_index_list.append(right)
    with open(ori_test_file) as ori_test_file_handle:
        for line in ori_test_file_handle:
            edge_type, left, right, label = line[:-1].split(' ')
            ori_index_list.append(left)
            ori_index_list.append(right)

    ori_index_list=list(set(ori_index_list))
    reindex_dic_file_handle = open(reindex_dic_file, 'w')
    reindex = 0
    for ori_index in ori_index_list:
        reindex_dic_file_handle.write(f'{ori_index}\t{reindex}\n')
        reindex+=1
    print(f'reindex {reindex} nodes')
    return


gen_reindex_dic()
gen_hom_data(data_name, dataset_type='train')
gen_hom_data(data_name, dataset_type='valid')
gen_hom_data(data_name, dataset_type='test')

