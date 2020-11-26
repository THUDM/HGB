#!/user/bin/env python
# -*- coding: utf-8 -*-

import collections
import networkx
# import matplotlib.pyplot as plt
import json
import os
import pickle
import gensim
import numpy as np
from tqdm import tqdm
from scipy import sparse as spr

from utils import sample

DATASETS = 'MR'


rootpath = './'
outpath = rootpath + 'model/data/{}/'.format(DATASETS)
datapath = rootpath + 'data/{}/'.format(DATASETS)
SIM = 0.5
LP = 0.75
RHO = 0.3
TopK_for_Topics = 2


def cnt_nodes(g):
    text_nodes, entity_nodes, topic_nodes = set(), set(), set()
    for i in g.nodes():
        if i.isdigit():
            text_nodes.add(i)
        elif i[:6] == 'topic_':
            topic_nodes.add(i)
        else:
            entity_nodes.add(i)
    print("# text_nodes: {}     # entity_nodes: {}     # topic_nodes: {}".format(
        len(text_nodes), len(entity_nodes), len(topic_nodes)))
    return text_nodes, entity_nodes, topic_nodes


with open(datapath+'model_network_sampled.pkl', 'rb') as f:
    g = pickle.load(f)
text_nodes, entity_nodes, topic_nodes = cnt_nodes(g)

with open(datapath+"features_BOW.pkl", 'rb') as f:
    features_BOW = pickle.load(f)
with open(datapath+"features_TFIDF.pkl", 'rb') as f:
    features_TFIDF = pickle.load(f)
with open(datapath+"features_index.pkl", 'rb') as f:
    features_index_BOWTFIDF = pickle.load(f)
with open(datapath+"features_entity_descBOW.pkl", 'rb') as f:
    features_entity_BOW = pickle.load(f)
with open(datapath+"features_entity_descTFIDF.pkl", 'rb') as f:
    features_entity_TFIDF = pickle.load(f)
with open(datapath+"features_entity_index_desc.pkl", 'rb') as f:
    features_entity_index_desc = pickle.load(f)

feature = features_TFIDF
features_index = features_index_BOWTFIDF
entityF = features_entity_TFIDF
features_entity_index = features_entity_index_desc
textShape = feature.shape
entityShape = entityF.shape
print("Shape of text feature:", textShape,
      'Shape of entity feature:', entityShape)

# 删掉没有特征的实体
notinind = set()

entitySet = set(entity_nodes)
print(len(entitySet))

for i in entitySet:
    if i not in features_entity_index:
        notinind.add(i)
print(len(g.nodes()), len(notinind))
g.remove_nodes_from(notinind)
entitySet = entitySet - notinind
print(len(entitySet), len(features_entity_index))
N = len(g.nodes())
print(len(g.nodes()), len(g.edges()))
text_nodes, entity_nodes, topic_nodes = cnt_nodes(g)


# 删掉一些边
cnt = 0
nodes = g.nodes()
print(len(g.edges()))
for node in tqdm(nodes):
    try:
        cache = [j for j in g[node]
                 if ('sim' in g[node][j] and g[node][j]['sim'] < SIM)    # 0.5
                 or ('link_probability' in g[node][j] and g[node][j]['link_probability'] <= LP)
                 or ('rho' in g[node][j] and g[node][j]['rho'] < RHO)
                 ]
        if len(cache) != 0:
            g.remove_edges_from([(node, i) for i in cache])
        cnt += len(cache)
    except:
        print(g[node])
        break
print(len(g.edges()), cnt)

# 删掉孤立点（实体）
delete = [n for n in g.nodes() if len(g[n]) == 0 and n not in text_nodes]
print("Num of 孤立点：", len(delete))
g.remove_nodes_from(delete)


train, vali, test, alltext = sample(datapath, DATASETS)


# topic
with open(datapath + 'topic_word_distribution.pkl', 'rb') as f:
    topic_word = pickle.load(f)
with open(datapath + 'doc_topic_distribution.pkl', 'rb') as f:
    doc_topic = pickle.load(f)
with open(datapath + 'doc_index_LDA.pkl', 'rb') as f:
    doc_idx_list = pickle.load(f)

topic_num = topic_word.shape[0]
topics = []
for i in range(topic_num):
    topicName = 'topic_' + str(i)
    topics.append(topicName)


def naive_arg_topK(matrix, K, axis=0):
    """
    perform topK based on np.argsort
    :param matrix: to be sorted
    :param K: select and sort the top K items
    :param axis: dimension to be sorted.
    :return:
    """
    full_sort = np.argsort(-matrix, axis=axis)
    return full_sort.take(np.arange(K), axis=axis)


topK_topics = naive_arg_topK(doc_topic, TopK_for_Topics, axis=1)
for i in range(topK_topics.shape[0]):
    for j in range(TopK_for_Topics):
        g.add_edge(doc_idx_list[i], topics[topK_topics[i, j]])

print("gnodes:", len(g.nodes()), "gedges:", len(g.edges()))


# build Edges data
cnt = 0
nodes = g.nodes()
graphdict = collections.defaultdict(list)
for node in tqdm(nodes):
    try:
        cache = [j for j in g[node]
                 # 0.5
                 if ('sim' in g[node][j] and g[node][j]['sim'] >= SIM) or ('sim' not in g[node][j])
                 if ('link_probability' in g[node][j] and g[node][j]['link_probability'] > LP) or ('link_probability' not in g[node][j])
                 if ('rho' in g[node][j] and g[node][j]['rho'] > RHO) or ('rho' not in g[node][j])
                 ]
        if len(cache) != 0:
            graphdict[node] = cache
        cnt += len(cache)
    except:
        print(g[node])
        break
print('edges: ', cnt)


def normalizeF(mx):
    sup = np.absolute(mx).max()
    if sup == 0:
        return mx
    return mx / sup


text_nodes, entity_nodes, topic_nodes = cnt_nodes(g)

mapindex = dict()
cnt = 0
for i in text_nodes | entity_nodes | topic_nodes:
    mapindex[i] = cnt
    cnt += 1
print(len(g.nodes()), len(mapindex))

if not os.path.exists(outpath):
    os.makedirs(outpath)


# build feature data
gnodes = set(g.nodes())
print(gnodes, mapindex)
with open(outpath + 'train.map', 'w') as f:
    f.write('\n'.join([str(mapindex[i]) for i in train if i in gnodes]))
with open(outpath + 'vali.map', 'w') as f:
    f.write('\n'.join([str(mapindex[i]) for i in vali if i in gnodes]))
with open(outpath + 'test.map', 'w') as f:
    f.write('\n'.join([str(mapindex[i]) for i in test if i in gnodes]))

flag_zero = False

input_type = 'text&entity&topic2hgcn'
if input_type == 'text&entity&topic2hgcn':
    node_with_feature = set()
    DEBUG = False

    # text node
    content = dict()
    for i in tqdm(range(textShape[0])):
        ind = features_index[i]
        if (ind) not in text_nodes:
            continue
        content[ind] = feature[i, :].toarray()[0].tolist()  #
        if DEBUG:
            content[ind] = feature[i, :10].toarray()[0].tolist()
        if flag_zero:
            entityFlen = entityShape[1]
            content[ind] += [0] * (entityFlen + topic_word.shape[1])
    with open(datapath + '{}.txt'.format(DATASETS), 'r') as f:
        for line in tqdm(f):
            ind, cat = line.strip('\n').split('\t')[:2]
            ind = (ind)
            if ind not in text_nodes:
                continue
            content[ind] += [cat]
            alllen = len(content[ind])
    with open(outpath + '{}.content.text'.format(DATASETS), 'w') as f:
        for ind in tqdm(content):
            f.write(str(mapindex[ind]) + '\t' +
                    '\t'.join(map(str, content[ind])) + '\n')
            node_with_feature.add(ind)
    cache = len(content)
    print("共{}个文本".format(len(content)))

    # entity node
    content = dict()
    for i in tqdm(range(entityShape[0])):
        name = features_entity_index[i]
        if name not in entity_nodes:
            continue
        content[name] = entityF[i, :].toarray()[0].tolist() + ['entity']
        if flag_zero:
            content[name] = [0] * textShape[1] + content[name] + \
                [0] * topic_word.shape[1] + ['entity']
    with open(outpath + '{}.content.entity'.format(DATASETS), 'w') as f:
        for ind in tqdm(content):
            f.write(str(mapindex[ind]) + '\t' +
                    '\t'.join(map(str, content[ind])) + '\n')
            node_with_feature.add(ind)
    cache += len(content)
    print("共{}个实体".format(len(content)))

    # topic node
    content = dict()
    for i in range(topic_num):
        #         zero_num = textShape[1] + entityFlen - topic_num
        topicName = topics[i]
        if topicName not in topic_nodes:
            continue
        one_hot = [0] * topic_num
        one_hot[i] = 1
        content[topicName] = one_hot
        content[topicName] = topic_word[i].tolist() + ['topic']
        if flag_zero:
            zero_num = textShape[1] + entityFlen
            content[topicName] = [0] * zero_num + \
                content[topicName] + ['topic']

    with open(outpath + '{}.content.topic'.format(DATASETS), 'w') as f:
        for ind in tqdm(content):
            f.write(str(mapindex[ind]) + '\t' +
                    '\t'.join(map(str, content[ind])) + '\n')
            node_with_feature.add(ind)
    cache += len(content)
    print("共{}个主题".format(len(content)))

    print(cache, len(mapindex))
    print("nodes with features:", len(node_with_feature))


# save mappings
with open(outpath+'mapindex.txt', 'w') as f:
    for i in mapindex:
        f.write("{}\t{}\n".format(i, mapindex[i]))


# save adj matrix
with open(outpath+'{}.cites'.format(DATASETS), 'w') as f:
    doneSet = set()
    nodeSet = set()
    for node in graphdict:
        for i in graphdict[node]:
            if (node, i) not in doneSet:
                f.write(str(mapindex[node])+'\t'+str(mapindex[i])+'\n')
                doneSet.add((i, node))
                doneSet.add((node, i))
                nodeSet.add(node)
                nodeSet.add(i)
    for i in tqdm(range(len(mapindex))):
        f.write(str(i)+'\t'+str(i)+'\n')

print('Num of nodes with edges: ', len(nodeSet))
