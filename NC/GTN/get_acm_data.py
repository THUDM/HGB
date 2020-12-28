from scipy import io
import numpy as np
from scipy.sparse import csr_matrix
import random

random.seed(2020)
np.random.seed(2020)

mat_file = io.loadmat('ACM.mat')

nodedat = open('ACM/node.dat', 'w', encoding='utf-8')
linkdat = open('ACM/link.dat', 'w', encoding='utf-8')
infodat = open('ACM/info.dat', 'w', encoding='utf-8')
labeldat = open('ACM/label.dat', 'w', encoding='utf-8')
labeltest = open('ACM/label.dat.test', 'w', encoding='utf-8')

info = {'node.dat':{}, 'link.dat':{}, 'label.dat':{}}

paper_conf = mat_file['PvsC'].nonzero()[1]

# DataBase
paper_db = np.isin(paper_conf,[1,13])
paper_db_idx = np.where(paper_db == True)[0]
paper_db_idx = np.sort(np.random.choice(paper_db_idx,994,replace=False))
# Data Mining
paper_dm = np.isin(paper_conf,[0])
paper_dm_idx = np.where(paper_dm == True)[0]
# Wireless Communication
paper_wc = np.isin(paper_conf,[9,10])
paper_wc_idx = np.where(paper_wc == True)[0]

paper_idx = np.sort(list(paper_db_idx)+list(paper_dm_idx)+list(paper_wc_idx))

info['label.dat'] = {'node type': {0: {0:'database', 1:'wireless communication', 2:'data mining'}}}

# 0 : database, 1: wireless communication, 2: data mining
paper_target = []
for idx in paper_idx:
    if idx in paper_db_idx:
        paper_target.append(0)
    elif idx in paper_wc_idx:
        paper_target.append(1)
    else:
        paper_target.append(2)
paper_target = np.array(paper_target)

def write_label(paper_target):
    labels = list(enumerate(paper_target))
    random.shuffle(labels)
    ratio = 0.3
    L = len(labels)
    for x,y in labels[:int(L*ratio)]:
        labeldat.write('{}\t\t{}\t{}\n'.format(x, 0, y))
    for x,y in labels[int(L*ratio):]:
        labeltest.write('{}\t\t{}\t{}\n'.format(x, 0, y))
write_label(paper_target)

paper_dic = {}
for i,paper in enumerate(paper_idx):
    paper_dic[paper] = i

authors = mat_file['PvsA'][paper_idx].nonzero()[1]
author_dic = {}
re_authors = []
for author in authors:
    if author not in author_dic:
        author_dic[author] = len(author_dic) + len(paper_idx)
    re_authors.append(author_dic[author])
re_authors = np.array(re_authors)

subjects = mat_file['PvsL'][paper_idx].nonzero()[1]
subject_dic = {}
re_subjects = []
for subject in subjects:
    if subject not in subject_dic:
        subject_dic[subject] = len(subject_dic) + len(paper_idx) + len(author_dic)
    re_subjects.append(subject_dic[subject])
re_subjects = np.array(re_subjects)

terms = mat_file['TvsP'].transpose()[paper_idx].nonzero()[1]
term_dic = {}
re_terms = []
for term in terms:
    if term not in term_dic:
        term_dic[term] = len(term_dic) + len(subject_dic) + len(author_dic) + len(paper_idx)
    re_terms.append(term_dic[term])
re_terms = np.array(re_terms)

node_num = len(term_dic) + len(subject_dic) + len(author_dic) + len(paper_idx)
print(node_num)

info['node.dat'] = {'node type':{0:'paper', 1:'author', 2:'subject', 3:'term'}}

papers = mat_file['PvsA'][paper_idx].nonzero()[0]
data = np.ones_like(papers)
A_pa = csr_matrix((data, (papers, re_authors)), shape=(node_num,node_num))

papers = mat_file['PvsL'][paper_idx].nonzero()[0]
data = np.ones_like(papers)
A_ps = csr_matrix((data, (papers, re_subjects)), shape=(node_num,node_num))

papers = mat_file['TvsP'].transpose()[paper_idx].nonzero()[0]
data = np.ones_like(papers)
A_pt = csr_matrix((data, (papers, re_terms)), shape=(node_num,node_num))

A_ap = A_pa.transpose()
A_sp = A_ps.transpose()
A_tp = A_pt.transpose()

A_cite = mat_file['PvsP'][paper_idx][:, paper_idx]
A_ref = A_cite.transpose()

info['link.dat'] = {'link type':{}}

def write_link(A, fr, to, name, type_id):
    info['link.dat']['link type'][type_id] = {'start':fr, 'end':to, 'meaning':name}
    for i, j in zip(*A.nonzero()):
        linkdat.write('{}\t{}\t{}\t{}\n'.format(i, j, type_id, '1.0'))
write_link(A_cite, 0, 0, 'paper-cite-paper', 0)
write_link(A_ref, 0, 0, 'paper-ref-paper', 1)
write_link(A_pa, 0, 1, 'paper-author', 2)
write_link(A_ap, 1, 0, 'author-paper', 3)
write_link(A_ps, 0, 2, 'paper-subject', 4)
write_link(A_sp, 2, 0, 'subject-paper', 5)
write_link(A_pt, 0, 3, 'paper-term', 6)
write_link(A_tp, 3, 0, 'term-paper', 7)

paper_feat = np.array(A_pt[:len(paper_idx),-len(term_dic):].toarray()>0, dtype=np.int)
author_feat = np.array(A_ap.dot(A_pt)[len(paper_idx):len(paper_idx)+len(author_dic),-len(term_dic):].toarray()>0, dtype=np.int)
subject_feat = np.array(A_sp.dot(A_pt)[len(paper_idx)+len(author_dic):len(paper_idx)+len(author_dic)+len(subject_dic),-len(term_dic):].toarray()>0, dtype=np.int)
all_feat = np.concatenate((paper_feat,author_feat,subject_feat))

def write_node(dic, mat, feat, type_id):
    print(mat.shape)
    for k in dic:
        s = mat[k][0][0].replace('\n','').replace('\r','').replace('\t',' ')
        new_idx = dic[k]
        if feat is not None:
            feat_s = ','.join(map(str, feat[new_idx]))
            nodedat.write('{}\t{}\t{}\t{}\n'.format(new_idx, s, type_id, feat_s))
        else:
            nodedat.write('{}\t{}\t{}\n'.format(new_idx, s, type_id))

write_node(paper_dic, mat_file['P'], all_feat, 0)
write_node(author_dic, mat_file['A'], all_feat, 1)
write_node(subject_dic, mat_file['L'], all_feat, 2)
write_node(term_dic, mat_file['T'], None, 3)

import json
infodat.write(json.dumps(info, indent=4))

nodedat.close()
linkdat.close()
infodat.close()
labeldat.close()
labeltest.close()
