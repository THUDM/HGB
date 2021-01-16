import pathlib
import pickle

import numpy as np
import scipy.sparse
import scipy.io
import pandas as pd

path = 'data/raw/LastFM/'

user_artist = pd.read_csv('data/raw/LastFM/user_artist.dat', encoding='utf-8', delimiter='\t', names=['userID', 'artistID', 'weight'])
user_friend = pd.read_csv('data/raw/LastFM/user_user(original).dat', encoding='utf-8', delimiter='\t', names=['userID', 'friendID'])
artist_tag = pd.read_csv('data/raw/LastFM/artist_tag.dat', encoding='utf-8', delimiter='\t', names=['artistID', 'tagID'])
num_user = 1892
num_artist = 17632
num_tag = 11945

train_val_test_idx = np.load('data/raw/LastFM/train_val_test_idx.npz')
train_idx = train_val_test_idx['train_idx']
val_idx = train_val_test_idx['val_idx']
test_idx = train_val_test_idx['test_idx']
train_idx = np.sort(np.concatenate([train_idx, val_idx]))
 
user_artist = user_artist.loc[train_idx].reset_index(drop=True)

# build the adjacency matrix
# 0 for user, 1 for artist, 2 for tag
dim = num_user + num_artist + num_tag

type_mask = np.zeros((dim), dtype=int)
type_mask[num_user:num_user+num_artist] = 1
type_mask[num_user+num_artist:] = 2

adjM = np.zeros((dim, dim), dtype=int)
for _, row in user_artist.iterrows():
    uid = row['userID'] - 1
    aid = num_user + row['artistID'] - 1
    adjM[uid, aid] = max(1, row['weight'])
    adjM[aid, uid] = max(1, row['weight'])
for _, row in user_friend.iterrows():
    uid = row['userID'] - 1
    fid = row['friendID'] - 1
    adjM[uid, fid] = 1
for _, row in artist_tag.iterrows():
    aid = num_user + row['artistID'] - 1
    tid = num_user + num_artist + row['tagID'] - 1
    adjM[aid, tid] += 1
    adjM[tid, aid] += 1

# filter out artist-tag links with counts less than 2
adjM[num_user:num_user+num_artist, num_user+num_artist:] = adjM[num_user:num_user+num_artist, num_user+num_artist:] * (adjM[num_user:num_user+num_artist, num_user+num_artist:] > 1)
adjM[num_user+num_artist:, num_user:num_user+num_artist] = np.transpose(adjM[num_user:num_user+num_artist, num_user+num_artist:])

valid_tag_idx = adjM[num_user:num_user+num_artist, num_user+num_artist:].sum(axis=0).nonzero()[0]
num_tag = len(valid_tag_idx)
dim = num_user + num_artist + num_tag
type_mask = np.zeros((dim), dtype=int)
type_mask[num_user:num_user+num_artist] = 1
type_mask[num_user+num_artist:] = 2

adjM_reduced = np.zeros((dim, dim), dtype=int)
adjM_reduced[:num_user+num_artist, :num_user+num_artist] = adjM[:num_user+num_artist, :num_user+num_artist]
adjM_reduced[num_user:num_user+num_artist, num_user+num_artist:] = adjM[num_user:num_user+num_artist, num_user+num_artist:][:, valid_tag_idx]
adjM_reduced[num_user+num_artist:, num_user:num_user+num_artist] = np.transpose(adjM_reduced[num_user:num_user+num_artist, num_user+num_artist:])

adjM = adjM_reduced

user_artist_list = {i: adjM[i, num_user:num_user+num_artist].nonzero()[0] for i in range(num_user)}
# artist_user_list = {i: adjM[num_user + i, :num_user].nonzero()[0] for i in range(num_artist)}
user_user_list = {i: adjM[i, :num_user].nonzero()[0] for i in range(num_user)}
artist_tag_list = {i: adjM[num_user + i, num_user+num_artist:].nonzero()[0] for i in range(num_artist)}
# tag_artist_list = {i: adjM[num_user + num_artist + i, num_user:num_user+num_artist].nonzero()[0] for i in range(num_tag)}

links = []

for user in user_artist_list:
    for artist in user_artist_list[user]:
        links.append((user, artist+num_user, 0, 1.0))

# ratio = 0.8
# split = int(len(links)*ratio)
# import random
# random.seed(2021)
# random.shuffle(links)
# links_test = links[split:]
# links = links[:split]
user_artist = pd.read_csv('data/raw/LastFM/user_artist.dat', encoding='utf-8', delimiter='\t', names=['userID', 'artistID', 'weight'])
train_val_test_idx = np.load('data/raw/LastFM/train_val_test_idx.npz')
# train_idx = train_val_test_idx['train_idx']
# val_idx = train_val_test_idx['val_idx']
test_idx = train_val_test_idx['test_idx']
user_artist = user_artist.loc[test_idx].reset_index(drop=True)
links_test = []
for _, row in user_artist.iterrows():
    uid = row['userID'] - 1
    aid = num_user + row['artistID'] - 1
    links_test.append((uid, aid, 0, 1.0))

for user in user_user_list:
    for friend in user_user_list[user]:
        links.append((user, friend, 1, 1.0))
for artist in artist_tag_list:
    for tag in artist_tag_list[artist]:
        links.append((num_user+artist, tag+num_user+num_artist, 2, 1.0))


info = {
    'node.dat': {
        '0': 'user',
        '1': 'artist',
        '2': 'tag'
    },
    'link.dat': {
        '0': {'start':'0', 'end':'1', 'meaning':'user-artist'},
        '1': {'start':'0', 'end':'0', 'meaning':'user-user'},
        '2': {'start':'1', 'end':'2', 'meaning':'artist-tag'}
    }
}

prefix = './LastFM_magnn/'

import json
with open(prefix+'info.dat', 'w', encoding='utf-8') as f:
    f.write(json.dumps(info, indent=4))

g = open(prefix+'link.dat', 'w', encoding='utf-8')
for u,v,t,w in links:
    g.write('{}\t{}\t{}\t{}\n'.format(u,v,t,w))
g.close()

g = open(prefix+'link.dat.test', 'w', encoding='utf-8')
for u,v,t,w in links_test:
    g.write('{}\t{}\t{}\t{}\n'.format(u,v,t,w))
g.close()

g = open(prefix+'node.dat', 'w', encoding='utf-8')
for i in range(num_user):
    g.write('{}\t\t0\n'.format(i))
for i in range(num_artist):
    g.write('{}\t\t1\n'.format(num_user+i))
for i in range(num_tag):
    g.write('{}\t\t2\n'.format(num_user+num_artist+i))
g.close()
