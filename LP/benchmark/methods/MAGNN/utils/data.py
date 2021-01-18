import networkx as nx
import scipy.sparse as sp
import numpy as np
import scipy
import pickle

def load_IMDB_data(prefix='data/preprocessed/IMDB_processed'):
    G00 = nx.read_adjlist(prefix + '/0/0-1-0.adjlist', create_using=nx.MultiDiGraph)
    G01 = nx.read_adjlist(prefix + '/0/0-2-0.adjlist', create_using=nx.MultiDiGraph)
    G10 = nx.read_adjlist(prefix + '/1/1-0-1.adjlist', create_using=nx.MultiDiGraph)
    G11 = nx.read_adjlist(prefix + '/1/1-0-2-0-1.adjlist', create_using=nx.MultiDiGraph)
    G20 = nx.read_adjlist(prefix + '/2/2-0-2.adjlist', create_using=nx.MultiDiGraph)
    G21 = nx.read_adjlist(prefix + '/2/2-0-1-0-2.adjlist', create_using=nx.MultiDiGraph)
    idx00 = np.load(prefix + '/0/0-1-0_idx.npy')
    idx01 = np.load(prefix + '/0/0-2-0_idx.npy')
    idx10 = np.load(prefix + '/1/1-0-1_idx.npy')
    idx11 = np.load(prefix + '/1/1-0-2-0-1_idx.npy')
    idx20 = np.load(prefix + '/2/2-0-2_idx.npy')
    idx21 = np.load(prefix + '/2/2-0-1-0-2_idx.npy')
    features_0 = scipy.sparse.load_npz(prefix + '/features_0.npz')
    features_1 = scipy.sparse.load_npz(prefix + '/features_1.npz')
    features_2 = scipy.sparse.load_npz(prefix + '/features_2.npz')
    adjM = scipy.sparse.load_npz(prefix + '/adjM.npz')
    type_mask = np.load(prefix + '/node_types.npy')
    labels = np.load(prefix + '/labels.npy')
    train_val_test_idx = np.load(prefix + '/train_val_test_idx.npz')
    return [[G00, G01], [G10, G11], [G20, G21]], \
           [[idx00, idx01], [idx10, idx11], [idx20, idx21]], \
           [features_0, features_1, features_2],\
           adjM, \
           type_mask,\
           labels,\
           train_val_test_idx

def get_adjlist_pkl(dl, meta, type_id=0, return_dic=True, symmetric=False, return_tmp=False):
    meta010 = dl.get_meta_path(meta).tocoo()
    if return_tmp:
        tmp1 = meta010.copy()
    adjlist00 = [[] for _ in range(dl.nodes['count'][type_id])]
    for i,j,v in zip(meta010.row, meta010.col, meta010.data):
        adjlist00[i-dl.nodes['shift'][type_id]].extend([j-dl.nodes['shift'][type_id]]*int(v))
    adjlist00 = [' '.join(map(str, [i]+sorted(x))) for i,x in enumerate(adjlist00)]
    meta010 = dl.get_full_meta_path(meta, symmetric=symmetric)
    if return_tmp:
        tmp2 = meta010.copy()
    idx00 = {}
    for k in meta010:
        idx00[k] = np.array(sorted([tuple(reversed(i)) for i in meta010[k]]), dtype=np.int32).reshape([-1, len(meta)+1])
    if not return_dic:
        idx00 = np.concatenate(list(idx00.values()), axis=0)
    if return_tmp:
        return adjlist00, idx00, tmp1, tmp2
    return adjlist00, idx00

def load_IMDB_data_new():
    from scripts.data_loader import data_loader
    dl = data_loader('../../data/IMDB')
    adjlist00, idx00 = get_adjlist_pkl(dl, [(0,1), (1,0)], 0, False)
    G00 = nx.readwrite.adjlist.parse_adjlist(adjlist00, create_using=nx.MultiDiGraph)
    print('meta path 1 done')
    adjlist01, idx01 = get_adjlist_pkl(dl, [(0,2), (2,0)], 0, False)
    G01 = nx.readwrite.adjlist.parse_adjlist(adjlist01, create_using=nx.MultiDiGraph)
    print('meta path 2 done')
    adjlist10, idx10 = get_adjlist_pkl(dl, [(1,0), (0,1)], 1, False)
    G10 = nx.readwrite.adjlist.parse_adjlist(adjlist10, create_using=nx.MultiDiGraph)
    print('meta path 3 done')
    adjlist11, idx11 = get_adjlist_pkl(dl, [(1,0), (0,2), (2,0), (0, 1)], 1, False)
    G11 = nx.readwrite.adjlist.parse_adjlist(adjlist11, create_using=nx.MultiDiGraph)
    print('meta path 4 done')
    adjlist20, idx20 = get_adjlist_pkl(dl, [(2,0), (0,2)], 2, False)
    G20 = nx.readwrite.adjlist.parse_adjlist(adjlist20, create_using=nx.MultiDiGraph)
    print('meta path 5 done')
    adjlist21, idx21 = get_adjlist_pkl(dl, [(2,0), (0,1), (1,0), (0,2)], 2, False)
    G21 = nx.readwrite.adjlist.parse_adjlist(adjlist21, create_using=nx.MultiDiGraph)
    print('meta path 6 done')
    features = []
    types = len(dl.nodes['count'])
    for i in range(types):
        th = dl.nodes['attr'][i]
        if th is None:
            features.append(np.eye(dl.nodes['count'][i], dtype=np.float32))
        else:
            if type(th) is np.ndarray:
                features.append(th)
            else:
                features.append(th.toarray())
    #features_0, features_1, features_2, features_3 = features
    adjM = sum(dl.links['data'].values())
    type_mask = np.zeros(dl.nodes['total'], dtype=np.int32)
    for i in range(types):
        type_mask[dl.nodes['shift'][i]:dl.nodes['shift'][i]+dl.nodes['count'][i]] = i
    labels = np.zeros((dl.nodes['count'][0], dl.labels_train['num_classes']), dtype=int)
    val_ratio = 0.2
    train_idx = np.nonzero(dl.labels_train['mask'])[0]
    np.random.shuffle(train_idx)
    split = int(train_idx.shape[0]*val_ratio)
    val_idx = train_idx[:split]
    train_idx = train_idx[split:]
    train_idx = np.sort(train_idx)
    val_idx = np.sort(val_idx)
    test_idx = np.nonzero(dl.labels_test['mask'])[0]
    labels[train_idx] = dl.labels_train['data'][train_idx]
    labels[val_idx] = dl.labels_train['data'][val_idx]
    #labels = labels.argmax(axis=1)
    train_val_test_idx = {}
    train_val_test_idx['train_idx'] = train_idx
    train_val_test_idx['val_idx'] = val_idx
    train_val_test_idx['test_idx'] = test_idx
    return [[G00, G01], [G10, G11], [G20, G21]], \
           [[idx00, idx01], [idx10, idx11], [idx20, idx21]], \
           features, \
           adjM, \
           type_mask,\
           labels,\
           train_val_test_idx,\
            dl

def load_Freebase_data():
    from scripts.data_loader import data_loader
    dl = data_loader('../../data/Freebase')
    import json
    adjlists, idxs = [], []
    for fn in ['meta1.json', 'meta2.json']:
      with open('meta1.json', 'r', encoding='utf-8') as f:
        meta = json.loads(''.join(f.readlines()))
      for i, x in enumerate(meta['node_0'][:5]):
        path = list(map(int, x['path'].split(',')))
        path = path + [-x-1 for x in reversed(path)]
        th_adj, th_idx = get_adjlist_pkl(dl, path)
        adjlists.append(th_adj)
        idxs.append(th_idx)
        print('meta path {}-{} done'.format(fn, i))
    features = []
    types = len(dl.nodes['count'])
    for i in range(types):
        th = dl.nodes['attr'][i]
        if th is None:
            features.append(np.eye(dl.nodes['count'][i], dtype=np.float32))
        else:
            if type(th) is np.ndarray:
                features.append(th)
            else:
                features.append(th.toarray())
    #features_0, features_1, features_2, features_3 = features
    adjM = sum(dl.links['data'].values())
    adjM = adjM + adjM.T
    type_mask = np.zeros(dl.nodes['total'], dtype=np.int32)
    for i in range(types):
        type_mask[dl.nodes['shift'][i]:dl.nodes['shift'][i]+dl.nodes['count'][i]] = i
    labels = np.zeros((dl.nodes['count'][0], dl.labels_train['num_classes']), dtype=int)
    val_ratio = 0.2
    train_idx = np.nonzero(dl.labels_train['mask'])[0]
    np.random.shuffle(train_idx)
    split = int(train_idx.shape[0]*val_ratio)
    val_idx = train_idx[:split]
    train_idx = train_idx[split:]
    train_idx = np.sort(train_idx)
    val_idx = np.sort(val_idx)
    test_idx = np.nonzero(dl.labels_test['mask'])[0]
    labels[train_idx] = dl.labels_train['data'][train_idx]
    labels[val_idx] = dl.labels_train['data'][val_idx]
    labels = labels.argmax(axis=1)
    train_val_test_idx = {}
    train_val_test_idx['train_idx'] = train_idx
    train_val_test_idx['val_idx'] = val_idx
    train_val_test_idx['test_idx'] = test_idx
    return adjlists, \
           idxs, \
           features, \
           adjM, \
           type_mask,\
           labels,\
           train_val_test_idx,\
            dl

def load_ACM_data():
    from scripts.data_loader import data_loader
    dl = data_loader('../../data/ACM')
    dl.get_sub_graph([0,1,2])
    #dl.links['data'][0] += sp.eye(dl.nodes['total'])
    for i in range(dl.nodes['count'][0]):
        if dl.links['data'][0][i].sum() == 0:
            dl.links['data'][0][i,i] = 1
        if dl.links['data'][1][i].sum() == 0:
            dl.links['data'][1][i,i] = 1
    adjlist00, idx00 = get_adjlist_pkl(dl, [(0,1), (1,0)])
    print('meta path 1 done')
    adjlist01, idx01 = get_adjlist_pkl(dl, [(0,2), (2,0)])
    print('meta path 2 done')
    adjlist02, idx02 = get_adjlist_pkl(dl, [0, (0,1), (1,0)])
    print('meta path 3 done')
    adjlist03, idx03 = get_adjlist_pkl(dl, [0, (0,2), (2,0)])
    print('meta path 4 done')
    adjlist04, idx04 = get_adjlist_pkl(dl, [1, (0,1), (1,0)])
    print('meta path 5 done')
    adjlist05, idx05 = get_adjlist_pkl(dl, [1, (0,2), (2,0)])
    print('meta path 6 done')
    features = []
    types = len(dl.nodes['count'])
    for i in range(types):
        th = dl.nodes['attr'][i]
        if th is None:
            features.append(np.eye(dl.nodes['count'][i], dtype=np.float32))
        else:
            if type(th) is np.ndarray:
                features.append(th)
            else:
                features.append(th.toarray())
    #features_0, features_1, features_2, features_3 = features
    adjM = sum(dl.links['data'].values())
    type_mask = np.zeros(dl.nodes['total'], dtype=np.int32)
    for i in range(types):
        type_mask[dl.nodes['shift'][i]:dl.nodes['shift'][i]+dl.nodes['count'][i]] = i
    labels = np.zeros((dl.nodes['count'][0], dl.labels_train['num_classes']), dtype=int)
    val_ratio = 0.2
    train_idx = np.nonzero(dl.labels_train['mask'])[0]
    np.random.shuffle(train_idx)
    split = int(train_idx.shape[0]*val_ratio)
    val_idx = train_idx[:split]
    train_idx = train_idx[split:]
    train_idx = np.sort(train_idx)
    val_idx = np.sort(val_idx)
    test_idx = np.nonzero(dl.labels_test['mask'])[0]
    labels[train_idx] = dl.labels_train['data'][train_idx]
    labels[val_idx] = dl.labels_train['data'][val_idx]
    labels = labels.argmax(axis=1)
    train_val_test_idx = {}
    train_val_test_idx['train_idx'] = train_idx
    train_val_test_idx['val_idx'] = val_idx
    train_val_test_idx['test_idx'] = test_idx
    return [adjlist00, adjlist01, adjlist02, adjlist03, adjlist04, adjlist05], \
           [idx00, idx01, idx02, idx03, idx04, idx05], \
           features, \
           adjM, \
           type_mask,\
           labels,\
           train_val_test_idx,\
            dl

def load_DBLP_data():
    from scripts.data_loader import data_loader
    dl = data_loader('../../data/DBLP')
    adjlist00, idx00 = get_adjlist_pkl(dl, [(0,1), (1,0)])
    print('meta path 1 done')
    adjlist01, idx01 = get_adjlist_pkl(dl, [(0,1), (1,2), (2,1), (1,0)])
    print('meta path 2 done')
    adjlist02, idx02 = get_adjlist_pkl(dl, [(0,1), (1,3), (3,1), (1,0)])
    print('meta path 3 done')
    features = []
    for i in range(4):
        th = dl.nodes['attr'][i]
        if th is None:
            features.append(np.eye(dl.nodes['count'][i], dtype=np.float32))
        else:
            if type(th) is np.ndarray:
                features.append(th)
            else:
                features.append(th.toarray())
    features_0, features_1, features_2, features_3 = features
    adjM = sum(dl.links['data'].values())
    type_mask = np.zeros(dl.nodes['total'], dtype=np.int32)
    for i in range(4):
        type_mask[dl.nodes['shift'][i]:dl.nodes['shift'][i]+dl.nodes['count'][i]] = i
    labels = np.zeros((dl.nodes['count'][0], dl.labels_train['num_classes']), dtype=int)
    val_ratio = 0.2
    train_idx = np.nonzero(dl.labels_train['mask'])[0]
    np.random.shuffle(train_idx)
    split = int(train_idx.shape[0]*val_ratio)
    val_idx = train_idx[:split]
    train_idx = train_idx[split:]
    train_idx = np.sort(train_idx)
    val_idx = np.sort(val_idx)
    test_idx = np.nonzero(dl.labels_test['mask'])[0]
    labels[train_idx] = dl.labels_train['data'][train_idx]
    labels[val_idx] = dl.labels_train['data'][val_idx]
    labels = labels.argmax(axis=1)
    train_val_test_idx = {}
    train_val_test_idx['train_idx'] = train_idx
    train_val_test_idx['val_idx'] = val_idx
    train_val_test_idx['test_idx'] = test_idx
    """
    in_file = open(prefix + '/0/0-1-0.adjlist', 'r')
    adjlist00 = [line.strip() for line in in_file]
    adjlist00 = adjlist00[3:]
    in_file.close()
    in_file = open(prefix + '/0/0-1-2-1-0.adjlist', 'r')
    adjlist01 = [line.strip() for line in in_file]
    adjlist01 = adjlist01[3:]
    in_file.close()
    in_file = open(prefix + '/0/0-1-3-1-0.adjlist', 'r')
    adjlist02 = [line.strip() for line in in_file]
    adjlist02 = adjlist02[3:]
    in_file.close()

    in_file = open(prefix + '/0/0-1-0_idx.pickle', 'rb')
    idx00 = pickle.load(in_file)
    in_file.close()
    in_file = open(prefix + '/0/0-1-2-1-0_idx.pickle', 'rb')
    idx01 = pickle.load(in_file)
    in_file.close()
    in_file = open(prefix + '/0/0-1-3-1-0_idx.pickle', 'rb')
    idx02 = pickle.load(in_file)
    in_file.close()

    features_0 = scipy.sparse.load_npz(prefix + '/features_0.npz').toarray()
    features_1 = scipy.sparse.load_npz(prefix + '/features_1.npz').toarray()
    features_2 = np.load(prefix + '/features_2.npy')
    features_3 = np.eye(20, dtype=np.float32)

    adjM = scipy.sparse.load_npz(prefix + '/adjM.npz')
    type_mask = np.load(prefix + '/node_types.npy')
    labels = np.load(prefix + '/labels.npy')
    train_val_test_idx = np.load(prefix + '/train_val_test_idx.npz')
    """
    return [adjlist00, adjlist01, adjlist02], \
           [idx00, idx01, idx02], \
           [features_0, features_1, features_2, features_3],\
           adjM, \
           type_mask,\
           labels,\
           train_val_test_idx,\
            dl

def get_adjlist_pkl_special(dl, meta, tmp1, tmp2):
    from collections import defaultdict
    rel = dl.get_edge_type(meta[-1])
    meta010 = defaultdict(list)
    mat010 = np.zeros((dl.nodes['count'][0], dl.nodes['count'][0]))
    for k in tmp2:
        for tri in tmp2[k]:
            li1 = dl.re_cache[rel][tri[0]]
            li2 = dl.re_cache[rel][tri[-1]]
            if len(li1) == 0 or len(li2) == 0:
                continue
            candidate_u1_list = np.random.choice(len(li1), int(0.2 * len(li1)), replace=False)
            candidate_u1_list = li1[candidate_u1_list]
            candidate_u2_list = np.random.choice(len(li2), int(0.2 * len(li2)), replace=False)
            candidate_u2_list = li2[candidate_u2_list]
            for u1 in candidate_u1_list:
                for u2 in candidate_u2_list:
                    meta010[u1].append((u1, tri[0], tri[1], tri[2], u2))
                    mat010[u1,u2] += 1
    mat010 = sp.coo_matrix(mat010)
    adjlist00 = [[] for _ in range(dl.nodes['count'][0])]
    for i,j,v in zip(mat010.row, mat010.col, mat010.data):
        adjlist00[i].extend([j]*int(v))
    adjlist00 = [' '.join(map(str, [i]+sorted(x))) for i,x in enumerate(adjlist00)]
    idx00 = {}
    for k in meta010:
        idx00[k] = np.array(sorted([tuple(reversed(i)) for i in meta010[k]]), dtype=np.int32).reshape([-1, len(meta)+1])
    return adjlist00, idx00

def load_LastFM_data(dataset='LastFM'):
    from scripts.data_loader import data_loader
    dl = data_loader('../../data/'+dataset)
    import time
    last = time.time()
    adjlist00, idx00 = get_adjlist_pkl(dl, [(0,1), (1,0)], symmetric=True)
    pay = time.time() - last
    last = time.time()
    print('meta paht 1 done', pay)
    adjlist11, idx11, tmp1, tmp2 = get_adjlist_pkl(dl, [(1,2), (2,1)], type_id=1, symmetric=True, return_tmp=True)
    pay = time.time() - last
    last = time.time()
    print('meta paht 5 done', pay)
    adjlist01, idx01 = get_adjlist_pkl_special(dl, [(0,1), (1,2), (2,1), (1,0)], tmp1, tmp2)
    pay = time.time() - last
    last = time.time()
    print('meta paht 2 done', pay)
    adjlist02, idx02 = get_adjlist_pkl(dl, [(0,0)])
    pay = time.time() - last
    last = time.time()
    print('meta paht 3 done', pay)
    adjlist10, idx10 = get_adjlist_pkl(dl, [(1,0), (0,1)], type_id=1, symmetric=True)
    pay = time.time() - last
    last = time.time()
    print('meta paht 4 done', pay)
    adjlist12, idx12 = get_adjlist_pkl(dl, [(1,0), (0,0), (0,1)], type_id=1)
    pay = time.time() - last
    last = time.time()
    print('meta paht 6 done', pay)
    """
    in_file = open(prefix + '/0/0-1-0.adjlist', 'r')
    adjlist00 = [line.strip() for line in in_file]
    adjlist00 = adjlist00
    in_file.close()
    in_file = open(prefix + '/0/0-1-2-1-0.adjlist', 'r')
    adjlist01 = [line.strip() for line in in_file]
    adjlist01 = adjlist01
    in_file.close()
    in_file = open(prefix + '/0/0-0.adjlist', 'r')
    adjlist02 = [line.strip() for line in in_file]
    adjlist02 = adjlist02
    in_file.close()
    in_file = open(prefix + '/1/1-0-1.adjlist', 'r')
    adjlist10 = [line.strip() for line in in_file]
    adjlist10 = adjlist10
    in_file.close()
    in_file = open(prefix + '/1/1-2-1.adjlist', 'r')
    adjlist11 = [line.strip() for line in in_file]
    adjlist11 = adjlist11
    in_file.close()
    in_file = open(prefix + '/1/1-0-0-1.adjlist', 'r')
    adjlist12 = [line.strip() for line in in_file]
    adjlist12 = adjlist12
    in_file.close()

    in_file = open(prefix + '/0/0-1-0_idx.pickle', 'rb')
    idx00 = pickle.load(in_file)
    in_file.close()
    in_file = open(prefix + '/0/0-1-2-1-0_idx.pickle', 'rb')
    idx01 = pickle.load(in_file)
    in_file.close()
    in_file = open(prefix + '/0/0-0_idx.pickle', 'rb')
    idx02 = pickle.load(in_file)
    in_file.close()
    in_file = open(prefix + '/1/1-0-1_idx.pickle', 'rb')
    idx10 = pickle.load(in_file)
    in_file.close()
    in_file = open(prefix + '/1/1-2-1_idx.pickle', 'rb')
    idx11 = pickle.load(in_file)
    in_file.close()
    in_file = open(prefix + '/1/1-0-0-1_idx.pickle', 'rb')
    idx12 = pickle.load(in_file)
    in_file.close()"""

    adjM = sum(dl.links['data'].values())
    adjM = adjM + adjM.T
    type_mask = np.zeros(dl.nodes['total'], dtype=np.int32)
    for i in range(len(dl.nodes['count'])):
        type_mask[dl.nodes['shift'][i]:dl.nodes['shift'][i]+dl.nodes['count'][i]] = i
    #train_val_test_pos_user_artist = np.load(prefix + '/train_val_test_pos_user_artist.npz')
    #train_val_test_neg_user_artist = np.load(prefix + '/train_val_test_neg_user_artist.npz')

    return [[adjlist00, adjlist01, adjlist02],[adjlist10, adjlist11, adjlist12]],\
           [[idx00, idx01, idx02], [idx10, idx11, idx12]],\
           adjM, type_mask, dl #, train_val_test_pos_user_artist, train_val_test_neg_user_artist


# load skipgram-format embeddings, treat missing node embeddings as zero vectors
def load_skipgram_embedding(path, num_embeddings):
    count = 0
    with open(path, 'r') as infile:
        _, dim = list(map(int, infile.readline().strip().split(' ')))
        embeddings = np.zeros((num_embeddings, dim))
        for line in infile.readlines():
            count += 1
            line = line.strip().split(' ')
            embeddings[int(line[0])] = np.array(list(map(float, line[1:])))
    print('{} out of {} nodes have non-zero embeddings'.format(count, num_embeddings))
    return embeddings


# load metapath2vec embeddings
def load_metapath2vec_embedding(path, type_list, num_embeddings_list, offset_list):
    count = 0
    with open(path, 'r') as infile:
        _, dim = list(map(int, infile.readline().strip().split(' ')))
        embeddings_dict = {type: np.zeros((num_embeddings, dim)) for type, num_embeddings in zip(type_list, num_embeddings_list)}
        offset_dict = {type: offset for type, offset in zip(type_list, offset_list)}
        for line in infile.readlines():
            line = line.strip().split(' ')
            # drop </s> token
            if line[0] == '</s>':
                continue
            count += 1
            embeddings_dict[line[0][0]][int(line[0][1:]) - offset_dict[line[0][0]]] = np.array(list(map(float, line[1:])))
    print('{} node embeddings loaded'.format(count))
    return embeddings_dict


def load_glove_vectors(dim=50):
    print('Loading GloVe pretrained word vectors')
    file_paths = {
        50: 'data/wordvec/GloVe/glove.6B.50d.txt',
        100: 'data/wordvec/GloVe/glove.6B.100d.txt',
        200: 'data/wordvec/GloVe/glove.6B.200d.txt',
        300: 'data/wordvec/GloVe/glove.6B.300d.txt'
    }
    f = open(file_paths[dim], 'r', encoding='utf-8')
    wordvecs = {}
    for line in f.readlines():
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        wordvecs[word] = embedding
    print('Done.', len(wordvecs), 'words loaded!')
    return wordvecs
