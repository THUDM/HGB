import sys
sys.path.append('../../')

from scripts.data_loader import data_loader

for dataset in ['LastFM', 'LastFM_magnn']:
    prefix = 'data/raw/'+dataset+'/'
    dl = data_loader('../../data/'+dataset)
    node_type = {0:'user', 1:'artist', 2:'tag'}
    for k in dl.links['data']:
        rel_meta = dl.links['meta'][k]
        fn = node_type[rel_meta[0]] + '_' + node_type[rel_meta[1]]
        if rel_meta[0] == rel_meta[1]:
            fn = fn + '(original)'
        fn = fn + '.dat'
        g = open(prefix+fn, 'w', encoding='utf-8')
        train_idx = []
        mat = dl.links['data'][k].nonzero()
        for i, (x,y) in enumerate(zip(*mat)):
            train_idx.append(i)
            g.write('{}\t{}'.format(x-dl.nodes['shift'][rel_meta[0]]+1, y-dl.nodes['shift'][rel_meta[1]]+1))
            if rel_meta[0] == 0 and rel_meta[1] == 1:
                g.write('\t{}'.format(1))
            g.write('\n')
        if k in dl.links_test['data']:
            shift = len(train_idx)
            for i, (x,y) in enumerate(zip(*dl.valid_pos[k])):
                train_idx.append(shift+i)
                g.write('{}\t{}'.format(x-dl.nodes['shift'][rel_meta[0]]+1, y-dl.nodes['shift'][rel_meta[1]]+1))
                if rel_meta[0] == 0 and rel_meta[1] == 1:
                    g.write('\t{}'.format(1))
                g.write('\n')
            test_idx = []
            shift = len(train_idx)
            mat_test = dl.links_test['data'][k].nonzero()
            for i, (x,y) in enumerate(zip(*mat_test)):
                test_idx.append(shift+i)
                g.write('{}\t{}'.format(x-dl.nodes['shift'][rel_meta[0]]+1, y-dl.nodes['shift'][rel_meta[1]]+1))
                if rel_meta[0] == 0 and rel_meta[1] == 1:
                    g.write('\t{}'.format(1))
                g.write('\n')
            import numpy as np
            import random
            random.seed(2021)
            g.close()
            links = []
            with open(prefix+fn, 'r', encoding='utf-8') as f:
                for line in f:
                    th = line.split('\t')
                    links.append(tuple(map(int, th)))
            arg = [i for (v, i) in sorted((v, i) for (i, v) in enumerate(links))]
            links = sorted(links)
            g = open(prefix+fn, 'w', encoding='utf-8')
            for i,j,k in links:
                g.write('{}\t{}\t{}\n'.format(i,j,k))
            old2new = {}
            for new_id, old_id in enumerate(arg):
                old2new[old_id] = new_id
            for i in range(len(train_idx)):
                train_idx[i] = old2new[train_idx[i]]
            for i in range(len(test_idx)):
                test_idx[i] = old2new[test_idx[i]]
            np.random.shuffle(train_idx)
            split = int(0.1*len(train_idx))
            val_idx = sorted(train_idx[:split])
            train_idx = sorted(train_idx[split:])
            np.savez(prefix+'train_val_test_idx.npz', train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)
        g.close()
