import networkx as nx
import numpy as np
import scipy
import pickle
import scipy.sparse as sp

def load_data(prefix='DBLP'):
    from data_loader import data_loader
    dl = data_loader(prefix)
    features = []
    for i in range(len(dl.nodes['count'])):
        th = dl.nodes['attr'][i]
        if th is None:
            features.append(sp.eye(dl.nodes['count'][i]))
        else:
            features.append(th)
    adjM = sum(dl.links['data'].values())
    return features,\
           adjM, \
            dl
