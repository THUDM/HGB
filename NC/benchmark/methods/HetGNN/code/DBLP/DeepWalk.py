import string;
import re;
import random
import math
import numpy as np
from gensim.models import Word2Vec
from itertools import *
import sys

dimen = 128
window = 5


def read_random_walk_corpus():
    walks = []
    inputfile = open(sys.path[0] + "/het_random_walk.txt", "r")
    for line in inputfile:
        path = re.split(' ', line)
        walks.append(path)
    inputfile.close()
    return walks


def gen_net_embed():
    walk_corpus = read_random_walk_corpus()
    model = Word2Vec(walk_corpus, size=dimen, window=window, min_count=0, workers=2, sg=1, hs=0, negative=5)
    file_ = sys.path[0] + "/node_net_embedding.txt"
    model.wv.save_word2vec_format(file_)
    print(f"Generate {file_} done.")
