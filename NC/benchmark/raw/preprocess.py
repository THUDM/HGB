import json
from os import PRIO_PGRP, ctermid, write
import numpy as np
from numpy.core.defchararray import count, replace, title
from numpy.core.numeric import NaN, moveaxis
from numpy.lib.npyio import save
from numpy.testing._private.utils import print_assert_equal
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

save_prefix = '../data/IMDB/'
num_ntypes = 3

# load raw data, delete movies with no actor or director
movies = pd.read_csv('movie_metadata.csv', encoding='utf-8').dropna(
    axis=0, subset=['actor_1_name', 'director_name']).reset_index(drop=True)
color_list = list(set(movies['color']))
lan_list = list(set(movies['language']))
cty_list = list(set(movies['country']))
rat_list = list(set(movies['content_rating']))

all_label = ["Romance", "Thriller", "Comedy", "Action", "Drama"]
label_cnt = [0 for i in range(len(all_label))]

# extract labels
labels = []
for movie_idx, genres in movies['genres'].iteritems():
    labels.append([])
    for genre in genres.split('|'):
        if genre in all_label:
            index = all_label.index(genre)
            labels[movie_idx].append(str(index))
            label_cnt[index] += 1
# extract attr
graph_field = ["director_name", 'actor_1_name', 'actor_2_name',
               'actor_3_name', 'genres', 'plot_keywords', 'movie_title', 'movie_imdb_link', 'color', 'language', 'country', 'content_rating']

attr_list = []
color = []
language = []
country = []
rating = []
for movie_idx, row in movies.iterrows():
    color.append(color_list.index(row['color']))
    language.append(lan_list.index(row['language']))
    country.append(cty_list.index(row['country']))
    rating.append(rat_list.index(row['content_rating']))
    attr_list.append(row.drop(graph_field))
    attr_list[movie_idx] = attr_list[movie_idx].fillna(0)
attr_list = pd.DataFrame(attr_list)
attr_list = (attr_list - attr_list.mean()) / (attr_list.std())
attr_list = attr_list.to_numpy()
color = np.eye(len(color_list))[color]
language = np.eye(len(lan_list))[language]
country = np.eye(len(cty_list))[country]
rating = np.eye(len(rat_list))[rating]

attr_list = np.hstack(
    (attr_list, color, language, country, rating))

# get director list and actor list
directors = list(set(movies['director_name'].dropna()))
directors.sort()
actors = list(set(movies['actor_1_name'].dropna().to_list() +
                  movies['actor_2_name'].dropna().to_list() +
                  movies['actor_3_name'].dropna().to_list()))
actors.sort()


# extract keywords
keywords = [[] for i in range(len(movies))]
for movie_idx, words in movies['plot_keywords'].iteritems():
    if not pd.isna(words):
        keywords[movie_idx] = words.split('|')

keywords_list = set()

for i in keywords:
    keywords_list = keywords_list.union(set(i))
keywords_list = list(keywords_list)


def write_file(f, l):
    l = [str(i) for i in l]
    f.write("\t".join(l))
    f.write("\n")


print("Generate Nodes")
open(save_prefix+"node.dat", "w")
cnt = 0
type_cnt = 0
for movie_idx, title in movies['movie_title'].iteritems():
    attr_str = ",".join([str(i) for i in attr_list[movie_idx]])
    with open(save_prefix+"node.dat", "a+") as f:
        write_file(f, [cnt, title.replace(
            " ", "_").strip(), type_cnt, attr_str])
    cnt += 1
type_cnt = 1
for j in [directors, actors, keywords_list]:
    for i in j:
        with open(save_prefix+"node.dat", "a+") as f:
            write_file(f, [cnt, i.replace(" ", "_").strip(), type_cnt])
            cnt += 1
    type_cnt += 1


# build the adjacency matrix for the graph consisting of movies, directors and actors
# 0 for movies, 1 for directors, 2 for actors, 3 for keyword

print("Generate Link")
open(save_prefix+"link.dat", "w")
edge_cnt = [0, 0, 0, 0, 0, 0]
with open(save_prefix+"link.dat", "a+") as f:
    for movie_idx, row in movies.iterrows():
        if row['director_name'] in directors:
            director_idx = directors.index(row['director_name'])
            director_idx += len(movies)
            write_file(f, [movie_idx, director_idx, 0, 1])
            write_file(f, [director_idx, movie_idx, 1, 1])
            edge_cnt[0] += 1
            edge_cnt[1] += 1
        if row['actor_1_name'] in actors:
            actor_idx = actors.index(row['actor_1_name'])
            actor_idx += len(movies)+len(directors)
            write_file(f, [movie_idx, actor_idx, 2, 1])
            write_file(f, [actor_idx, movie_idx, 3, 1])
            edge_cnt[2] += 1
            edge_cnt[3] += 1
        if row['actor_2_name'] in actors:
            actor_idx = actors.index(row['actor_2_name'])
            actor_idx += len(movies)+len(directors)
            write_file(f, [movie_idx, actor_idx, 2, 1])
            write_file(f, [actor_idx, movie_idx, 3, 1])
            edge_cnt[2] += 1
            edge_cnt[3] += 1
        if row['actor_3_name'] in actors:
            actor_idx = actors.index(row['actor_3_name'])
            actor_idx += len(movies)+len(directors)
            write_file(f, [movie_idx, actor_idx, 2, 1])
            write_file(f, [actor_idx, movie_idx, 3, 1])
            edge_cnt[2] += 1
            edge_cnt[3] += 1
        if pd.isna(row['plot_keywords']):
            continue
        for word in row['plot_keywords'].split("|"):
            if word in keywords_list:
                word_idx = keywords_list.index(word)
                word_idx += len(movies)+len(directors)+len(actors)
                write_file(f, [movie_idx, word_idx, 4, 1])
                write_file(f, [word_idx, movie_idx, 5, 1])
                edge_cnt[4] += 1
                edge_cnt[5] += 1

print("Generate Label")
all_data = []
for movie_idx, row in movies.iterrows():
    if not len(labels[movie_idx]) == 0:
        label = ",".join(labels[movie_idx])
        title = row['movie_title']
        data = [movie_idx, title.replace(" ", "_").strip(), 0, label]
        all_data.append(data)

label_train, label_test = train_test_split(
    all_data, test_size=0.7, random_state=42)
open(save_prefix+"label.dat", "w")
with open(save_prefix+"label.dat", "a+") as f:
    for i in label_train:
        write_file(f, i)
open(save_prefix+"label.dat.test", "w")
with open(save_prefix+"label.dat.test", "a+") as f:
    for i in label_test:
        write_file(f, i)

print("Generate Meta")
open(save_prefix+"meta.dat", "w")
dim = len(movies) + len(directors) + len(actors) + len(keywords_list)
meta_dict = {}
meta_dict["Node Total"] = dim
meta_dict["Node Type_0"] = len(movies)
meta_dict["Node Type_1"] = len(directors)
meta_dict["Node Type_2"] = len(actors)
meta_dict["Node Type_3"] = len(keywords)
meta_dict["Edge Total"] = sum(edge_cnt)
meta_dict["Edge Type_0"] = edge_cnt[0]
meta_dict["Edge Type_1"] = edge_cnt[1]
meta_dict["Edge Type_2"] = edge_cnt[2]
meta_dict["Edge Type_3"] = edge_cnt[3]
meta_dict["Edge Type_4"] = edge_cnt[4]
meta_dict["Edge Type_5"] = edge_cnt[5]
meta_dict["Label Total"] = sum(label_cnt)
for i in range(len(all_label)):
    meta_dict["Label Class_0_Type_"+str(i)] = label_cnt[i]
with open(save_prefix+"meta.dat", "a+") as f:
    json.dump(meta_dict, f, indent=1)

print("Generate Info")
open(save_prefix+"info.dat", "w")
info_dict = {}
info_dict['node.dat'] = {"node type": {}, "Attributes": {}, "keyword": {
}, "color": {}, "language": {}, "country": {}, "content_rating": {}}
info_dict['node.dat']['node type'][0] = "movie"
info_dict['node.dat']['node type'][1] = "director"
info_dict['node.dat']['node type'][2] = "actor"
info_dict['node.dat']['node type'][3] = "keyword"
info_dict['node.dat']['Attributes'][0] = "num_critic_for_reviews"
info_dict['node.dat']['Attributes'][1] = "duration"
info_dict['node.dat']['Attributes'][2] = "director_facebook_likes"
info_dict['node.dat']['Attributes'][3] = "actor_1_facebook_likes"
info_dict['node.dat']['Attributes'][4] = "actor_3_facebook_likes"
info_dict['node.dat']['Attributes'][5] = "gross"
info_dict['node.dat']['Attributes'][6] = "num_voted_users"
info_dict['node.dat']['Attributes'][7] = "cast_total_facebook_likes"
info_dict['node.dat']['Attributes'][8] = "facenumber_in_poster"
info_dict['node.dat']['Attributes'][9] = "num_user_for_reviews"
info_dict['node.dat']['Attributes'][10] = "budget"
info_dict['node.dat']['Attributes'][11] = "title_year"
info_dict['node.dat']['Attributes'][12] = "actor_2_facebook_likes"
info_dict['node.dat']['Attributes'][13] = "imdb_score"
info_dict['node.dat']['Attributes'][14] = "aspect_ration"
info_dict['node.dat']['Attributes'][15] = "movie_facebook_likes"
info_dict['node.dat']['Attributes']["16:19"] = "color"
info_dict['node.dat']['Attributes']["19:67"] = "language"
info_dict['node.dat']['Attributes']["67:132"] = "country"
info_dict['node.dat']['Attributes']["132:148"] = "content_rating"
for i in range(len(color_list)):
    info_dict['node.dat']['color'][i] = color_list[i]
for i in range(len(lan_list)):
    info_dict['node.dat']['language'][i] = lan_list[i]
for i in range(len(cty_list)):
    info_dict['node.dat']['country'][i] = cty_list[i]
for i in range(len(rat_list)):
    info_dict['node.dat']['content_rating'][i] = rat_list[i]
for i in range(len(keywords_list)):
    info_dict['node.dat']['keyword'][i] = keywords_list[i]
info_dict['link.dat'] = {'link type': {}}
info_dict['link.dat']['link type'][0] = {
    "start": 0, "end": 1, "meaning": "movie->director"}
info_dict['link.dat']['link type'][1] = {
    "start": 1, "end": 0, "meaning": "director->movie"}
info_dict['link.dat']['link type'][2] = {
    "start": 0, "end": 2, "meaning": "movie->actorh"}
info_dict['link.dat']['link type'][3] = {
    "start": 2, "end": 0, "meaning": "actor->movie"}
info_dict['link.dat']['link type'][4] = {
    "start": 0, "end": 3, "meaning": "movie->keyword"}
info_dict['link.dat']['link type'][5] = {
    "start": 3, "end": 0, "meaning": "keyword->movie"}
info_dict['label.dat'] = {'node type': {0: {}}}
for i in range(len(all_label)):
    info_dict['label.dat']['node type'][0][i] = all_label[i]
with open(save_prefix+"info.dat", "a+") as f:
    json.dump(info_dict, f, indent=4)

print("Generate Urls")
open(save_prefix+"url.dat", "w")
for movie_idx, url in movies['movie_imdb_link'].iteritems():
    with open(save_prefix+"url.dat", "a+") as f:
        write_file(f, [movie_idx, url.replace(" ", "_").strip()])
