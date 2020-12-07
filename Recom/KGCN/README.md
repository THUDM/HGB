# KGCN code

Adapted from [hwwang55/KGCN](https://github.com/hwwang55/KGCN).

We replace the GNN module in paper by GCN and GAT for comparison.

## running environment

* Python 3.6
* tensorflow-gpu 1.9.0

## running procedure

* Download data from [tsinghua-cloud](https://cloud.tsinghua.edu.cn/d/eee49039f99e4984a30a/)
* unzip and move data folder to KGCN/
* cd to src/
* run main.py

```bash
python main.py --model gcn
python main.py --model gat
python main.py --model kgcn
```

## performance report

For MovieLens-20M dataset: (Other datasets to be completed)

|      | AUC       | F1        |
|------|-----------|-----------|
| KGCN | 0.977     | 0.930     |
| GAT  | 0.978     | 0.932     |
| GGN  | **0.980** | **0.937** |

***The following content is from the initial hwwang55/KGCN repo.***

# KGCN

This repository is the implementation of [KGCN](https://dl.acm.org/citation.cfm?id=3313417) ([arXiv](https://arxiv.org/abs/1904.12575)):

> Knowledge Graph Convolutional Networks for Recommender Systems  
Hongwei Wang, Miao Zhao, Xing Xie, Wenjie Li, Minyi Guo.  
In Proceedings of The 2019 Web Conference (WWW 2019)

![](https://github.com/hwwang55/KGCN/blob/master/framework.png)

KGCN is **K**nowledge **G**raph **C**onvolutional **N**etworks for recommender systems, which uses the technique of graph convolutional networks (GCN) to proces knowledge graphs for the purpose of recommendation.


### Files in the folder

- `data/`
  - `movie/`
    - `item_index2entity_id.txt`: the mapping from item indices in the raw rating file to entity IDs in the KG;
    - `kg.txt`: knowledge graph file;
  - `music/`
    - `item_index2entity_id.txt`: the mapping from item indices in the raw rating file to entity IDs in the KG;
    - `kg.txt`: knowledge graph file;
    - `user_artists.dat`: raw rating file of Last.FM;
- `src/`: implementations of KGCN.




### Running the code
- Movie  
  (The raw rating file of MovieLens-20M is too large to be contained in this repository.
  Download the dataset first.)
  ```
  $ wget http://files.grouplens.org/datasets/movielens/ml-20m.zip
  $ unzip ml-20m.zip
  $ mv ml-20m/ratings.csv data/movie/
  $ cd src
  $ python preprocess.py -d movie
  ```
- Music
  - ```
    $ cd src
    $ python preprocess.py -d music
    ```
  - open `src/main.py` file;
    
  - comment the code blocks of parameter settings for MovieLens-20M;
    
  - uncomment the code blocks of parameter settings for Last.FM;
    
  - ```
    $ python main.py
    ```
