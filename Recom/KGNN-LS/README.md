# KGNN-LS code
Adapted from [hwwang55/KGCN](https://github.com/hwwang55/KGNN-LS).

We replace the GNN module in paper by GCN and GAT for comparison.

## running environment

* Python 3.6
* tensorflow-gpu 1.9.0

## running procedure

* Download data from [tsinghua-cloud](https://cloud.tsinghua.edu.cn/d/02a5659200d34348aae8/)
* unzip file, rename the folder to "data", and move data folder to KGCN/
* cd to src/
* run main.py

```bash
python main.py --dataset=movie --n_epochs=10 --neighbor_sample_size=16 --dim=32 --n_iter=2 --batch_size=65536 --l2_weight=1e-7 --ls_weight=1.0 --lr=2e-2 --model=kgnn_ls
python main.py --dataset=music --n_epochs=10 --neighbor_sample_size=8 --dim=16 --n_iter=1 --batch_size=128 --l2_weight=1e-4 --ls_weight=0.1 --lr=5e-4 --model=kgnn_ls 
python main.py --dataset=restaurant --n_epochs=10 --neighbor_sample_size=4 --dim=8 --n_iter=2 --batch_size=65536 --l2_weight=1e-7 --ls_weight=0.5 --lr=2e-2 --model=kgnn_ls

python main.py --dataset=movie --n_epochs=10 --neighbor_sample_size=16 --dim=32 --n_iter=2 --batch_size=65536 --l2_weight=1e-7 --ls_weight=1.0 --lr=2e-2 --model=kgcn
python main.py --dataset=music --n_epochs=10 --neighbor_sample_size=8 --dim=16 --n_iter=1 --batch_size=128 --l2_weight=1e-4 --ls_weight=0.1 --lr=5e-4 --model=kgcn 
python main.py --dataset=restaurant --n_epochs=10 --neighbor_sample_size=4 --dim=8 --n_iter=2 --batch_size=65536 --l2_weight=1e-7 --ls_weight=0.5 --lr=2e-2 --model=kgcn

python main.py --dataset=movie --n_epochs=10 --neighbor_sample_size=16 --dim=32 --n_iter=1 --batch_size=65536 --l2_weight=1e-7 --ls_weight=1.0 --lr=2e-2 --model=gcn
python main.py --dataset=music --n_epochs=10 --neighbor_sample_size=8 --dim=16 --n_iter=1 --batch_size=128 --l2_weight=1e-4 --ls_weight=0.1 --lr=5e-4 --model=gcn
python main.py --dataset=restaurant --n_epochs=10 --neighbor_sample_size=4 --dim=8 --n_iter=2 --batch_size=65536 --l2_weight=1e-7 --ls_weight=0.5 --lr=2e-2 --model=gcn

python main.py --dataset=movie --n_epochs=10 --neighbor_sample_size=16 --dim=32 --n_iter=1 --batch_size=65536 --l2_weight=1e-7 --ls_weight=1.0 --lr=2e-2 --model=gat
python main.py --dataset=music --n_epochs=10 --neighbor_sample_size=8 --dim=16 --n_iter=1 --batch_size=128 --l2_weight=1e-6 --ls_weight=0.1 --lr=5e-4 --model=gat
python main.py --dataset=restaurant --n_epochs=10 --neighbor_sample_size=4 --dim=8 --n_iter=2 --batch_size=65536 --l2_weight=1e-7 --ls_weight=0.5 --lr=2e-2 --model=gat

```

## performance report

For MovieLens-20M dataset: 

|          |  R@2   |  R@10  |  R@50  | R@100  |  AUC   |
| :------: | :----: | :----: | :----: | :----: | :----: |
| 文中结果 | 0.043  | 0.155  | 0.321  | 0.458  | 0.979  |
| KGCN-LS  | 0.0449 | 0.1286 | 0.3343 | 0.4596 | 0.9785 |
|   GCN    | 0.0446 | 0.1523 | 0.3603 | 0.5246 | 0.9805 |
|   GAT    | 0.0476 | 0.1433 | 0.3567 | 0.4843 | 0.9772 |
|   KGCN   | 0.032  | 0.1155 | 0.2959 | 0.4238 | 0.9774 |

For Last.FM dataset:
|          |  R@2   |  R@10  |  R@50  | R@100  |  AUC   |
| :------: | :----: | :----: | :----: | :----: | :----: |
| 文中结果 | 0.044  | 0.122  | 0.277  |  0.37  | 0.744  |
| KGCN-LS  | 0.0439 | 0.0946 | 0.2612 | 0.3483 | 0.798  |
|   GCN    | 0.0388 | 0.1146 | 0.259  | 0.3575 | 0.7938 |
|   GAT    | 0.0107 | 0.071  | 0.2459 | 0.326  | 0.7353 |
|   KGCN   | 0.0388 | 0.1196 | 0.245  | 0.3721 | 0.7909 |

For Dianping-Food dataset:
|          |  R@2   |  R@10  |  R@50  | R@100  |  AUC   |
| :------: | :----: | :----: | :----: | :----: | :----: |
| 文中结果 | 0.047  |  0.17  |  0.34  | 0.487  |  0.85  |
| KGCN-LS  | 0.0317 | 0.1457 | 0.3779 | 0.4832 | 0.8454 |
|   GCN    | 0.0393 | 0.138  | 0.3276 | 0.4436 | 0.8524 |
|   GAT    | 0.0503 | 0.1527 | 0.3608 | 0.4794 | 0.8497 |
|   KGCN   | 0.0267 |  0.14  | 0.3662 | 0.4993 | 0.8407 |

***The following content is from the initial hwwang55/KGCN repo.***
# KGNN-LS

This repository is the implementation of KGNN-LS ([arXiv](http://arxiv.org/abs/1905.04413)):

> Knowledge-aware Graph Neural Networks with Label Smoothness Regularization for Recommender Systems
Hongwei Wang, Fuzheng Zhang, Mengdi Zhang, Jure Leskovec, Miao Zhao, Wenjie Li, Zhongyuan Wang.  
In Proceedings of The 25th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD 2019)

![](https://github.com/hwwang55/KGNN-LS/blob/master/framework.png)

KGNN-LS applies the technique of graph neural networks (GNNs) to proces knowledge graphs for the purpose of recommendation.
The model is enhanced by adding a label smoothness regularizer for more powerful and adaptive learning.


### Files in the folder

- `data/`
  - `movie/`
    - `item_index2entity_id.txt`: the mapping from item indices in the raw rating file to entity IDs in the KG;
    - `kg.txt`: knowledge graph file;
  - `music/`
    - `item_index2entity_id.txt`: the mapping from item indices in the raw rating file to entity IDs in the KG;
    - `kg.txt`: knowledge graph file;
    - `user_artists.dat`: raw rating file of Last.FM;
  - `restaurant/`
    - `Dianping-Food.zip`: containing the final rating file and the final KG file;
- `src/`: implementations of KGNN-LS.




### Running the code
- Movie  
  (The raw rating file of MovieLens-20M is too large to be contained in this repository.
  Download the dataset first.)
  ```
  $ wget http://files.grouplens.org/datasets/movielens/ml-20m.zip
  $ unzip ml-20m.zip
  $ mv ml-20m/ratings.csv data/movie/
  $ cd src
  $ python preprocess.py --dataset movie
  $ python main.py
  ```
- Music
  - ```
    $ cd src
    $ python preprocess.py --dataset music
    ```
  - open `src/main.py` file;
    
  - comment the code blocks of parameter settings for MovieLens-20M;
    
  - uncomment the code blocks of parameter settings for Last.FM;
    
  - ```
    $ python main.py
    ```
- Restaurant  
  ```
  $ cd data/restaurant
  $ unzip Dianping-Food.zip
  ```
  - open `src/main.py` file;
    
  - comment the code blocks of parameter settings for MovieLens-20M;
    
  - uncomment the code blocks of parameter settings for Dianping-Food;
    
  - ```
    $ python main.py
    ```
