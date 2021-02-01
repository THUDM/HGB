# HetSANN code
Adapted from [didi/hetsann](https://github.com/didi/hetsann)
We produced the result of HetSANN.M.R.V on our DBLP, ACM, IMDB dataset.

## running environment
- python 3.6.12
- nvidia-tensorflow 1.15.4+nv20.11

## run
- download dataset
- run execute_sparse.py
```shell
cd HetSANN_MRV
CUDA_VISIBLE_DEVICES=0 python -u execute_sparse.py --dataset ACM --train_rate 0.8 --lr 0.001 --l2_coef 0.0001 --hid 64 --heads 8 8 1 --epochs 10000 --patience 100 --target_node 0 --target_is_multilabels 0 --layers 3 --inv_coef 1e-3 --feats-type 2

CUDA_VISIBLE_DEVICES=0 python -u execute_sparse.py --repeat=5 --dataset IMDB --train_rate 0.8 --lr 0.005 --l2_coef 0 --hid 32 --heads 4 1 --epochs 10000 --patience 100 --target_node 0 --target_is_multilabels 1 --layers 2 --inv_coef 1e-3 --feats-type $type

CUDA_VISIBLE_DEVICES=0 python -u execute_sparse.py --repeat=5 --dataset DBLP --train_rate 0.8 --lr 0.005 --l2_coef 0.0001 --hid 64 --heads 4 1 --epochs 10000 --patience 100 --target_node 0 --target_is_multilabels 0 --layers 2 --inv_coef 1e-3 --feats-type $type
```

The following contnet is from initial `didi/hetsann` repo

# HetSANN: Heterogeneous Graph Structural Attention Neural Network
---------------

This is basic implementation of our AAAI'20 paper:

Huiting Hong, Hantao Guo, Yucheng Lin, Xiaoqing Yang, Zang Li, Jieping Ye. 2020. An Attention-based Graph Nerual Network for Heterogeneous Structural Learning. In Proceedings of AAAI conference (AAAI’20).

HetSANN framework            
:-------------------------:
![](https://github.com/didi/hetsann/raw/master/fig/model.png)

Type-aware attention layer (TAL) in HetSANN            
:-------------------------:
![](https://github.com/didi/hetsann/raw/master/fig/attention.png)



Dependencies
------------
The script has been tested running under Python 3.5.2, with the following packages installed (along with their dependencies):

- `argparse==1.4.0`
- `numpy==1.14.1`
- `scipy==1.0.0`
- `networkx==2.1`
- `tensorflow-gpu==1.6.0`

In addition, CUDA 9.0 and cuDNN 7 have been used.


Overview
--------------
Here we provide the implementation of HetSANN.M, HetSANN.M.R and HetSANN.M.R.V in HetSANN_M, HetSANN_MR and HetSANN_MRV respectively.

M, R and V mean HetSANN with multi-task learning, voices-sharing product and cycle-consistency loss.

Each model folder is organised as follows:
- `models/` contains the implementation of the HetSANN network (`sp_hgat.py`);
- `pre_trained/` the process will save the trained model in this directory;
- `utils/` contains:
    * an implementation of an attention head, along with an experimental sparse version (`layers.py`);
    * preprocessing subroutines (`process.py`);

Finally, `execute_sparse.py` puts all of the above together and may be used to execute a full training run on the dataset.

How to run
---------------
**Network Input**

The input of a heterogeneous network consists of node features, node labels and edges between nodes in the HIN. The edges are preferred to saved as sparse adjacency matrixes, each adjacency matrix preserves one type of relationship. Here is an input example of aminer network:
```python
data = scipy.io.loadmat(Aminer.mat)
print(data.keys())  # ['__header__', '__version__', '__globals__', 'PvsF', 'AvsF', 'AvsC', 'PvsC', 'PvsA', 'AvsA', 'PvsP'], where '__header__', '__version__' and '__globals__' are automatic generated when save matfile with scipy.io.savemat.

print(data["PvsF"].shape)  # PvsF matrix is the feature of paper nodes, the shape is [paper_number, paper_features_dimension]

print(data["AvsF"].shape)  # AvsF matrix is the feature of author nodes, the shape is [author_number, author_features_dimension]

print(data["PvsA"].shape)  # PvsA is the adjacency matrix with shape of [paper_number, author_number], which preserves the publishing relationship between the paper an the author.

print(data["PvsP"].shape)  # PvsP is the adjacency matrix with shape of [paper_number, paper_number], which preserves the citation relationship between papers.

print(data["AvsA"].shape)  # AvsA is the adjacency matrix with shape of [author_number, author_number], which preserves the collaboration relationship between authors.

print(data["AvsC"].shape)  # AvsC matrix saves the label information of authors. The shape is [author_number, class_number]. Here an author can be multi labeled.

print(data["PvsC"].shape)  # PvsC matrix saves the label information of papers. The shape is [paper_number, class_number]. Here a paper is labeled as one class.
```


**Run**

```shell
cd HetSANN_M
python -u execute_sparse.py --seed 0 --dataset aminer --train_rate 0.8 --hid 8 --heads 8 --epochs 1000 --patience 30 --target_node 0 1 --target_is_multilabels 0 1 --layers 3
```
- seed, random seed;
- dataset, the dataset name;
- train_rate, the ratio of label samples for training;
- hid, number of hidden units per head in each TAL.
- heads, number of attention heads in each TAL.
- epochs, epochs for training;
- patience, patience for early stopping;
- target_node, type indices of target nodes for classification. When only one index is set (--target_node 0), HetSANN without multi-task learning is used. Here 0 indicates the paper and 1 indicates the author;
- target_is_multilabels, each type of target node for classification is multi-labels or not (0 means no, the other means yes);
- layers, number of TAL.

License
----------
AI Labs, Didi Chuxing, Beijing, China.
