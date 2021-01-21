# RSHN code

Adapted from [CheriseZhu/RSHN](https://github.com/CheriseZhu/RSHN).

We add GCN and GAT comparison and tried to reproduce the result in the RSHN paper.

## running environment

* Python 3.8.5
* torch 1.4.0 cuda 10.1
* dgl 0.5.2 cuda 10.1
* torch_geometric 1.6.1 cuda 10.1 with latest torch_sparse etc. (Install as guided in [here](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html))

## running procedure

* Download data.zip from [tsinghua-cloud](https://cloud.tsinghua.edu.cn/d/68cde69f3bdd401ebaa9/files/?p=%2Fdata.zip)
* unzip data.zip to the current directory
* cd model
* run RSHN_gnn.py

```bash
python RSHN.py --dataset IMDB --lr 0.01 --weight_decay 5e-4 --dim 16 --num_node_layer 2 --num_edge_layer 2 --dropout 0.6 --epoch 100
python RSHN.py --dataset DBLP --lr 0.01 --weight_decay 5e-4 --dim 16 --num_node_layer 2 --num_edge_layer 2 --dropout 0.6 --epoch 100
python RSHN.py --dataset ACM --lr 0.01 --weight_decay 5e-6 --dim 32 --num_node_layer 3 --num_edge_layer 3 --dropout 0.6 --epoch 100
```

## performance report

It's very unreasonable that no validation set is split and the best score in test set among all epoches are reported. However, we follow this setting as in paper in this experiment.

|       | GCN   | GAT       | RSHN      |
|-------|-------|-----------|-----------|
| AIFB  | 97.22 | **100.0** | 97.22     |
| MUTAG | 79.41 | 80.88     | **82.35** |
| BGS   | 96.55 | **100.0** | 93.1      |

***The following content is from the initial CheriseZhu/RSHN repo.***

# RSHN
The implementation of our ICDM 2019 paper "Relation Structure-Aware Heterogeneous Graph Neural Network" [RSHN](https://www.researchgate.net/publication/337473241_Relation_Structure-Aware_Heterogeneous_Graph_Neural_Network). [Slides](http://ddl.escience.cn/f/UW3L).

# Requirements
python == 3.6.2<br>
torch == 1.1.0<br>
numpy == 1.16.4<br>
scipy == 1.2.0<br>
torch_geometric == 1.0.0<br>
numba == 0.42.1

# How to use
  ### Dataset
  The data folder includes our propocessed data for training and testing. <br>
  The orginal datasets can be founded from [here](https://s3.us-east-2.amazonaws.com/dgl.ai/dataset).

  ### Model
  The model folder includes our proposed model "RSHN".<br>
  The build_coarsened_line_graph folder includes utils used in model.<br>
  The torch_geometeric/nn/conv folder includes the designed convolution layers used in model. 
  
  ### Training/Testing
  ```
  cd model
  python RSHN.py --dataset AIFB --lr 0.01 --weight_decay 5e-4 --dim 16 --num_node_layer 2 --num_edge_layer 1 --dropout 0.6 --epoch 50
  ```
  
  
# Citation
```
@inproceedings{zhu2019RSHN
author={Shichao Zhu and Chuan Zhou and Shirui Pan and Xingquan Zhu and Bin Wang},
title={Relation Structure-Aware Heterogeneous Graph Neural Network},
journal={IEEE International Conference On Data Mining (ICDM)},
year={2019}
}
```
