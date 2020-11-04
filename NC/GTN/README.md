# GTN code

Adapted from [seongjunyun/Graph_Transformer_Networks](https://github.com/seongjunyun/Graph_Transformer_Networks).

We add GCN and GAT comparison and tried to reproduce the result in the GTN paper.

## running environment

* Python 3.8.5
* torch 1.4.0 cuda 10.1
* torch_geometric 1.6.1 cuda 10.1 with latest torch_sparse etc. (Install as guided in [here](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html))
* torch-sparse-old cuda 10.1 (Build from source as guided in [here](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html))

## running procedure

* Download data.zip from [tsinghua-cloud](https://cloud.tsinghua.edu.cn/d/c105c8b92c5549fa95cc/files/?p=%2Fdata.zip)
* mkdir data
* Move data.zip to data folder and unzip it
* run main_gnn.py

```bash
python main_gnn.py --dataset DBLP --model gcn
python main_gnn.py --dataset DBLP --model gat
python main_gnn.py --dataset ACM --model gcn --num_layers 2
python main_gnn.py --dataset ACM --model gat --num_layers 2
python main_gnn.py --dataset IMDB --model gcn
python main_gnn.py --dataset IMDB --model gat --weight_decay 0.03
```

## performance report

We repeat 5 times and report the average Macro-F1 for each model and each dataset.

|      | GCN       | GAT       | GTN              |
|------|-----------|-----------|------------------|
| DBLP | **91.48** | 94.18     | running with cpu |
| ACM  | 92.28     | **92.49** | running with cpu |
| IMDB | **59.11** | 58.86     | 57.53            |

***The following content is from the initial seongjunyun/Graph_Transformer_Networks repo.***

# Graph Transformer Networks
This repository is the implementation of [Graph Transformer Networks(GTN)](https://arxiv.org/abs/1911.06455).

> Seongjun Yun, Minbyul Jeong, Raehyun Kim, Jaewoo Kang, Hyunwoo J. Kim, Graph Transformer Networks, In Advances in Neural Information Processing Systems (NeurIPS 2019).

![](https://github.com/seongjunyun/Graph_Transformer_Networks/blob/master/GTN.png)

## Installation

Install [pytorch](https://pytorch.org/get-started/locally/)

Install [torch_geometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)
``` 
$ pip install torch-sparse-old
```
** The latest version of torch_geometric removed the backward() of the multiplication of sparse matrices (spspmm), so to solve the problem, we uploaded the old version of torch-sparse with backward() on pip under the name torch-sparse-old.

## Data Preprocessing
We used datasets from [Heterogeneous Graph Attention Networks](https://github.com/Jhy1993/HAN) (Xiao Wang et al.) and uploaded the preprocessing code of acm data as an example.

## Running the code
``` 
$ mkdir data
$ cd data
```
Download datasets (DBLP, ACM, IMDB) from this [link](https://drive.google.com/file/d/1qOZ3QjqWMIIvWjzrIdRe3EA4iKzPi6S5/view?usp=sharing) and extract data.zip into data folder.
```
$ cd ..
```
- DBLP
```
$ python main.py --dataset DBLP --num_layers 3
```
- ACM
```
 $ python main.py --dataset ACM --num_layers 2 --adaptive_lr true
```
- IMDB
```
 $ python main_sparse.py --dataset IMDB --num_layers 3 --adaptive_lr true
```

## Citation
If this work is useful for your research, please cite our [paper](https://arxiv.org/abs/1911.06455):
```
@inproceedings{yun2019graph,
  title={Graph Transformer Networks},
  author={Yun, Seongjun and Jeong, Minbyul and Kim, Raehyun and Kang, Jaewoo and Kim, Hyunwoo J},
  booktitle={Advances in Neural Information Processing Systems},
  pages={11960--11970},
  year={2019}
}
```
