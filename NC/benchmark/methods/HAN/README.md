# HAN code

Adapted from [dgl/han](https://github.com/dmlc/dgl/tree/master/examples/pytorch/han).

We add benchmark support.

## running environment

* Python 3.8.5
* torch 1.4.0 cuda 10.1
* dgl 0.5.2 cuda 10.1


```bash
python main.py --dataset DBLP
python main.py --dataset ACM
python main.py --dataset Freebase --device cpu --num_epochs 1000
python main_multi.py --dataset IMDB
```

***The following content is from the initial dgl/han repo.***

# Heterogeneous Graph Attention Network (HAN) with DGL

This is an attempt to implement HAN with DGL's latest APIs for heterogeneous graphs.
The authors' implementation can be found [here](https://github.com/Jhy1993/HAN).

## Usage

`python main.py` for reproducing HAN's work on their dataset.

`python main.py --hetero` for reproducing HAN's work on DGL's own dataset from
[here](https://github.com/Jhy1993/HAN/tree/master/data/acm).  The dataset is noisy
because there are same author occurring multiple times as different nodes.

## Performance

Reference performance numbers for the ACM dataset:

|                     | micro f1 score | macro f1 score |
| ------------------- | -------------- | -------------- |
| Paper               | 89.22          | 89.40          |
| DGL                 | 88.99          | 89.02          |
| Softmax regression (own dataset) | 89.66  | 89.62     |
| DGL (own dataset)   | 91.51          | 91.66          |

We ran a softmax regression to check the easiness of our own dataset.  HAN did show some improvements.
