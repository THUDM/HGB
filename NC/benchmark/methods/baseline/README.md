# new baseline for benchmark

For message passing with relation attention version:

```
python run_new.py --dataset DBLP
python run_new.py --dataset ACM --feats-type 2
python run_multi.py --dataset IMDB --feats-type 2
```

For relational gat version:

```
python run.py --dataset DBLP
```

## running environment

* torch 1.6.0 cuda 10.1
* dgl 0.4.3 cuda 10.1
* networkx 2.3
* scikit-learn 0.23.2
* scipy 1.5.2
