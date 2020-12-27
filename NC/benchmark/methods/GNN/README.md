# GCN and GAT for benchmark

```
python run_DBLP.py --model-type gat
python run_DBLP.py --model-type gcn --weight-decay 1e-6 --lr 1e-3
```

## running environment

* torch 1.6.0 cuda 10.1
* dgl 0.4.3 cuda 10.1
* networkx 2.3
* scikit-learn 0.23.2
* scipy 1.5.2