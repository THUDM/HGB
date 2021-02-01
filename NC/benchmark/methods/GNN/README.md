# GCN and GAT for benchmark

(To be tuned)

```
python run.py --dataset DBLP --model-type gat
python run.py --dataset DBLP --model-type gcn --weight-decay 1e-6 --lr 1e-3

python run.py --dataset ACM --model-type gat --feats-type 2
python run.py --dataset ACM --model-type gcn --weight-decay 1e-6 --lr 1e-3 --feats-type=0

python run.py --dataset Freebase --model-type gat
python run.py --dataset Freebase --model-type gcn

python run_multi.py --dataset IMDB --model-type gat --feats-type 0 --num-layers 4
python run_multi.py --dataset IMDB --model-type gcn --feats-type 0 --num-layers 3
```

## running environment

* torch 1.6.0 cuda 10.1
* dgl 0.4.3 cuda 10.1
* networkx 2.3
* scikit-learn 0.23.2
* scipy 1.5.2
