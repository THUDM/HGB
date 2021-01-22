# new baseline for benchmark


```
python run_new.py --dataset LastFM
python run_new.py --dataset LastFM_magnn
python run_dist.py --dataset amazon
python run_dist.py --dataset PubMed --batch-size 8192
```

## running environment

* torch 1.6.0 cuda 10.1
* dgl 0.4.3 cuda 10.1
* networkx 2.3
* scikit-learn 0.23.2
* scipy 1.5.2
