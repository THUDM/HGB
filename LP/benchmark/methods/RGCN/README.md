# RGCN code

Adapted from [DGL example](https://github.com/dmlc/dgl/tree/master/examples/pytorch/rgcn).

We replace the GNN module in paper by GCN for comparison.

## running environment

* Python 3.7
* torch 1.7.0
* dgl 0.5.2
* nvidia-ml-py3 7.352.0

## running procedure

* Dataset will be downloaded automatically at ~/.dgl/.
* or you can download data from [tsinghua-cloud](https://cloud.tsinghua.edu.cn/d/8b9644cfa8344f26878c/)
* unzip all zip files
* move them to ~/.dgl/
* cd to RGCN/
* run python file

```bash
python link_predict.py --dataset=LastFM
python link_predict.py --dataset=amazon  --hidden-dim=60
python link_predict.py --dataset=PubMed  --hidden-dim=60
python link_predict.py --dataset=LastFM_magnn
```