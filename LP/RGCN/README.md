# RGCN code

Adapted from [DGL example](https://github.com/dmlc/dgl/tree/master/examples/pytorch/rgcn).

We replace the Benchmark.

## running environment

* Python 3.7
* torch 1.7.0
* dgl 0.5.2

## running procedure

* Dataset will be downloaded automatically at ~/.dgl/.
* or you can download data from [tsinghua-cloud](https://cloud.tsinghua.edu.cn/d/8b9644cfa8344f26878c/) or [google-drive](https://drive.google.com/drive/folders/13o5dYuvpZWzgeUPVTLLtpYHAGss2sk_x?usp=sharing)
* unzip all zip files
* move them to ./data/
* cd to RGCN/
* run python file

```bash
python3 link_predict.py -d FB15k-237 --gpu 0 
python3 gnn_link_predict.py -d FB15k-237 --gpu 0 --model=gcn
python3 gnn_link_predict.py -d FB15k-237 --gpu 0 --model=gat
```