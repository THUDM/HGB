# HGT code

Adapted from [HGT-DGL](https://github.com/acbull/HGT-DGL).

## running environment

* Python 3.7
* torch 1.7.0
* dgl 0.5.2

## running procedure

* download data from [tsinghua-cloud](https://cloud.tsinghua.edu.cn/d/8b9644cfa8344f26878c/)
* cd to HGT/
* unzip all zip files
* run scripts
* mkdir checkpoint

```scripts
sh run_LastFM.sh
sh run_PubMed.sh
sh run_amazon.sh
sh run_LastFM_magnn.sh
```