# Heterogeneous Graph Benchmark

Revisiting, benchmarking, and refining Heterogeneous Graph Neural Networks.

## Roadmap

We organize our repo by task, and one sub-folder per task. Currently, we have four tasks, i.e., node classification (NC), link prediction (LP), knowledge-aware recommendation (Recom) and text classification (TC).

### Revisiting

This part refers to Section 3 and Table 1 in our paper.

* [HAN](./NC/HAN)
* [GTN](./NC/GTN)
* [RSHN](./NC/RSHN)
* [HetGNN](./NC/HetGNN)
* [MAGNN-nc](./NC/MAGNN) and [MAGNN-lp](./LP/benchmark/methods/MAGNN_ini)

### Benchmarking and Refining

This part refers to Section 4,5,6 in our paper.

* [Node classification](./NC/benchmark)
* [Link prediction](./LP/benchmark)
* [Knowledge-aware recommendation](./Recom/baseline)

**You should notice that the test data labels are randomly replaced to prevent data leakage issues.** If you want to obtain test scores, you need to submit your prediction to our [website](https://www.biendata.xyz/hgb/).

### More

**This repo is actively under development.** Therefore, there are some extra experiments in this repo beyond our paper, such as graph-based text classification. For more information, see our [website](https://www.biendata.xyz/hgb/). Welcome contribute new tasks, datasets, methods to HGB!

Moreover, we also have an implementation of Simple-HGN in [cogdl](https://github.com/THUDM/cogdl/tree/master/examples/simple_hgn).


## Citation

* **Title:** Are we really making much progress? Revisiting, benchmarking and refining the Heterogeneous Graph Neural Networks.
* **Authors:** Qingsong Lv\*, Ming Ding\*, Qiang Liu, Yuxiang Chen, Wenzheng Feng, Siming He, Chang Zhou, Jianguo Jiang, Yuxiao Dong, Jie Tang.
* **In preceedings:** KDD 2021.
