# Heterogeneous Graph Benchmark

Revisiting, benchmarking, and refining Heterogeneous Graph Neural Networks.

**2023.3.2 update**: We make benchmark data including test set pulic. You can download data as follows:

* Node Classification: https://cloud.tsinghua.edu.cn/d/a2728e52cd4943efa389/
* Link Prediction: https://cloud.tsinghua.edu.cn/d/10974f42a5ab46b99b88/

Therefore, you can get your metric scores locally. Anyway, try not to overfit.

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

* [Node classification benchmark](./NC/benchmark), [NC-baseline](./NC/benchmark/methods/baseline)
* [Link prediction benchmark](./LP/benchmark), [LP-baseline](./LP/benchmark/methods/baseline)
* [Knowledge-aware recommendation benchmark](./Recom), [Recom-baseline](./Recom/baseline)

**We only make half of test labels public to prevent data leakage issues.** The public half data is to help you debug locally. If you want to obtain test scores, you need to **rename all the xxx_full to xxx in data folder** and submit your prediction to our [website](https://www.biendata.xyz/hgb/).

For node classification and link prediction tasks, you can submit online. But for recommendation task, since the prediction files are too large to submit, you have to test offline by yourself.

**If you want to show your method on our official leaderboard on HGB website, you can make an issue on this repo. Once your code or paper is verified, your method will be displayed on the official leaderboard.**

### FAQ

1. **Where is the code for all HGNNs in paper for HGB?**

Follow this roadmap in this repo:

```
NC or LP
|--benchmark
    |--methods
        |--HGNNs
```

2. **How to take part in HGB?**

See [Get Started](https://www.biendata.xyz/hgb/#/about).

3. **Why can't I obtain test score locally?**

To prevent data leakage issues, we randomly replace the test set labels. Therefore, you can only get test scores after you submit your prediction to biendata competitions.

4. **What is the format of training data and submission files?**

You can read the instructions in biendata competitions in "Data" and "Evaluation" panel. What should be noticed is that, your prediction files should be on the top level of the zipped file.

For example, you should submit a zip like this:

```
submit.zip
|--ACM_1.txt
|--ACM_2.txt
|--...
```

Instead of

```
submit.zip
|--submit/
    |--ACM_1.txt
    |--ACM_2.txt
    |--...
```

It is recommended that zip a file using ```zip``` command rather than right click. Because subfolder may be automatically built for some operating systems when using right click.

### More

**This repo is actively under development.** Therefore, there are some extra experiments in this repo beyond our paper, such as graph-based text classification. For more information, see our [website](https://www.biendata.xyz/hgb/). Welcome contribute new tasks, datasets, methods to HGB!

Moreover, we also have an implementation of Simple-HGN in [cogdl](https://github.com/THUDM/cogdl/tree/master/examples/simple_hgn).


## Citation

* **Title:** Are we really making much progress? Revisiting, benchmarking and refining the Heterogeneous Graph Neural Networks.
* **Authors:** Qingsong Lv\*, Ming Ding\*, Qiang Liu, Yuxiang Chen, Wenzheng Feng, Siming He, Chang Zhou, Jianguo Jiang, Yuxiao Dong, Jie Tang.
* **In proceedings:** KDD 2021.
