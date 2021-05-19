# Heterogeneous Graph Benchmark

Revisiting, benchmarking, and refining Heterogeneous Graph Neural Networks.

**This repo is actively under-development.** Therefore, there are some extra experiments in this repo beyond our paper, such as graph-based text classification. Moreover, we are adapting HGB with [cogdl](https://github.com/THUDM/cogdl). For more information, see our website (under construction).

## Roadmap

We organize our repo by task. Each folder contains a task and the structure is similar, i.e. reproducing experiments for each method in each sub-folder and benchmark experiments in benchmark sub-folder. Currently, we have four tasks, i.e. node classification (NC), link prediction (LP), knowledge-aware recommendation (Recom) and text classification (TC).

```bash
task/
    method1/ (experiments in the original paper)
    method2/
    ...
    method_n/
    benchmark/ (experiments for benchmark)
        scripts/
            data_loader.py
        methods/
            method1/
            method2/
            ...
            method_n/
```

## Citation

**Title:** Are we really making much progress? Revisiting, benchmarking and refining the Heterogeneous Graph Neural Networks.

**Authors:** Qingsong Lv\*, Ming Ding\*, Qiang Liu, Yuxiang Chen, Wenzheng Feng, Siming He, Chang Zhou, Jian-guo Jiang, Yuxiao Dong, Jie Tang.

**In preceedings:** KDD 2021.
