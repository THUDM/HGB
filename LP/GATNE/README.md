# GATNE code

Adapted from [THUDM/GATNE](git@github.com/THUDM/GATNE).

We add GCN and GAT comparison.

## running environment

* Python 3.6.11
* numpy 1.19.2
* torch 1.6.0 cuda 10.1
* torch_geometric 1.6.1

## running procedure

* Download data.zip from [tsinghua-cloud](https://cloud.tsinghua.edu.cn/d/9d5d74a6d1884919a0be/)
* mkdir data
* Unzip data.zip to the data/
* cd src
* run gen_hom_data.py
```bash
python gen_hom_data.py twitter/amazon/youtube
```
* run homGNN.py
```bash
python homGNN.py twitter/amazon/youtube GCN/GAT
```

## performance report
dataset: Amazon   

|                      |     ROC-AUC   |    PR-AUC     | F1 |
| ---------------------- | --------- | --------- | --------- |
| GATNE-T(in paper)   | 97.44     | 97.05   | 92.87  |
| GATNE-T(reproduction)   | 97.00     | 96.55   | 91.79  |
| GCN        | **98.96** | **98.70**|  95.60 |
| GAT        | 98.85 | 98.44|  **95.62** |
train history:  
dismult GCN avg_auc:0.9893726487499943, avg_pr:0.9866278993230313, avg_f1:0.9550082769803161  
dismult GCN avg_auc:0.9888561379644261, avg_pr:0.9862258321495022, avg_f1:0.954416872065084  
dismult GCN avg_auc:0.9892315075449153, avg_pr:0.9870857320071876, avg_f1:0.9550479138984123  
dismult GCN avg_auc:0.9901102838328097, avg_pr:0.9876929230388979, avg_f1:0.9574298671000376  
dismult GCN avg_auc:0.9901687745103358, avg_pr:0.9877416651704218, avg_f1:0.9580603841239973  

dismult GAT avg_auc:0.9880027092297872, avg_pr:0.9839058255584743, avg_f1:0.9548638162962442  
dismult GAT avg_auc:0.9889904320924259, avg_pr:0.9854195267710879, avg_f1:0.9575927605622812  
dismult GAT avg_auc:0.9886742488496425, avg_pr:0.9846178597635857, avg_f1:0.956501737128217  
dismult GAT avg_auc:0.9883389311074906, avg_pr:0.9843169010797554, avg_f1:0.9548464331368097  
dismult GAT avg_auc:0.9885978249510003, avg_pr:0.9839523820100493, avg_f1:0.9570627215144385  

dataset: Youtube

|                      |     ROC-AUC   |    PR-AUC     | F1 |
| ---------------------- | --------- | --------- | --------- |
| GATNE-T(in paper)   |84.61     | 81.93   | 76.83  |
| GATNE-T(reproduction)   | 83.86     | 81.70   | 76.33  |
| GCN        |  90.74 | **90.35**|  83.58 |
| GAT        |  **90.79** | 89.70|  **85.43** |
train history:  
dismult GCN avg_auc:0.9094494361436617, avg_pr:0.9087728290404339, avg_f1:0.8388703931687967  
dismult GCN avg_auc:0.9242133636057691, avg_pr:0.9160412505739988, avg_f1:0.8489381353724885  
dismult GCN avg_auc:0.9058114262218171, avg_pr:0.9035770438139551, avg_f1:0.8354748445599363  
dismult GCN avg_auc:0.9011731963059322, avg_pr:0.8969737772606987, avg_f1:0.8309293242595783  
dismult GCN avg_auc:0.8966789444129539, avg_pr:0.8924226168006543, avg_f1:0.8249906791590677  

dismult GAT avg_auc:0.9033363254015534, avg_pr:0.8915774041961573, avg_f1:0.8404788327644999  
dismult GAT avg_auc:0.9139027818107369, avg_pr:0.902460352548726, avg_f1:0.8565797802312906  
dismult GAT avg_auc:0.9068145692214784, avg_pr:0.8979265857018127, avg_f1:0.8446799217179949  
dismult GAT avg_auc:0.9157110929358415, avg_pr:0.9029699204084523, avg_f1:0.8609902329960291  
dismult GAT avg_auc:0.899973953630429, avg_pr:0.8898476160506157, avg_f1:0.8386576275549906  

dataset: Twitter  

|                      |     ROC-AUC   |    PR-AUC     | F1 |
| ---------------------- | --------- | --------- | --------- |
| GATNE-T(in paper)   | 92.30    | 91.77   | 84.96  |
| GATNE-T(reproduction)   | 92.45     | 92.36   | 85.43  |
| GCN        |  **97.13** | **96.81** |  **91.77** |
| GAT        |  97.05** | 95.70 |  91.68 |
train hitory:  
dismult GCN avg_auc:0.9715256880161214, avg_pr:0.9681166609804589, avg_f1:0.919419160400621  
dismult GCN avg_auc:0.9703256890922464, avg_pr:0.9666125976775855, avg_f1:0.9151572346190143  
dismult GCN avg_auc:0.9716783403138014, avg_pr:0.968904670499224, avg_f1:0.9178970440500472  
dismult GCN avg_auc:0.971561759796554, avg_pr:0.9690990187942997, avg_f1:0.9175926207799325  
dismult GCN avg_auc:0.9714450926295584, avg_pr:0.9680136386144598, avg_f1:0.9183536789552194  
 
dismult GAT avg_auc:0.9704444261224763, avg_pr:0.9667555006599379, avg_f1:0.9164358123534961  
dismult GAT avg_auc:0.9707723134029902, avg_pr:0.9667198824487546, avg_f1:0.9190538524764833  
dismult GAT avg_auc:0.9709207870513206, avg_pr:0.9681788907203988, avg_f1:0.9181710249931505  
dismult GAT avg_auc:0.970365882064442, avg_pr:0.9674064814816721, avg_f1:0.9146092727328076  
dismult GAT avg_auc:0.9698625628698485, avg_pr:0.9660000674366716, avg_f1:0.9159487351213127  
***The following content is from the initial THUDM/GATNE repo.***

# GATNE

### [Project](https://sites.google.com/view/gatne) | [Arxiv](https://arxiv.org/abs/1905.01669)

Representation Learning for Attributed Multiplex Heterogeneous Network.

[Yukuo Cen](https://sites.google.com/view/yukuocen), Xu Zou, Jianwei Zhang, [Hongxia Yang](https://sites.google.com/site/hystatistics/home), [Jingren Zhou](http://www.cs.columbia.edu/~jrzhou/), [Jie Tang](http://keg.cs.tsinghua.edu.cn/jietang/)

Accepted to KDD 2019 Research Track!

## ❗ News

Recent Updates (Nov. 2020):
- Use multiprocessing to speedup the random walk procedure (by `--num-workers`)
- Support saving/loading walk file (by `--walk-file`)
- The PyTorch version now supports node features (by `--features`)

Some Tips:
- Running on large-scale datasets needs to set a larger value for `batch-size` to speedup training (e.g., several hundred or thousand).
- If **out of memory (OOM)** occurs, you may need to decrease the values of `dimensions` and `att-dim`.

Our GATNE models have been implemented by many popular graph toolkits:
- Deep Graph Library ([DGL](https://github.com/dmlc/dgl)): see https://github.com/dmlc/dgl/tree/master/examples/pytorch/GATNE-T 
- Paddle Graph Learning ([PGL](https://github.com/PaddlePaddle/PGL)): see https://github.com/PaddlePaddle/PGL/tree/main/examples/GATNE
- [CogDL](https://github.com/THUDM/cogdl): see https://github.com/THUDM/cogdl/blob/master/cogdl/models/emb/gatne.py

Some recent papers have listed GATNE models as a strong baseline:
- [Deep Adversarial Completion for Sparse Heterogeneous Information Network Embedding](https://dl.acm.org/doi/pdf/10.1145/3366423.3380134) (WWW'20)
- [Decoupled Graph Convolution Network for Inferring Substitutable and Complementary Items](https://dl.acm.org/doi/pdf/10.1145/3340531.3412695) (CIKM'20)
- [Graph Attention Networks over Edge Content-Based Channels](https://dl.acm.org/doi/pdf/10.1145/3394486.3403233) (KDD'20)
- [Temporal heterogeneous interaction graph embedding for next-item recommendation](http://shichuan.org/doc/84.pdf) (PKDD'20)
- [Link Inference via Heterogeneous Multi-view Graph Neural Networks](https://link.springer.com/chapter/10.1007/978-3-030-59410-7_48) (DASFAA 2020)
- [Multi-View Collaborative Network Embedding](https://arxiv.org/pdf/2005.08189.pdf) (Arxiv, May 2020)

Please let me know if your toolkit includes GATNE models or your paper uses GATNE models as baselines. 

## Prerequisites

- Python 3
- TensorFlow >= 1.8 or PyTorch

## Getting Started

### Installation

Clone this repo.

```bash
git clone https://github.com/THUDM/GATNE
cd GATNE
```

Please first install TensorFlow or PyTorch, and then install other dependencies by

```bash
pip install -r requirements.txt
```

### Dataset

These datasets are sampled from the original datasets.

- Amazon contains 10,166 nodes and 148,865 edges. [Source](http://jmcauley.ucsd.edu/data/amazon)
- Twitter contains 10,000 nodes and 331,899 edges. [Source](https://snap.stanford.edu/data/higgs-twitter.html)
- YouTube contains 2,000 nodes and 1,310,617 edges. [Source](http://socialcomputing.asu.edu/datasets/YouTube)
- Alibaba contains 6,163 nodes and 17,865 edges.

### Training

#### Training on the existing datasets

You can use `./scripts/run_example.sh` or `python src/main.py --input data/example` or `python src/main_pytorch.py --input data/example` to train GATNE-T model on the example data. (If you share the server with others or you want to use the specific GPU(s), you may need to set `CUDA_VISIBLE_DEVICES`.) 

If you want to train on the Amazon dataset, you can run `python src/main.py --input data/amazon` or `python src/main.py --input data/amazon --features data/amazon/feature.txt` to train GATNE-T model or GATNE-I model, respectively. 

You can use the following commands to train GATNE-T on Twitter and YouTube datasets: `python src/main.py --input data/twitter --eval-type 1` or `python src/main.py --input data/youtube`. We only evaluate the edges of the first edge type on Twitter dataset as the number of edges of other edge types is too small.

As Twitter and YouTube datasets do not have node attributes, you can generate heuristic features for them, such as DeepWalk embeddings. Then you can train GATNE-I model on these two datasets by adding the `--features` argument.

#### Training on your own datasets

If you want to train GATNE-T/I on your own dataset, you should prepare the following three(or four) files:
- train.txt: Each line represents an edge, which contains three tokens `<edge_type> <node1> <node2>` where each token can be either a number or a string.
- valid.txt: Each line represents an edge or a non-edge, which contains four tokens `<edge_type> <node1> <node2> <label>`, where `<label>` is either 1 or 0 denoting an edge or a non-edge
- test.txt: the same format with valid.txt
- feature.txt (optional): First line contains two number `<num> <dim>` representing the number of nodes and the feature dimension size. From the second line, each line describes the features of a node, i.e., `<node> <f_1> <f_2> ... <f_dim>`.

If your dataset contains several node types and you want to use meta-path based random walk, you should also provide an additional file as follows:
- node_type.txt: Each line contains two tokens `<node> <node_type>`, where `<node_type>` should be consistent with the meta-path schema in the training command, i.e., `--schema node_type_1-node_type_2-...-node_type_k-node_type_1`. (Note that the first node type in the schema should equals to the last node type.)


If you have ANY difficulties to get things working in the above steps, feel free to open an issue. You can expect a reply within 24 hours.

## Cite

Please cite our paper if you find this code useful for your research:

```
@inproceedings{cen2019representation,
  title = {Representation Learning for Attributed Multiplex Heterogeneous Network},
  author = {Cen, Yukuo and Zou, Xu and Zhang, Jianwei and Yang, Hongxia and Zhou, Jingren and Tang, Jie},
  booktitle = {Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining},
  year = {2019},
  pages = {1358--1368},
  publisher = {ACM},
}
```