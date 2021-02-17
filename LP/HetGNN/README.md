# HAN code

Adapted from [chuxuzhang/KDD2019_HetGNN](https://github.com/chuxuzhang/KDD2019_HetGNN).

We add GCN and GAT comparison.

## running environment

* Python 3.6.11
* numpy 1.19.2
* torch 1.6.0 cuda 10.1
* torch_geometric 1.6.1

## running procedure

* Download academic_test.rar from [Google Drive](https://drive.google.com/file/d/1BEb34-06XSwNZZHF4Gi4n8-buzoJpZs0/view?usp=sharing)
* Unzip academic_test.rar to the ./data/
* run homoGNN.py

```bash
python homoGNN.py --model GCN --data_name colab/cite 
python homoGNN.py --model GAT --data_name colab/cite
``` 

## performance report

| LP:author-author colab |     AUC   |    F1     |
| ---------------------- | --------- | --------- |
| HetGNN(reproduction)   | 0.7594     | 0.7130     |
| GCN                    | 0.8180| 0.7529| 
| GAT                    | **0.8397**    | **0.7700**    |   

| LP:author-paper cite     |    AUC       |    F1     |
| ------------------------ | ------------ | --------- |
| HetGNN(reproduction)     | 0.7839       | **0.7661**     |
| GCN                     | 0.7789   | 0.7328|  
| GAT                      | **0.8433**         | 0.7420      | 

***The following content is from the initial chuxuzhang/KDD2019_HetGNN repo.***

<1> Introduction 

code of HetGNN in KDD2019 paper: Heterogeneous Graph Neural Network 


<2> How to use

python HetGNN.py [parameters]

(enable GPU: python HetGNN.py --cuda 1)

#test data used in academic_test folder (academic-2 data used in this paper, T_s = 2012): (author) A_n - 28646, (paper) P_n - 21044, (venue) V_n - 18

test data link: https://drive.google.com/file/d/1N6GWsniacaT-L0GPXpi1D3gM2LVOih-A/view?usp=sharing


<3> Data requirement

a_p_list_train.txt: paper neighbor list of each author in training data

p_a_list_train.txt: author neighbor list of each paper in training data

p_p_citation_list.txt: paper citation neighbor list of each paper 

v_p_list_train.txt: paper neighbor list of each venue in training data

p_v.txt: venue of each paper

p_title_embed.txt: pre-trained paper title embedding

p_abstract_embed.txt: pre-trained paper abstract embedding

node_net_embedding.txt: pre-trained node embedding by network embedding

het_neigh_train.txt: generated neighbor set of each node by random walk with re-start 

het_random_walk.txt: generated random walks as node sequences (corpus) for model training


<4> Model evaluation for different applications

4-1 link prediction (author-author collaboration, type-1)

step-1: run input_data_class.a_a_collaborate_train_test() in input_data_process.py to set author-author collaboration train/test data 

step-2: run [author collaboration link prediction] part in application.py to obtain evalution result 

4-2 link prediction (author-paper citation, type-2)

step-1: run input_data_class.a_p_citation_train_test() in input_data_process.py to set author-paper citation train/test data

step-2: run [author paper citation link prediction] part in application.py to obtain evalution result 

4-3 node recommendation (venue recommendation)

step-1: run input_data_class.a_v_train_test() in input_data_process.py to set author-venue train/test data

step-2: run [venue recommendation] part in application.py to obtain evalution result 

4-4 node classification/clustering (author classification/clustering)

step-1: run [author classification/clustering] part in application.py to obtain evalution result 

<5> Others

5-1 raw_data_process.py: raw data (academic_small.txt) processing and data preparation

5-2 input_data_process.py: generating het_rand_walk for model training, train/test data for different tasks, etc. 

5-3 DeepWalk.py: generating pre-train node embedding 

5-4 If you find code useful, please consider citing our work.

Heterogeneous Graph Neural Network

Zhang, Chuxu and Song, Dongjin and Huang, Chao and Swami, Ananthram and Chawla, Nitesh V.

Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, KDD '19

5-5 For more information, contact: Chuxu Zhang (chuxuzhang@gmail.com)