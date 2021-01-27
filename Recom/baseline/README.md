# new baseline for recommendation

Adapted from [xiangwang1223/knowledge_graph_attention_network](https://github.com/xiangwang1223/knowledge_graph_attention_network/tree/master/Model).

## running environment

* torch and dgl latest

## running procedure

* Download Data folder from [tsinghua-cloud](https://cloud.tsinghua.edu.cn/d/2bafd2674d5d43299dfa/) and unzip to current folder
* Download pretrain folder from [tsinghua-cloud](https://cloud.tsinghua.edu.cn/d/2bafd2674d5d43299dfa/) and unzip to Model folder
* cd to Model folder and run

## run

```bash
python main.py --model_type baseline --dataset movie-lens --layer_size [64,32,16] --embed_size 64 --lr 0.0001 --epoch 1000 --verbose 1 --save_flag 1 --pretrain -1 --batch_size 8192 --gpu_id 0
python main.py --model_type baseline --dataset last-fm --layer_size [64,32,16] --embed_size 64 --lr 0.0001 --epoch 1000 --verbose 1 --save_flag 1 --pretrain -1 --batch_size 8192 --gpu_id 0
python main.py --model_type baseline --dataset yelp2018 --layer_size [64,32,16] --embed_size 64 --lr 0.0001 --epoch 1000 --verbose 1 --save_flag 1 --pretrain -1 --batch_size 8192 --gpu_id 0
python main.py --model_type baseline --dataset amazon-book --layer_size [64,32,16] --embed_size 64 --lr 0.0001 --epoch 1000 --verbose 1 --save_flag 1 --pretrain -1 --batch_size 8192 --gpu_id 0
```
