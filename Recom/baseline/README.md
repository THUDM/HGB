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
python main.py --model_type baseline --dataset movie-lens --layer_size [64,32,16] --embed_size 64 --lr 0.0001 --epoch 1000 --verbose 1 --save_flag 1 --pretrain -1 --batch_size 1024 --alpha 0 --gpu_id 0
python main.py --model_type baseline --dataset last-fm --layer_size [64,32,16] --embed_size 64 --lr 0.0001 --epoch 1000 --verbose 1 --save_flag 1 --pretrain -1 --batch_size 1024 --alpha 0 --gpu_id 0
python main.py --model_type baseline --dataset yelp2018 --layer_size [64,32,16] --embed_size 64 --lr 0.0001 --epoch 1000 --verbose 1 --save_flag 1 --pretrain -1 --batch_size 1024 --alpha 1 --gpu_id 0
python main.py --model_type baseline --dataset amazon-book --layer_size [64,32,16] --embed_size 64 --lr 0.0001 --epoch 1000 --verbose 1 --save_flag 1 --pretrain -1 --batch_size 1024 --alpha 1 --gpu_id 0
```

This dgl implemenation is less efficient.

For alpha=1, you can just run the following lines in [here](https://github.com/null-id/HeterZoo/tree/master/Recom/KGAT), which is a tensorflow implementation equivalent to here, but more efficient.

```bash
python Main.py --model_type kgat --alg_type gcn --dataset yelp2018 --regs [1e-5,1e-5] --layer_size [64,32,16] --embed_size 64 --lr 0.0001 --epoch 1000 --verbose 50 --save_flag 1 --pretrain -1 --batch_size 1024 --node_dropout [0.1] --mess_dropout [0.1,0.1,0.1] --use_att True --att_type rgat --gpu_id 0
python Main.py --model_type kgat --alg_type gcn --dataset amazon-book --regs [1e-5,1e-5] --layer_size [64,32,16] --embed_size 64 --lr 0.0001 --epoch 1000 --verbose 50 --save_flag 1 --pretrain -1 --batch_size 1024 --node_dropout [0.1] --mess_dropout [0.1,0.1,0.1] --use_att True --att_type rgat --gpu_id 0
```

For alpha=0, you can use larger batch size to accelerate, but will get a slightly lower performance.

```bash
python main.py --model_type baseline --dataset movie-lens --layer_size [64,32,16] --embed_size 64 --lr 0.0001 --epoch 1000 --verbose 1 --save_flag 1 --pretrain -1 --batch_size 8192 --alpha 0 --gpu_id 0
python main.py --model_type baseline --dataset last-fm --layer_size [64,32,16] --embed_size 64 --lr 0.0001 --epoch 1000 --verbose 1 --save_flag 1 --pretrain -1 --batch_size 8192 --alpha 0 --gpu_id 0
```