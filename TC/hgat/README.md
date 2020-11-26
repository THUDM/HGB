An implement of EMNLP 2019 paper "[Heterogeneous Graph Attention Networks for Semi-supervised Short Text Classification](http://shichuan.org/doc/74.pdf)". 

Thank you for your interest in our work!  :smile:



# Requirements

- Anaconda3 (python 3.6)
- Pytorch 1.3.1
- gensim  3.6.0



# Easy Run

```
cd ./model/code/
python train.py
```

You may change the dataset by modifying the variable "dataset = 'example'" in the top of the code "train.py" or use arguments (see train.py). 

Our datasets can be downloaded from [Google Drive](https://drive.google.com/open?id=1pz1IMdJqkKidD7eEc3T_2-VkrUhkUKd4).   PS: I have accidentally deleted some files, but I tried to restore them, hope they will run correctly. 



# Prepare for your own dataset

The following files are required:

    ./model/data/YourData/
        ---- YourData.cites                // the adjcencies
        ---- YourData.content.text         // the features of texts
        ---- YourData.content.entity       // the features of entities
        ---- YourData.content.topic        // the features of topics
        ---- train.map                     // the index of the training node
        ---- vali.map                      // the index of the validation nodes
        ---- test.map                      // the index of the testing nodes

The format is as following:

- **YourData.cites**

  Each line contains an edge:     "idx1\tidx2\n".        eg: "98	13"

- **YourData.content.text**

  Each line contains a node:    "idx\t[features]\t[category]\n", note that the [features] is a list of floats with '\t' as the delimiter.      eg:    "59	1.0	0.5	0.751	0.0	0.659	0.0	computers"
  If used for multi-label classification,  [category] must be one-hot with space as delimiter,       eg:   "59	1.0	0.5	0.751	0.0	0.659	0.0	0 1 1 0 1 0".

 - **YourData.content.entity**

   Similar with .text, just change the [category] to "entity".		eg: "13	0.0	0.0	1.0	0.0	0.0	entity"

 - **YourData.content.topic**

   Similar with .text, just change the [category] to "topic".		eg: "64	0.10	1.21	8.09	0.10	topic"

 - ***.map**

   Each line contains an index:     "idx\n".              eg:  "98"

You can see the example in ./model/data/example/*

----

A simple data preprocessing code is provided. Successfully running it requires a token of [tagme](https://sobigdata.d4science.org/web/tagme/tagme-help "TagMe")'s account  (my personal token is provided  in tagme.py, but may be invalid in the future), [Wikipedia](https://dumps.wikimedia.org/ "WikiPedia")'s entity descriptions, and a word2vec model containing entity embeddings. You can prepare them yourself or obtain our files from [Google Drive](https://drive.google.com/open?id=1v9GD5ezHGbekoLDw5aAzh6-C-QUS-j93) and unzip them to ./data/ .

Then, you should prepare a data file like ./data/example/example.txt, whose format is:         "[idx]\t[category]\t[content]\n". 

Finally, modify the variable "dataset = 'example'" in the top of following codes and run:

```
python tagMe.py
python build_network.py
python build_features.py
python build_data.py
```



# Use HGAT as GNN

If you just wanna use the HGAT model as a graph neural network, you can just prepare some files following the above format:

     ./model/data/YourData/
        ---- YourData.cites                // the adjcencies
        ---- YourData.content.*            // the features of *, namely node_type1, node_type2, ...
        ---- train.map                     // the index of the training node
        ---- vali.map                      // the index of the validation nodes
        ---- test.map                      // the index of the testing nodes

And change the   "load_data()"  in ./model/code/utils.py

```
type_list = [node_type1, node_type2, ...]
type_have_label = node_type
```

See the codes for more details.



# Citation

If you make advantage of the HGAT model in your research, please cite the following in your manuscript:

```
@inproceedings{linmei2019heterogeneous,
  title={Heterogeneous graph attention networks for semi-supervised short text classification},
  author={Linmei, Hu and Yang, Tianchi and Shi, Chuan and Ji, Houye and Li, Xiaoli},
  booktitle={Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)},
  pages={4823--4832},
  year={2019}
}
```