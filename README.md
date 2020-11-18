# HeterZoo
Revisiting and Benchmarking Homo- and Hetero-geneous Network Embedding Methods.

## Roadmap

The code under differenct tasks (even different datasets) are separatedly organized at present. (Maybe tidyup is needed later on, but maybe not necessary, since different models may have conflicts on relied packages).

### Node classification

This part is in NC folder.

We mainly focuse on supervised learning setting for node classification task for now. (Maybe unsupervised setting in [HNE](https://github.com/yangji9181/HNE) should also be added here later.)

If we compare our method on the existing datasets with published methods, we will use the same settings in the corresponding papers.

If we compare our method on our own datasets:

* For unsupervised embedding methods, we train an SVM by separating labeled nodes with 2:1:7 (train:valid:test) on the learnt embedding.
* For semi-supervised embedding methods, we train the model by separating labeled nodes with 2:1:7 (train:valid:test).

### Link prediction

todo

### Text classification

This part is in TC folder.

### Recommendation

This part is in Recom folder.

### Alignment

todo
