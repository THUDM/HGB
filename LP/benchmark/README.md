# benchmark

benchmark data loader and evaluation scripts

## data

Data can be downloaded from [tsinghua-cloud](https://cloud.tsinghua.edu.cn/d/fc10cb35d19047a88cb1/)

## data format

* All ids begin from 0.
* Each node type takes a continuous range of node_id.
* node_id and node_type id are with same order. I.e. nodes with node_type 0 take the first range of node_ids, nodes with node_type 1 take the second range, and so on.
* One-hot node features can be omited.