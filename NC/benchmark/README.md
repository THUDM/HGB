# benchmark

benchmark data loader and evaluation scripts

## data

Warning: As we have opened test data, you should try not to overfit or leak data during training.

## data format

* All ids begin from 0.
* Each node type takes a continuous range of node_id.
* node_id and node_type id are with same order. I.e. nodes with node_type 0 take the first range of node_ids, nodes with node_type 1 take the second range, and so on.
* One-hot node features can be omited.