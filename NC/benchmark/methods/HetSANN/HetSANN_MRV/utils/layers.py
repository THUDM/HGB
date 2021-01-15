import numpy as np
import tensorflow as tf

import pdb

conv1d = tf.layers.conv1d

def attn_head(seq, out_sz, bias_mat, activation, in_drop=0.0, coef_drop=0.0, residual=False):
    with tf.name_scope('my_attn'):
        if in_drop != 0.0:
            seq = tf.nn.dropout(seq, 1.0 - in_drop)

        seq_fts = tf.layers.conv1d(seq, out_sz, 1, use_bias=False)

        # simplest self-attention possible
        f_1 = tf.layers.conv1d(seq_fts, 1, 1)
        f_2 = tf.layers.conv1d(seq_fts, 1, 1)
        logits = f_1 + tf.transpose(f_2, [0, 2, 1])
        coefs = tf.nn.softmax(tf.nn.leaky_relu(logits) + bias_mat)

        if coef_drop != 0.0:
            coefs = tf.nn.dropout(coefs, 1.0 - coef_drop)
        if in_drop != 0.0:
            seq_fts = tf.nn.dropout(seq_fts, 1.0 - in_drop)

        vals = tf.matmul(coefs, seq_fts)
        ret = tf.contrib.layers.bias_add(vals)

        # residual connection
        if residual:
            if seq.shape[-1] != ret.shape[-1]:
                ret = ret + conv1d(seq, ret.shape[-1], 1) # activation
            else:
                ret = ret + seq

        return activation(ret)  # activation

# Experimental sparse attention head (for running on datasets such as Pubmed)
# N.B. Because of limitations of current TF implementation, will work _only_ if batch_size = 1!
def sp_attn_head(seq, out_sz, adj_mat, activation, nb_nodes, in_drop=0.0, coef_drop=0.0, residual=False):
    with tf.name_scope('sp_attn'):
        if in_drop != 0.0:
            seq = tf.nn.dropout(seq, 1.0 - in_drop)

        seq_fts = tf.layers.conv1d(seq, out_sz, 1, use_bias=False)

        # simplest self-attention possible
        f_1 = tf.layers.conv1d(seq_fts, 1, 1)
        f_2 = tf.layers.conv1d(seq_fts, 1, 1)
        
        f_1 = tf.reshape(f_1, (nb_nodes, 1))
        f_2 = tf.reshape(f_2, (nb_nodes, 1))

        f_1 = adj_mat*f_1
        f_2 = adj_mat * tf.transpose(f_2, [1,0])

        logits = tf.sparse_add(f_1, f_2)
        lrelu = tf.SparseTensor(indices=logits.indices, 
                values=tf.nn.leaky_relu(logits.values), 
                dense_shape=logits.dense_shape)
        coefs = tf.sparse_softmax(lrelu)

        if coef_drop != 0.0:
            coefs = tf.SparseTensor(indices=coefs.indices,
                    values=tf.nn.dropout(coefs.values, 1.0 - coef_drop),
                    dense_shape=coefs.dense_shape)
        if in_drop != 0.0:
            seq_fts = tf.nn.dropout(seq_fts, 1.0 - in_drop)

        # As tf.sparse_tensor_dense_matmul expects its arguments to have rank-2,
        # here we make an assumption that our input is of batch size 1, and reshape appropriately.
        # The method will fail in all other cases!
        coefs = tf.sparse_reshape(coefs, [nb_nodes, nb_nodes])
        seq_fts = tf.squeeze(seq_fts)
        vals = tf.sparse_tensor_dense_matmul(coefs, seq_fts)
        vals = tf.expand_dims(vals, axis=0)
        vals.set_shape([1, nb_nodes, out_sz])
        ret = tf.contrib.layers.bias_add(vals)

        # residual connection
        if residual:
            if seq.shape[-1] != ret.shape[-1]:
                ret = ret + conv1d(seq, ret.shape[-1], 1) # activation
            else:
                ret = ret + seq

        return activation(ret)  # activation


def sp_hete_attn_head(seq, out_sz, adj_mat, adj_type, edge_list, activation, nb_nodes, in_drop=0.0, coef_drop=0.0, residual=False):
    # input adjacency matrices are TRANSPOSED before feeding!
    # nb_nodes_j * nb_nodes_i
    with tf.name_scope('sp_hete_attn'):
        if in_drop != 0.0:
            seq = [tf.nn.dropout(seq_i, 1.0 - in_drop) for seq_i in seq]
        
        # seq_fts[j][i]: hidden features from group i to group j, center node is j
        # 1 * nb_nodes_i * out_sz_j
        W_list = [[None for _ in seq] for _ in seq]
        seq_fts = [[None for _ in seq] for _ in seq]
        #pdb.set_trace()
        W_inv_list = [tf.Variable(tf.glorot_uniform_initializer()(shape=(1,
                                  out_sz,
                                  int(s.shape[2])))) for s in seq]
        for W_type in adj_type:
            #pdb.set_trace()
            i, j = W_type
            if W_list[i][j] is None:
                W_list[i][j] = tf.Variable(tf.glorot_uniform_initializer()(shape=(1, 
                                                                   int(seq[i].shape[2]),
                                                                   out_sz)))
                seq_fts[j][i] = tf.matmul(seq[i], W_list[i][j])
        for i in range(len(seq)):
            for j in range(len(seq)):
                if W_list[i][j] is not None and W_list[j][i] is not None:
                    #pdb.set_trace()
                    #print("seq_fts: ")
                    #print(seq_fts[j][i])
                    #print(seq_fts[i][i])
                    #print("W_inv_list: ")
                    #print(W_inv_list[j])
                    #print("W_list: ")
                    #print(W_list[j][i])
                    #pdb.set_trace()
                    loss1 = seq_fts[j][i] @ (W_inv_list[j] @ W_list[j][i]) - seq_fts[i][i]
                    # 1 * nb_nodes_i * out_sz_i
                    loss1 = loss1 * loss1
                    loss1 = tf.reduce_sum(loss1) / nb_nodes[i]
                    tf.add_to_collection('loss_loop', loss1)
        for j in range(len(seq)):
            #pdb.set_trace()
            if out_sz >= seq[j].shape[1].value:
                loss2 = W_list[j][j] @ W_inv_list[j] - tf.eye(seq[j].shape[2].value)
            else:
                loss2 = W_inv_list[j] @ W_list[j][j] - tf.eye(out_sz)
            loss2 = loss2 * loss2
            loss2 = tf.reduce_sum(loss2)
            tf.add_to_collection('loss_inv', loss2)
        attn_biases = [None for _ in adj_type]
        for dir_edge in edge_list:
            attn_bias = tf.Variable(tf.random_normal(shape=(1, out_sz)))
            attn_biases[dir_edge[0]] = attn_bias
            if len(dir_edge) == 2:
                attn_biases[dir_edge[1]] = -attn_bias
    
        # for out_sz_j in out_sz
        coefs_lists = [[] for _ in range(len(seq))]
        seq_fts_lists = [[] for _ in range(len(seq))]
        
        # simplest self-attention possible
        for adj_ij, type_ij, attn_bias in zip(adj_mat, adj_type, attn_biases):
            # adj_ij is transposed, nb_nodes_j * nb_nodes_i
            i, j = type_ij
             
            f_1 = tf.reshape(seq_fts[j][j], (nb_nodes[j], out_sz))
            f_1 = tf.gather(f_1, adj_ij.indices[:, 0])
            f_2 = tf.reshape(seq_fts[j][i], (nb_nodes[i], out_sz))
            if attn_bias is not None:
                f_2 = f_2 + attn_bias
            f_2 = tf.gather(f_2, adj_ij.indices[:, 1])
            f = tf.reduce_sum(tf.multiply(f_1, f_2), 1)
    
            coefs = tf.SparseTensor(indices=adj_ij.indices, 
                    values=tf.nn.leaky_relu(f), 
                    dense_shape=adj_ij.dense_shape)
    
            if coef_drop != 0.0:
                coefs = tf.SparseTensor(indices=coefs.indices,
                        values=tf.nn.dropout(coefs.values, 1.0 - coef_drop),
                        dense_shape=coefs.dense_shape)
            coefs_lists[j].append(coefs) # transposed, nb_nodes_j * nb_nodes_i
            if in_drop != 0.0:
                seq_fts_ij = tf.nn.dropout(seq_fts[j][i], 1.0 - in_drop)
            seq_fts_lists[j].append(tf.squeeze(seq_fts_ij)) # nb_nodes_i * out_sz_j
            
    
        # As tf.sparse_tensor_dense_matmul expects its arguments to have rank-2,
        # here we make an assumption that our input is of batch size 1, and reshape appropriately.
        # The method will fail in all other cases!
        coefs = [tf.sparse_concat(1, coefs_list) for coefs_list in coefs_lists]
        coefs = [tf.sparse_softmax(coef) for coef in coefs]
        seq_fts = [tf.concat(seq_fts_list, 0) for seq_fts_list in seq_fts_lists]
        vals = [tf.sparse_tensor_dense_matmul(coef, seq_ft) for coef, seq_ft in zip(coefs, seq_fts)]
        # nb_nodes_j * out_sz_j
        vals = [tf.expand_dims(val, axis=0) for val in vals]
        for i, val in enumerate(vals):
            val.set_shape([1, nb_nodes[i], out_sz])
        ret = [tf.contrib.layers.bias_add(val) for val in vals]
    
        # residual connection
        if residual:
            ret2 = []
            for r, s in zip(ret, seq):
                if s.shape[-1] != r.shape[-1]:
                    ret2.append(r + tf.layers.conv1d(s, r.shape[-1], 1)) 
                else:
                    ret2.append(r + s)
            ret = ret2
        ret = [activation(r) for r in ret]
        return ret  # activation

def full_connection(seq, out_sz, target_node, activation, in_drop=0.0, use_bias=True):
    with tf.name_scope('full_connection_layer'):
        if in_drop != 0.0:
            seq = [tf.nn.dropout(seq_i, 1.0 - in_drop) for seq_i in seq]
        
        seq_fc = [tf.layers.conv1d(seq[target_node[i]], out_sz[i], 1, use_bias=use_bias) for i in range(len(target_node))]
        seq_fc = [tf.squeeze(seq_i) for seq_i in seq_fc] # remove the bach_size which is set as 1
        ret = [activation(s) for s in seq_fc]
        return ret
