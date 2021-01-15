import numpy as np
import tensorflow as tf

from utils import layers
from models.base_gattn import BaseGAttN

class SpHGAT(BaseGAttN):
    def inference(inputs, nb_classes, nb_nodes, training, attn_drop, ffd_drop,
            bias_mat, adj_type, edge_list, hid_units, n_heads,
            activation=tf.nn.elu, residual=False, target_nodes=[0]):
        attns = []
        for _ in range(n_heads[0]):
            attns.append(layers.sp_hete_attn_head(inputs,  
                adj_mat=bias_mat, adj_type=adj_type, edge_list=edge_list,
                out_sz=hid_units[0], activation=activation, nb_nodes=nb_nodes,
                in_drop=ffd_drop, coef_drop=attn_drop, residual=False))
        h_1 = [tf.concat(attn, axis=-1) for attn in zip(*attns)]
        for i in range(1, len(hid_units)):
            h_old = h_1
            attns = []
            head_act = activation
            is_residual = residual
            for _ in range(n_heads[i]):
                attns.append(layers.sp_hete_attn_head(h_1,  
                    adj_mat=bias_mat, adj_type=adj_type, edge_list=edge_list,
                    out_sz=hid_units[i], activation=head_act, nb_nodes=nb_nodes,
                    in_drop=ffd_drop, coef_drop=attn_drop, residual=is_residual))
            h_1 = [tf.concat(attn, axis=-1) for attn in zip(*attns)]
        # here now we have the output embedding of multi-head attention
        logits = layers.full_connection(h_1, nb_classes, target_nodes, activation=lambda x:x, in_drop=ffd_drop, use_bias=True)
        return logits
