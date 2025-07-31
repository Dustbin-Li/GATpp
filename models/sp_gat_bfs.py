import numpy as np
import tensorflow as tf
from collections import deque

from utils import layers
from models.base_gattn import BaseGAttN

class SpGATbfs(BaseGAttN):
    @staticmethod
    def create_bfs_bias(atten_coefs, epsilon, max_hops, nb_nodes):
        adj_bool = (atten_coefs >= epsilon)
        adj_bool = adj_bool | tf.eye(nb_nodes, dtype=tf.bool)
        
        adj_int = tf.cast(adj_bool, tf.int32)
        
        reachability = tf.eye(nb_nodes, dtype=tf.int32)
        
        for _ in range(max_hops):
            next_reach = tf.matmul(reachability, adj_int)
            reachability = tf.clip_by_value(reachability + next_reach, 0, 1)
        
        reachability_bool = reachability > 0
        
        bias_mat = tf.where(
            reachability_bool,
            tf.zeros_like(reachability, dtype=tf.float32),
            tf.ones_like(reachability, dtype=tf.float32) * -1e9
        )
        
        return bias_mat[tf.newaxis, :, :]

    @staticmethod
    def inference(inputs, nb_classes, nb_nodes, training, attn_drop, ffd_drop,
                 bias_mat, hid_units, n_heads, activation=tf.nn.elu, residual=False,
                 epsilon=0.2, max_hops=2):
        all_attention_maps = []
        attns_output = []
        attns_coefs = []
        
        for _ in range(n_heads[0]):
            output, coef = layers.sp_attn_head(
                inputs,
                out_sz=hid_units[0],
                adj_mat=bias_mat,
                activation=activation,
                nb_nodes=nb_nodes,
                in_drop=ffd_drop,
                coef_drop=attn_drop,
                residual=False,
                return_attn=True
            )
            attns_output.append(output)
            attns_coefs.append(tf.sparse.to_dense(coef))
        
        h_1 = tf.concat(attns_output, axis=-1)
        avg_attention = tf.reduce_mean(tf.stack(attns_coefs), axis=0)
        all_attention_maps.append(avg_attention)
        
        new_bias_mat = SpGATbfs.create_bfs_bias(
            avg_attention,
            epsilon,
            max_hops,
            nb_nodes
        )
        
        for i in range(1, len(hid_units)):
            h_old = h_1
            attns_output = []
            attns_coefs = []
            for _ in range(n_heads[i]):
                output, coef = layers.sp_attn_head(
                    h_1, 
                    adj_mat=new_bias_mat,
                    out_sz=hid_units[i], 
                    activation=activation,
                    nb_nodes=nb_nodes,
                    in_drop=ffd_drop, 
                    coef_drop=attn_drop, 
                    residual=residual, 
                    return_attn=True
                )
                attns_output.append(output)
                attns_coefs.append(tf.sparse.to_dense(coef))
            h_1 = tf.concat(attns_output, axis=-1)
            avg_attention = tf.reduce_mean(tf.stack(attns_coefs), axis=0)
            all_attention_maps.append(avg_attention)
            
            new_bias_mat = SpGATbfs.create_bfs_bias(
                avg_attention,
                epsilon,
                max_hops,
                nb_nodes
            )
            
        out = []
        for i in range(n_heads[-1]):
            out.append(layers.sp_attn_head(
                h_1, 
                adj_mat=new_bias_mat,
                out_sz=nb_classes, 
                activation=lambda x: x,
                nb_nodes=nb_nodes,
                in_drop=ffd_drop, 
                coef_drop=attn_drop, 
                residual=False,
                return_attn=False
            ))
        
        logits = tf.add_n(out) / n_heads[-1]
        
        return logits, all_attention_maps