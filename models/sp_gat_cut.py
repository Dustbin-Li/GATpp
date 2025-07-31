import numpy as np
import tensorflow as tf

from utils import layers
from models.base_gattn import BaseGAttN

class SpGATcut(BaseGAttN):
    @staticmethod
    def create_filtered_bias(atten_coefs, neighbor_threshold, top_k_neighbors, nb_nodes):
        if len(atten_coefs.shape) == 3:
            atten_coefs = tf.squeeze(atten_coefs, axis=0)
        
        new_adj = tf.zeros_like(atten_coefs, dtype=tf.bool)
        if neighbor_threshold > 0:
            threshold_mask = atten_coefs > neighbor_threshold
            new_adj = tf.logical_or(new_adj, threshold_mask)
            
        if top_k_neighbors > 0:
            top_k = min(top_k_neighbors, nb_nodes - 1)
            top_values, top_indices = tf.math.top_k(atten_coefs, k=top_k+1, sorted=True)
            row_indices = tf.tile(
                tf.range(nb_nodes)[:, tf.newaxis],
                [1, top_k+1]
            )
            flat_indices = tf.stack([
                tf.reshape(row_indices, [-1]),
                tf.reshape(top_indices, [-1])
            ], axis=1)
            flat_updates = tf.ones([nb_nodes * (top_k+1)], dtype=tf.bool)
            # top_indices = tf.squeeze(top_indices, axis=0)
            top_k_mask = tf.scatter_nd(
                # tf.stack([row_indices, top_indices], axis=-1),
                # tf.ones_like(top_values, dtype=tf.bool),
                flat_indices,
                flat_updates,
                shape=[nb_nodes, nb_nodes]
            )
            diag_mask = tf.logical_not(tf.eye(nb_nodes, dtype=tf.bool))
            top_k_mask = tf.logical_and(top_k_mask, diag_mask)
            new_adj = tf.logical_or(new_adj, top_k_mask)
            
            bias_mat = tf.where(
                new_adj,
                tf.zeros_like(atten_coefs),
                tf.ones_like(atten_coefs) * -1e9
            )
            return bias_mat[tf.newaxis, :, :]
        
    @staticmethod
    def inference(inputs, nb_classes, nb_nodes, training, attn_drop, ffd_drop,
            bias_mat, hid_units, n_heads, activation=tf.nn.elu, residual=False,
            neighbor_threshold=0.0, top_k_neighbors=0.0):
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
        
        if neighbor_threshold > 0 or top_k_neighbors > 0:
            new_bias_mat = SpGATcut.create_filtered_bias(
                avg_attention,
                neighbor_threshold,
                top_k_neighbors,
                nb_nodes
            )
            bias_mat = new_bias_mat
        for i in range(1, len(hid_units)):
            h_old = h_1
            attns_output = []
            attns_coefs = []
            for _ in range(n_heads[i]):
                output, coef = layers.sp_attn_head(
                    h_1, 
                    adj_mat=bias_mat,
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
            if neighbor_threshold > 0 or top_k_neighbors > 0:
                new_bias_mat = SpGATcut.create_filtered_bias(
                    avg_attention,
                    neighbor_threshold,
                    top_k_neighbors,
                    nb_nodes
                )
                bias_mat = new_bias_mat
        out = []
        for i in range(n_heads[-1]):
            out.append(layers.sp_attn_head(h_1, adj_mat=bias_mat,
                out_sz=nb_classes, activation=lambda x: x, nb_nodes=nb_nodes,
                in_drop=ffd_drop, coef_drop=attn_drop, residual=False, return_attn=False))
        logits = tf.add_n(out) / n_heads[-1]
    
        return logits, all_attention_maps
