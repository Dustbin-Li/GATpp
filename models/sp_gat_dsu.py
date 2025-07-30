import numpy as np
import tensorflow as tf
from utils import layers
from models.base_gattn import BaseGAttN
from scipy.sparse.csgraph import connected_components

class GATdsu(BaseGAttN):
    @staticmethod
    def cluster_nodes(attention_matrix, epsilon):
        if not isinstance(attention_matrix, np.ndarray):
            attention_matrix = np.array(attention_matrix)
        
        num_nodes = attention_matrix.shape[0]
        
        adj_matrix = (attention_matrix > epsilon).astype(np.float32)
        
        n_components, labels = connected_components(
            csgraph=adj_matrix, 
            directed=False,
            return_labels=True
        )
        
        clusters = {}
        for node_idx, cluster_id in enumerate(labels):
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(node_idx)
        
        return clusters

    @staticmethod
    def create_cluster_features(h_features, clusters):
        if not isinstance(h_features, np.ndarray):
            h_features = np.array(h_features)
        
        num_nodes = h_features.shape[0]
        feature_dim = h_features.shape[1]
        
        cluster_feature_matrix = np.zeros((num_nodes, feature_dim), dtype=np.float32)
        
        for cluster_id, node_indices in clusters.items():
            if not node_indices:
                continue
                
            cluster_nodes = h_features[node_indices]
            cluster_mean = np.mean(cluster_nodes, axis=0)
            for idx in node_indices:
                cluster_feature_matrix[idx] = cluster_mean
        
        return cluster_feature_matrix

    @staticmethod
    def inference(inputs, nb_classes, nb_nodes, training, attn_drop, ffd_drop,
                 bias_mat, hid_units, n_heads, activation=tf.nn.elu, residual=False,
                 epsilon=0.1, fusion_method='gate'):
        first_layer_outputs = []
        first_layer_attentions = []
        for _ in range(n_heads[0]):
            output, attn_coef = layers.sp_attn_head(
                inputs, 
                bias_mat=bias_mat,
                out_sz=hid_units[0], 
                activation=activation,
                in_drop=ffd_drop, 
                coef_drop=attn_drop, 
                residual=False,
                return_attn=True
            )
            first_layer_outputs.append(output)
            first_layer_attentions.append(attn_coef)
        
        h_1 = tf.concat(first_layer_outputs, axis=-1)
        
        avg_attention = tf.reduce_mean(tf.stack(first_layer_attentions), axis=0)
        
        avg_attention = tf.squeeze(avg_attention, axis=0)
        
        feature_dim = hid_units[0] * n_heads[0]
        
        def cluster_and_fuse(h_features, attn_matrix, eps):
            h_features_2d = np.squeeze(h_features, axis=0)
            
            clusters = GATdsu.cluster_nodes(attn_matrix, eps)
            cluster_features = GATdsu.create_cluster_features(h_features_2d, clusters)
            
            return np.expand_dims(cluster_features, axis=0)
        
        cluster_features = tf.py_func(
            func=lambda h, a, e: cluster_and_fuse(h, a, e),
            inp=[h_1, avg_attention, epsilon],
            Tout=tf.float32,
            stateful=False
        )
        
        cluster_features.set_shape([1, nb_nodes, feature_dim])
        
        if fusion_method == 'add':
            fused_features = h_1 + cluster_features
        elif fusion_method == 'concat':
            fused_features = tf.concat([h_1, cluster_features], axis=-1)
        elif fusion_method == 'gate':
            combined = tf.concat([h_1, cluster_features], axis=-1)
            gate = tf.layers.dense(
                inputs=combined,
                units=feature_dim,
                activation=tf.sigmoid
            )
            fused_features = gate * h_1 + (1 - gate) * cluster_features
        else:
            fused_features = h_1
        
        current_input = fused_features
        for i in range(1, len(hid_units)):
            attns = []
            for _ in range(n_heads[i]):
                attns.append(layers.sp_attn_head(
                    current_input, 
                    bias_mat=bias_mat,
                    out_sz=hid_units[i], 
                    activation=activation,
                    in_drop=ffd_drop, 
                    coef_drop=attn_drop, 
                    residual=residual
                ))
            current_input = tf.concat(attns, axis=-1)
        
        out = []
        for i in range(n_heads[-1]):
            out.append(layers.sp_attn_head(
                current_input, 
                bias_mat=bias_mat,
                out_sz=nb_classes, 
                activation=lambda x: x,
                in_drop=ffd_drop, 
                coef_drop=attn_drop, 
                residual=False
            ))
        
        logits = tf.add_n(out) / n_heads[-1]
        
        return logits