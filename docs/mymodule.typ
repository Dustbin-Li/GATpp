#import "../template.typ": *

== 实验与分析

=== 实验思路

在此部分，我们对于 *GAT* 进行修改，期待修改后的 *GAT* 能够展现出更好的性能。因为是在 *GAT* 的基础上进行修改，所以我们将其命名为 *GAT++*。 *GAT++* 的主要修改思路共有三种：1. 利用“砍边”法，只保留关联度较高的边；2. 基于广度优先搜索，将感知野通过广度搜索扩大；3. 基于并查集算法，得到一个集合的特征，再将集合特征与点的特征做融合。在之后的篇幅中，我们将对这三种思路进行详细的介绍。在此之前，我们先在符号上进行一些约定，方便后续讨论。

#list(
  tight: true,
  [
    $ cal(N)_i^(k)$ : 从节点 $i$ 辐射范围为 $k$ 的邻域，$k in bb(N)^+ $
  ],
  [
    $h_i$ ~: 节点 $i$ 的特征向量，其维度为 $cal(F)$
  ],
  [
    $h'_i$: 经过 $M$ 映射之后的节点 $i$ 的特征向量，其维度为 $cal(F')$
  ],
  [
    $M$ : 通过学习出来的一个映射函数，将 $h_i$ 映射到 $h'_i$
  ],
  [
    $G$ : 原本的图结构；$V$ $->$ 图中节点的集合；$E$ $->$ 图中边的集合
  ],
  [
    $a$ : 注意力机制
  ],
  [
    $e_(i j)$ : 点 $j$ 对于点 $i$ 的注意力值
  ],
  [
    $alpha_(i j)$ : 点 $j$ 对点 $i$ 的归一化后的注意力值
  ],
  [
    $G' $ : 权重为 $alpha_(i j)$ 的图
  ]
)

==== “砍边”法
在原本的 *GAT* 中，对于每个节点 $v_i$，所有与其相邻的点 $v_j$ 都会被纳入到 $cal(N)_i^(ast)$ 中。但是实际中，其实有些点虽然是相邻的，但是其相关性较低，这类点在最终算聚合特征时可能并不重要，甚至在某些情况下会起到相反的作用。因此，我们并不需要将所有与 $v_i$ 相邻的点都纳入到 $cal(N)_i^(ast)$ 中。

在“砍边”法中，我们采取两种方法对于边进行筛选：
- 第一种方法，我们关注边的相关性，即只保留 $alpha_(i j) > epsilon$ 的边。每次 *GAT* 结束后，对于 $v_i in V'$，只将 $alpha_(i j) > epsilon$ 的相邻点 $v_j$ 纳入 $cal(N)_i^(ast)$ 中，再进行后续的 *GAT*$(dot)$ 操作。
- 第二种方法，我们关注点的数量，即只保留与 $v_i$ 相关性较高的 $k$ 个点。对于 $v_i in V'$，只将 $alpha_(i, dot)$ 的前 $k$ 大所对应的相邻点 $v_j$ 纳入 $cal(N)_i^(ast)$ 中，再进行后续的 *GAT*$(dot)$ 操作。

#pseudocode-list[
+ *Function* prune_edges(G, node_features, attention_weights, epsilon, top_k):
  + *For* each node v_i in G.nodes:
    + neighbors = G.get_neighbors(v_i)
    + attn_scores = [attention_weights[(v_i, v_j)] for v_j in neighbors]
    + *Method 1 (Threshold):*
      + pruned_neighbors = [v_j for v_j in neighbors if attention_weights[(v_i, v_j)] > epsilon]
    + *Method 2 (Top-k):*
      + pruned_neighbors = select top_k neighbors v_j with largest attention_weights[(v_i, v_j)]
    + aggregated_feature = Aggregate([node_features[v_j] for v_j in pruned_neighbors])
    + update node_features[v_i] = aggregated_feature
  + *Return* updated node_features
+ *End Function*
]

==== 基于广度优先搜索
在原本的 *GAT* 中，通常我们只会将邻域内的点纳入到 $cal(N)_i^(ast)$ 中。但是只用图中的邻居关系，并不能完全反映出节点之间的关系。在实际中，有些点可能在邻居关系上与其他点之间存在着更长的距离，但是其实其路径上的 $alpha_(i j)$ 的值仍然很大。而两点之间的关系又有传递性，根据传递性，这些点很可能与目标点的相关性仍然很高，我们可以将这些点也纳入到 $cal(N)_i^(ast)$ 中。然而，有些点虽然在图结构中，与目标点之间有边相连，但是其 $alpha_(i j)$ 的值却很小，这时我们并不希望将其纳入到 $cal(N)_i^(ast)$ 中。

基于广度优先搜索，我们可以将感知野进行扩大，使得感知野中的点都纳入到 $cal(N)_i^(ast)$ 中。具体的做法是，在进行一次 $bold("GAT")(dot)$ 后，在 $G'$ 上进行操作。对于点 $v_i in V'$，利用 BFS 算法找与它相关性高的点。

对于在图结构上不相邻的两点 $v_i$ 和 $v_p$，如果 $v_i$ 和 $v_p$ 之间存在一条路径：$v_i -> v_j -> v_k -> v_p$，则我们定义 $v_i$ 与 $v_p$ 之间的相关性为：
  $ alpha_(i p) = alpha_(i j) dot alpha_(j k) dot alpha_(k p) $

如果 $alpha_(i p) > epsilon$，则认为 $v_i$ 与 $v_p$ 之间存在较强的相关性，将 $v_p$ 纳入到 $cal(N)_i^(ast)$ 中。即最终的剪枝条件为如果从 $v_i$ 出发的路径，$exists q$，经过该点 $v_q$ 时，$alpha_(i q) < epsilon$，则该路径中断。

将路径上所有经过的点都纳入 $cal(N)_i^(ast)$ 中，最终计算得到的$h_i^(2) = bold("GAT")( \{ h_k | v_k in cal(N)_i^(ast) \} ), text(bold("where")) h_k = h_k^(1))$


#pseudocode-list[
+ *Class* GATbfs:
  + *Static Method* create_bfs_bias(atten_coefs, epsilon, max_hops, nb_nodes):
    + adj_bool = (atten_coefs >= epsilon) 
    + adj_bool = adj_bool OR identity_matrix(nb_nodes)  
    + adj_int = convert_to_int(adj_bool) 
    + reachability = identity_matrix(nb_nodes) 
    
    + *Loop* max_hops times:
      + next_reach = matrix_multiply(reachability, adj_int)
      + reachability = clip(reachability + next_reach, 0, 1)  
    
    + reachability_bool = (reachability > 0) 
    + bias_mat = where(
      + reachability_bool, 
      + 0, 
      + -inf
    + )
    + *Return* expand_dims(bias_mat)
  
  + *Static Method* inference(inputs, nb_classes, nb_nodes, ...):
    + all_attention_maps = []  
    + new_bias_mat = initial_bias_mat  
    
    + attns_output = []
    + attns_coefs = []
    + *For* n_heads[0] times:
      + output, coef = attn_head(
        + inputs, 
        + bias_mat=new_bias_mat,
        + ...
      + )
      + attns_output.append(output)
      + attns_coefs.append(coef)
    + h = concatenate(attns_output)
    + avg_attention = mean(attns_coefs)
    + all_attention_maps.append(avg_attention)
    + new_bias_mat = create_bfs_bias(avg_attention, ...) 
    
    + *For* each subsequent layer:
      + attns_output = []
      + attns_coefs = []
      + *For* n_heads[i] times:
        + output, coef = attn_head(...) 
        + ...
      + h = concatenate(attns_output)
      + avg_attention = mean(attns_coefs)
      + all_attention_maps.append(avg_attention)
      + new_bias_mat = create_bfs_bias(avg_attention, ...)  
    
    + outputs = []
    + *For* n_heads[-1] times:
      + out = attn_head(h, new_bias_mat, ...)
      + outputs.append(out)
    + logits = average(outputs)
    
    + *Return* logits, all_attention_maps
+ *End Class*
]

==== 基于并查集算法

在原本的 *GAT* 中，我们只是用了图结构中相邻的关系，即 $cal(N)_i^(ast)$ 作为 $h_i$ 的输入，而没有考虑到节点之间的关系。而在实际中，节点之间的关系往往是复杂的。我们可以将所有的节点按照并查集的思路进行合并，为节点根据它们的相似性进行分类，得到一个特征向量。

具体的做法是，在进行一次 $bold("GAT")(dot)$ 后，在 $G'$ 上进行操作。首先，我们给出划分集合的规则：对于 $G'$ 中相邻的两个节点 $v_i$ 和 $v_j$，如果 $alpha_(i j) > epsilon$，那么则认为 $v_i$ 与 $v_j$ 同属于一个集合 $cal(A)_p$。特别的，如果 $G'$ 中一个节点 $v_k$，$alpha_(k, dot) <= epsilon$，则 $cal(A)_q = \{v_k\}$。

通过此操作，我们得到了 $cal(A)_1$、$cal(A)_2$...$cal(A)_m$。对于集合 $cal(A)_s$，令 $H_s = frac(1, |cal(A)_s|) sum_(v_k in cal(A)_s) h'_k$，为表征该集合的特征向量。可将 $H_s$ 与 $h'_k$ 融合，去做后续操作。

在这里，我们将集合特征与点的特征融合时，共实现了三种方法：

- 第一种方法，我们将 $H_s$ 与 $h'_k$ 直接相加，得到 $h'_s$，即为 fused_features。
- 第二种方法，我们将 $H_s$ 与 $h'_k$ 拼接，得到 $[h'_k, H_s]$，即为 fused_features。
- 第三种方法，我们将 $h'_k$ 与 $H_s$ 拼接，得到 $[h'_k, H_s]$，然后通过全连接层和 sigmoid 激活得到门控系数 $g$，得到最后的 fused_features 为 $g dot h'_k + (1-g) dot H_s$。


#pseudocode-list[
+ *Class* GATdsu(*BaseGAttN*):
    + *Static Method* cluster_nodes(attention_matrix, epsilon):
        + *If* attention_matrix is not np.ndarray:
            + attention_matrix = np.array(attention_matrix)
        + num_nodes = attention_matrix.shape[0]
        + adj_matrix = (attention_matrix > epsilon).astype(float)
        + n_components, labels = connected_components(adj_matrix, directed=False)
        + clusters = {}
        + *For* node_idx, cluster_id in enumerate(labels):
            + *If* cluster_id not in clusters:
                + clusters[cluster_id] = []
            + clusters[cluster_id].append(node_idx)
        + *Return* clusters

    + *Static Method* create_cluster_features(h_features, clusters):
        + *If* h_features is not np.ndarray:
            + h_features = np.array(h_features)
        + num_nodes = h_features.shape[0]
        + feature_dim = h_features.shape[1]
        + cluster_feature_matrix = zeros((num_nodes, feature_dim))
        + *For* cluster_id, node_indices in clusters.items():
            + *If* node_indices is empty: *continue*
            + cluster_nodes = h_features[node_indices]
            + cluster_mean = mean(cluster_nodes, axis=0)
            + *For* idx in node_indices:
                + cluster_feature_matrix[idx] = cluster_mean
        + *Return* cluster_feature_matrix

    + *Static Method* inference
        + *For* n_heads[0] times:
            + output, attn_coef = *attn_head*
            + first_layer_outputs.append
            + first_layer_attentions.append
        + h_1 = concatenate(first_layer_outputs, axis=-1)
        + avg_attention = mean(stack(first_layer_attentions), axis=0)
        + avg_attention = squeeze(avg_attention, axis=0)
        + feature_dim = hid_units[0] \* n_heads[0]
        + *Define* cluster_and_fuse(h_features, attn_matrix, eps):
            + h_features_2d = squeeze(h_features, axis=0)
            + clusters = cluster_nodes(attn_matrix, eps)
            + cluster_features = create_cluster_features(h_features_2d, clusters)
            + *Return* expand_dims(cluster_features, axis=0)
        + cluster_features = py_func(cluster_and_fuse, [h_1, avg_attention, epsilon])
        + cluster_features.set_shape([1, nb_nodes, feature_dim])
        + *If* fusion_method == 'add':
            + fused_features = h_1 + cluster_features
        + *Elif* fusion_method == 'concat':
            + fused_features = concatenate([h_1, cluster_features], axis=-1)
        + *Elif* fusion_method == 'gate':
            + combined = concatenate([h_1, cluster_features], axis=-1)
            + gate = dense(combined, units=feature_dim, activation=sigmoid)
            + fused_features = gate * h_1 + (1 - gate) * cluster_features
        + *Else*:
            + fused_features = h_1
        + current_input = fused_features
        + *For* i in 1 to len(hid_units)-1:
            + attns = []
            + *For* n_heads[i] times:
                + attns.append
            + current_input = concatenate(attns, axis=-1)
        + out = []
        + *For* n_heads[-1] times:
            + out.append
        + logits = average(out)
        + *Return* logits
+ *End Class*
]


=== 实验结果与分析

在实验中，我们对 *GAT* 进行了修改，并进行了性能测试。实验结果表明，*GAT++* 能够在某些情况下展现出更好的性能。

==== “砍边”法


#table3(
  align: center,
  columns: (1fr, 1fr, 1fr, 1fr),
  inset: 10pt,
  [$epsilon$], [top k], [Val Acc], [Test Acc],
  [0.0], [5], [0.7940], [0.8150],
  [0.0], [10], [0.7680], [0.8000],
  [0.0], [15], [0.6140], [0.6250],
  [0.1], [5], [0.8100], [0.8210],
  [0.1], [10], [0.7880], [0.8140],
  [0.1], [15], [0.7100], [0.7190],
  [0.15], [5], [0.8140], [0.8250],
  [0.15], [10], [0.7880], [0.8090],
  [0.15], [15], [0.7580], [0.7760],
  [0.2], [5], [0.7940], [0.8120],
  [0.2], [10], [0.7680], [0.7950],
  [0.2], [15], [0.7760], [0.7740],
)




==== 基于广度优先搜索

#table3(
  align: center,
  columns: (1fr, 1fr, 1fr, 1fr),
  inset: 10pt,
  [$epsilon$], [max\_hops], [Val Acc], [Test Acc],
  [0.0], [2], [0.3160], [0.3190],
  [0.0], [3], [0.3160], [0.3190],
  [0.1], [2], [0.8120], [0.8120],
  [0.1], [3], [0.7960], [0.8160],
  [0.15], [2], [0.8180], [0.8110],
  [0.15], [3], [0.8140], [0.7960],
  [0.2], [2], [0.8140], [0.8110],
  [0.2], [3], [0.8140], [0.8140],
  [0.25], [2], [0.8220], [0.8190],
  [0.25], [3], [0.8120], [0.8280],
  [0.3], [2], [0.8100], [0.8100],
  [0.3], [3], [0.8200], [0.8210],
)


==== 基于并查集算法

#table3(
  align: center,
  columns: (1fr, 1fr, 1fr, 1fr),
  inset: 10pt,
  [$epsilon$], [融合方式], [Val Acc], [Test Acc],
  [0.05], [add], [0.8200], [0.8250],
  [0.05], [concat], [0.8140], [0.8180],
  [0.05], [gate], [0.8240], [0.8350],
  [0.1], [add], [0.8140], [0.7850],
  [0.1], [concat], [0.8160], [0.8160],
  [0.1], [gate], [0.8260], [0.8280],
  [0.15], [add], [0.8140], [0.8190],
  [0.15], [concat], [0.8180], [0.8060],
  [0.15], [gate], [0.8240], [0.8400],
  [0.2], [add], [0.8140], [0.8220],
  [0.2], [concat], [0.8140], [0.8160],
  [0.2], [gate], [0.8120], [0.8190],
  [0.25], [add], [0.8200], [0.8240],
  [0.25], [concat], [0.8160], [0.8330],
  [0.25], [gate], [0.8240], [0.8230],
  [0.3], [add], [0.8140], [0.8250],
  [0.3], [concat], [0.8180], [0.8340],
  [0.3], [gate], [0.8260], [0.8070],
)


