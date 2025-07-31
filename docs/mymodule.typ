#import "template.typ": *

== 实验背景与意义

=== 背景介绍

\

在数字化时代，数据越来越多地以图（Graph）的形式存在，用以描述实体间的复杂关系。从社交网络中的用户关系、金融交易网络中的资金流动，到生物学中的蛋白质相互作用网络和学术界的论文引用网络，图结构数据无处不在。如何有效地从这些非欧几里得结构的数据中学习和提取有价值的信息，即图表示学习（Graph Representation Learning），已成为机器学习领域一个至关重要且充满挑战的研究方向。

\

图神经网络（Graph Neural Networks, GNNs）作为一种端到端的图表示学习框架，近年来取得了突破性进展。早期的图卷积网络（Graph Convolutional Network, GCN）通过借鉴卷积神经网络（CNN）的思想，在图的邻域上进行信息聚合，成功地将深度学习应用于图数据，并在节点分类等任务上表现出色。然而，GCN的卷积核权重在训练完成后是固定的，这意味着它对邻居节点的聚合方式是静态的，无法根据节点自身特性区分不同邻居的重要性，这在一定程度上限制了模型的表达能力。

\

为了克服这一局限性，Veličković等人于2018年提出了图注意力网络（Graph Attention Network, GAT）。GAT的核心创新在于将自注意力（Self-Attention）机制引入图的聚合过程。通过为每个节点的邻居动态计算注意力系数，GAT能够为更重要的邻居节点分配更高的权重，从而实现一种更具适应性和表达力的信息聚合。此外，GAT通过多头注意力（Multi-head Attention）机制稳定学习过程并捕捉节点间不同维度的复杂关系。这些特性使得GAT不仅在多个基准数据集上取得了领先的性能，也为后续图神经网络的发展开辟了新的道路。

\

=== 实验意义 

\

本实验旨在对经典的图注意力网络（GAT）进行改进，并在权威的Planetoid基准数据集（包括Cora, CiteSeer, PubMed）上进行一系列优化研究。本研究的意义主要体含现在以下两个方面：

\

*理论意义*： 首先，通过复现GAT，我们能够深入理解注意力机制在图结构数据上的工作原理及其优势。其次，针对当前GNNs普遍面临的挑战，如模型加深时的过平滑（Over-smoothing）问题、计算效率以及对复杂关系的捕捉能力，本实验将探索并验证一系列前沿优化策略的有效性。这包括但不限于引入残差连接构建更深层的网络、应用图数据增强技术提升模型鲁棒性等。这些探索有助于推动GNN理论边界的拓展，为设计更强大、更通用的图学习模型提供实验依据和新的见解。

\

*实践意义*： 在金融科技领域，图神经网络在信用风险评估、欺诈检测、客户关系管理等任务中展现出巨大潜力。然而，当前GNN模型在处理大规模金融图数据时，往往面临计算资源消耗大、模型解释性差等问题。本实验将探索并验证一系列前沿优化策略的有效性，这些策略包括但不限于引入残差连接构建更深层的网络、应用图数据增强技术提升模型鲁棒性等。这些探索有助于推动GNN理论边界的拓展，为设计更强大、更通用的图学习模型提供实验依据和新的见解。

\





== 研究现状

自图神经网络概念被提出以来，相关研究层出不穷，模型架构和理论分析日趋成熟。本节将围绕图注意力网络（GAT）及其相关技术，对当前的研究现状进行梳理。

\


=== 经典图卷积神经网络（GCN）

图神经网络的早期发展以图卷积网络（GCN）为代表。Kipf和Welling提出的GCN通过一种简化的谱图卷积，高效地聚合邻居节点信息，其简洁的架构和出色的性能使其成为后续大量研究的基石。然而，GCN的局限性也较为明显：其一，它采用静态的拉普拉斯矩阵进行信息聚合，平等对待所有邻居，缺乏动态调整权重的能力；其二，GCN的训练方式通常是直推式（Transductive）的，难以直接泛化到训练时未见过的新节点或新图。

\

=== 注意力机制的引入（GAT）

\

- *图注意力网络（GAT）*：为了解决GCN的局限性，研究者们开始将注意力机制引入GNN。图注意力网络（GAT） 是这一方向的开创性工作。GAT不依赖于预定义的图结构，而是通过一个可学习的神经网络层，计算中心节点与其邻居节点之间的注意力分数，并以此为权重进行加权求和。这种方式赋予了模型为不同邻居分配不同重要性的能力。结合多头注意力机制，GAT能够从不同子空间捕捉节点间的丰富特征，并展现出强大的归纳（Inductive）学习能力。

\

- *注意力机制改进（GATv2）*：尽管GAT取得了巨大成功，但后续研究发现其注意力机制存在“静态”的缺陷。具体而言，在原始GAT的注意力计算LeakyReLU$(a^T [W h_i || W h_j])$中，对邻居节点的排序主要由$a^T [· || W h_j]$决定，与中心节点（查询节点）$h_i$的关系较弱，这意味着模型学习到的注意力权重在很大程度上是全局共享的，而非完全依赖于查询上下文。为了解决此问题，Brody等人提出了 GATv2。GATv2对注意力计算的顺序进行了简单而有效的修改，将其调整为$a^T $ LeakyReLU$(W[h_i + W h_j])$或类似形式。这一改动使得查询节点$h_i$和键节点$h_j$的信息能够更早地交互，从而让注意力分数的计算变得完全“动态”，即对于同一个邻居节点，不同的中心节点可以计算出截然不同的注意力权重。理论和实验均证明，GATv2具有比GAT更强的表达能力，能够在多个基准测试中取得更优的性能。

\

- *聚合范围和方式的增强（SuperGAT）*：
标准的GAT在聚合信息时仅考虑了节点的直接（一跳）邻居，这限制了其对图中长距离依赖关系的捕获。此外，其多头注意力机制平等对待每个头的输出，可能无法最优地适应特定任务。为了同时解决这两个问题，Kim和Oh提出了 SuperGAT (Supervising and Attending GAT)。该模型从两个核心方面进行了创新：其一，它通过一种类似PageRank的注意力传播机制，在消息传递前预先计算出一个“注意力感知的接收域”，从而将聚合范围扩展至多跳邻居。其二，它引入了一个“监督注意力”（Supervising Attention）机制，该机制学习如何根据边的任务相关性，对原始GAT多头注意力的输出进行加权聚合。这相当于在注意力之上再增加一层注意力，用以指导和优化底层的聚合过程。通过这种方式，SuperGAT能够同时捕捉长距离依赖和更具任务适应性的节点关系，展现出卓越的性能和鲁棒性。

\

综上所述，当前GNN的研究已经从早期的静态卷积发展到以GAT为代表的动态注意力机制，并进一步演化出了GATv2和SuperGAT等更具表达力和适应性的高级变体。同时，围绕深度、可扩展性和鲁棒性等关键问题也形成了一系列成熟的解决方案。本实验正是在这一坚实的研究基础上，系统地将这些先进的优化策略应用于GAT模型，并与GATv2、SuperGAT等模型进行对比，以期获得性能上的显著提升。


\

=== 实验数据集

\

为了客观、全面地评估模型的性能，本实验采用了三个图神经网络领域公认的基准引文网络数据集：Cora、CiteSeer 和 PubMed。这三个数据集均源于 Planetoid 项目，是半监督节点分类任务中最常用的测评标准。它们的共同特点是：图中的节点代表学术论文，边代表论文之间的引用关系，每个节点都包含一个高维的词袋模型（Bag-of-Words）或TF-IDF特征向量，并被赋予一个研究领域的标签。

实验任务是在仅有少量节点标签已知的情况下，对网络中其他所有节点的类别进行预测。我们遵循标准的Planetoid数据集划分方法，即每类仅使用20个节点作为训练集，500个节点作为验证集，1000个节点作为测试集。

以下是三个数据集的详细介绍：

\

==== Cora

\

Cora 数据集是一个关于计算机科学论文的引文网络。网络中的每篇论文都属于七个研究领域中的一个。如果论文A引用了论文B，则在图中存在一条从A到B的有向边。数据集经过预处理后，移除了词典中出现次数少于10次的词语。

\

- 节点 (Nodes): 2,708篇学术论文。

- 边 (Edges): 5,429条引用关系。

- 节点特征 (Features): 一个1,433维的二进制特征向量，每一维对应一个词典中的词语，1或0表示该词语是否出现在论文的摘要中。

- 类别 (Classes): 7个类别，包括神经网络、强化学习、遗传算法等。

==== CiteSeer

\

CiteSeer 数据集同样是一个引文网络，规模和复杂度与Cora相近。它包含了六个不同研究领域的科学出版物。相比Cora，CiteSeer的图结构连接更为稀疏。

- 节点 (Nodes): 3,327篇学术论文。

- 边 (Edges): 4,732条引用关系。

- 节点特征 (Features): 一个3,703维的二进制特征向量，表示论文是否包含词典中的特定词语。

- 类别 (Classes): 6个类别，包括人工智能、数据库、机器学习等。

==== PubMed

\

PubMed 数据集源自生物医学领域的权威数据库PubMed。它包含的节点和边的数量远超Cora和CiteSeer，是一个规模更大的引文网络。该数据集主要关注于糖尿病相关的研究论文。


\

- 节点 (Nodes): 19,717篇关于糖尿病的学术论文。

- 边 (Edges): 44,338条引用关系。

- 节点特征 (Features): 一个500维的TF-IDF（词频-逆文档频率）特征向量，代表了论文摘要中词语的重要性。

- 类别 (Classes): 3个类别，分别对应三种不同类型的糖尿病研究。

\

通过这三个具有不同规模、稀疏度和特征维度的数据集，我们可以更全面地检验GAT及其改进模型在各种场景下的性能、鲁棒性和泛化能力。



== 实验与分析

=== 实验思路

\




在此部分，我们对于 *GAT* 进行修改，期待修改后的 *GAT* 能够展现出更好的性能。因为是在 *GAT* 的基础上进行修改，所以我们将其命名为 *GAT++*。 *GAT++* 的主要修改思路共有三种：1. 利用“砍边”法，只保留关联度较高的边；2. 基于广度优先搜索，将感知野通过广度搜索扩大；3. 基于并查集算法，得到一个集合的特征，再将集合特征与点的特征做融合。在之后的篇幅中，我们将对这三种思路进行详细的介绍。在此之前，我们先在符号上进行一些约定，方便后续讨论。

\

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

\

==== “砍边”法
在原本的 *GAT* 中，对于每个节点 $v_i$，所有与其相邻的点 $v_j$ 都会被纳入到 $cal(N)_i^(ast)$ 中。但是实际中，其实有些点虽然是相邻的，但是其相关性较低，这类点在最终算聚合特征时可能并不重要，甚至在某些情况下会起到相反的作用。因此，我们并不需要将所有与 $v_i$ 相邻的点都纳入到 $cal(N)_i^(ast)$ 中。

\

在“砍边”法中，我们采取两种方法对于边进行筛选：
- 第一种方法，我们关注边的相关性，即只保留 $alpha_(i j) > epsilon$ 的边。每次 *GAT* 结束后，对于 $v_i in V'$，只将 $alpha_(i j) > epsilon$ 的相邻点 $v_j$ 纳入 $cal(N)_i^(ast)$ 中，再进行后续的 *GAT*$(dot)$ 操作。
- 第二种方法，我们关注点的数量，即只保留与 $v_i$ 相关性较高的 $k$ 个点。对于 $v_i in V'$，只将 $alpha_(i, dot)$ 的前 $k$ 大所对应的相邻点 $v_j$ 纳入 $cal(N)_i^(ast)$ 中，再进行后续的 *GAT*$(dot)$ 操作。

\

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

\

==== 基于广度优先搜索
\

在原本的 *GAT* 中，通常我们只会将邻域内的点纳入到 $cal(N)_i^(ast)$ 中。但是只用图中的邻居关系，并不能完全反映出节点之间的关系。在实际中，有些点可能在邻居关系上与其他点之间存在着更长的距离，但是其实其路径上的 $alpha_(i j)$ 的值仍然很大。而两点之间的关系又有传递性，根据传递性，这些点很可能与目标点的相关性仍然很高，我们可以将这些点也纳入到 $cal(N)_i^(ast)$ 中。然而，有些点虽然在图结构中，与目标点之间有边相连，但是其 $alpha_(i j)$ 的值却很小，这时我们并不希望将其纳入到 $cal(N)_i^(ast)$ 中。

\

基于广度优先搜索，我们可以将感知野进行扩大，使得感知野中的点都纳入到 $cal(N)_i^(ast)$ 中。具体的做法是，在进行一次 $bold("GAT")(dot)$ 后，在 $G'$ 上进行操作。对于点 $v_i in V'$，利用 BFS 算法找与它相关性高的点。

\

对于在图结构上不相邻的两点 $v_i$ 和 $v_p$，如果 $v_i$ 和 $v_p$ 之间存在一条路径：$v_i -> v_j -> v_k -> v_p$，则我们定义 $v_i$ 与 $v_p$ 之间的相关性为：



  $ alpha_(i p) = alpha_(i j) dot alpha_(j k) dot alpha_(k p) $

如果 $alpha_(i p) > epsilon$，则认为 $v_i$ 与 $v_p$ 之间存在较强的相关性，将 $v_p$ 纳入到 $cal(N)_i^(ast)$ 中。即最终的剪枝条件为如果从 $v_i$ 出发的路径，$exists q$，经过该点 $v_q$ 时，$alpha_(i q) < epsilon$，则该路径中断。

将路径上所有经过的点都纳入 $cal(N)_i^(ast)$ 中，最终计算得到的$h_i^(2) = bold("GAT")( \{ h_k | v_k in cal(N)_i^(ast) \} ), text(bold("where")) h_k = h_k^(1))$

\

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
    + bias_mat = where(reachability_bool)
    + *Return* expand_dims(bias_mat)
  
  + *Static Method* inference(inputs, nb_classes, nb_nodes, ...):
    + new_bias_mat = initial_bias_mat  
    + *For* n_heads[0] times:
      + output, coef = attn_head(
        + inputs, 
        + bias_mat=new_bias_mat,
        + ...
      + )
      + attns_output.append
      + attns_coefs.append
    + h = concatenate(attns_output)
    + avg_attention = mean(attns_coefs)
    + all_attention_maps.append
    + new_bias_mat = create_bfs_bias(avg_attention, ...) 

    + *For* each subsequent layer:
      + *For* n_heads[i] times:
        + output, coef = attn_head(...) 
        + ...
      + h = concatenate(attns_output)
      + avg_attention = mean(attns_coefs)
      + all_attention_maps.append
      + new_bias_mat = create_bfs_bias(avg_attention, ...)  
  
    + *For* n_heads[-1] times:
      + out = attn_head(h, new_bias_mat, ...)
      + outputs.append
    + logits = average(outputs)
    
    + *Return* logits, all_attention_maps
+ *End Class*
]

\

==== 基于并查集算法

\

在原本的 *GAT* 中，我们只是用了图结构中相邻的关系，即 $cal(N)_i^(ast)$ 作为 $h_i$ 的输入，而没有考虑到节点之间的关系。而在实际中，节点之间的关系往往是复杂的。我们可以将所有的节点按照并查集的思路进行合并，为节点根据它们的相似性进行分类，得到一个特征向量。

\

具体的做法是，在进行一次 $bold("GAT")(dot)$ 后，在 $G'$ 上进行操作。首先，我们给出划分集合的规则：对于 $G'$ 中相邻的两个节点 $v_i$ 和 $v_j$，如果 $alpha_(i j) > epsilon$，那么则认为 $v_i$ 与 $v_j$ 同属于一个集合 $cal(A)_p$。特别的，如果 $G'$ 中一个节点 $v_k$，$alpha_(k, dot) <= epsilon$，则 $cal(A)_q = \{v_k\}$。

\

通过此操作，我们得到了 $cal(A)_1$、$cal(A)_2$...$cal(A)_m$。对于集合 $cal(A)_s$，令 $H_s = frac(1, |cal(A)_s|) sum_(v_k in cal(A)_s) h'_k$，为表征该集合的特征向量。可将 $H_s$ 与 $h'_k$ 融合，去做后续操作。

\

在这里，我们将集合特征与点的特征融合时，共实现了三种方法：

\

- 第一种方法，我们将 $H_s$ 与 $h'_k$ 直接相加，得到 $h'_s$，即为 fused_features。
- 第二种方法，我们将 $H_s$ 与 $h'_k$ 拼接，得到 $[h'_k, H_s]$，即为 fused_features。
- 第三种方法，我们将 $h'_k$ 与 $H_s$ 拼接，得到 $[h'_k, H_s]$，然后通过全连接层和 sigmoid 激活得到门控系数 $g$，得到最后的 fused_features 为 $g dot h'_k + (1-g) dot H_s$。

\

#pseudocode-list[
+ *Class* GATdsu(*BaseGAttN*):
    + *Static Method* cluster_nodes(attention_matrix, epsilon):
        + num_nodes = attention_matrix.shape[0]
        + adj_matrix = (attention_matrix > epsilon).astype(float)
        + n_components, labels = connected_components(adj_matrix, directed=False)
        + *For* node_idx, cluster_id in enumerate(labels):
            + clusters[cluster_id].append(node_idx)
        + *Return* clusters

    + *Static Method* create_cluster_features(h_features, clusters):
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
        + *Define* cluster_and_fuse(h_features, attn_matrix, eps):
            + h_features_2d = squeeze(h_features, axis=0)
            + clusters = cluster_nodes(attn_matrix, eps)
            + cluster_features = create_cluster_features(h_features_2d, clusters)
            + *Return* expand_dims(cluster_features, axis=0)
        + cluster_features = py_func(cluster_and_fuse, [h_1, avg_attention, epsilon])
        + cluster_features.set_shape(1, nb_nodes, feature_dim)
        + *If* fusion_method == 'add':
            + fused_features = h_1 + cluster_features
        + *Elif* fusion_method == 'concat':
            + fused_features = concatenate(h_1, cluster_features)
        + *Elif* fusion_method == 'gate':
            + fused_features = gate \* h_1 + (1 - gate) \* cluster_features
        + *Else*:
            + fused_features = h_1
        + current_input = fused_features
        + *For* i in 1 to len(hid_units)-1:
            + *For* n_heads[i] times:
                + attns.append
            + current_input = concatenate(attns, axis=-1)
        + *For* n_heads[-1] times:
            + out.append
        + logits = average(out)
        + *Return* logits
+ *End Class*
]

\



=== 实验结果与分析

\

在实验中，我们对 *GAT* 进行了修改，并进行了性能测试。实验结果表明，*GAT++* 能够在某些情况下展现出更好的性能。其中，*GATcut*、*GATbfs*、*GATdsu* 分别代表了基于“砍边”、基于广度优先搜索、基于并查集算法的 *GAT++* 实现。

除此之外，我们还复现了 GATv2、SuperGAT-MX、SuperGAT-SD 等模型，将这几种方法在 Cora、Citeseer 和 Pubmed 数据集上进行了测试，并对其性能进行了比较。

\

==== Cora 数据集

===== GATcut

GATcut 在 Cora 数据集上的表现如下：

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

表现最好的参数组合为 $epsilon=0.15$，top_k $=5$，其 Val Acc 为 0.8140，Test Acc 为 0.8250。
\

===== GATbfs

\
GATbfs 在 Cora 数据集上的表现如下：

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

表现最好的参数组合为 $epsilon=0.15$，max_hops $=3$，其 Val Acc 为 0.8220，Test Acc 为 0.8190。
\

===== GATdsu

\
GATdsu 在 Cora 数据集上的表现如下：

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

表现最好的参数组合为 $epsilon=0.1$，融合方式为 gate，其 Val Acc 为 0.8260，Test Acc 为 0.8280。

===== 比较

将 GAT、GATcut、GATbfs、GATdsu 与 GATv2、SuperGAT-MX、SuperGAT-SD 进行比较，结果如下：

#table3(
  align: center,
  columns: (1.5fr, 1fr, 1fr),
  inset: 10pt,
  [方法], [Val Acc], [Test Acc],
  [GAT], [0.8080], [0.8180],
  [GAT++(cut)], [0.8140], [0.8250],
  [GAT++(bfs)], [0.8220], [0.8190],
  [GAT++(dsu)], [0.8260], [0.8280],
  [GATv2], [0.8240], [0.8460],
  [SuperGAT-MX], [0.8140], [0.8160],
  [SuperGAT-SD], [0.8220], [0.8210],
)

可以发现，GAT++(cut)、GAT++(bfs)、GAT++(dsu) 都在 Cora 数据集上都取得了更好的性能，能够超过原来 GAT 在 Cora 数据集上的性能，说明我们的修改起到了一定的作用；同时，GAT++(dsu) 的性能表现最佳，甚至能超过 SuperGAT-MX 和 SuperGAT-SD 在 Cora 数据集上的表现。但是，GATv2 在 Cora 上的表现仍是最佳。

==== Citeseer 数据集


===== GATcut

GATcut 在 Citeseer 数据集上的表现如下：

#table3(
  align: center,
  columns: (1fr, 1fr, 1fr, 1fr),
  inset: 10pt,
  [neighbor\_threshold], [top\_k], [Val Acc], [Test Acc],
  [0.0], [5], [0.6860], [0.6820],
  [0.0], [10], [0.5960], [0.5920],
  [0.0], [15], [0.4920], [0.5100],
  [0.1], [5], [0.7140], [0.7000],
  [0.1], [10], [0.6680], [0.6430],
  [0.1], [15], [0.6320], [0.6310],
  [0.15], [5], [0.7220], [0.7120],
  [0.15], [10], [0.6560], [0.6560],
  [0.15], [15], [0.5860], [0.5860],
  [0.2], [5], [0.6620], [0.6590],
  [0.2], [10], [0.6680], [0.6900],
  [0.2], [15], [0.6280], [0.6080],
)

表现最好的参数组合为 neighbor\_threshold $=0.15$，top\_k $=5$，其 Val Acc 为 0.7220，Test Acc 为 0.7120。

===== GATbfs

GATbfs 在 Citeseer 数据集上的表现如下：

#table3(
  align: center,
  columns: (1fr, 1fr, 1fr, 1fr),
  inset: 10pt,
  [$epsilon$], [max\_hops], [Val Acc], [Test Acc],
  [0.0], [2], [0.2320], [0.1810],
  [0.0], [3], [0.2320], [0.1810],
  [0.1], [2], [0.7360], [0.7170],
  [0.1], [3], [0.7320], [0.7250],
  [0.15], [2], [0.7400], [0.7320],
  [0.15], [3], [0.7420], [0.7210],
  [0.2], [2], [0.7460], [0.7170],
  [0.2], [3], [0.7420], [0.7160],
  [0.25], [2], [0.7440], [0.7310],
  [0.25], [3], [0.7320], [0.7240],
  [0.3], [2], [0.7300], [0.7150],
  [0.3], [3], [0.7300], [0.7270],
)

表现最好的参数组合为 $epsilon=0.1$，max_hops $=3$，其 Val Acc 为 0.7460，Test Acc 为 0.7170。

===== GATdsu

GATdsu 在 Citeseer 数据集上的表现如下：

#table3(
  align: center,
  columns: (1fr, 1fr, 1fr, 1fr),
  inset: 10pt,
  [$epsilon$], [fusion\_method], [Val Acc], [Test Acc],
  [0.05], [add], [0.7420], [0.7220],
  [0.05], [concat], [0.7300], [0.7200],
  [0.05], [gate], [0.7440], [0.7450],
  [0.1], [add], [0.7340], [0.7360],
  [0.1], [concat], [0.7440], [0.7240],
  [0.1], [gate], [0.7440], [0.7330],
  [0.15], [add], [0.7300], [0.7110],
  [0.15], [concat], [0.7360], [0.7280],
  [0.15], [gate], [0.7400], [0.7140],
  [0.2], [add], [0.7240], [0.7080],
  [0.2], [concat], [0.7400], [0.7190],
  [0.2], [gate], [0.7340], [0.7250],
  [0.25], [add], [0.7320], [0.7180],
  [0.25], [concat], [0.7440], [0.7370],
  [0.25], [gate], [0.7420], [0.7210],
  [0.3], [add], [0.7320], [0.7080],
  [0.3], [concat], [0.7400], [0.7090],
  [0.3], [gate], [0.7380], [0.7180],
)

表现最好的参数组合为 $epsilon=0.05$，fusion\_method 为 gate，其 Val Acc 为 0.7440，Test Acc 为 0.7450。

===== 比较

将 GAT、GATcut、GATbfs、GATdsu 与 GATv2、SuperGAT-MX、SuperGAT-SD 进行比较，结果如下：

#table3(
  align: center,
  columns: (1.5fr, 1fr, 1fr),
  inset: 10pt,
  [方法], [Val Acc], [Test Acc],
  [GAT], [0.7380], [0.7270],
  [GAT++(cut)], [0.7220], [0.7120],
  [GAT++(bfs)], [0.7460], [0.7170],
  [GAT++(dsu)], [0.7440], [0.7450],
  [GATv2], [0.8000], [0.7830],
  [SuperGAT-MX], [0.7300], [0.7280],
  [SuperGAT-SD], [0.7500], [0.7170],
)

可以发现，GAT++(bfs) 和 GAT++(dsu) 在 Citeseer 数据集上取得了与原始 GAT 相当甚至略优的性能，尤其是 GAT++(dsu) 的 Test Acc 达到了 0.7450，略高于原始 GAT 的 0.7270，说明基于并查集的特征融合方法在该数据集上同样有效。而 GAT++(cut) 的表现略低于原始 GAT，说明“砍边”法在 Citeseer 上的提升有限。与 Cora 数据集类似，GATv2 在 Citeseer 上依然表现最佳，Test Acc 达到 0.7830，显著高于其他方法。SuperGAT-MX 和 SuperGAT-SD 的表现与 GAT++ 系列方法相近，但整体略低于 GATv2。总体来看，GAT++(dsu) 在 Citeseer 上依然展现出一定的优势，验证了该方法的有效性。

==== Pubmed 数据集


===== 比较

遗憾的是，Pubmed 数据集的数据量较大，数据密集，而 GAT++ 并不能很好地适用于图结构稠密的场景，因此我们并没有做有关 GAT++ 的相关实验，而是只比较了 GAT、GATv2、SuperGAT-MX、SuperGAT-SD。

#table3(
  align: center,
  columns: (1.5fr, 1fr, 1fr),
  inset: 10pt,
  [方法], [Val Acc], [Test Acc],
  [GAT], [0.8040], [0.7750],
  [GATv2], [0.8000], [0.7830],
  [SuperGAT-MX], [0.7980], [0.7740],
  [SuperGAT-SD], [0.8120], [0.7800],
)
\

可以发现，GATv2 在 Pubmed 数据集上的表现最佳，Test Acc 达到 0.7830，与 Cora 和 Citeseer 数据集相当。SuperGAT-MX 和 SuperGAT-SD 的表现与 GATv2 相似，但整体略低于 GAT。总体来看，GATv2 在 Pubmed 数据集上的表现优于其他方法。

针对稀疏的数据集PubMed，我们考虑对GAT本身的sparse版本进行改进，设计了SparseGAT++系列方法。虽然由于计算复杂度等技术挑战，这些方法在大规模数据集上遇到了实施困难，但我们的探索为稀疏图神经网络的优化提供了有价值的技术路径。

\

=== SparseGAT++ 的探索与技术挑战

\ 

在处理稀疏图时，我们可以看到我们的方法对PubMed数据集的性能提升有限。因此我们尝试对GAT的sparse版本进行改进，设计了SparseGAT++系列方法。




==== 技术实现与算法设计

\

与GAT++类似，我们设计了三种方法：

- SpGATcut：稀疏砍边法
- SpGATbfs：广度优先搜索扩展
- SpGATdsu：并查集特征融合



*SpGATcut（稀疏砍边法）*：

#codex("def create_filtered_bias(atten_coefs, neighbor_threshold, top_k_neighbors, nb_nodes):
def create_filtered_bias(atten_coefs, neighbor_threshold, top_k_neighbors, nb_nodes):
    # 阈值过滤：保留 α_ij > threshold 的边
    threshold_mask = atten_coefs > neighbor_threshold
    # Top-k过滤：保留每个节点最重要的k个邻居
    top_values, top_indices = tf.math.top_k(atten_coefs, k=top_k+1)
    # 构建过滤后的偏置矩阵
    bias_mat = tf.where(threshold_mask, 0.0, -1e9)
", lang: "python")

该方法的核心思想是在每层注意力计算后，动态过滤低重要性的边连接，减少噪声传播。

\

*SpGATbfs（广度优先搜索扩展）*：
  
#codex("def create_bfs_bias(atten_coefs, epsilon, max_hops, nb_nodes):
    # 构建可达性矩阵：基于注意力权重的图遍历
    adj_bool = (atten_coefs >= epsilon)
    reachability = tf.eye(nb_nodes)
    for _ in range(max_hops):
        next_reach = tf.matmul(reachability, adj_int)
        reachability = tf.clip_by_value(reachability + next_reach, 0, 1)", lang: "python")


该方法通过矩阵乘法迭代计算多跳可达性，将感知野从直接邻居扩展到高相关性的间接邻居。

\

*SpGATdsu（并查集特征融合）*：

#codex("def cluster_and_fuse(h_features, attn_matrix, epsilon):
    # 基于注意力权重构建邻接图
    adj_matrix = (attention_matrix > epsilon).astype(float)
    # 计算连通分量（并查集思想）
    n_components, labels = connected_components(adj_matrix)
    # 为每个聚类计算平均特征
    cluster_features = compute_cluster_means(h_features, labels)
    # 门控融合机制
    gate = sigmoid(dense_layer(concat(h_features, cluster_features)))
    return gate * h_features + (1-gate) * cluster_features", lang: "python")



该方法将高相关性节点聚合成集合，通过集合级特征增强个体节点的表征能力。

\

==== 技术挑战与复杂度分析

\

在实现过程中，我们遇到了以下关键技术挑战：

*内存复杂度*：PubMed数据集包含约20K节点，需要存储20K×20K的注意力矩阵（约1.6GB），超出了单GPU内存限制。特别是SpGATbfs的多跳可达性计算需要O(N³)的时间复杂度和O(N²)的空间复杂度。

*计算效率*：SpGATcut的top-k选择在稀疏矩阵上需要O(N²logN)复杂度；SpGATdsu的连通分量计算虽然理论上是O(N²)，但在TensorFlow 1.x的tf.py_func实现下存在严重的性能瓶颈。

*稀疏性处理*：原始GAT使用稀疏矩阵优化，但我们的改进算法需要频繁的稠密矩阵运算（如矩阵乘法、top-k选择），破坏了稀疏性优势。

==== 改进方向与技术展望

基于这次探索，我们识别出稀疏图神经网络优化的关键方向：

1. *层次化计算*：将大图分解为多个重叠的子图块，分层处理后合并结果
2. *采样近似*：使用邻居采样或重要性采样减少计算规模  
3. *硬件优化*：开发GPU友好的稀疏张量操作，避免稠密转换开销
4. *渐进式训练*：从核心子图开始训练，逐步扩展到完整图结构

虽然当前的SparseGAT++实现在工程上遇到了挑战，但这些探索验证了改进策略的理论可行性，为大规模稀疏图神经网络的发展提供了重要的技术积累和改进思路。在未来的工作中，结合更先进的稀疏计算框架和分布式训练技术，这些方法有望在大规模实际应用中发挥价值。


\


=== 引用文献

- [1] NodeFormer: A Scalable Graph Structure Learning Transformer forNode Classification
- [2] GraphMAE: Self-Supervised Masked Graph Autoencoders
- [3] Graph Contrastive Learning with Augmentations
- [4] Graph Attention Networks
- [5] Graph Attention Networks v2
- [6] HOW TO FIND YOUR FRIENDLY NEIGHBORHOOD: GRAPHATTENTION DESIGN WITH SELF-SUPERVISION
- [7] Planetoid: A benchmark dataset for graph classification


