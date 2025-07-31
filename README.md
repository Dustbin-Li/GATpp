# GAT++

## Overview
本项目实现了 GAT++，即在原始图注意力网络（GAT）的基础上进行改进的图神经网络模型。GAT++ 主要包括以下三种创新思路：

1. **“砍边”法（Edge Pruning）**：通过注意力分数筛选邻居节点，只保留与目标节点相关性较高的边，从而减少无关信息的干扰，提升模型性能。
2. **基于广度优先搜索的感知野扩展（BFS Neighborhood Expansion）**：利用广度优先搜索算法，扩大每个节点的感知野，使节点能够聚合来自更远距离的信息。
3. **基于并查集的特征融合（DSU Feature Fusion）**：采用并查集算法，将强相关的节点划分为集合，并对集合内的特征进行融合，增强节点表示能力。

本仓库包含了上述三种 GAT++ 变体的实现，并在 Cora、Citeseer 等常用图数据集上进行了实验，验证了各方法的有效性。

### 目录结构

- `data/`：包含 Cora、Citeseer 等数据集的原始文件。
- `models/`：GAT 及 GAT++ 各变体的模型实现。
- `utils/`：数据预处理、评估指标等工具函数。
- `test_super.py`：基于 PyTorch Geometric 的 SuperGAT 测试脚本，可用于对比实验。
- `test_v2.py`：基于 PyTorch Geometric 的 GATv2 测试脚本，可用于对比实验。 
- `execute.py`：GAT 及 GAT++ 的测试窗口。

### 快速开始

1. 安装依赖（推荐使用 Python 3.7 以及 CUDA-10）：

   ```bash
   pip install -r requirements.txt
   ```

2. 运行示例：

   ```bash
   python excute_cora.py
   ```
   注意，如要更换数据集或者更换方法或者更换参数，需要在 excute_cora.py 中手动修改。

   可更换的数据集有：Cora、Citeseer 和 Pubmed
   可更换的方法有：gat、cut、bfs 和 dsu

3. 如需要测试SuperGAT和GATv2，请参考 reproduce.txt 中的环境要求，并使用`test_super.py`和`test_v2.py`进行测试。


### 参考文献

- [Graph Attention Networks (GAT)](https://arxiv.org/abs/1710.10903)
- [GATv2：HOW ATTENTIVE GRAPH ATTENTION NETWORKS?](https://arxiv.org/abs/2105.14491)
- [SuperGAT: Self-supervised Graph Attention Networks](https://arxiv.org/abs/2104.09864)

如有问题欢迎 issue 或 PR！