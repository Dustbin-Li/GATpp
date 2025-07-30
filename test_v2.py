#!/usr/bin/env python3
"""
GATv2测试脚本
使用PyTorch Geometric中的GATv2Conv实现在Cora数据集上进行测试
基于SuperGAT测试脚本修改
"""

import os
import time
import pickle as pkl
import numpy as np
import networkx as nx
import scipy.sparse as sp
import sys

import torch
import torch.nn.functional as F
from torch.nn import Linear, Dropout
from torch_geometric.nn import GATv2Conv
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx
from sklearn.metrics import accuracy_score, f1_score
import argparse

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=bool)

def load_cora_data(dataset_str='cora'):
    """
    加载Cora数据集，兼容原项目的数据格式
    """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y)+500)

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    return adj, features, labels, train_mask, val_mask, test_mask

def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features

def convert_to_pytorch_geometric(adj, features, labels, train_mask, val_mask, test_mask):
    """
    将数据转换为PyTorch Geometric格式
    """
    # 转换邻接矩阵为edge_index
    adj_coo = adj.tocoo()
    edge_index = torch.tensor(np.vstack([adj_coo.row, adj_coo.col]), dtype=torch.long)
    
    # 转换特征矩阵
    if sp.issparse(features):
        features = features.toarray()
    x = torch.tensor(features, dtype=torch.float)
    
    # 转换标签
    y = torch.tensor(np.argmax(labels, axis=1), dtype=torch.long)
    
    # 转换mask
    train_mask = torch.tensor(train_mask, dtype=torch.bool)
    val_mask = torch.tensor(val_mask, dtype=torch.bool)
    test_mask = torch.tensor(test_mask, dtype=torch.bool)
    
    # 创建PyTorch Geometric数据对象
    data = Data(x=x, edge_index=edge_index, y=y, 
                train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)
    
    return data

class GATv2(torch.nn.Module):
    """
    GATv2模型实现
    使用PyTorch Geometric的GATv2Conv
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_heads=8, dropout=0.6, 
                 concat=True, add_self_loops=True, bias=True):
        super(GATv2, self).__init__()
        
        self.dropout = dropout
        self.num_heads = num_heads
        self.concat = concat
        
        # 第一层GATv2
        self.conv1 = GATv2Conv(
            input_dim, 
            hidden_dim,
            heads=num_heads,
            dropout=dropout,
            concat=concat,
            add_self_loops=add_self_loops,
            bias=bias
        )
        
        # 计算第二层的输入维度
        conv1_output_dim = hidden_dim * num_heads if concat else hidden_dim
        
        # 第二层GATv2  
        self.conv2 = GATv2Conv(
            conv1_output_dim,
            output_dim,
            heads=1,
            concat=False,  # 最后一层通常不concat
            dropout=dropout,
            add_self_loops=add_self_loops,
            bias=bias
        )
        
    def forward(self, x, edge_index, batch=None):
        # 第一层
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        
        # 第二层
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        
        return F.log_softmax(x, dim=1)

def train_epoch(model, data, optimizer, device):
    """训练一个epoch"""
    model.train()
    optimizer.zero_grad()
    
    logits = model(data.x, data.edge_index)
    
    # 计算节点分类损失（GATv2不需要额外的自监督损失）
    loss = F.nll_loss(logits[data.train_mask], data.y[data.train_mask])
    
    loss.backward()
    optimizer.step()
    
    return loss.item()

def evaluate(model, data, mask):
    """评估模型"""
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        pred = logits[mask].max(1)[1]
        acc = accuracy_score(data.y[mask].cpu().numpy(), pred.cpu().numpy())
        f1 = f1_score(data.y[mask].cpu().numpy(), pred.cpu().numpy(), average='macro')
    return acc, f1

def main():
    parser = argparse.ArgumentParser(description='GATv2测试脚本')
    parser.add_argument('--dataset', type=str, default='cora', help='数据集名称')
    parser.add_argument('--hidden_dim', type=int, default=8, help='隐藏层维度')
    parser.add_argument('--num_heads', type=int, default=8, help='注意力头数')
    parser.add_argument('--epochs', type=int, default=1000, help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.005, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='权重衰减')
    parser.add_argument('--dropout', type=float, default=0.6, help='dropout率')
    parser.add_argument('--patience', type=int, default=100, help='早停耐心值')
    parser.add_argument('--concat', action='store_true', default=True, help='是否在第一层concat多头注意力')
    parser.add_argument('--no_self_loops', action='store_true', help='是否不添加自环')
    parser.add_argument('--device', type=str, default='auto', help='设备选择')
    
    args = parser.parse_args()
    
    # 设备选择
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    print(f"GATv2配置: heads={args.num_heads}, concat={args.concat}, self_loops={not args.no_self_loops}")
    
    # 加载数据
    print("加载Cora数据集...")
    adj, features, labels, train_mask, val_mask, test_mask = load_cora_data(args.dataset)
    features = preprocess_features(features)
    data = convert_to_pytorch_geometric(adj, features, labels, train_mask, val_mask, test_mask)
    data = data.to(device)
    
    print(f"数据信息:")
    print(f"  节点数: {data.x.shape[0]}")
    print(f"  特征维度: {data.x.shape[1]}")
    print(f"  边数: {data.edge_index.shape[1]}")
    print(f"  类别数: {torch.max(data.y).item() + 1}")
    print(f"  训练节点: {data.train_mask.sum().item()}")
    print(f"  验证节点: {data.val_mask.sum().item()}")
    print(f"  测试节点: {data.test_mask.sum().item()}")
    
    # 创建模型
    num_classes = torch.max(data.y).item() + 1
    model = GATv2(
        input_dim=data.x.shape[1],
        hidden_dim=args.hidden_dim,
        output_dim=num_classes,
        num_heads=args.num_heads,
        dropout=args.dropout,
        concat=args.concat,
        add_self_loops=not args.no_self_loops
    ).to(device)
    
    print(f"\n模型参数数量: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # 训练
    print("\n开始训练...")
    best_val_acc = 0
    best_test_acc = 0
    patience_counter = 0
    
    start_time = time.time()
    
    for epoch in range(args.epochs):
        # 训练
        loss = train_epoch(model, data, optimizer, device)
        
        # 评估
        train_acc, train_f1 = evaluate(model, data, data.train_mask)
        val_acc, val_f1 = evaluate(model, data, data.val_mask)
        test_acc, test_f1 = evaluate(model, data, data.test_mask)
        
        # 早停检查
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc
            patience_counter = 0
            # 保存最佳模型
            torch.save(model.state_dict(), 'best_gatv2_model.pth')
        else:
            patience_counter += 1
        
        # 打印进度
        if epoch % 20 == 0:
            print(f'Epoch {epoch:3d}: Loss={loss:.4f}, '
                  f'Train={train_acc:.4f}, Val={val_acc:.4f}, Test={test_acc:.4f}')
        
        # 早停
        if patience_counter >= args.patience:
            print(f"\n早停在epoch {epoch}, 最佳验证准确率: {best_val_acc:.4f}")
            break
    
    # 加载最佳模型并进行最终测试
    model.load_state_dict(torch.load('best_gatv2_model.pth'))
    final_test_acc, final_test_f1 = evaluate(model, data, data.test_mask)
    
    training_time = time.time() - start_time
    
    print(f"\n=== 最终结果 ===")
    print(f"训练时间: {training_time:.2f}秒")
    print(f"最佳验证准确率: {best_val_acc:.4f}")
    print(f"对应测试准确率: {best_test_acc:.4f}")
    print(f"最终测试准确率: {final_test_acc:.4f}")
    print(f"最终测试F1分数: {final_test_f1:.4f}")
    
    # 对比分析
    print(f"\n=== GATv2性能分析 ===")
    print(f"- 使用了GATv2注意力机制（改进的注意力计算）")
    print(f"- 隐藏层维度: {args.hidden_dim}")
    print(f"- 注意力头数: {args.num_heads}")
    print(f"- 多头注意力concat: {args.concat}")
    print(f"- 总参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    # 清理临时文件
    if os.path.exists('best_gatv2_model.pth'):
        os.remove('best_gatv2_model.pth')

if __name__ == "__main__":
    main()