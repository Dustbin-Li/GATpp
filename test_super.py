#!/usr/bin/env python3
"""
SuperGAT测试脚本
使用PyTorch Geometric中的SuperGAT实现在Cora数据集上进行测试
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
from torch_geometric.nn import SuperGATConv
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

class SuperGAT(torch.nn.Module):
    """
    SuperGAT模型实现
    使用PyTorch Geometric的SuperGATConv
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_heads=8, dropout=0.6, 
                 attention_type='MX', neg_sample_ratio=0.5):
        super(SuperGAT, self).__init__()
        
        self.dropout = dropout
        self.num_heads = num_heads
        
        # 第一层SuperGAT
        self.conv1 = SuperGATConv(
            input_dim, 
            hidden_dim,
            heads=num_heads,
            dropout=dropout,
            attention_type=attention_type,
            neg_sample_ratio=neg_sample_ratio,
            edge_sample_ratio=1.0,
            is_undirected=True
        )
        
        # 第二层SuperGAT  
        self.conv2 = SuperGATConv(
            hidden_dim * num_heads,
            output_dim,
            heads=1,
            concat=False,
            dropout=dropout,
            attention_type=attention_type,
            neg_sample_ratio=neg_sample_ratio,
            edge_sample_ratio=1.0,
            is_undirected=True
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

def get_supergat_loss(model):
    """获取SuperGAT的自监督损失"""
    total_loss = 0
    for layer in [model.conv1, model.conv2]:
        if hasattr(layer, 'get_attention_loss'):
            loss = layer.get_attention_loss()
            if loss is not None:
                total_loss += loss
    return total_loss

def train_epoch(model, data, optimizer, device):
    """训练一个epoch"""
    model.train()
    optimizer.zero_grad()
    
    logits = model(data.x, data.edge_index)
    
    # 计算节点分类损失
    classification_loss = F.nll_loss(logits[data.train_mask], data.y[data.train_mask])
    
    # 计算SuperGAT的自监督损失
    super_gat_loss = get_supergat_loss(model)
    
    # 总损失
    total_loss = classification_loss + 0.001 * super_gat_loss  # 可调整权重
    
    total_loss.backward()
    optimizer.step()
    
    return total_loss.item(), classification_loss.item()

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
    parser = argparse.ArgumentParser(description='SuperGAT测试脚本')
    parser.add_argument('--dataset', type=str, default='cora', help='数据集名称')
    parser.add_argument('--hidden_dim', type=int, default=8, help='隐藏层维度')
    parser.add_argument('--num_heads', type=int, default=8, help='注意力头数')
    parser.add_argument('--epochs', type=int, default=1000, help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.005, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='权重衰减')
    parser.add_argument('--dropout', type=float, default=0.6, help='dropout率')
    parser.add_argument('--patience', type=int, default=100, help='早停耐心值')
    parser.add_argument('--attention_type', type=str, default='MX', 
                       choices=['SD', 'MX'], help='SuperGAT注意力类型')
    parser.add_argument('--device', type=str, default='auto', help='设备选择')
    
    args = parser.parse_args()
    
    # 设备选择
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    print(f"SuperGAT类型: {args.attention_type}")
    
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
    model = SuperGAT(
        input_dim=data.x.shape[1],
        hidden_dim=args.hidden_dim,
        output_dim=num_classes,
        num_heads=args.num_heads,
        dropout=args.dropout,
        attention_type=args.attention_type
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
        total_loss, class_loss = train_epoch(model, data, optimizer, device)
        
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
            torch.save(model.state_dict(), 'best_supergat_model.pth')
        else:
            patience_counter += 1
        
        # 打印进度
        if epoch % 20 == 0:
            print(f'Epoch {epoch:3d}: Loss={total_loss:.4f} (Class={class_loss:.4f}), '
                  f'Train={train_acc:.4f}, Val={val_acc:.4f}, Test={test_acc:.4f}')
        
        # 早停
        if patience_counter >= args.patience:
            print(f"\n早停在epoch {epoch}, 最佳验证准确率: {best_val_acc:.4f}")
            break
    
    # 加载最佳模型并进行最终测试
    model.load_state_dict(torch.load('best_supergat_model.pth'))
    final_test_acc, final_test_f1 = evaluate(model, data, data.test_mask)
    
    training_time = time.time() - start_time
    
    print(f"\n=== 最终结果 ===")
    print(f"训练时间: {training_time:.2f}秒")
    print(f"最佳验证准确率: {best_val_acc:.4f}")
    print(f"对应测试准确率: {best_test_acc:.4f}")
    print(f"最终测试准确率: {final_test_acc:.4f}")
    print(f"最终测试F1分数: {final_test_f1:.4f}")
    
    # 对比不同SuperGAT类型的结果
    print(f"\n=== SuperGAT-{args.attention_type} 性能分析 ===")
    print(f"- 使用了 {args.attention_type} 类型的注意力机制")
    print(f"- 隐藏层维度: {args.hidden_dim}")
    print(f"- 注意力头数: {args.num_heads}")
    print(f"- 总参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    # 清理临时文件
    if os.path.exists('best_supergat_model.pth'):
        os.remove('best_supergat_model.pth')

if __name__ == "__main__":
    main()