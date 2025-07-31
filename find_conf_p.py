import time
import numpy as np
import tensorflow as tf
import itertools
import os
from collections import defaultdict
from models import SpGAT, SpGATcut, SpGATdsu, SpGATbfs
from utils import process

RESULTS_DIR = "results"
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

# 数据集
DATASET = 'pubmed'

# 训练参数
BATCH_SIZE = 1
NB_EPOCHS = 100000  # 为了快速调参，减少epoch数
PATIENCE = 100
LR = 0.005
L2_COEF = 0.0005
HID_UNITS = [8]  # 隐藏单元数
N_HEADS = [8, 1]  # 注意力头数
RESIDUAL = False
NONLINEARITY = tf.nn.elu

# 每种方法的参数搜索空间
PARAM_SPACES = {
    'gat': [{}],  # 基础GAT没有额外参数
    'cut': [
        {'neighbor_threshold': 0.0, 'top_k_neighbors': 0},
        {'neighbor_threshold': 0.0, 'top_k_neighbors': 5},
        {'neighbor_threshold': 0.0, 'top_k_neighbors': 10},
        {'neighbor_threshold': 0.0, 'top_k_neighbors': 15},
        {'neighbor_threshold': 0.1, 'top_k_neighbors': 0},
        {'neighbor_threshold': 0.1, 'top_k_neighbors': 5},
        {'neighbor_threshold': 0.1, 'top_k_neighbors': 10},
        {'neighbor_threshold': 0.1, 'top_k_neighbors': 15},
        {'neighbor_threshold': 0.15, 'top_k_neighbors': 0},
        {'neighbor_threshold': 0.15, 'top_k_neighbors': 5},
        {'neighbor_threshold': 0.15, 'top_k_neighbors': 10},
        {'neighbor_threshold': 0.15, 'top_k_neighbors': 15},
        {'neighbor_threshold': 0.2, 'top_k_neighbors': 0}, 
        {'neighbor_threshold': 0.2, 'top_k_neighbors': 5},
        {'neighbor_threshold': 0.2, 'top_k_neighbors': 10},
        {'neighbor_threshold': 0.2, 'top_k_neighbors': 15},
    ],
    'bfs': [
        {'epsilon': 0.0, 'max_hops': 2},
        {'epsilon': 0.0, 'max_hops': 3},
        {'epsilon': 0.1, 'max_hops': 2},
        {'epsilon': 0.1, 'max_hops': 3},
        {'epsilon': 0.15, 'max_hops': 2},
        {'epsilon': 0.15, 'max_hops': 3},
        {'epsilon': 0.2, 'max_hops': 2},
        {'epsilon': 0.2, 'max_hops': 3},
        {'epsilon': 0.25, 'max_hops': 2},
        {'epsilon': 0.25, 'max_hops': 3},
        {'epsilon': 0.3, 'max_hops': 2},
        {'epsilon': 0.3, 'max_hops': 3},
    ],
    'dsu': [
        {'epsilon': 0.05, 'fusion_method': 'add'},
        {'epsilon': 0.05, 'fusion_method': 'concat'},
        {'epsilon': 0.05, 'fusion_method': 'gate'},
        {'epsilon': 0.1, 'fusion_method': 'add'},
        {'epsilon': 0.1, 'fusion_method': 'concat'},
        {'epsilon': 0.1, 'fusion_method': 'gate'},
        {'epsilon': 0.15, 'fusion_method': 'add'},
        {'epsilon': 0.15, 'fusion_method': 'concat'},
        {'epsilon': 0.15, 'fusion_method': 'gate'},
        {'epsilon': 0.2, 'fusion_method': 'add'},
        {'epsilon': 0.2, 'fusion_method': 'concat'},
        {'epsilon': 0.2, 'fusion_method': 'gate'},
        {'epsilon': 0.25, 'fusion_method': 'add'},
        {'epsilon': 0.25, 'fusion_method': 'concat'},
        {'epsilon': 0.25, 'fusion_method': 'gate'},
        {'epsilon': 0.3, 'fusion_method': 'add'},
        {'epsilon': 0.3, 'fusion_method': 'concat'},
        {'epsilon': 0.3, 'fusion_method': 'gate'},
    ]
}

def load_and_preprocess_data(dataset):
    """加载并预处理数据"""
    adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = process.load_data(dataset)
    features, spars = process.preprocess_features(features)
    
    nb_nodes = features.shape[0]
    ft_size = features.shape[1]
    nb_classes = y_train.shape[1]
     
    # 添加批次维度
    features = features[np.newaxis]
    y_train = y_train[np.newaxis]
    y_val = y_val[np.newaxis]
    y_test = y_test[np.newaxis]
    train_mask = train_mask[np.newaxis]
    val_mask = val_mask[np.newaxis]
    test_mask = test_mask[np.newaxis]
    
    biases = process.preprocess_adj_bias(adj)
    
    return {
        'features': features,
        'biases': biases,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test,
        'train_mask': train_mask,
        'val_mask': val_mask,
        'test_mask': test_mask,
        'nb_nodes': nb_nodes,
        'ft_size': ft_size,
        'nb_classes': nb_classes
    }

def train_model(method, params, data):
    """训练并评估单个模型"""
    # 根据方法选择模型类
    if method == 'gat':
        model_class = SpGAT
    elif method == 'cut':
        model_class = SpGATcut
    elif method == 'bfs':
        model_class = SpGATbfs
    elif method == 'dsu':
        model_class = SpGATdsu
    
    # 创建检查点目录
    checkpt_dir = f"checkpoints/{method}"
    if not os.path.exists(checkpt_dir):
        os.makedirs(checkpt_dir)
    
    # 创建唯一的检查点文件名
    param_str = "_".join([f"{k}_{v}" for k, v in params.items()])
    checkpt_file = f"{checkpt_dir}/model_{param_str}.ckpt"
    
    # 提取数据
    features = data['features']
    biases = data['biases']
    y_train = data['y_train']
    y_val = data['y_val']
    y_test = data['y_test']
    train_mask = data['train_mask']
    val_mask = data['val_mask']
    test_mask = data['test_mask']
    nb_nodes = data['nb_nodes']
    ft_size = data['ft_size']
    nb_classes = data['nb_classes']
    
    best_val_acc = 0.0
    best_test_acc = 0.0
    best_params = params.copy()
    
    with tf.Graph().as_default():
        with tf.name_scope('input'):
            ftr_in = tf.placeholder(dtype=tf.float32, shape=(BATCH_SIZE, nb_nodes, ft_size))
            bias_in = tf.sparse_placeholder(dtype=tf.float32)
            lbl_in = tf.placeholder(dtype=tf.int32, shape=(BATCH_SIZE, nb_nodes, nb_classes))
            msk_in = tf.placeholder(dtype=tf.int32, shape=(BATCH_SIZE, nb_nodes))
            attn_drop = tf.placeholder(dtype=tf.float32, shape=())
            ffd_drop = tf.placeholder(dtype=tf.float32, shape=())
            is_train = tf.placeholder(dtype=tf.bool, shape=())
        
        # 创建模型
        if method == 'gat':
            logits = model_class.inference(
                ftr_in, nb_classes, nb_nodes, is_train,
                attn_drop, ffd_drop,
                bias_mat=bias_in,
                hid_units=HID_UNITS, n_heads=N_HEADS,
                residual=RESIDUAL, activation=NONLINEARITY
            )
        elif method == 'cut':
            logits, _ = model_class.inference(
                ftr_in, nb_classes, nb_nodes, is_train,
                attn_drop, ffd_drop,
                bias_mat=bias_in,
                hid_units=HID_UNITS, n_heads=N_HEADS,
                residual=RESIDUAL, activation=NONLINEARITY,
                **params
            )
            # logits = tf.sparse.to_dense(logits)
        elif method == 'bfs':
            logits, _ = model_class.inference(
                ftr_in, nb_classes, nb_nodes, is_train,
                attn_drop, ffd_drop,
                bias_mat=bias_in,
                hid_units=HID_UNITS, n_heads=N_HEADS,
                residual=RESIDUAL, activation=NONLINEARITY,
                **params
            )
            # logits = tf.sparse.to_dense(logits)
        elif method == 'dsu':
            logits = model_class.inference(
                ftr_in, nb_classes, nb_nodes, is_train,
                attn_drop, ffd_drop,
                bias_mat=bias_in,
                hid_units=HID_UNITS, n_heads=N_HEADS,
                residual=RESIDUAL, activation=NONLINEARITY,
                **params
            )
            # logits = tf.sparse.to_dense(logits)
        
        # 损失和准确率计算
        log_resh = tf.reshape(logits, [-1, nb_classes])
        lab_resh = tf.reshape(lbl_in, [-1, nb_classes])
        msk_resh = tf.reshape(msk_in, [-1])
        loss = model_class.masked_softmax_cross_entropy(log_resh, lab_resh, msk_resh)
        accuracy = model_class.masked_accuracy(log_resh, lab_resh, msk_resh)
        
        # 训练操作
        train_op = model_class.training(loss, LR, L2_COEF)
        
        saver = tf.train.Saver()
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        
        with tf.Session() as sess:
            sess.run(init_op)
            
            best_val_acc = 0.0
            best_test_acc = 0.0
            curr_step = 0
            best_epoch = 0
            
            for epoch in range(NB_EPOCHS):
                # 训练
                _, loss_value_tr, acc_tr = sess.run([train_op, loss, accuracy],
                    feed_dict={
                        ftr_in: features,
                        bias_in: biases,
                        lbl_in: y_train,
                        msk_in: train_mask,
                        is_train: True,
                        attn_drop: 0.6, ffd_drop: 0.6
                    })
                
                # 验证
                loss_value_val, acc_val = sess.run([loss, accuracy],
                    feed_dict={
                        ftr_in: features,
                        bias_in: biases,
                        lbl_in: y_val,
                        msk_in: val_mask,
                        is_train: False,
                        attn_drop: 0.0, ffd_drop: 0.0
                    })
                
                # 测试（仅用于记录，不用于模型选择）
                loss_value_test, acc_test = sess.run([loss, accuracy],
                    feed_dict={
                        ftr_in: features,
                        bias_in: biases,
                        lbl_in: y_test,
                        msk_in: test_mask,
                        is_train: False,
                        attn_drop: 0.0, ffd_drop: 0.0
                    })
                
                # 打印进度
                if epoch % 50 == 0:
                    print(f"Method: {method}, Params: {params}, Epoch: {epoch}, "
                          f"Train Loss: {loss_value_tr:.4f}, Train Acc: {acc_tr:.4f}, "
                          f"Val Acc: {acc_val:.4f}, Test Acc: {acc_test:.4f}")
                
                # 检查是否是最佳验证准确率
                if acc_val > best_val_acc:
                    best_val_acc = acc_val
                    best_test_acc = acc_test
                    best_epoch = epoch
                    saver.save(sess, checkpt_file)
                    curr_step = 0
                else:
                    curr_step += 1
                    if curr_step >= PATIENCE:
                        print(f"Early stopping at epoch {epoch}, best epoch: {best_epoch}")
                        break
            
            # 恢复最佳模型
            saver.restore(sess, checkpt_file)
            
            # 最终评估
            final_loss_test, final_acc_test = sess.run([loss, accuracy],
                feed_dict={
                    ftr_in: features,
                    bias_in: biases,
                    lbl_in: y_test,
                    msk_in: test_mask,
                    is_train: False,
                    attn_drop: 0.0, ffd_drop: 0.0
                })
            
            print(f"Method: {method}, Params: {params}, "
                  f"Final Test Acc: {final_acc_test:.4f}, Best Val Acc: {best_val_acc:.4f}")
            
            # 删除检查点文件
            try:
                os.remove(checkpt_file + ".meta")
                os.remove(checkpt_file + ".index")
                os.remove(checkpt_file + ".data-00000-of-00001")
            except:
                pass
            
            return best_val_acc, final_acc_test

def run_parameter_search():
    """运行参数搜索实验"""
    # 加载数据
    print("Loading and preprocessing data...")
    data = load_and_preprocess_data(DATASET)
    print(f"Data loaded. Nodes: {data['nb_nodes']}, Features: {data['ft_size']}, Classes: {data['nb_classes']}")
    
    # 存储结果
    results = defaultdict(list)
    best_results = {}
    
    # 为每种方法运行参数搜索
    for method in ['cut', 'dsu', 'gat', 'bfs']:
        print(f"\n{'='*50}")
        print(f"Starting parameter search for {method.upper()}")
        print(f"Parameter space: {PARAM_SPACES[method]}")
        print(f"{'='*50}")
        
        best_val_acc = 0.0
        best_test_acc = 0.0
        best_params = {}
        
        for params in PARAM_SPACES[method]:
            print(f"\nTraining with params: {params}")
            start_time = time.time()
            
            try:
                val_acc, test_acc = train_model(method, params, data)
                results[method].append({
                    'params': params,
                    'val_acc': val_acc,
                    'test_acc': test_acc
                })
                
                # 更新最佳结果
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_test_acc = test_acc
                    best_params = params
                
                elapsed = time.time() - start_time
                print(f"Completed in {elapsed:.2f} seconds. Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}")
            
            except Exception as e:
                print(f"Error training model with params {params}: {str(e)}")
                results[method].append({
                    'params': params,
                    'error': str(e)
                })
        
        # 保存该方法的最终结果
        best_results[method] = {
            'params': best_params,
            'val_acc': best_val_acc,
            'test_acc': best_test_acc
        }
        
        print(f"\nBest for {method.upper()}: Params: {best_params}, Val Acc: {best_val_acc:.4f}, Test Acc: {best_test_acc:.4f}")
    
    # 保存所有结果
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    results_file = os.path.join(RESULTS_DIR, f"results_{timestamp}.txt")
    
    with open(results_file, 'w') as f:
        f.write("GAT Model Parameter Search Results\n")
        f.write(f"Dataset: {DATASET}\n")
        f.write(f"Date: {timestamp}\n\n")
        
        f.write("Fixed Parameters:\n")
        f.write(f"  Batch Size: {BATCH_SIZE}\n")
        f.write(f"  Learning Rate: {LR}\n")
        f.write(f"  L2 Coef: {L2_COEF}\n")
        f.write(f"  Hidden Units: {HID_UNITS}\n")
        f.write(f"  Attention Heads: {N_HEADS}\n")
        f.write(f"  Residual: {RESIDUAL}\n")
        f.write(f"  Nonlinearity: {NONLINEARITY.__name__}\n")
        f.write(f"  Max Epochs: {NB_EPOCHS}\n")
        f.write(f"  Patience: {PATIENCE}\n\n")
        
        # 写入每种方法的最佳结果
        f.write("Best Results Summary:\n")
        for method, res in best_results.items():
            f.write(f"{method.upper()}:\n")
            f.write(f"  Params: {res['params']}\n")
            f.write(f"  Validation Accuracy: {res['val_acc']:.4f}\n")
            f.write(f"  Test Accuracy: {res['test_acc']:.4f}\n\n")
        
        # 写入所有详细结果
        f.write("Detailed Results:\n")
        for method, runs in results.items():
            f.write(f"\n{method.upper()} Results:\n")
            for i, run in enumerate(runs):
                f.write(f"  Run {i+1}:\n")
                f.write(f"    Params: {run['params']}\n")
                if 'error' in run:
                    f.write(f"    Error: {run['error']}\n")
                else:
                    f.write(f"    Val Acc: {run['val_acc']:.4f}\n")
                    f.write(f"    Test Acc: {run['test_acc']:.4f}\n")
    
    print(f"\nAll results saved to {results_file}")
    
    # 打印最佳结果摘要
    print("\nBest Results Summary:")
    for method, res in best_results.items():
        print(f"{method.upper()}:")
        print(f"  Params: {res['params']}")
        print(f"  Validation Accuracy: {res['val_acc']:.4f}")
        print(f"  Test Accuracy: {res['test_acc']:.4f}\n")

if __name__ == "__main__":
    run_parameter_search()