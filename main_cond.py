from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json, logging
import os, sys
import scipy.stats as stats
import random
import numpy as np
import torch
import torch.nn.functional as F

import time
from model.model import *
from data_process import get_dataset, PyG

from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.metrics import average_precision_score, ndcg_score
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from scipy.linalg import sqrtm
from scipy.stats import norm
from torch_geometric.utils import negative_sampling as pyg_negative_sampling

from utils import *
from model.base_gnn import *
from model.model_cond import *

from evaluate_embedding import evaluate_embedding  # 导入评估函数
from torch_geometric.utils import to_dense_adj, to_dense_batch

torch.set_printoptions(profile='full')
np.set_printoptions(precision=4, threshold=sys.maxsize)

def mean_trials(out_list, name='', log_file=None):
    ci = 1.96 * st.sem(out_list) / math.sqrt(len(out_list))
    log = name + ' Mean: {:.3f} Std: {:.3f}' \
            .format(np.mean(out_list), st.sem(out_list)) 
    print(log)
    return log

def evaluate(model, data, args, device):
    model.eval()
    with torch.no_grad():
        embeddings, _ = model.encoder.get_embeddings(data)
        embeddings = embeddings.cpu().numpy()  # 获取节点嵌入

    if args.task == 'link':
        # 链路预测评估，使用 MRR、NDCG、H1 指标
        edge_index = data.test_pos_edge_index.cpu().numpy()
        edge_labels = np.ones(edge_index.shape[1])

        # 负采样的边（需要与正样本对应）
        neg_edge_index = data.test_neg_edge_index.cpu().numpy()
        # 注意，这里假设 neg_edge_index 的形状为 [2, num_samples * neg_samples]
        # 并且每个正样本对应 neg_samples 个负样本

        results = evaluate_embedding(
            embeddings,
            labels=None,
            edge_index=edge_index,
            edge_labels=edge_labels,
            neg_edge_index=neg_edge_index,
            task='link',
            metrics='ranking'
        )

        MRR = results['MRR']
        NDCG = results['NDCG']
        H1 = results['H1']

        print(f"Link Prediction MAP/MRR: {MRR:.4f}, NDCG: {NDCG:.4f}, H1: {H1:.4f}")
        return MRR,NDCG,H1  # 可以根据需要返回其他指标
        
    elif args.task == 'node':
        # 节点分类评估
        labels = data.y.cpu().numpy()
        embeddings = embeddings.cpu().numpy()
        acc_val, acc = evaluate_embedding(embeddings, labels, task='node')
        print(f"Node Classification Accuracy (Val): {acc_val:.4f}, Accuracy (Test): {acc:.4f}")
        return acc
    else:
        raise ValueError("Invalid task type. Choose 'node' or 'link'.")

# def train_diffusion(embeddings, neigh, n_nodes, num, d_optimizer, diffusion_model, device, args):
#     nei_output = embeddings[neigh]
#     n_output = embeddings[np.repeat(np.arange(n_nodes), num)]
#     for _ in range(args.d_epoch):
#         d_optimizer.zero_grad()
#         dif_loss = diffusion_model(nei_output, n_output, device)
#         dif_loss.backward(retain_graph=True)
#         d_optimizer.step()

# def get_synthetic_negatives(q, epoch, total_epochs, diffusion_model):
#     max_timesteps = diffusion_model.max_timesteps
#     lamda = 1/2
#     time_step = int(((epoch + 1) / total_epochs) ** lamda * max_timesteps)
#     h_syn = diffusion_model.sample(q.shape, q, time_steps=time_step)
#     return h_syn

def run(args, data, save_path, seed, device):

    #split batch
    num_nodes = data.feat.size(0)
    batch_size = 21 
    indices = torch.arange(num_nodes)  
    node_batches = torch.chunk(indices, batch_size)

    val_loader = DataLoader([data], batch_size=128, shuffle=True)

    print(f"Running task: {args.task}")

    model_path = os.path.join(save_path, 'model_{:d}.pkl'.format(seed))
    out_feat = data.feat.shape[1]

    dataset_num_features = out_feat
    dataset_num = data.num_ent  # 假设数据中有 num_ent 属性
    hidden_dim = args.nhid
    num_gc_layers = args.num_layers  # 假设 args 中有 num_layers 参数

    # 实例化模型,确保传入的参数正确
    
    diffusion_model = Diffusion_Cond(batch_size, dataset_num_features, args, device)
    
    d_optimizer = torch.optim.Adam(diffusion_model.parameters(), lr=0.01, weight_decay=args.decay)

    data.to_device(device)

    # 准备数据
    n_nodes = data.feat.shape[0]
    num = 20  # 邻居数量

    # 生成用于 diffusion 训练的邻居索引
    no_self = data.adj - torch.eye(data.num_ent, device=device)
    neigh = torch.zeros(num * n_nodes, dtype=int, device=device)

    for i in range(n_nodes):
        nei = torch.where(no_self[i] > 0)[0]
        nei = nei[torch.randperm(nei.shape[0])]
        size = nei.shape[0]
        if size >= num:
            nei1 = nei[:num]
        elif size == 0:
            nei1 = torch.tensor([i] * num, device=device)
        else:
            repeats = int(np.ceil(num / size))
            nei1 = nei.repeat(repeats)[:num]
        neigh[i * num:(i + 1) * num] = nei1

    # 训练 Diffusion_Cond 模型
    if not args.no_train:
        print('Training Diffusion_Cond ...')
        y_dim = data.y.size(-1)  # 使用全量数据的标签维度
        diffusion_model.initialize_gcn_model(args, x_dim=data.feat.size(-1), y_dim=y_dim)
        for epoch in range(args.epoch):
            diffusion_model.train()
            train_loss_epoch = 0.0

            x = data.feat
            y = data.y

            # 提取子图邻接矩阵
            adj = data.adj
            num_ent = adj.size(0)  # 当前子图节点数量
            drop_prob = 0.3
            node_mask = torch.rand((batch_size, num_ent), device=x.device) > drop_prob


            # 使用扩散模型生成增强后的节点特征和边信息
            noisy_data = diffusion_model.apply_noise(x, adj, y, node_mask)
            # print("noisy_data['X_t'] mean:", noisy_data['X_t'].float().mean().item(), "std:", noisy_data['X_t'].float().std().item())
            # print("noisy_data['E_t'] mean:", noisy_data['E_t'].float().mean().item(), "std:", noisy_data['E_t'].float().std().item())
            x_aug, edge_aug = diffusion_model.forward(args, noisy_data, adj)
                
            # torch.autograd.set_detect_anomaly(True)
            # 计算扩散模型的损失
            diffusion_loss = diffusion_model.compute_train_loss(x_aug, edge_aug, data, adj, noisy_data)
            before_params = {name: param.clone() for name, param in diffusion_model.named_parameters()}
            d_optimizer.zero_grad()
            diffusion_loss.backward()
            d_optimizer.step()

            train_loss_epoch += diffusion_loss.item()

            # train_loss_epoch /= len(node_batches)
            print(f"Epoch {epoch+1}, Training Loss (Diffusion_Cond): {train_loss_epoch:.4f}")

    # 固定 Diffusion_Cond 参数
    for param in diffusion_model.parameters():
        param.requires_grad = False
    model = DiffusionSimCLR(batch_size, dataset_num_features, num_gc_layers, dataset_num, args, diffusion_model).to_device(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)
    # 训练 DiffusionSimCLR 模型
    if not args.no_train:
        best = 0
        cur = 0

        # 节点分类任务
        if args.task == 'node':
            print('Training DiffusionSimCLR for node classification ...')
            for epoch in range(args.epoch):
                model.train()
                optimizer.zero_grad()

                # 获取数据增强
                x = data.feat
                edge_index = data.edges
                batch = torch.zeros(dataset_num, dtype=torch.long).to(device)

                # 使用扩散模型生成增强后的节点特征和边信息，作为对比学习中的正样本和负样本
                diffusion_model.eval()  # 使用之前训练好的扩散模型来生成增强特征
                with torch.no_grad():
                    _, node_mask = to_dense_batch(x=x, batch=batch)
                    noisy_data = diffusion_model.apply_noise(x, edge_index, data.y, node_mask)
                    x_aug, edge_aug = diffusion_model.forward(noisy_data, edge_index)

                # 获取两种视图下的嵌入
                q = model(x, edge_index, batch, num_graphs=1)
                q_aug = model(x_aug, edge_aug, batch, num_graphs=1)

                # 生成合成负样本
                h_syn = model.get_synthetic_negatives(q, epoch, args.epoch)

                # 计算负样本相似度
                negative_sim = torch.mm(q, q_aug.t())
                negative_sim = negative_sim[~torch.eye(q.size(0), dtype=bool).to(device)].view(q.size(0), -1)

                # 计算损失
                contrastive_loss = model.loss_cal(q, q_aug, negative_sim, syn_negatives=h_syn)

                total_loss = contrastive_loss + args.alpha * diffusion_loss

                # 反向传播和优化
                total_loss.backward()
                optimizer.step()

                # 验证和早停逻辑
                if (epoch + 1) % args.eval_interval == 0:
                    test_metric = evaluate(model, data, args, device)
                    if best <= test_metric:
                        best = test_metric
                        cur = 0
                        torch.save({'model': model.state_dict(), 'diffusion': diffusion_model.state_dict()}, model_path)
                    else:
                        cur += 1

                    if cur > args.patience:
                        print('Early stopping!')
                        break

        # 链接预测任务
        elif args.task == 'link':
            print('Training DiffusionSimCLR for link prediction ...')
            pass  # 链接预测的具体实现逻辑可以在这里进行补充

    # 测试
    print('Testing ...')

    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model'])
    diffusion_model.load_state_dict(checkpoint['diffusion'])

    test_metric = evaluate(model, data, args, device)

    # 保存结果
    with open(os.path.join(save_path, 'results.txt'), 'w') as f:
        if args.task == 'node':
            f.write(f'Accuracy: {test_metric}\n')
        elif args.task == 'link':
            f.write(f'AUC: {test_metric}\n')

    return test_metric


if __name__ == '__main__':
    args = parse_args()

    device = "cuda:" + str(args.gpu) if torch.cuda.is_available() else 'cpu'

    np.random.seed(args.seed)
    seed = np.random.choice(100, args.n_runs, replace=False)
    print('Seed: ', seed)

    metrics_list = []

    print('Processing data ...')        
    data = get_dataset(args, args.dataset)
    save_path = prepare_saved_path(args)

    for i in range(args.n_runs): 
        np.random.seed(seed[i])
        torch.manual_seed(seed[i])
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed[i])

        metric = run(args, data, save_path, seed[i], device)
        metrics_list.append(metric)

    if args.task == 'node':
        mean_trials(metrics_list, name='Accuracy')
    elif args.task == 'link':
        MRR_list = [metrics['MRR'] for metrics in metrics_list]
        NDCG_list = [metrics['NDCG'] for metrics in metrics_list]
        H1_list = [metrics['H1'] for metrics in metrics_list]

        mean_trials(MRR_list, name='MRR')
        mean_trials(NDCG_list, name='NDCG')
        mean_trials(H1_list, name='H@1')