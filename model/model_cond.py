from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# from turtle import forward

import tqdm
import logging
import math
from os import path
import re
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader
from utils import *
import time
from model.gcn import *
# from noise_schedule import DiscreteUniformTransition, PredefinedNoiseScheduleDiscrete,\
#     MarginalUniformTransition

def cosine_beta_schedule_discrete(timesteps, s=0.008):
    """ Cosine schedule as proposed in https://openreview.net/forum?id=-NEXDKk8gZ. """
    steps = timesteps + 2
    x = np.linspace(0, steps, steps)

    alphas_cumprod = np.cos(0.5 * np.pi * ((x / steps) + s) / (1 + s)) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    alphas = (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = 1 - alphas
    return betas.squeeze()

def custom_beta_schedule_discrete(timesteps, average_num_nodes=50, s=0.008):
    """ Cosine schedule as proposed in https://openreview.net/forum?id=-NEXDKk8gZ. """
    steps = timesteps + 2
    x = np.linspace(0, steps, steps)

    alphas_cumprod = np.cos(0.5 * np.pi * ((x / steps) + s) / (1 + s)) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    alphas = (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = 1 - alphas

    assert timesteps >= 100

    p = 4 / 5       # 1 - 1 / num_edge_classes
    num_edges = average_num_nodes * (average_num_nodes - 1) / 2

    # First 100 steps: only a few updates per graph
    updates_per_graph = 1.2
    beta_first = updates_per_graph / (p * num_edges)

    betas[betas < beta_first] = beta_first
    return np.array(betas)


class PredefinedNoiseScheduleDiscrete(torch.nn.Module):
    """
    Predefined noise schedule. Essentially creates a lookup array for predefined (non-learned) noise schedules.
    """

    def __init__(self, noise_schedule, timesteps):
        super(PredefinedNoiseScheduleDiscrete, self).__init__()
        self.timesteps = timesteps

        if noise_schedule == 'cosine':
            betas = cosine_beta_schedule_discrete(timesteps)
        elif noise_schedule == 'custom':
            betas = custom_beta_schedule_discrete(timesteps)
        else:
            raise NotImplementedError(noise_schedule)

        self.register_buffer('betas', torch.from_numpy(betas).float())

        self.alphas = 1 - torch.clamp(self.betas, min=0, max=0.9999)

        log_alpha = torch.log(self.alphas)
        log_alpha_bar = torch.cumsum(log_alpha, dim=0)
        self.alphas_bar = torch.exp(log_alpha_bar)
        # print(f"[Noise schedule: {noise_schedule}] alpha_bar:", self.alphas_bar)

    def forward(self, t_normalized=None, t_int=None):
        assert int(t_normalized is None) + int(t_int is None) == 1
        if t_int is None:
            t_int = torch.round(t_normalized * self.timesteps)
        return self.betas[t_int.long()]

    def get_alpha_bar(self, t_normalized=None, t_int=None):
        assert int(t_normalized is None) + int(t_int is None) == 1
        if t_int is None:
            t_int = torch.round(t_normalized * self.timesteps)
        return self.alphas_bar.to(t_int.device)[t_int.long()]
    
def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)


def quadratic_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2

def sigmoid_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start



def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


def normalize_to_neg_one_to_one(emb):
    return emb * 2 - 1

def sample_discrete_features(probX, probE, node_mask):
    ''' Sample features from multinomial distribution with given probabilities (probX, probE, proby)
        :param probX: bs, n, dx_out        node features
        :param probE: bs, n, n, de_out     edge features(we don't have this)
        :param proby: bs, dy_out           global features.
    '''
    bs, n, _ = probX.shape
    # Noise X
    # The masked rows should define probability distributions as well
    probX[~node_mask] = 1 / probX.shape[-1]

    # Flatten the probability tensor to sample with multinomial
    probX = probX.reshape(bs * n, -1)       # (bs * n, dx_out)

    # Sample X
    X_t = probX.multinomial(1)                                  # (bs * n, 1)
    X_t = X_t.reshape(bs, n)     # (bs, n)

    # Noise E
    # The masked rows should define probability distributions as well
    # print(f"node_mask shape: {node_mask.shape}")
    inverse_edge_mask = ~(node_mask.unsqueeze(1) * node_mask.unsqueeze(2))
    diag_mask = torch.eye(n).unsqueeze(0).expand(bs, -1, -1)
    # print(f"inverse_edge_mask shape: {inverse_edge_mask.shape}")
    # print(f"diag_mask shape: {diag_mask.shape}")

    probE[inverse_edge_mask] = 1 / probE.shape[-1]
    probE[diag_mask.bool()] = 1 / probE.shape[-1]

    probE = probE.reshape(bs * n * n, -1)    # (bs * n * n, de_out)

    # Sample E
    E_t = probE.multinomial(1).reshape(bs, n, n)   # (bs, n, n)
    E_t = torch.triu(E_t, diagonal=1)
    E_t = (E_t + torch.transpose(E_t, 1, 2))

    return PlaceHolder(X=X_t, E=E_t)

def compute_posterior_distribution(M, M_t, Qt_M, Qsb_M, Qtb_M):
    ''' M: X or E
        Compute xt @ Qt.T * x0 @ Qsb / x0 @ Qtb @ xt.T
    '''
    # Flatten feature tensors
    M = M.flatten(start_dim=1, end_dim=-2).to(torch.float32)        # (bs, N, d) with N = n or n * n
    M_t = M_t.flatten(start_dim=1, end_dim=-2).to(torch.float32)    # same

    Qt_M_T = torch.transpose(Qt_M, -2, -1)      # (bs, d, d)

    left_term = M_t @ Qt_M_T   # (bs, N, d)
    right_term = M @ Qsb_M     # (bs, N, d)
    product = left_term * right_term    # (bs, N, d)

    denom = M @ Qtb_M     # (bs, N, d) @ (bs, d, d) = (bs, N, d)
    denom = (denom * M_t).sum(dim=-1)   # (bs, N, d) * (bs, N, d) + sum = (bs, N)
    # denom = product.sum(dim=-1)
    # denom[denom == 0.] = 1

    prob = product / denom.unsqueeze(-1)    # (bs, N, d)

    return prob

def mask_distributions(true_X, true_E, pred_X, pred_E, node_mask):
    """
    Set masked rows to arbitrary distributions, so it doesn't contribute to loss
    :param true_X: bs, n, dx_out
    :param true_E: bs, n, n, de_out
    :param pred_X: bs, n, dx_out
    :param pred_E: bs, n, n, de_out
    :param node_mask: bs, n
    :return: same sizes as input
    """

    row_X = torch.zeros(true_X.size(-1), dtype=torch.float, device=true_X.device)
    row_X[0] = 1.
    row_E = torch.zeros(true_E.size(-1), dtype=torch.float, device=true_E.device)
    row_E[0] = 1.

    diag_mask = ~torch.eye(node_mask.size(1), device=node_mask.device, dtype=torch.bool).unsqueeze(0)
    true_X[~node_mask] = row_X
    pred_X[~node_mask] = row_X
    true_E[~(node_mask.unsqueeze(1) * node_mask.unsqueeze(2) * diag_mask), :] = row_E
    pred_E[~(node_mask.unsqueeze(1) * node_mask.unsqueeze(2) * diag_mask), :] = row_E

    true_X = true_X + 1e-7
    pred_X = pred_X + 1e-7
    true_E = true_E + 1e-7
    pred_E = pred_E + 1e-7

    true_X = true_X / torch.sum(true_X, dim=-1, keepdim=True)
    pred_X = pred_X / torch.sum(pred_X, dim=-1, keepdim=True)
    true_E = true_E / torch.sum(true_E, dim=-1, keepdim=True)
    pred_E = pred_E / torch.sum(pred_E, dim=-1, keepdim=True)

    return true_X, true_E, pred_X, pred_E


def posterior_distributions(X, E, y, X_t, E_t, y_t, Qt, Qsb, Qtb):
    prob_X = compute_posterior_distribution(M=X, M_t=X_t, Qt_M=Qt.X, Qsb_M=Qsb.X, Qtb_M=Qtb.X)   # (bs, n, dx)
    prob_E = compute_posterior_distribution(M=E, M_t=E_t, Qt_M=Qt.E, Qsb_M=Qsb.E, Qtb_M=Qtb.E)   # (bs, n * n, de)

    return PlaceHolder(X=prob_X, E=prob_E, y=y_t)

def compute_batched_over0_posterior_distribution(X_t, Qt, Qsb, Qtb):
    """
    Computes the posterior distribution p(x_s, x_t | x_0) using the transition matrices.

    Args:
        X_t: bs, n, dt          or bs, n, n, dt
        Qt: bs, d_t-1, dt
        Qsb: bs, d0, d_t-1
        Qtb: bs, d0, dt.

    Returns:
        torch.Tensor: The posterior distribution p(x_s, x_t | x_0) with shape (bs, n, dx_out, dx_out).
    """
    # bs, n, dx = X_t.shape
    # dx_out = Qt.shape[1] #但是get qt输出是没错的
    # print("Qt shape:", Qt.shape) #wrong shape

    # # Expand transition matrices to match the batch and node dimensions
    # Qt_expanded = Qt.expand(bs, n, dx, dx_out)
    # Qsb_expanded = Qsb.expand(bs, n, dx, dx_out)
    # Qtb_expanded = Qtb.expand(bs, n, dx, dx_out)

    # # Compute the posterior distribution
    # p_s_and_t_given_0 = (Qt_expanded * Qsb_expanded) / Qtb_expanded

    # return p_s_and_t_given_0
    X_t = X_t.flatten(start_dim=1, end_dim=-2).to(torch.float32)            # bs x N x dt

    Qt_T = Qt.transpose(-1, -2)                 # bs, dt, d_t-1
    left_term = X_t @ Qt_T                      # bs, N, d_t-1, 这个有问题
    left_term = left_term.unsqueeze(dim=2)      # bs, N, 1, d_t-1

    right_term = Qsb.unsqueeze(1)               # bs, 1, d0, d_t-1
    # print(f"Shape of left_term: {left_term.shape}")  # 打印left_term的shape
    # print(f"Shape of right_term: {right_term.shape}")  # 打印right_term的shape
    numerator = left_term * right_term          # bs, N, d0, d_t-1

    X_t_transposed = X_t.transpose(-1, -2)      # bs, dt, N

    prod = Qtb @ X_t_transposed                 # bs, d0, N
    prod = prod.transpose(-1, -2)               # bs, N, d0
    denominator = prod.unsqueeze(-1)            # bs, N, d0, 1
    denominator[denominator == 0] = 1e-6

    out = numerator / denominator
    return out


class DiscreteUniformTransition:
    def __init__(self, x_classes: int, e_classes: int, y_classes: int):
        self.X_classes = x_classes
        self.E_classes = e_classes
        self.y_classes = y_classes
        self.u_x = torch.ones(1, self.X_classes, self.X_classes)
        if self.X_classes > 0:
            self.u_x = self.u_x / self.X_classes

        self.u_e = torch.ones(1, self.E_classes, self.E_classes)
        if self.E_classes > 0:
            self.u_e = self.u_e / self.E_classes

        self.u_y = torch.ones(1, self.y_classes, self.y_classes)
        if self.y_classes > 0:
            self.u_y = self.u_y / self.y_classes

    def get_Qt(self, beta_t, device):
        """ Returns one-step transition matrices for X and E, from step t - 1 to step t.
        Qt = (1 - beta_t) * I + beta_t / K

        beta_t: (bs)                         noise level between 0 and 1
        returns: qx (bs, dx, dx), qe (bs, de, de), qy (bs, dy, dy).
        """
        beta_t = beta_t.unsqueeze(1)
        beta_t = beta_t.to(device)
        self.u_x = self.u_x.to(device)
        self.u_e = self.u_e.to(device)
        self.u_y = self.u_y.to(device)

        q_x = beta_t * self.u_x + (1 - beta_t) * torch.eye(self.X_classes, device=device).unsqueeze(0)
        q_e = beta_t * self.u_e + (1 - beta_t) * torch.eye(self.E_classes, device=device).unsqueeze(0)
        q_y = beta_t * self.u_y + (1 - beta_t) * torch.eye(self.y_classes, device=device).unsqueeze(0)

        return PlaceHolder(X=q_x, E=q_e, y=q_y)
    
    def get_Qt_bar(self, alpha_bar_t, device):
        """ Returns t-step transition matrices for X and E, from step 0 to step t.
        Qt = prod(1 - beta_t) * I + (1 - prod(1 - beta_t)) / K

        alpha_bar_t: (bs)         Product of the (1 - beta_t) for each time step from 0 to t.
        returns: qx (bs, dx, dx), qe (bs, de, de), qy (bs, dy, dy).
        """
        alpha_bar_t = alpha_bar_t.unsqueeze(1)
        alpha_bar_t = alpha_bar_t.to(device)
        self.u_x = self.u_x.to(device)
        self.u_e = self.u_e.to(device)
        self.u_y = self.u_y.to(device)

        q_x = alpha_bar_t * torch.eye(self.X_classes, device=device).unsqueeze(0) + (1 - alpha_bar_t) * self.u_x
        q_e = alpha_bar_t * torch.eye(self.E_classes, device=device).unsqueeze(0) + (1 - alpha_bar_t) * self.u_e
        q_y = alpha_bar_t * torch.eye(self.y_classes, device=device).unsqueeze(0) + (1 - alpha_bar_t) * self.u_y

        return PlaceHolder(X=q_x, E=q_e)
    

# 定义位置编码类
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = (self.dim // 2) + 1
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb[:, :self.dim]

# Block类定义
class Block(nn.Module):
    def __init__(self, in_ft, out_ft):
        super(Block, self).__init__()
        self.lin = nn.Linear(in_ft, out_ft)
        self.time = nn.Sequential(
            nn.SiLU(),
            nn.Linear(out_ft, out_ft * 2)
        )

    def forward(self, h, t):
        t = self.time(t)
        scale, shift = t.chunk(2, dim=1)
        h = (scale + 1) * h + shift
        return h

# Encoder类定义
class Encoder(nn.Module):
    def __init__(self, in_ft, out_ft, y=None):
        super(Encoder, self).__init__()
        self.l1 = Block(in_ft, out_ft)
        self.l2 = Block(out_ft, out_ft)
        sinu_pos_emb = SinusoidalPosEmb(out_ft)
        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(out_ft, out_ft),
            nn.GELU(),
            nn.Linear(out_ft, out_ft)
        )

    def forward(self, h, t, y):
        t = self.time_mlp(t)
        if y is not None:
            t += y
        h = self.l1(h, t)
        h = self.l2(h, t)
        return h

# 扩展Diffusion_Cond类，加入图结构扩散模型的增强功能
class Diffusion_Cond(nn.Module):
    def __init__(self, in_feat, out_feat, args, device):
        super(Diffusion_Cond, self).__init__()
        self.device = device
        
        self.timesteps = args.timesteps
        self.betas = self.linear_beta_schedule(timesteps=self.timesteps).to(self.device)
        
        # Compute cumulative product of alpha
        self.alphas = 1. - self.betas
        alphas_cumprod = torch.cumprod(self.alphas, axis=0).to(self.device)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0).to(self.device)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas).to(self.device)

        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod).to(self.device)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod).to(self.device)
        self.posterior_variance = (self.betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)).to(self.device)

        input_dims = {'X': 1433, 'E': 1, 'y': 129}
        hidden_dims = {'dx': args.nhid, 'de': args.nhid, 'dy': args.nhid}
        output_dims = {'X': 1433, 'E': 1, 'y': 129}

        # 初始化 gcn_model
        self.gcn = GraphConvolutionalNetwork(n_layers=args.num_layers,
                                            input_dims=input_dims,
                                            hidden_dims=hidden_dims,
                                            output_dims=output_dims).to(self.device)

        self.noise_schedule = PredefinedNoiseScheduleDiscrete(args.diffusion_noise_schedule,
                                                              timesteps=args.timesteps).to(self.device)

        self.Xdim_output = out_feat
        self.Edim_output = 1
        # self.transition_model = DiscreteUniformTransition(
        #     x_classes=data.feat.shape[1],
        #     e_classes=data.edges.shape[1],
        #     y_classes=y.shape[1]
        # )

        # x_limit = torch.ones(data.feat.shape[1]) / data.feat.shape[1]
        # e_limit = torch.ones(data.edges.shape[1]) / data.edges.shape[1]
        # y_limit = torch.ones(y.shape[1]) / y.shape[1]
        # self.limit_dist = PlaceHolder(X=x_limit, E=e_limit, y=y_limit)

        self.transition_model = None  # Will be initialized dynamically
        self.limit_dist = None  # Will be initialized dynamically

        # self.optimizer = torch.optim.AdamW(self.parameters(), lr=args.lr)
        self.best_model_params = None
        self.best_val_loss = float('inf')
        self.train_loss = TrainLossDiscrete(args.lambda_train)# in utils.py

    def initialize_gcn_model(self, args, x_dim=1433, e_dim=1, y_dim=None):
        # Use the GraphConvolutionalNetwork model (similar to GraphTransformer)
        input_dims = {'X': x_dim, 'E': e_dim, 'y': y_dim}
        hidden_dims = {'dx': args.nhid, 'de': args.nhid, 'dy': args.nhid}
        output_dims = {'X': x_dim, 'E': e_dim, 'y': y_dim}

        # 初始化 gcn_model
        self.gcn = GraphConvolutionalNetwork(n_layers=args.num_layers,
                                            input_dims=input_dims,
                                            hidden_dims=hidden_dims,
                                            output_dims=output_dims).to(self.device)

        
    def initialize_transition_model(self, x_dim, e_dim, y_dim):

            # 初始化 transition_model
            self.transition_model = DiscreteUniformTransition(
                x_classes=x_dim,
                e_classes=e_dim,
                y_classes=y_dim
            )

            # 初始化 limit distribution
            x_limit = torch.ones(x_dim, device=self.device) / x_dim
            e_limit = torch.ones(e_dim, device=self.device) / e_dim
            y_limit = torch.ones(y_dim, device=self.device) / y_dim
            self.limit_dist = PlaceHolder(X=x_limit, E=e_limit)

    def linear_beta_schedule(self, timesteps):
        scale = 1000 / timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)

    def apply_noise(self, X, E, y, node_mask):
        """ Sample noise and apply it to the data. """
        self.initialize_transition_model(x_dim=X.size(-1), e_dim=E.size(-1), y_dim=y.size(-1))
        
        # print(f"E shape: {E.shape}")
        # print(f"X shape: {X.shape}")
        # Sample a timestep t.
        lowest_t = 0 if self.training else 1
        t_int = torch.randint(lowest_t, self.timesteps + 1, size=(21, 1), device=X.device).float()  # (bs=21, 1)
        # print(f"t_int shape: {t_int.shape}")
        s_int = t_int - 1

        t_float = t_int / self.timesteps
        # print(f"t_float shape: {t_float.shape}")
        s_float = s_int / self.timesteps

        # beta_t and alpha_s_bar are used for denoising/loss computation
        beta_t = self.noise_schedule(t_normalized=t_float)                         # (bs, 1)
        alpha_s_bar = self.noise_schedule.get_alpha_bar(t_normalized=s_float)      # (bs, 1)
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_normalized=t_float)      # (bs, 1)

        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, device=self.device)
        assert (abs(Qtb.X.sum(dim=2) - 1.) < 1e-4).all(), Qtb.X.sum(dim=2) - 1
        assert (abs(Qtb.E.sum(dim=2) - 1.) < 1e-4).all()

        # Compute transition probabilities
        probX = X @ Qtb.X  # (bs, n, dx_out)
        probE = E @ Qtb.E.unsqueeze(1)   # (bs, n, n, de_out)
        # 确保 probE 非负并进行归一化
        probE = torch.clamp(probE, min=0)
        if probE.sum(dim=-1).min().item() <= 0:
            # 给概率和为 0 的行加一个小的正数
            probE[probE.sum(dim=-1) == 0] = 1e-10
        probE = probE / probE.sum(dim=-1, keepdim=True)  # 归一化
        

        sampled_t = sample_discrete_features(probX=probX, probE=probE, node_mask=node_mask)

        X_t = F.one_hot(sampled_t.X, num_classes=self.Xdim_output)
        E_t = F.one_hot(sampled_t.E, num_classes=self.Edim_output)

        if E_t.float().std() == 0:
            E_t = torch.randn_like(E_t.float())
            E_t = E_t.squeeze(-1)  # 去掉最后一维
            E_t = (E_t + E_t.transpose(-2, -1)) / 2  # 对称化
            E_t = E_t.unsqueeze(-1)  # 恢复原始形状
        assert (X.shape == X_t.shape) and (E.shape == E_t.shape)

        z_t = PlaceHolder(X=X_t, E=E_t).type_as(X_t).mask(node_mask)
        # print(f"Original X mean={X.mean().item()}, std={X.std().item()}")
        # print(f"Noisy X_t mean={z_t.X.float().mean().item()}, std={z_t.X.float().std().item()}")

        noisy_data = {'t_int': t_int, 't': t_float, 'beta_t': beta_t, 'alpha_s_bar': alpha_s_bar,
                      'alpha_t_bar': alpha_t_bar, 'X_t': z_t.X, 'E_t': z_t.E, 'node_mask': node_mask}
        return noisy_data

    def forward(self, args, noisy_data, edge_index):
        self.initialize_gcn_model(args, x_dim=noisy_data['X_t'].shape[-1], e_dim=noisy_data['E_t'].shape[-1])
        # Fetch noisy data
        X_t = noisy_data['X_t'].float().to(self.device)
        E_t = noisy_data['E_t'].float().to(self.device)
        # y_t = noisy_data['y_t'].float().to(self.device)
        node_mask = noisy_data['node_mask']

        # Use the GraphConvolutionalNetwork model for feature transformation
        output = self.gcn(X_t, E_t, node_mask=node_mask)

        return output.X, output.E

    def sample_p_zs_given_zt(self, s, t, pred_X, pred_E, node_mask, noisy_data):
        
        bs, n, dxs = pred_X.shape
        # Neural net predictions
        X_t = noisy_data['X_t'].float().to(self.device)
        E_t = noisy_data['E_t'].float().to(self.device)
        # y_t = noisy_data['y_t'].float().to(self.device)

        # Normalize predictions
        pred_probs_X = F.softmax(pred_X, dim=-1)
        pred_probs_E = F.softmax(pred_E, dim=-1)

        # Compute posterior distributions
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_normalized=t)
        alpha_s_bar = self.noise_schedule.get_alpha_bar(t_normalized=s)
        beta_t = self.noise_schedule(t_normalized=t)

        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, self.device)
        Qsb = self.transition_model.get_Qt_bar(alpha_s_bar, self.device)
        Qt = self.transition_model.get_Qt(beta_t, self.device)
        # print(f"Shape of E_t: {E_t.shape}")
        # print(f"Shape of Qt.E: {Qt.E.shape}")
        # print(f"Shape of Qsb.E: {Qsb.E.shape}")
        # print(f"Shape of Qtb.E: {Qtb.E.shape}")

        p_s_and_t_given_0_X = compute_batched_over0_posterior_distribution(X_t=X_t, Qt=Qt.X, Qsb=Qsb.X, Qtb=Qtb.X)
        p_s_and_t_given_0_E = compute_batched_over0_posterior_distribution(X_t=E_t, Qt=Qt.E, Qsb=Qsb.E, Qtb=Qtb.E)
        # print(f"Shape of p_s_and_t_given_0_E: {p_s_and_t_given_0_E.shape}")
        # print(f"Shape of p_s_and_t_given_0_X: {p_s_and_t_given_0_X.shape}")

        # Weighted sum to get probabilities
        weighted_X = pred_probs_X.unsqueeze(-1) * p_s_and_t_given_0_X
        unnormalized_prob_X = weighted_X.sum(dim=2)
        unnormalized_prob_X[torch.sum(unnormalized_prob_X, dim=-1) == 0] = 1e-5
        prob_X = unnormalized_prob_X / torch.sum(unnormalized_prob_X, dim=-1, keepdim=True)

        pred_probs_E = pred_probs_E.reshape((bs, -1, pred_probs_E.shape[-1]))
        weighted_E = pred_probs_E.unsqueeze(-1) * p_s_and_t_given_0_E        # bs, N, d0, d_t-1
        unnormalized_prob_E = weighted_E.sum(dim=-2)
        unnormalized_prob_E[torch.sum(unnormalized_prob_E, dim=-1) == 0] = 1e-5
        pred_probs_E = unnormalized_prob_E / torch.sum(unnormalized_prob_E, dim=-1, keepdim=True)
        prob_E = pred_probs_E.reshape(bs, n, n, pred_probs_E.shape[-1])

        # Sample new states
        sampled_s = sample_discrete_features(prob_X, prob_E, node_mask=node_mask)

        X_s = F.one_hot(sampled_s.X, num_classes=self.Xdim_output).float()
        E_s = F.one_hot(sampled_s.E, num_classes=self.Edim_output).float()

        return PlaceHolder(X=X_s, E=E_s), PlaceHolder(X=X_s, E=E_s)

    def kl_prior(self, X, E, node_mask):
        """计算扩散过程初始阶段的 KL 散度，即 q(z1 | x) 与先验 p(z1) 之间的 KL 散度。"""
        
        ones = torch.ones((X.size(0), 1), device=X.device)
        Ts = self.timesteps * ones
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_int=Ts)  # (bs, 1)

        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, self.device)

        # 计算转移概率
        probX = X @ Qtb.X  # (bs, n, dx_out)
        probE = E @ Qtb.E.unsqueeze(1)  # (bs, n, n, de_out)

        # 计算 KL 散度
        bs, n, _ = probX.shape
        limit_X = self.limit_dist.X[None, None, :].expand(bs, n, -1).type_as(probX)
        limit_E = self.limit_dist.E[None, None, None, :].expand(bs, n, n, -1).type_as(probE)

        kl_distance_X = F.kl_div(input=probX.log(), target=limit_X, reduction='batchmean')
        kl_distance_E = F.kl_div(input=probE.log(), target=limit_E, reduction='batchmean')

        return kl_distance_X + kl_distance_E

    # def reconstruction_logp(self, t, X, E, node_mask):
    #     """计算从 t=0 状态重构输入数据的对数概率，度量模型的重构能力。"""
    #     t_zeros = torch.zeros_like(t)
    #     beta_0 = self.noise_schedule(t_zeros)
    #     Q0 = self.transition_model.get_Qt(beta_t=beta_0, device=self.device)

    #     probX0 = X @ Q0.X  # (bs, n, dx_out)
    #     probE0 = E @ Q0.E.unsqueeze(1)  # (bs, n, n, de_out)

    #     sampled0 = sample_discrete_features(probX=probX0, probE=probE0, node_mask=node_mask)

    #     X0 = F.one_hot(sampled0.X, num_classes=self.Xdim_output).float()
    #     E0 = F.one_hot(sampled0.E, num_classes=self.Edim_output).float()

    #     # 预测
    #     noisy_data = {'X_t': X0, 'E_t': E0, 'node_mask': node_mask, 't': torch.zeros(X0.shape[0], 1).type_as(t)}
    #     extra_data = self.compute_extra_data(noisy_data)
    #     pred0 = self.forward(noisy_data, extra_data, node_mask)

    #     # 归一化预测
    #     probX0 = F.softmax(pred0[0], dim=-1)
    #     probE0 = F.softmax(pred0[1], dim=-1)

    #     return PlaceHolder(X=probX0, E=probE0)

    def compute_train_loss(self, pred_X, pred_E, data, edges, noisy_data):
        
        
        # Calculate train loss using the simplified TrainLossDiscrete
        train_loss = self.train_loss(masked_pred_X=pred_X, masked_pred_E=pred_E,
                                     true_X=data.feat, true_E=data.adj)
        
        return train_loss

    def compute_validation_loss(self, data, pred_X, pred_E, node_mask, noisy_data):
        """通过逐步去噪对模型进行验证，计算验证损失 (KL 散度 + 重构损失)。"""
        val_loss = 0.0
        lowest_t = 0 if self.training else 1
        t_int = torch.randint(lowest_t, self.timesteps + 1, size=(21, 1), device=self.device).float()  # (bs=21, 1)
        # print(f"t_int shape: {t_int.shape}")
        s_int = t_int - 1

        t_float = t_int / self.timesteps
        # print(f"t_float shape: {t_float.shape}")
        s_float = s_int / self.timesteps
        
        # 从最初的噪声逐步去噪到干净状态
        # X_t, E_t, y_t = data.feat, data.adj, data.y
        for t in reversed(range(1, self.timesteps + 1)):
            # print("the denoised process:", t)
            s = t - 1
            sampled_s, discrete_sampled_s = self.sample_p_zs_given_zt(s_float, t_float, pred_X, pred_E, node_mask, noisy_data)
            val_loss += self.kl_prior(discrete_sampled_s.X, discrete_sampled_s.E, node_mask)  # 使用KL散度来衡量模型性能
        
        return val_loss

    def load_best_model(self):
        if self.best_model_params is not None:
            self.load_state_dict(self.best_model_params)

class DiffusionSimCLR(nn.Module):
    def __init__(self, in_feat, out_feat, num_gc_layers, dataset_num, args, diffusion_model, temperature=0.8):
        super(DiffusionSimCLR, self).__init__()
        device = "cuda:" + str(args.gpu) if torch.cuda.is_available() else 'cpu'

        # 修改 embedding_dim 的定义
        self.embedding_dim = args.nhid * num_gc_layers

        # 基础 encoder 和投影头
        self.encoder = Encoder(in_feat, args.nhid, num_gc_layers)

        # 投影头的维度调整，使输入到嵌入层的维度更加合理
        self.proj_head = nn.Sequential(
            nn.Linear(in_feat, self.embedding_dim),  # 从输入特征维度映射到嵌入维度
            nn.ReLU(inplace=True),
            nn.Linear(self.embedding_dim, out_feat)  # 从嵌入维度映射到最终的输出特征维度
        )

        # Diffusion 模型用于负采样，在 model_cond.py 实现
        # 确保传入一致的 in_feat 和 out_feat
        self.diffusion_model = diffusion_model

        self.temperature = temperature
        self.classification_head = nn.Linear(out_feat, dataset_num)

        self.init_emb()

    def init_emb(self):
        initrange = -1.5 / self.embedding_dim
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)
    
    def forward(self, x, edge_index, batch, num_graphs, epoch, total_epochs):
        # 编码节点特征并得到基本的节点嵌入
        X, M = self.encoder(x, edge_index, batch)#这个传入batch还是有点问题
        X = self.proj_head(X)

        # 使用扩散模型生成负样本
        h_syn = self.get_synthetic_negatives(X, epoch, total_epochs)

        # 排序负样本队列，确保课程学习
        sorted_negatives = self.rank_negative_queue(X, h_syn)
        
        return X, sorted_negatives


    def get_synthetic_negatives(self, q, epoch, total_epochs):
        """使用扩散模型生成负样本，包含对时间步的控制"""
        max_timesteps = self.diffusion_model.timesteps
        lamda = 1 / 2
        # 根据 epoch 计算当前时间步
        time_step = int(((epoch + 1) / total_epochs) ** lamda * max_timesteps)
        
        # 获取去噪后的图数据作为负样本
        noisy_data = {'X_t': q, 't': torch.tensor([time_step], dtype=torch.float32, device=q.device)}
        extra_data = self.diffusion_model.compute_extra_data(noisy_data)
        h_syn, _ = self.diffusion_model.forward(noisy_data, extra_data, edge_index=None)
        return h_syn

    def rank_negative_queue(self, x1, x2):
        """保持Curriculum的负样本排序逻辑"""
        x2 = x2.t()
        x = x1.mm(x2)

        x1_frobenius = x1.norm(dim=1).unsqueeze(0).t()
        x2_frobenins = x2.norm(dim=0).unsqueeze(0)
        x_frobenins = x1_frobenius.mm(x2_frobenins)

        final_value = x.mul(1 / x_frobenins)
        sort_queue, _ = torch.sort(final_value, dim=0, descending=False)
        
        return sort_queue

    def loss_cal(self, q_batch, q_aug_batch, negative_samples):
        """结合diffusion的timestep权重和对比学习loss"""
        T = self.temperature
        weights = [0, 1, 0.9, 0.8, 0.7]  # DMNS的timestep权重
        
        # 计算正样本相似度
        positive_sim = torch.cosine_similarity(q_batch, q_aug_batch, dim=1)
        positive_exp = torch.exp(positive_sim / T)
        
        # 对不同timestep的负样本分别计算loss
        total_loss = 0
        total_weight = sum(w for w in weights if w != 0)
        
        for i, neg in enumerate(negative_samples):
            if weights[i] == 0:
                continue
                
            negative_sim = torch.cosine_similarity(q_batch.unsqueeze(1), neg, dim=2)
            negative_exp = torch.exp(negative_sim / T)
            negative_sum = torch.sum(negative_exp, dim=1)
            
            current_loss = -torch.log(positive_exp / (positive_exp + negative_sum))
            total_loss += weights[i] * current_loss.mean()
            
        return total_loss / total_weight

    def node_loss(self, x, edge_index, batch, epoch, total_epochs, y):
        """训练步骤: 计算总的loss，但不更新梯度"""
        # 获取 anchor 和 positive 的 embeddings
        q, sorted_negatives = self.forward(x, edge_index, batch, num_graphs=1, epoch=epoch, total_epochs=total_epochs)

        # 使用 diffusion 生成负样本
        neg_samples = [sorted_negatives]  # 对应 get_synthetic_negatives 生成的负样本

        # 计算对比学习的 loss
        contrastive_loss = self.loss_cal(q, q, neg_samples)

        # 计算节点分类损失 (添加监督信号)
        labels = y  # 假设已经传入相应的标签
        classification_logits = self.classification_head(q)  # 使用一个简单的分类头
        classification_loss_fn = nn.CrossEntropyLoss()
        classification_loss = classification_loss_fn(classification_logits, labels)

        # 合并两种损失
        total_loss = contrastive_loss + args.alpha * classification_loss

        return total_loss