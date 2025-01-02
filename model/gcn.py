import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *
from torch_geometric.nn import GCNConv
from torch_geometric.utils import dense_to_sparse

class GraphConvolutionalNetwork(nn.Module):
    def __init__(self, n_layers: int, input_dims: dict, hidden_dims: dict, output_dims: dict, act_fn=nn.LeakyReLU()):
        super().__init__()
        self.n_layers = n_layers
        self.out_dim_X = output_dims['X']
        self.out_dim_E = output_dims['E']

        # Input MLP for X
        self.mlp_in_X = nn.Sequential(
            nn.Linear(input_dims['X'], hidden_dims['dx']),
            act_fn
        )

        # PyG GCN layers
        self.gcn_layers = nn.ModuleList([
            GCNConv(hidden_dims['dx'], hidden_dims['dx']) for _ in range(n_layers)
        ])

        # Output MLP for X
        self.mlp_out_X = nn.Sequential(
            nn.Linear(hidden_dims['dx'], hidden_dims['dx']),
            act_fn,
            nn.Linear(hidden_dims['dx'], output_dims['X'])
        )

    def forward(self, X, adj, edge_weight=None, node_mask=None):
        """
        :param X: Node features, shape (num_nodes, feature_dim)
        :param adj: Adjacency matrix, shape (bs, n, n, de_out)
        :param edge_weight: Edge weights, shape (num_edges) (optional)
        :param node_mask: Mask indicating valid nodes, shape (batch_size, num_nodes) (optional)
        """
        batch_size, num_nodes, _, _ = adj.size()

        # Initial MLP transformation
        X = self.mlp_in_X(X)

        # Loop over batch
        outputs = []
        for batch_idx in range(batch_size):
            # Extract adjacency matrix for the current graph
            adj_single = adj[batch_idx].squeeze(-1)  # (num_nodes, num_nodes)

            # Convert dense adjacency matrix to sparse format
            edge_index, edge_weight = dense_to_sparse(adj_single)
            # 打印 edge_index 和 edge_weight 的形状，确认其匹配
            
            # Apply GCN layers
            X_single = X[batch_idx]  # (num_nodes, feature_dim)
            for layer in self.gcn_layers:
                torch.nn.init.xavier_uniform_(layer.lin.weight)
                X_single = layer(X_single, edge_index, edge_weight)
                X_single = F.leaky_relu(X_single, negative_slope=0.01)

            # Output transformation
            X_out = self.mlp_out_X(X_single)

            # If node_mask is provided, mask invalid nodes
            if node_mask is not None:
                mask_single = node_mask[batch_idx]  # (num_nodes)
                X_out = X_out * mask_single.unsqueeze(-1)

            outputs.append(X_out)
            X_out = torch.stack(outputs, dim=0)

        # Stack outputs for the batch
        return PlaceHolder(X=X_out, E=adj).mask(node_mask)
