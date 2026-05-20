"""
Financial GNN Model - NODE-LEVEL PREDICTIONS VERSION
Makes individual predictions for each stock, not graph-level
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool
from typing import Dict, Optional


class FinancialGNN(nn.Module):
    """
    Fixed GNN with reduced over-smoothing for node-level predictions.
    """
    
    def __init__(self, 
                 num_node_features: int,
                 hidden_dim: int = 128,
                 num_heads: int = 8,
                 num_layers: int = 3,
                 dropout: float = 0.2,
                 edge_dim: int = 3,
                 node_level: bool = True):
        super().__init__()
        
        self.dropout = dropout
        self.node_level = node_level
        
        # Input projection
        self.input_projection = nn.Linear(num_node_features, hidden_dim)
        
        # Reduced GAT layers - only 2 instead of 3
        self.gat1 = GATConv(hidden_dim, hidden_dim // 4, heads=4, dropout=dropout, edge_dim=edge_dim, concat=True)
        self.gat2 = GATConv(hidden_dim, hidden_dim, heads=1, dropout=dropout, edge_dim=edge_dim, concat=False)
        
        # Batch norms
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        
        # Skip connection layer
        self.skip_projection = nn.Linear(num_node_features, hidden_dim)
        
        # Prediction heads - NO pooling dimensions
        self.risk_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        self.return_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        
        self.volatility_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Softplus()
        )
        
    def forward(self, x: torch.Tensor, 
                edge_index: torch.Tensor,
                edge_attr: Optional[torch.Tensor] = None,
                batch: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        
        # Skip connection
        x_skip = self.skip_projection(x)
        
        # Graph processing
        x = self.input_projection(x)
        x = F.relu(x)
        
        # GAT layer 1
        x = self.gat1(x, edge_index, edge_attr)
        x = self.bn1(x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # GAT layer 2
        x = self.gat2(x, edge_index, edge_attr)
        x = self.bn2(x)
        
        # Strong skip connection - preserve original features
        x = 0.5 * x + 0.5 * x_skip
        
        # NODE-LEVEL predictions - no pooling
        outputs = {
            'risk': self.risk_head(x),
            'return': self.return_head(x),
            'volatility': self.volatility_head(x)
        }
        
        return outputs