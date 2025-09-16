"""
REAL Graph Neural Network Implementation
This actually uses graph structure for predictions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, SAGEConv, global_mean_pool, global_max_pool
from torch_geometric.data import Data, Batch
from typing import Optional, Tuple, Dict
import numpy as np


class FinancialGNN(nn.Module):
    """
    A REAL Graph Neural Network that actually uses graph structure.
    
    This model:
    1. Takes in node features AND edge connections
    2. Performs message passing between connected nodes
    3. Aggregates neighborhood information
    4. Uses attention mechanisms to weight important connections
    
    This is fundamentally different from an MLP because:
    - Companies influence each other through edges
    - Sector nodes aggregate company information
    - Economic indicators propagate through the entire graph
    """
    
    def __init__(
        self,
        num_node_features: int,
        num_edge_features: int = 3,
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 3,
        dropout: float = 0.2,
        aggregation: str = 'mean'
    ):
        super().__init__()
        
        # Store configuration
        self.num_node_features = num_node_features
        self.num_edge_features = num_edge_features
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout
        self.aggregation = aggregation
        
        # Initial projection of node features
        self.node_encoder = nn.Sequential(
            nn.Linear(num_node_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Graph Attention Network layers - THIS IS WHAT MAKES IT A GNN
        self.gat_layers = nn.ModuleList()
        
        # First GAT layer
        self.gat_layers.append(
            GATConv(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                heads=num_heads,
                dropout=dropout,
                edge_dim=num_edge_features,  # Use edge features
                add_self_loops=True,
                negative_slope=0.2
            )
        )
        
        # Middle GAT layers
        for _ in range(num_layers - 2):
            self.gat_layers.append(
                GATConv(
                    in_channels=hidden_dim * num_heads,
                    out_channels=hidden_dim,
                    heads=num_heads,
                    dropout=dropout,
                    edge_dim=num_edge_features,
                    add_self_loops=True,
                    negative_slope=0.2
                )
            )
        
        # Final GAT layer (no concatenation of heads)
        self.gat_layers.append(
            GATConv(
                in_channels=hidden_dim * num_heads,
                out_channels=hidden_dim,
                heads=1,
                concat=False,
                dropout=dropout,
                edge_dim=num_edge_features,
                add_self_loops=True,
                negative_slope=0.2
            )
        )
        
        # Batch normalization for each layer
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim * num_heads if i < num_layers - 1 else hidden_dim)
            for i in range(num_layers)
        ])
        
        # Skip connections
        self.skip_connections = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim * num_heads if i < num_layers - 1 else hidden_dim)
            for i in range(num_layers)
        ])
        
        # Risk prediction head for company nodes
        self.risk_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()  # Output risk score in [0, 1]
        )
        
        # Auxiliary heads for multi-task learning
        self.volatility_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Softplus()  # Volatility is positive
        )
        
        self.return_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Tanh()  # Returns can be negative
        )
        
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        node_type: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the graph neural network.
        
        Args:
            x: Node features [num_nodes, num_features]
            edge_index: Graph connectivity [2, num_edges]
            edge_attr: Edge features [num_edges, num_edge_features]
            node_type: Type of each node (0=company, 1=sector, 2=economic)
            batch: Batch assignment for multiple graphs
            
        Returns:
            Dictionary with risk scores and auxiliary predictions
        """
        
        # Encode initial node features
        h = self.node_encoder(x)
        initial_h = h
        
        # Message passing through GAT layers
        for i, (gat_layer, batch_norm, skip_conn) in enumerate(
            zip(self.gat_layers, self.batch_norms, self.skip_connections)
        ):
            # Store residual
            residual = skip_conn(h if i == 0 else h_prev)
            h_prev = h
            
            # Graph attention convolution
            h = gat_layer(h, edge_index, edge_attr=edge_attr)
            
            # Batch normalization
            h = batch_norm(h)
            
            # Add residual connection
            h = h + residual
            
            # Activation and dropout (except last layer)
            if i < len(self.gat_layers) - 1:
                h = F.elu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
        
        # Now h contains graph-aware representations for all nodes
        
        # If we have node types, we can apply type-specific processing
        if node_type is not None:
            # Get company nodes only for risk prediction
            company_mask = (node_type == 0)
            company_features = h[company_mask]
            
            # Predict risk for company nodes
            risk_scores = self.risk_predictor(company_features)
            volatility = self.volatility_predictor(company_features)
            returns = self.return_predictor(company_features)
            
            # For sector and economic nodes, we might want different outputs
            sector_mask = (node_type == 1)
            economic_mask = (node_type == 2)
            
            # Aggregate sector information (optional)
            if sector_mask.any():
                sector_features = h[sector_mask]
                # Could add sector-specific predictions here
            
            # Aggregate economic information (optional)
            if economic_mask.any():
                economic_features = h[economic_mask]
                # Could add macro predictions here
                
        else:
            # If no node types, treat all nodes as companies
            risk_scores = self.risk_predictor(h)
            volatility = self.volatility_predictor(h)
            returns = self.return_predictor(h)
        
        return {
            'risk': risk_scores,
            'volatility': volatility,
            'returns': returns,
            'node_embeddings': h  # Return learned representations
        }
    
    def get_attention_weights(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None
    ) -> List[torch.Tensor]:
        """
        Extract attention weights from each GAT layer.
        This shows which connections are most important.
        """
        attention_weights = []
        h = self.node_encoder(x)
        
        for gat_layer in self.gat_layers:
            # Get attention weights from this layer
            h, (edge_index_out, attention) = gat_layer(
                h, edge_index, edge_attr=edge_attr, return_attention_weights=True
            )
            attention_weights.append(attention)
        
        return attention_weights


class TemporalFinancialGNN(nn.Module):
    """
    Temporal extension of the Financial GNN.
    This model handles time-varying graphs and sequential predictions.
    """
    
    def __init__(
        self,
        base_gnn: FinancialGNN,
        sequence_length: int = 5,
        lstm_hidden: int = 128,
        lstm_layers: int = 2
    ):
        super().__init__()
        
        self.base_gnn = base_gnn
        self.sequence_length = sequence_length
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=base_gnn.hidden_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=0.2 if lstm_layers > 1 else 0
        )
        
        # Final prediction from temporal features
        self.temporal_predictor = nn.Sequential(
            nn.Linear(lstm_hidden, lstm_hidden // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(lstm_hidden // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(
        self,
        graph_sequence: List[Data]
    ) -> Dict[str, torch.Tensor]:
        """
        Process a sequence of graphs over time.
        
        Args:
            graph_sequence: List of Data objects representing graphs at different times
            
        Returns:
            Temporal risk predictions
        """
        
        # Process each graph in the sequence
        temporal_embeddings = []
        
        for graph in graph_sequence:
            # Get embeddings from base GNN
            outputs = self.base_gnn(
                graph.x,
                graph.edge_index,
                graph.edge_attr if hasattr(graph, 'edge_attr') else None,
                graph.node_type if hasattr(graph, 'node_type') else None
            )
            
            # Get company node embeddings
            if hasattr(graph, 'node_type'):
                company_mask = (graph.node_type == 0)
                company_embeddings = outputs['node_embeddings'][company_mask]
            else:
                company_embeddings = outputs['node_embeddings']
            
            # Pool embeddings for this timestamp
            pooled = company_embeddings.mean(dim=0)
            temporal_embeddings.append(pooled)
        
        # Stack temporal embeddings
        temporal_features = torch.stack(temporal_embeddings, dim=0).unsqueeze(0)
        
        # Process through LSTM
        lstm_out, _ = self.lstm(temporal_features)
        
        # Final prediction
        temporal_risk = self.temporal_predictor(lstm_out[:, -1, :])
        
        return {
            'temporal_risk': temporal_risk,
            'sequence_embeddings': lstm_out
        }


def create_gnn_model(config: Dict) -> FinancialGNN:
    """
    Factory function to create GNN model from configuration.
    """
    model = FinancialGNN(
        num_node_features=config['num_node_features'],
        num_edge_features=config.get('num_edge_features', 3),
        hidden_dim=config.get('hidden_dim', 128),
        num_heads=config.get('num_heads', 4),
        num_layers=config.get('num_layers', 3),
        dropout=config.get('dropout', 0.2),
        aggregation=config.get('aggregation', 'mean')
    )
    
    # Initialize weights properly
    for module in model.modules():
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    return model