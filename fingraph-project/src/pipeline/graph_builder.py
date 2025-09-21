"""
Graph Construction Module for Phase 2
Builds temporal graphs from feature vectors
"""

import torch
import numpy as np
import pandas as pd
from torch_geometric.data import Data, Batch
from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class TemporalGraphBuilder:
    """Build graphs with temporal integrity"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.correlation_threshold = config.get('graph', {}).get('correlation_threshold', 0.3)
        self.max_edges_per_node = config.get('graph', {}).get('max_edges_per_node', 10)
        
    def build_graph_from_features(self, 
                                  features_df: pd.DataFrame,
                                  date: pd.Timestamp) -> Data:
        """
        Build graph for a specific date from features.
        
        Args:
            features_df: Feature dataframe from Phase 1
            date: Date to build graph for
            
        Returns:
            PyTorch Geometric Data object
        """
        # Filter features for this date
        date_features = features_df[features_df['date'] == date].copy()
        
        if len(date_features) == 0:
            logger.warning(f"No features for date {date}")
            return None
            
        # Create node features (excluding metadata columns)
        feature_cols = [col for col in date_features.columns 
                       if col not in ['date', 'symbol', 'forward_return', 
                                     'forward_volatility', 'forward_max_drawdown', 
                                     'risk_score']]
        
        node_features = date_features[feature_cols].values
        x = torch.FloatTensor(node_features)
        
        # Create edges based on correlation
        edges, edge_attr = self._create_edges(date_features, feature_cols)
        
        # Create labels
        y_risk = torch.FloatTensor(date_features['risk_score'].values)
        y_return = torch.FloatTensor(date_features['forward_return'].values)
        y_vol = torch.FloatTensor(date_features['forward_volatility'].values)
        
        # Create graph
        graph = Data(
            x=x,
            edge_index=edges,
            edge_attr=edge_attr,
            y_risk=y_risk,
            y_return=y_return,
            y_volatility=y_vol,
            symbols=date_features['symbol'].tolist(),
            date=date
        )
        
        return graph
    
    def _create_edges(self, 
                     features_df: pd.DataFrame,
                     feature_cols: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create edges based on feature correlation"""
        
        features = features_df[feature_cols].values
        n_nodes = len(features)
        
        # Calculate correlation matrix
        corr_matrix = np.corrcoef(features)
        
        # Create edges for correlations above threshold
        edges = []
        edge_weights = []
        
        for i in range(n_nodes):
            # Get top-k correlated nodes
            correlations = corr_matrix[i]
            correlations[i] = -1  # Exclude self
            
            # Get indices of top correlations
            top_indices = np.argsort(np.abs(correlations))[-self.max_edges_per_node:]
            
            for j in top_indices:
                if np.abs(correlations[j]) > self.correlation_threshold:
                    edges.append([i, j])
                    edge_weights.append([
                        correlations[j],
                        np.abs(correlations[j]),
                        1.0 if correlations[j] > 0 else 0.0
                    ])
        
        if edges:
            edge_index = torch.LongTensor(edges).t().contiguous()
            edge_attr = torch.FloatTensor(edge_weights)
        else:
            # Create minimal connectivity
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros((0, 3), dtype=torch.float)
            
        return edge_index, edge_attr
    
    def create_temporal_graphs(self, 
                              features_df: pd.DataFrame,
                              max_graphs: Optional[int] = None) -> List[Data]:
        """Create graphs for all dates in features"""
        
        unique_dates = features_df['date'].unique()
        if max_graphs:
            unique_dates = unique_dates[:max_graphs]
            
        graphs = []
        for date in unique_dates:
            graph = self.build_graph_from_features(features_df, date)
            if graph is not None:
                graphs.append(graph)
                
        logger.info(f"Created {len(graphs)} temporal graphs")
        return graphs