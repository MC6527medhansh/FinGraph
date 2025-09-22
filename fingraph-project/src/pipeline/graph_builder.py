"""
Graph Construction Module for Phase 2
Builds temporal graphs from feature vectors
"""

import torch
import numpy as np
import pandas as pd
from torch_geometric.data import Data
from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class TemporalGraphBuilder:
    """Build graphs with temporal integrity"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.correlation_threshold = config.get('graph', {}).get('correlation_threshold', 0.3)
        self.max_edges_per_node = config.get('graph', {}).get('max_edges_per_node', 10)
        
    def build_graph_from_features(self, features_df: pd.DataFrame, date: pd.Timestamp) -> Data:
        # Filter features for this date
        date_features = features_df[features_df['date'] == date].copy()
        if len(date_features) == 0:
            logger.warning(f"No features for date {date}")
            return None

        # === EXCLUSIONS (meta + any label/label-derived columns) ===
        META_COLS = {'date', 'symbol'}
        LABEL_COLS = {
            'forward_return', 'forward_volatility', 'forward_max_drawdown', 'risk_score',
            'forward_return_cs_z', 'forward_volatility_cs_z', 'risk_score_cs_pct'
        }
        feature_cols = [c for c in date_features.columns if c not in META_COLS | LABEL_COLS]

        # Node features
        x = torch.FloatTensor(date_features[feature_cols].values)

        # Edges
        edges, edge_attr = self._create_edges(date_features, feature_cols)

        # Labels (ok to read if present)
        y_risk = torch.FloatTensor(date_features['risk_score'].values) if 'risk_score' in date_features.columns else torch.zeros(len(date_features))
        y_return = torch.FloatTensor(date_features['forward_return'].values) if 'forward_return' in date_features.columns else torch.zeros(len(date_features))
        y_vol = torch.FloatTensor(date_features['forward_volatility'].values) if 'forward_volatility' in date_features.columns else torch.zeros(len(date_features))

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

        # Robust correlation (handle NaNs/constant columns)
        with np.errstate(invalid='ignore'):
            corr_matrix = np.corrcoef(features) if n_nodes > 1 else np.array([[1.0]])
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0, posinf=0.0, neginf=0.0)

        edges = []
        edge_weights = []
        for i in range(n_nodes):
            correlations = corr_matrix[i].copy()
            if correlations.shape[0] <= i:
                continue
            correlations[i] = -1  # exclude self

            top_indices = np.argsort(np.abs(correlations))[-self.max_edges_per_node:]
            for j in top_indices:
                if j < 0 or j >= n_nodes:
                    continue
                if np.abs(correlations[j]) > self.correlation_threshold:
                    edges.append([i, j])
                    edge_weights.append([
                        float(correlations[j]),
                        float(abs(correlations[j])),
                        1.0 if correlations[j] > 0 else 0.0
                    ])

        if edges:
            edge_index = torch.LongTensor(edges).t().contiguous()
            edge_attr = torch.FloatTensor(edge_weights)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros((0, 3), dtype=torch.float)
        return edge_index, edge_attr
    
    def create_temporal_graphs(self, 
                               features_df: pd.DataFrame,
                               max_graphs: Optional[int] = None) -> List[Data]:
        """Create graphs for all dates in features (TRAIN/EVAL path)."""
        unique_dates = pd.Series(features_df['date'].unique()).sort_values().to_list()
        if max_graphs:
            unique_dates = unique_dates[:max_graphs]
        graphs = []
        for date in unique_dates:
            g = self.build_graph_from_features(features_df, date)
            if g is not None:
                graphs.append(g)
        logger.info(f"Created {len(graphs)} temporal graphs")
        return graphs
    
    def build_graph_for_prediction(self, features_df: pd.DataFrame, date: pd.Timestamp) -> Data:
        # Filter features for this date
        date_features = features_df[features_df['date'] == date].copy()
        if len(date_features) == 0:
            logger.warning(f"No features for date {date}")
            return None

        META_COLS = {'date', 'symbol'}
        LABEL_COLS = {
            'forward_return', 'forward_volatility', 'forward_max_drawdown', 'risk_score',
            'forward_return_cs_z', 'forward_volatility_cs_z', 'risk_score_cs_pct'
        }
        feature_cols = [c for c in date_features.columns if c not in META_COLS | LABEL_COLS]

        x = torch.FloatTensor(date_features[feature_cols].values)
        edges, edge_attr = self._create_edges(date_features, feature_cols)

        graph = Data(
            x=x,
            edge_index=edges,
            edge_attr=edge_attr,
            symbols=date_features['symbol'].tolist(),
            date=date
        )

        # Dummy labels for shape safety (not used by inference)
        n = len(date_features)
        graph.y_risk = torch.zeros(n)
        graph.y_return = torch.zeros(n)
        graph.y_volatility = torch.zeros(n)
        return graph
