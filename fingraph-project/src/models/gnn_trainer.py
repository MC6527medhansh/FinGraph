"""
Production Training System for Graph Neural Networks
Uses REAL message passing, not fake MLPs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool
from torch_geometric.data import Data, DataLoader, Batch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
from pathlib import Path
import json
import logging
import pickle
from dataclasses import dataclass
import hashlib
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class RealFinancialGNN(nn.Module):
    """
    REAL Graph Neural Network for Financial Risk Prediction.
    
    This is NOT an MLP. It performs actual message passing between nodes.
    """
    
    def __init__(self,
                 num_node_features: int,
                 num_edge_features: int = 3,
                 hidden_dim: int = 128,
                 num_heads: int = 8,
                 num_layers: int = 3,
                 dropout: float = 0.2,
                 output_dim: int = 3):  # Risk, volatility, return
        super().__init__()
        
        self.num_node_features = num_node_features
        self.hidden_dim = hidden_dim
        
        # Node feature encoder
        self.node_encoder = nn.Sequential(
            nn.Linear(num_node_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Graph Attention Networks - THE CORE OF REAL GNN
        self.gat_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # First GAT layer
        self.gat_layers.append(
            GATConv(
                in_channels=hidden_dim,
                out_channels=hidden_dim // num_heads,
                heads=num_heads,
                dropout=dropout,
                edge_dim=num_edge_features,
                add_self_loops=True,
                negative_slope=0.2
            )
        )
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Middle layers
        for _ in range(num_layers - 2):
            self.gat_layers.append(
                GATConv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim // num_heads,
                    heads=num_heads,
                    dropout=dropout,
                    edge_dim=num_edge_features,
                    add_self_loops=True,
                    negative_slope=0.2
                )
            )
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Final GAT layer
        self.gat_layers.append(
            GATConv(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                heads=1,
                concat=False,
                dropout=dropout,
                edge_dim=num_edge_features,
                add_self_loops=True,
                negative_slope=0.2
            )
        )
        
        # Multi-task prediction heads
        self.risk_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # *2 for pooling concat
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self.volatility_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus()  # Volatility must be positive
        )
        
        self.return_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, x, edge_index, edge_attr=None, batch=None):
        """
        Forward pass through GNN with message passing.
        
        Args:
            x: Node features [num_nodes, num_features]
            edge_index: Graph connectivity [2, num_edges]
            edge_attr: Edge features [num_edges, edge_features]
            batch: Batch assignment for multiple graphs
        """
        # Encode node features
        h = self.node_encoder(x)
        
        # Message passing through GAT layers
        for i, (gat_layer, batch_norm) in enumerate(zip(self.gat_layers, self.batch_norms)):
            h_prev = h
            
            # Graph attention convolution - THIS IS WHERE THE MAGIC HAPPENS
            h = gat_layer(h, edge_index, edge_attr=edge_attr)
            
            # Apply batch norm (except last layer)
            if i < len(self.gat_layers) - 1:
                h = batch_norm(h)
                h = F.relu(h)
                h = F.dropout(h, p=0.2, training=self.training)
                
                # Residual connection
                if h.shape == h_prev.shape:
                    h = h + h_prev * 0.5
        
        # Graph-level pooling (critical for graph-level predictions)
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        # Combine mean and max pooling for richer representation
        h_mean = global_mean_pool(h, batch)
        h_max = global_max_pool(h, batch)
        h_graph = torch.cat([h_mean, h_max], dim=1)
        
        # Multi-task predictions
        risk = self.risk_head(h_graph)
        volatility = self.volatility_head(h_graph)
        returns = self.return_head(h_graph)
        
        return {
            'risk': risk,
            'volatility': volatility,
            'return': returns,
            'node_embeddings': h
        }
    
    def get_attention_weights(self, x, edge_index, edge_attr=None):
        """Extract attention weights to understand what the model focuses on."""
        attention_weights = []
        h = self.node_encoder(x)
        
        for gat_layer in self.gat_layers:
            h, (edge_index_out, alpha) = gat_layer(h, edge_index, edge_attr=edge_attr, return_attention_weights=True)
            attention_weights.append(alpha)
        
        return attention_weights


class DynamicGraphBuilder:
    """
    Builds time-varying graphs from market data.
    Graph structure changes based on market conditions.
    """
    
    def __init__(self, correlation_threshold: float = 0.3, top_k_edges: int = 50):
        self.correlation_threshold = correlation_threshold
        self.top_k_edges = top_k_edges
        self.node_mapping = {}
        
    def build_temporal_graph(self,
                            samples: List[Any],
                            timestamp: datetime,
                            lookback_window: int = 60) -> Data:
        """
        Build graph at specific timestamp using historical correlations.
        
        Args:
            samples: List of temporal samples
            timestamp: Current time point
            lookback_window: Days to look back for correlation
            
        Returns:
            PyTorch Geometric Data object
        """
        # Get samples in lookback window
        window_start = timestamp - timedelta(days=lookback_window)
        window_samples = [s for s in samples if window_start <= s.timestamp < timestamp]
        
        if len(window_samples) < 10:
            raise ValueError(f"Insufficient samples for graph construction: {len(window_samples)}")
        
        # Get unique symbols
        symbols = list(set([s.symbol for s in window_samples]))
        self.node_mapping = {symbol: i for i, symbol in enumerate(symbols)}
        
        # Create node features (latest features for each symbol)
        node_features = []
        for symbol in symbols:
            symbol_samples = [s for s in window_samples if s.symbol == symbol]
            if symbol_samples:
                latest_sample = max(symbol_samples, key=lambda x: x.timestamp)
                node_features.append(latest_sample.features)
            else:
                # Shouldn't happen, but handle gracefully
                node_features.append(np.zeros_like(window_samples[0].features))
        
        node_features = np.vstack(node_features)
        
        # Calculate correlations for edges
        # Group samples by symbol for correlation calculation
        returns_by_symbol = {}
        for symbol in symbols:
            symbol_samples = sorted([s for s in window_samples if s.symbol == symbol], 
                                   key=lambda x: x.timestamp)
            if len(symbol_samples) > 1:
                returns = [s.forward_return for s in symbol_samples]
                returns_by_symbol[symbol] = returns
        
        # Build edge list based on correlations
        edge_list = []
        edge_features = []
        
        for i, symbol1 in enumerate(symbols):
            for j, symbol2 in enumerate(symbols):
                if i >= j:  # Skip self-loops and duplicate edges
                    continue
                
                if symbol1 in returns_by_symbol and symbol2 in returns_by_symbol:
                    returns1 = returns_by_symbol[symbol1]
                    returns2 = returns_by_symbol[symbol2]
                    
                    # Align lengths
                    min_len = min(len(returns1), len(returns2))
                    if min_len > 3:
                        corr = np.corrcoef(returns1[:min_len], returns2[:min_len])[0, 1]
                        
                        if abs(corr) > self.correlation_threshold:
                            # Add bidirectional edges
                            edge_list.append([i, j])
                            edge_list.append([j, i])
                            
                            # Edge features: [correlation, abs_correlation, is_positive]
                            edge_feat = [corr, abs(corr), 1.0 if corr > 0 else 0.0]
                            edge_features.append(edge_feat)
                            edge_features.append(edge_feat)
        
        # If too few edges, add top correlations
        if len(edge_list) < self.top_k_edges * 2:
            all_correlations = []
            for i, symbol1 in enumerate(symbols):
                for j, symbol2 in enumerate(symbols):
                    if i >= j:
                        continue
                    if symbol1 in returns_by_symbol and symbol2 in returns_by_symbol:
                        returns1 = returns_by_symbol[symbol1]
                        returns2 = returns_by_symbol[symbol2]
                        min_len = min(len(returns1), len(returns2))
                        if min_len > 3:
                            corr = np.corrcoef(returns1[:min_len], returns2[:min_len])[0, 1]
                            all_correlations.append((i, j, corr))
            
            # Sort by absolute correlation
            all_correlations.sort(key=lambda x: abs(x[2]), reverse=True)
            
            # Add top K edges
            for i, j, corr in all_correlations[:self.top_k_edges]:
                if [i, j] not in edge_list:
                    edge_list.append([i, j])
                    edge_list.append([j, i])
                    edge_feat = [corr, abs(corr), 1.0 if corr > 0 else 0.0]
                    edge_features.append(edge_feat)
                    edge_features.append(edge_feat)
        
        # Convert to tensors
        x = torch.FloatTensor(node_features)
        
        if edge_list:
            edge_index = torch.LongTensor(edge_list).t().contiguous()
            edge_attr = torch.FloatTensor(edge_features)
        else:
            # Create a minimally connected graph if no correlations
            edge_index = torch.LongTensor([[0, 1], [1, 0]]).t()
            edge_attr = torch.FloatTensor([[0, 0, 0], [0, 0, 0]])
        
        # Create PyTorch Geometric Data object
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        data.num_nodes = len(symbols)
        
        return data


@dataclass
class TrainingConfig:
    """Training configuration"""
    batch_size: int = 32
    learning_rate: float = 0.001
    num_epochs: int = 100
    early_stopping_patience: int = 10
    weight_decay: float = 1e-5
    gradient_clip: float = 1.0
    warmup_epochs: int = 5
    
    # Loss weights for multi-task learning
    risk_weight: float = 1.0
    volatility_weight: float = 0.5
    return_weight: float = 0.3
    
    # Graph construction
    correlation_threshold: float = 0.3
    graph_lookback_window: int = 60
    
    # Model architecture
    hidden_dim: int = 128
    num_heads: int = 8
    num_layers: int = 3
    dropout: float = 0.2


class QuantGNNTrainer:
    """
    Production trainer for Graph Neural Networks.
    Handles multi-task learning, validation, and model versioning.
    """
    
    def __init__(self, config: TrainingConfig = None):
        self.config = config or TrainingConfig()
        self.model = None
        self.graph_builder = DynamicGraphBuilder(
            correlation_threshold=config.correlation_threshold
        )
        self.feature_scaler = StandardScaler()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.training_history = []
        
    def prepare_graph_data(self,
                          samples: List[Any],
                          reference_time: datetime) -> List[Data]:
        """
        Prepare graph data for training.
        
        Args:
            samples: Temporal samples
            reference_time: Time point for graph construction
            
        Returns:
            List of graph data objects
        """
        graphs = []
        
        # Build graphs at different time points
        unique_times = sorted(list(set([s.timestamp for s in samples])))
        
        # Sample time points to reduce computation
        step = max(1, len(unique_times) // 100)  # Max 100 graphs
        sampled_times = unique_times[::step]
        
        for timestamp in sampled_times:
            if timestamp < reference_time:  # Only use historical data
                try:
                    graph = self.graph_builder.build_temporal_graph(
                        samples, timestamp, self.config.graph_lookback_window
                    )
                    
                    # Add labels to graph
                    samples_at_time = [s for s in samples if s.timestamp == timestamp]
                    if samples_at_time:
                        # Average labels for graph-level prediction
                        avg_return = np.mean([s.forward_return for s in samples_at_time])
                        avg_volatility = np.mean([s.forward_volatility for s in samples_at_time])
                        avg_drawdown = np.mean([s.forward_max_drawdown for s in samples_at_time])
                        
                        graph.y_risk = torch.FloatTensor([avg_drawdown])  # Risk as drawdown
                        graph.y_volatility = torch.FloatTensor([avg_volatility])
                        graph.y_return = torch.FloatTensor([avg_return])
                        
                        graphs.append(graph)
                        
                except Exception as e:
                    logger.warning(f"Failed to build graph at {timestamp}: {e}")
                    continue
        
        logger.info(f"Prepared {len(graphs)} graphs for training")
        return graphs
    
    def train(self,
              train_samples: List[Any],
              val_samples: List[Any],
              num_features: int) -> Dict[str, Any]:
        """
        Train the GNN model.
        
        Args:
            train_samples: Training samples
            val_samples: Validation samples
            num_features: Number of input features
            
        Returns:
            Training results and metrics
        """
        logger.info("="*50)
        logger.info("TRAINING GRAPH NEURAL NETWORK")
        logger.info("="*50)
        
        # Prepare graph data
        train_cutoff = max([s.timestamp for s in train_samples])
        val_cutoff = max([s.timestamp for s in val_samples])
        
        logger.info("Building training graphs...")
        train_graphs = self.prepare_graph_data(train_samples, train_cutoff)
        
        logger.info("Building validation graphs...")
        val_graphs = self.prepare_graph_data(val_samples, val_cutoff)
        
        if len(train_graphs) < 10 or len(val_graphs) < 5:
            raise ValueError(f"Insufficient graphs: {len(train_graphs)} train, {len(val_graphs)} val")
        
        # Create data loaders
        train_loader = DataLoader(train_graphs, batch_size=self.config.batch_size, shuffle=True)
        val_loader = DataLoader(val_graphs, batch_size=self.config.batch_size, shuffle=False)
        
        # Initialize model
        self.model = RealFinancialGNN(
            num_node_features=num_features,
            num_edge_features=3,
            hidden_dim=self.config.hidden_dim,
            num_heads=self.config.num_heads,
            num_layers=self.config.num_layers,
            dropout=self.config.dropout
        ).to(self.device)
        
        # Optimizer with weight decay
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config.num_epochs):
            # Training phase
            self.model.train()
            train_losses = []
            
            for batch in train_loader:
                batch = batch.to(self.device)
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                
                # Multi-task loss
                loss_risk = F.mse_loss(outputs['risk'].squeeze(), batch.y_risk)
                loss_vol = F.mse_loss(outputs['volatility'].squeeze(), batch.y_volatility)
                loss_return = F.mse_loss(outputs['return'].squeeze(), batch.y_return)
                
                total_loss = (self.config.risk_weight * loss_risk +
                             self.config.volatility_weight * loss_vol +
                             self.config.return_weight * loss_return)
                
                # Backward pass
                total_loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
                
                optimizer.step()
                train_losses.append(total_loss.item())
            
            # Validation phase
            self.model.eval()
            val_losses = []
            val_predictions = []
            val_actuals = []
            
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(self.device)
                    outputs = self.model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                    
                    loss_risk = F.mse_loss(outputs['risk'].squeeze(), batch.y_risk)
                    loss_vol = F.mse_loss(outputs['volatility'].squeeze(), batch.y_volatility)
                    loss_return = F.mse_loss(outputs['return'].squeeze(), batch.y_return)
                    
                    total_loss = (self.config.risk_weight * loss_risk +
                                 self.config.volatility_weight * loss_vol +
                                 self.config.return_weight * loss_return)
                    
                    val_losses.append(total_loss.item())
                    val_predictions.extend(outputs['risk'].squeeze().cpu().numpy())
                    val_actuals.extend(batch.y_risk.cpu().numpy())
            
            # Calculate metrics
            avg_train_loss = np.mean(train_losses)
            avg_val_loss = np.mean(val_losses)
            
            # Correlation between predictions and actuals
            if len(val_predictions) > 1:
                correlation = np.corrcoef(val_predictions, val_actuals)[0, 1]
            else:
                correlation = 0.0
            
            # Log progress
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: Train Loss = {avg_train_loss:.4f}, "
                           f"Val Loss = {avg_val_loss:.4f}, Correlation = {correlation:.3f}")
            
            # Learning rate scheduling
            scheduler.step(avg_val_loss)
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                
                # Save best model
                self.save_checkpoint(epoch, avg_val_loss)
            else:
                patience_counter += 1
                if patience_counter >= self.config.early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
            
            # Store training history
            self.training_history.append({
                'epoch': epoch,
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'correlation': correlation,
                'learning_rate': optimizer.param_groups[0]['lr']
            })
        
        # Load best model
        self.load_best_checkpoint()
        
        # Final evaluation
        final_metrics = self.evaluate(val_graphs)
        
        logger.info("="*50)
        logger.info("TRAINING COMPLETE")
        logger.info(f"Best validation loss: {best_val_loss:.4f}")
        logger.info(f"Final correlation: {final_metrics['correlation']:.3f}")
        logger.info("="*50)
        
        return {
            'best_val_loss': best_val_loss,
            'final_metrics': final_metrics,
            'training_history': self.training_history,
            'model_params': sum(p.numel() for p in self.model.parameters())
        }
    
    def evaluate(self, test_graphs: List[Data]) -> Dict[str, float]:
        """
        Evaluate model on test graphs.
        
        Args:
            test_graphs: List of test graphs
            
        Returns:
            Evaluation metrics
        """
        self.model.eval()
        test_loader = DataLoader(test_graphs, batch_size=self.config.batch_size, shuffle=False)
        
        all_predictions = {
            'risk': [],
            'volatility': [],
            'return': []
        }
        all_actuals = {
            'risk': [],
            'volatility': [],
            'return': []
        }
        
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(self.device)
                outputs = self.model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                
                all_predictions['risk'].extend(outputs['risk'].squeeze().cpu().numpy())
                all_predictions['volatility'].extend(outputs['volatility'].squeeze().cpu().numpy())
                all_predictions['return'].extend(outputs['return'].squeeze().cpu().numpy())
                
                all_actuals['risk'].extend(batch.y_risk.cpu().numpy())
                all_actuals['volatility'].extend(batch.y_volatility.cpu().numpy())
                all_actuals['return'].extend(batch.y_return.cpu().numpy())
        
        # Calculate metrics
        metrics = {}
        
        for task in ['risk', 'volatility', 'return']:
            preds = np.array(all_predictions[task])
            actuals = np.array(all_actuals[task])
            
            if len(preds) > 0 and len(actuals) > 0:
                mse = np.mean((preds - actuals) ** 2)
                mae = np.mean(np.abs(preds - actuals))
                correlation = np.corrcoef(preds, actuals)[0, 1] if len(preds) > 1 else 0.0
                
                metrics[f'{task}_mse'] = float(mse)
                metrics[f'{task}_mae'] = float(mae)
                metrics[f'{task}_correlation'] = float(correlation)
        
        # Overall metrics
        metrics['correlation'] = np.mean([metrics.get(f'{t}_correlation', 0) for t in ['risk', 'volatility', 'return']])
        
        return metrics
    
    def save_checkpoint(self, epoch: int, val_loss: float):
        """Save model checkpoint."""
        checkpoint_dir = Path("models/checkpoints")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'val_loss': val_loss,
            'config': self.config,
            'feature_scaler': self.feature_scaler,
            'timestamp': datetime.now().isoformat()
        }
        
        path = checkpoint_dir / f"gnn_checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, path)
        
        # Also save as best if this is the best so far
        best_path = checkpoint_dir / "best_model.pt"
        torch.save(checkpoint, best_path)
        
        logger.info(f"Checkpoint saved: {path}")
    
    def load_best_checkpoint(self):
        """Load best model checkpoint."""
        best_path = Path("models/checkpoints/best_model.pt")
        if best_path.exists():
            checkpoint = torch.load(best_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Loaded best model from epoch {checkpoint['epoch']}")
    
    def save_production_model(self, version: str, metadata: Dict[str, Any]):
        """
        Save production-ready model with versioning.
        
        Args:
            version: Model version string
            metadata: Additional metadata to store
        """
        production_dir = Path("models/production")
        production_dir.mkdir(parents=True, exist_ok=True)
        
        # Create versioned filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"gnn_model_v{version}_{timestamp}.pt"
        
        # Prepare model package
        model_package = {
            'version': version,
            'timestamp': timestamp,
            'model_state_dict': self.model.state_dict(),
            'model_config': {
                'num_node_features': self.model.num_node_features,
                'hidden_dim': self.model.hidden_dim,
                'architecture': 'GAT'
            },
            'training_config': self.config.__dict__,
            'feature_scaler': self.feature_scaler,
            'metadata': metadata,
            'checksum': self._compute_model_checksum()
        }
        
        # Save model
        path = production_dir / filename
        torch.save(model_package, path)
        
        # Also save as latest
        latest_path = production_dir / "latest_model.pt"
        torch.save(model_package, latest_path)
        
        # Save metadata separately for easy access
        metadata_path = production_dir / f"model_metadata_v{version}.json"
        with open(metadata_path, 'w') as f:
            json.dump({
                'version': version,
                'timestamp': timestamp,
                'path': str(path),
                'metrics': metadata.get('metrics', {}),
                'checksum': model_package['checksum']
            }, f, indent=2)
        
        logger.info(f"Production model saved: {path}")
        logger.info(f"Model checksum: {model_package['checksum']}")
        
        return path
    
    def _compute_model_checksum(self) -> str:
        """Compute checksum of model weights."""
        if self.model is None:
            return ""
        
        # Get all model parameters
        params = []
        for param in self.model.parameters():
            params.append(param.cpu().detach().numpy().tobytes())
        
        # Compute SHA256
        combined = b''.join(params)
        return hashlib.sha256(combined).hexdigest()[:16]


# Example usage
if __name__ == "__main__":
    # Configuration
    config = TrainingConfig(
        batch_size=32,
        learning_rate=0.001,
        num_epochs=100,
        hidden_dim=128,
        num_heads=8,
        num_layers=3
    )
    
    # Initialize trainer
    trainer = QuantGNNTrainer(config)
    
    print("GNN Trainer initialized")
    print(f"Device: {trainer.device}")
    print(f"Config: {config.__dict__}")
    print("\nReady to train REAL Graph Neural Network")
    print("This uses actual message passing, not fake MLPs")