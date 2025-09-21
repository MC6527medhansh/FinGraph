"""
Model Training Module - Following roadmap specifications
Handles training, validation, and model persistence
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
# >>> CHANGED: ensure we have pooling available
from torch_geometric.nn import global_mean_pool  # >>> CHANGED
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import json
import logging

logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Production trainer with experiment tracking and validation.
    """
    
    def __init__(self, model: nn.Module, config: Dict, device: Optional[str] = None):
        """
        Initialize trainer.
        
        Args:
            model: GNN model to train
            config: Training configuration
            device: Device to use (auto-detect if None)
        """
        self.model = model
        self.config = config
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.model.to(self.device)
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_metrics': [],
            'val_metrics': []
        }
        
        # Best model tracking
        self.best_val_loss = float('inf')
        self.best_model_state = None
        self.patience_counter = 0
        
        logger.info(f"Initialized trainer on device: {self.device}")
    
    def train(self, 
             train_loader: DataLoader,
             val_loader: DataLoader,
             num_epochs: Optional[int] = None) -> nn.Module:
        """
        Train the model with validation.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs (uses config if None)
            
        Returns:
            Trained model with best weights
        """
        num_epochs = num_epochs or self.config['model']['num_epochs']
        
        # Setup optimizer
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.config['model']['learning_rate'],
            weight_decay=1e-5
        )
        
        # Learning rate scheduler
        scheduler = ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.5, 
            patience=5,
            min_lr=1e-6,
            # removed deprecated verbose=True
        )
        
        logger.info(f"Starting training for {num_epochs} epochs")
        logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
        
        for epoch in range(num_epochs):
            # Training phase
            train_loss, train_metrics = self._train_epoch(train_loader, optimizer)
            
            # Validation phase
            val_loss, val_metrics = self._validate_epoch(val_loader)
            
            # Record history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_metrics'].append(train_metrics)
            self.history['val_metrics'].append(val_metrics)
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Model checkpointing
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model_state = self.model.state_dict().copy()
                self.patience_counter = 0
                logger.info(f"  📈 New best model (val_loss: {val_loss:.4f})")
            else:
                self.patience_counter += 1
            
            # Logging
            if epoch % max(1, num_epochs // 20) == 0:
                logger.info(
                    f"Epoch {epoch}/{num_epochs} - "
                    f"Train Loss: {train_loss:.4f}, "
                    f"Val Loss: {val_loss:.4f}, "
                    f"Risk MSE: {val_metrics['risk_mse']:.4f}"
                )
            
            # Early stopping
            if self.patience_counter >= self.config['model']['early_stopping_patience']:
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break
        
        # Load best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            logger.info(f"Loaded best model with val_loss: {self.best_val_loss:.4f}")
        
        return self.model

    # >>> CHANGED: helper to compute MSE with node/graph alignment
    @staticmethod
    def _mse_with_shape_alignment(pred: torch.Tensor,
                                  target: torch.Tensor,
                                  batch) -> torch.Tensor:
        p = pred.squeeze().view(-1)
        t = target.squeeze().view(-1)

        n_graphs = int(batch.num_graphs)
        n_nodes  = int(batch.num_nodes)

        sp, st = p.shape[0], t.shape[0]

        if sp == st:
            return F.mse_loss(p, t)

        # pred graph-level, target node-level -> pool target
        if sp == n_graphs and st == n_nodes:
            t_g = global_mean_pool(t, batch.batch)
            return F.mse_loss(p, t_g)

        # pred node-level, target graph-level -> pool pred
        if sp == n_nodes and st == n_graphs:
            p_g = global_mean_pool(p, batch.batch)
            return F.mse_loss(p_g, t)

        # As fallback, broadcast graph <-> node and compute node loss
        if sp == n_graphs and st == n_nodes:
            p_node = p[batch.batch]
            return F.mse_loss(p_node, t)

        if sp == n_nodes and st == n_graphs:
            t_node = t[batch.batch]
            return F.mse_loss(p, t_node)

        raise RuntimeError(
            f"Cannot align shapes: pred={p.shape}, target={t.shape}, "
            f"num_graphs={n_graphs}, num_nodes={n_nodes}"
        )
    # >>> CHANGED END

    # >>> CHANGED: helper to convert any 1D vec to GRAPH-LEVEL for aggregation/eval
    @staticmethod
    def _to_graph_level(vec: torch.Tensor, batch) -> torch.Tensor:
        """
        Return graph-level vector (length = num_graphs).
        If vec is node-level, mean-pool by graph using batch.batch.
        """
        v = vec.squeeze().view(-1)
        if v.shape[0] == int(batch.num_graphs):
            return v
        if v.shape[0] == int(batch.num_nodes):
            return global_mean_pool(v, batch.batch)
        # try broadcast down to nodes then pool (graph-level → node-level case)
        if v.shape[0] == int(batch.num_graphs):
            v_node = v[batch.batch]
            return global_mean_pool(v_node, batch.batch)
        raise RuntimeError(f"Unexpected vector size for _to_graph_level: {v.shape}")
    # >>> CHANGED END

    def _train_epoch(self, loader: DataLoader, optimizer) -> Tuple[float, Dict]:
        """Train for one epoch"""
        self.model.train()
        losses = []
        all_metrics = {'risk_mse': [], 'return_mse': [], 'vol_mse': []}
        
        for batch in loader:
            batch = batch.to(self.device)
            optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            
            # Use shape-aligned losses
            loss_risk   = self._mse_with_shape_alignment(outputs['risk'],        batch.y_risk,       batch)
            loss_return = self._mse_with_shape_alignment(outputs['return'],      batch.y_return,     batch)
            loss_vol    = self._mse_with_shape_alignment(outputs['volatility'],  batch.y_volatility, batch)
            
            # Weighted multi-task loss
            total_loss = (
                self.config['model']['risk_weight'] * loss_risk +
                self.config['model']['return_weight'] * loss_return +
                self.config['model']['volatility_weight'] * loss_vol
            )
            
            # Backward pass
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Record metrics
            losses.append(total_loss.item())
            all_metrics['risk_mse'].append(loss_risk.item())
            all_metrics['return_mse'].append(loss_return.item())
            all_metrics['vol_mse'].append(loss_vol.item())
        
        return np.mean(losses), {k: np.mean(v) for k, v in all_metrics.items()}
    
    def _validate_epoch(self, loader: DataLoader) -> Tuple[float, Dict]:
        """Validate for one epoch"""
        self.model.eval()
        losses = []
        all_metrics = {'risk_mse': [], 'return_mse': [], 'vol_mse': []}
        
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                
                # Forward pass
                outputs = self.model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                
                # Use shape-aligned losses
                loss_risk   = self._mse_with_shape_alignment(outputs['risk'],        batch.y_risk,       batch)
                loss_return = self._mse_with_shape_alignment(outputs['return'],      batch.y_return,     batch)
                loss_vol    = self._mse_with_shape_alignment(outputs['volatility'],  batch.y_volatility, batch)
                
                total_loss = (
                    self.config['model']['risk_weight'] * loss_risk +
                    self.config['model']['return_weight'] * loss_return +
                    self.config['model']['volatility_weight'] * loss_vol
                )
                
                losses.append(total_loss.item())
                all_metrics['risk_mse'].append(loss_risk.item())
                all_metrics['return_mse'].append(loss_return.item())
                all_metrics['vol_mse'].append(loss_vol.item())
        
        return np.mean(losses), {k: np.mean(v) for k, v in all_metrics.items()}
    
    def _verify_predictions(self, outputs, batch):
        """Verify predictions are node-level"""
        num_nodes = batch.x.shape[0]
        
        for key in ['risk', 'return', 'volatility']:
            pred_shape = outputs[key].shape[0]
            if pred_shape != num_nodes:
                logger.warning(f"Expected {num_nodes} predictions for {key}, got {pred_shape}")
        
        return outputs
    
    def evaluate(self, test_loader: DataLoader) -> Dict:
        """
        Evaluate model on test set - FIXED version.
        """
        logger.info("Evaluating on test set...")
        test_loss, test_metrics = self._validate_epoch(test_loader)
        
        # Get predictions using fixed method
        predictions, actuals = self._get_predictions(test_loader)
        
        test_metrics['test_loss'] = test_loss
        
        # Calculate correlations safely
        for key in ['risk', 'return', 'volatility']:
            if len(predictions[key]) > 1 and len(actuals[key]) > 1:
                # Check for variance
                if np.std(predictions[key]) > 0 and np.std(actuals[key]) > 0:
                    corr = np.corrcoef(predictions[key], actuals[key])[0, 1]
                    test_metrics[f'{key}_correlation'] = corr
                else:
                    test_metrics[f'{key}_correlation'] = 0.0
            else:
                test_metrics[f'{key}_correlation'] = 0.0
        
        return test_metrics
    
    def _get_predictions(self, loader: DataLoader) -> Tuple[Dict, Dict]:
        """Get all predictions and actuals - FIXED to match debug script logic"""
        self.model.eval()
        
        all_predictions = {'risk': [], 'return': [], 'volatility': []}
        all_actuals = {'risk': [], 'return': [], 'volatility': []}
        
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                
                # Get model outputs
                outputs = self.model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                
                # YOUR DEBUG SCRIPT LOGIC - This works correctly!
                # The model outputs are already at graph-level after pooling
                
                # Get risk predictions (already graph-level from model)
                risk_pred = outputs['risk'].squeeze()
                return_pred = outputs['return'].squeeze()
                vol_pred = outputs['volatility'].squeeze()
                
                # Convert to list (matching your debug script's to_1d_list logic)
                if risk_pred.dim() == 0:
                    # Single value
                    pred_risk = [risk_pred.item()]
                    pred_return = [return_pred.item()]
                    pred_vol = [vol_pred.item()]
                else:
                    # Multiple values
                    pred_risk = risk_pred.cpu().numpy().tolist()
                    pred_return = return_pred.cpu().numpy().tolist()
                    pred_vol = vol_pred.cpu().numpy().tolist()
                
                # Get actual labels - these are node-level, need to aggregate per graph
                if batch.batch is not None:
                    # Multiple graphs in batch
                    num_graphs = batch.batch.max().item() + 1
                    
                    for i in range(num_graphs):
                        # Get mask for this graph's nodes
                        mask = (batch.batch == i)
                        
                        # Predictions are already per-graph
                        all_predictions['risk'].append(pred_risk[i] if i < len(pred_risk) else pred_risk[0])
                        all_predictions['return'].append(pred_return[i] if i < len(pred_return) else pred_return[0])
                        all_predictions['volatility'].append(pred_vol[i] if i < len(pred_vol) else pred_vol[0])
                        
                        # Actuals need to be averaged across nodes in the graph
                        all_actuals['risk'].append(batch.y_risk[mask].mean().item())
                        all_actuals['return'].append(batch.y_return[mask].mean().item())
                        all_actuals['volatility'].append(batch.y_volatility[mask].mean().item())
                else:
                    # Single graph
                    all_predictions['risk'].extend(pred_risk)
                    all_predictions['return'].extend(pred_return)
                    all_predictions['volatility'].extend(pred_vol)
                    
                    # Take mean of actuals across all nodes
                    all_actuals['risk'].append(batch.y_risk.mean().item())
                    all_actuals['return'].append(batch.y_return.mean().item())
                    all_actuals['volatility'].append(batch.y_volatility.mean().item())
        
        # Convert to arrays and clean
        for key in all_predictions:
            preds = np.array(all_predictions[key], dtype=float)
            acts = np.array(all_actuals[key], dtype=float)
            
            # Remove any NaN or infinite values
            valid_mask = np.isfinite(preds) & np.isfinite(acts)
            all_predictions[key] = preds[valid_mask]
            all_actuals[key] = acts[valid_mask]
        
        return all_predictions, all_actuals
    
    def save_model(self, save_path: str, metadata: Optional[Dict] = None):
        """
        Save model with metadata.
        
        Args:
            save_path: Path to save model
            metadata: Additional metadata to save
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'history': self.history,
            'best_val_loss': self.best_val_loss,
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {}
        }
        
        torch.save(checkpoint, save_path)
        logger.info(f"Model saved to {save_path}")
        
        # Also save training history as JSON
        history_path = save_path.parent / f"{save_path.stem}_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        logger.info(f"Training history saved to {history_path}")
