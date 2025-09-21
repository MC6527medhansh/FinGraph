#!/usr/bin/env python
"""
Main Pipeline Script - Phase 2 Clean Implementation
Following roadmap strictly with no legacy code
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import yaml
import logging
from datetime import datetime
import argparse
import pandas as pd
import torch
from torch_geometric.loader import DataLoader

# Import Phase 1 components
from src.core.data_manager import UnifiedDataManager
from src.core.feature_engine import UnifiedFeatureEngine

# Import Phase 2 components
from src.pipeline.graph_builder import TemporalGraphBuilder
from src.models.gnn_model import FinancialGNN
from src.pipeline.trainer import ModelTrainer


def setup_logging(config: dict):
    """Setup logging configuration"""
    log_config = config.get('logging', {})
    
    # Create logs directory
    log_dir = Path(log_config.get('file', 'logs/pipeline.log')).parent
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=getattr(logging, log_config.get('level', 'INFO')),
        format=log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
        handlers=[
            logging.FileHandler(log_config.get('file', 'logs/pipeline.log')),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_data_splits(graphs: list, config: dict, logger) -> tuple:
    """
    Create train/val/test splits with temporal gaps.
    
    Args:
        graphs: List of graph objects
        config: Configuration dictionary
        logger: Logger instance
        
    Returns:
        Tuple of (train_graphs, val_graphs, test_graphs)
    """
    total_graphs = len(graphs)
    
    # Calculate split indices
    train_size = int(total_graphs * config['validation']['train_pct'])
    val_size = int(total_graphs * config['validation']['val_pct'])
    
    # Apply temporal gaps (in terms of graph indices)
    gap = config['validation'].get('train_val_gap', 5)
    
    train_end = train_size
    val_start = train_end + gap
    val_end = val_start + val_size
    test_start = val_end + gap
    
    # Ensure we have enough data
    if test_start >= total_graphs:
        logger.warning("Adjusting splits due to temporal gaps")
        # Adjust without gaps if necessary
        train_graphs = graphs[:train_size]
        val_graphs = graphs[train_size:train_size + val_size]
        test_graphs = graphs[train_size + val_size:]
    else:
        train_graphs = graphs[:train_end]
        val_graphs = graphs[val_start:val_end]
        test_graphs = graphs[test_start:]
    
    logger.info(f"Data splits with temporal gaps:")
    logger.info(f"  Train: {len(train_graphs)} graphs")
    logger.info(f"  Val: {len(val_graphs)} graphs")
    logger.info(f"  Test: {len(test_graphs)} graphs")
    
    return train_graphs, val_graphs, test_graphs


def main(args):
    """
    Main pipeline execution - Phase 2 clean implementation.
    
    Pipeline flow (per roadmap):
    1. Data Collection (Phase 1) ✅
    2. Feature Engineering (Phase 1) ✅
    3. Graph Construction (Phase 2)
    4. Model Training (Phase 2)
    5. Validation (Phase 2)
    6. Save Results (Phase 2)
    """
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup logging
    logger = setup_logging(config)
    
    logger.info("=" * 60)
    logger.info("🚀 FINGRAPH PIPELINE v2.0 - PHASE 2")
    logger.info("=" * 60)
    logger.info(f"Configuration: {args.config}")
    logger.info(f"Mode: {args.mode}")
    
    try:
        # ========== PHASE 1 COMPONENTS ==========
        # Step 1: Data Management
        logger.info("\n📊 Step 1: Loading Data (Phase 1)")
        data_manager = UnifiedDataManager(config)
        data_package = data_manager.load_all_data(use_cache=not args.fresh)
        
        logger.info(f"✅ Loaded data for {len(data_package['metadata']['symbols'])} symbols")
        
        # Step 2: Feature Engineering
        logger.info("\n🔧 Step 2: Creating Features (Phase 1)")
        feature_engine = UnifiedFeatureEngine(config)
        features = feature_engine.create_features(data_package['prices'])
        
        logger.info(f"✅ Created {len(features)} feature vectors")
        # ✱ NEW — log presence of cross-sectional columns
        ### ✱ NEW
        expected_cols = ['forward_return_cs_z', 'forward_volatility_cs_z', 'risk_score_cs_pct']
        have = [c for c in expected_cols if c in features.columns]
        logger.info(f"Cross-sectional targets present: {have}")
        
        # Save features
        if args.save:
            saved_path = data_manager.save_processed_data(features, 'features')
            logger.info(f"💾 Saved features to {saved_path}")
        
        # ========== PHASE 2 COMPONENTS ==========
        # Step 3: Graph Construction
        logger.info("\n🌐 Step 3: Building Temporal Graphs (Phase 2)")
        graph_builder = TemporalGraphBuilder(config)
        graphs = graph_builder.create_temporal_graphs(features)
        
        logger.info(f"✅ Created {len(graphs)} temporal graphs")
        if graphs:
            logger.info(f"   Nodes per graph: {graphs[0].x.shape[0]}")
            logger.info(f"   Features per node: {graphs[0].x.shape[1]}")
            logger.info(f"   Average edges: {graphs[0].edge_index.shape[1]}")
        
        # Step 4: Create Data Splits
        logger.info("\n📊 Step 4: Creating Train/Val/Test Splits")
        train_graphs, val_graphs, test_graphs = create_data_splits(graphs, config, logger)
        
        # Step 5: Model Training
        if args.mode == 'train' and len(train_graphs) > 0:
            logger.info("\n🎯 Step 5: Training Model")
            
            # Create data loaders
            train_loader = DataLoader(
                train_graphs, 
                batch_size=config['model']['batch_size'], 
                shuffle=True
            )
            val_loader = DataLoader(
                val_graphs, 
                batch_size=config['model']['batch_size'], 
                shuffle=False
            )
            test_loader = DataLoader(
                test_graphs, 
                batch_size=config['model']['batch_size'], 
                shuffle=False
            )
            
            # Initialize model
            num_features = graphs[0].x.shape[1]
            model = FinancialGNN(
                num_node_features=num_features,
                hidden_dim=config['model']['hidden_dim'],
                num_heads=config['model']['num_heads'],
                num_layers=config['model']['num_layers'],
                dropout=config['model']['dropout']
            )
            
            logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
            
            # Initialize trainer
            trainer = ModelTrainer(model, config)
            
            # Train model
            trained_model = trainer.train(
                train_loader=train_loader,
                val_loader=val_loader,
                num_epochs=config['model']['num_epochs']
            )
            
            # Evaluate on test set
            logger.info("\n📈 Step 6: Evaluating Model")
            test_metrics = trainer.evaluate(test_loader)
            
            logger.info("✅ Test Results:")
            logger.info(f"   Risk MSE: {test_metrics['risk_mse']:.4f}")
            logger.info(f"   Risk Correlation: {test_metrics['risk_correlation']:.4f}")
            logger.info(f"   Return MSE: {test_metrics['return_mse']:.4f}")
            logger.info(f"   Return Correlation: {test_metrics['return_correlation']:.4f}")
            logger.info(f"   Volatility MSE: {test_metrics['vol_mse']:.4f}")
            logger.info(f"   Volatility Correlation: {test_metrics['volatility_correlation']:.4f}")
            
            # Save model
            if args.save:
                model_dir = Path(config['paths']['model_dir'])
                model_path = model_dir / f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
                
                metadata = {
                    'feature_names': feature_engine.feature_names,
                    'num_features': num_features,
                    'symbols': data_package['metadata']['symbols'],
                    'test_metrics': test_metrics
                }
                
                trainer.save_model(model_path, metadata)
                logger.info(f"💾 Saved model to {model_path}")
        
        logger.info("\n" + "=" * 60)
        logger.info("✅ PHASE 2 COMPLETE - READY FOR PHASE 3 (BACKTESTING)")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FinGraph Pipeline - Phase 2")
    parser.add_argument('--config', type=str, default='config/pipeline_config.yaml',
                      help='Path to configuration file')
    parser.add_argument('--mode', type=str, default='train',
                      choices=['train', 'evaluate', 'predict'],
                      help='Pipeline mode')
    parser.add_argument('--fresh', action='store_true',
                      help='Force fresh data download (ignore cache)')
    parser.add_argument('--save', action='store_true',
                      help='Save all outputs (features, models, results)')
    
    args = parser.parse_args()
    main(args)
