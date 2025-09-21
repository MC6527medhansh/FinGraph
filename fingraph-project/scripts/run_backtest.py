#!/usr/bin/env python
"""
Run backtest on trained model - NODE-LEVEL VERSION
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import pandas as pd
import numpy as np
from torch_geometric.loader import DataLoader

from src.models.gnn_model import FinancialGNN
from src.pipeline.graph_builder import TemporalGraphBuilder
from src.backtesting.backtester import FinGraphBacktester, BacktestConfig
from src.core.data_manager import UnifiedDataManager
import yaml

def run_backtest():
    print("=" * 60)
    print("📈 RUNNING BACKTEST - NODE-LEVEL PREDICTIONS")
    print("=" * 60)
    
    # Load model
    model_files = list(Path('data/models').glob('*.pt'))
    if not model_files:
        print("❌ No model found! Train a model first.")
        return
    
    latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
    print(f"Loading model: {latest_model.name}")
    
    checkpoint = torch.load(latest_model, map_location='cpu', weights_only=False)
    
    # Load features
    features = pd.read_parquet('data/processed/features_latest.parquet')
    print(f"Loaded {len(features)} feature vectors")
    
    # Load config
    with open('config/pipeline_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Build graphs
    graph_builder = TemporalGraphBuilder(config)
    graphs = graph_builder.create_temporal_graphs(features)
    print(f"Created {len(graphs)} graphs")
    
    # Initialize model with node_level=True
    model = FinancialGNN(
        num_node_features=graphs[0].x.shape[1],
        hidden_dim=config['model']['hidden_dim'],
        num_heads=config['model']['num_heads'],
        num_layers=config['model']['num_layers'],
        dropout=config['model']['dropout'],
        node_level=True  # IMPORTANT: Enable node-level predictions
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print("\n📊 Generating NODE-LEVEL predictions...")
    
    # Generate predictions
    predictions_list = []
    
    with torch.no_grad():
        for graph in graphs:
            # Get predictions for this graph
            outputs = model(graph.x, graph.edge_index, graph.edge_attr)
            
            # NOW WE HAVE NODE-LEVEL PREDICTIONS!
            # outputs['risk'] is [num_nodes, 1]
            
            # Extract predictions for each node/symbol
            num_nodes = len(graph.symbols)
            
            for i, symbol in enumerate(graph.symbols):
                predictions_list.append({
                    'date': graph.date,
                    'symbol': symbol,
                    'risk_pred': outputs['risk'][i].item(),      # Individual prediction for this stock
                    'return_pred': outputs['return'][i].item(),  # Individual prediction for this stock
                    'vol_pred': outputs['volatility'][i].item()  # Individual prediction for this stock
                })
    
    predictions_df = pd.DataFrame(predictions_list)
    
    # Verify we have different predictions for different stocks
    sample_date = predictions_df['date'].iloc[0]
    sample_preds = predictions_df[predictions_df['date'] == sample_date]
    print(f"\nSample predictions for {sample_date}:")
    print(sample_preds[['symbol', 'risk_pred', 'return_pred']].head())
    print(f"Unique risk predictions on this date: {sample_preds['risk_pred'].nunique()}")
    
    # Load price data
    print("\n📉 Loading price data...")
    data_manager = UnifiedDataManager(config)
    data_package = data_manager.load_all_data(use_cache=True)
    prices = data_package['prices'].reset_index()
    prices.columns = [col.lower() for col in prices.columns]
    
    print(f"Price columns: {prices.columns.tolist()}")
    print(f"Price shape: {prices.shape}")
    
    # Run backtest
    print("\n🎯 Running backtest with NODE-LEVEL predictions...")
    config = BacktestConfig()
    backtester = FinGraphBacktester(config)
    
    # Filter to test period only
    test_start = predictions_df['date'].quantile(0.7)  # Last 30% for testing
    test_predictions = predictions_df[predictions_df['date'] >= test_start]
    
    results = backtester.backtest(test_predictions, prices)
    
    # Display results
    print("\n" + "=" * 60)
    print("📊 BACKTEST RESULTS (NODE-LEVEL):")
    print("=" * 60)
    
    for key, value in results.items():
        if isinstance(value, float):
            if 'return' in key or 'drawdown' in key or 'rate' in key:
                print(f"{key:20s}: {value:>10.2%}")
            elif key == 'final_value':
                print(f"{key:20s}: ${value:>10,.2f}")
            else:
                print(f"{key:20s}: {value:>10.2f}")
        else:
            print(f"{key:20s}: {value}")
    
    print("\n" + "=" * 60)
    if results.get('sharpe_ratio', 0) > 1.5:
        print("✅ Strategy is PROFITABLE with good risk-adjusted returns!")
    elif results.get('sharpe_ratio', 0) > 0.8:
        print("⚠️ Strategy shows promise but needs optimization.")
    else:
        print("❌ Strategy needs fundamental improvements.")
    print("=" * 60)

if __name__ == "__main__":
    run_backtest()