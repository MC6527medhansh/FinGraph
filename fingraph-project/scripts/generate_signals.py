#!/usr/bin/env python
"""
Daily Signal Generation Service for FinGraph
Generates trading signals for dashboard consumption
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from typing import Dict

from src.models.gnn_model import FinancialGNN
from src.pipeline.graph_builder import TemporalGraphBuilder
from src.core.data_manager import DataManager
from src.core.feature_engine import FeatureEngine
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SignalGenerator:
    """Generate daily trading signals from trained model"""

    def __init__(self, config_path: str = 'config/pipeline_config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.data_manager = DataManager(self.config)
        self.feature_engine = FeatureEngine(self.config)
        self.graph_builder = TemporalGraphBuilder(self.config)

        # Load latest model
        self.model = self._load_latest_model()

    def _load_latest_model(self) -> FinancialGNN:
        """Load the most recent trained model"""
        model_files = list(Path('data/models').glob('*.pt'))
        if not model_files:
            raise FileNotFoundError("No trained model found!")

        latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
        logger.info(f"Loading model: {latest_model.name}")

        checkpoint = torch.load(latest_model, map_location='cpu', weights_only=False)

        # Initialize model using one dummy graph to get feature dimensionality
        dummy_features = pd.read_parquet('data/processed/features_latest.parquet')
        dummy_graphs = self.graph_builder.create_temporal_graphs(dummy_features, max_graphs=1)

        model = FinancialGNN(
            num_node_features=dummy_graphs[0].x.shape[1],
            hidden_dim=self.config['model']['hidden_dim'],
            num_heads=self.config['model']['num_heads'],
            num_layers=self.config['model']['num_layers'],
            dropout=self.config['model']['dropout'],
            node_level=True
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model

    def generate_current_signals(self) -> pd.DataFrame:
        """Generate signals for the most recent valid feature date"""
        logger.info("Generating current signals...")

        # 1) Load FULL price history so we meet min_history_days
        data_package = self.data_manager.load_all_data(use_cache=False)

        prices = data_package['prices'].copy()
        # Normalize timezone
        if hasattr(prices.index, 'tz') and prices.index.tz is not None:
            prices.index = prices.index.tz_localize(None)

        if prices.empty:
            logger.error("No price data available")
            return pd.DataFrame()

        # 2) Create features from full history (no slicing!)
        features = self.feature_engine.create_features(prices)
        if features.empty:
            logger.error("Feature creation failed")
            return pd.DataFrame()

        # 3) Use the most recent valid feature date
        latest_date = features['date'].max()
        logger.info(f"Generating signals for date: {latest_date}")

        # 4) Build a graph for that date
        graph = self.graph_builder.build_graph_from_features(features, latest_date)
        if graph is None:
            logger.error(f"Failed to build graph for date {latest_date}")
            return pd.DataFrame()

        # 5) Predict node-level scores
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(graph.x, graph.edge_index, graph.edge_attr)

        # 6) Assemble signal dataframe
        signals = pd.DataFrame({
            'date': [latest_date] * len(graph.symbols),
            'symbol': graph.symbols,
            'risk_score': outputs['risk'].squeeze().cpu().numpy(),
            'return_forecast': outputs['return'].squeeze().cpu().numpy(),
            'volatility_forecast': outputs['volatility'].squeeze().cpu().numpy()
        })

        # 7) Ranking & recommendations (safe math)
        signals['signal_strength'] = signals['return_forecast'] / (signals['risk_score'] + 1e-2)
        signals['rank'] = signals['signal_strength'].rank(ascending=False, method='first')

        signals['recommendation'] = 'HOLD'
        signals.loc[signals['rank'] <= 2, 'recommendation'] = 'STRONG_BUY'
        signals.loc[(signals['rank'] > 2) & (signals['rank'] <= 5), 'recommendation'] = 'BUY'
        signals.loc[signals['rank'] > max(8, len(signals) - 2), 'recommendation'] = 'SELL'

        # 8) Round for display
        for col in ['risk_score', 'return_forecast', 'volatility_forecast', 'signal_strength']:
            signals[col] = signals[col].astype(float).round(6)

        return signals

    def save_signals(self, signals: pd.DataFrame) -> Path:
        """Save signals to CSV and JSON for dashboard consumption"""
        signals_dir = Path('data/signals')
        signals_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        csv_path = signals_dir / f'signals_{timestamp}.csv'
        signals.to_csv(csv_path, index=False)

        latest_csv = signals_dir / 'latest_signals.csv'
        signals.to_csv(latest_csv, index=False)

        # JSON
        json_path = signals_dir / f'signals_{timestamp}.json'
        sj = signals.copy()
        sj['date'] = sj['date'].astype(str)
        sj.to_json(json_path, orient='records')
        sj.to_json(signals_dir / 'latest_signals.json', orient='records')

        logger.info(f"Signals saved to {csv_path}")
        return csv_path

    def generate_performance_metrics(self) -> Dict:
        """Lightweight summary for dashboard health checks"""
        signals_dir = Path('data/signals')
        latest = signals_dir / 'latest_signals.csv'
        if not latest.exists():
            return {'status': 'No signals available yet'}

        df = pd.read_csv(latest)
        if df.empty:
            return {'status': 'Signals file is empty'}

        return {
            'signal_date': str(df['date'].iloc[0]),
            'total_stocks': int(len(df)),
            'strong_buys': int((df['recommendation'] == 'STRONG_BUY').sum()),
            'buys': int((df['recommendation'] == 'BUY').sum()),
            'holds': int((df['recommendation'] == 'HOLD').sum()),
            'sells': int((df['recommendation'] == 'SELL').sum()),
            'avg_risk_score': float(df['risk_score'].mean()),
            'avg_return_forecast': float(df['return_forecast'].mean()),
            'top_pick': df.sort_values('rank').iloc[0]['symbol'] if not df.empty else 'N/A'
        }


def main():
    try:
        generator = SignalGenerator()

        signals = generator.generate_current_signals()
        if signals.empty:
            logger.error("No signals generated - market may be closed or data unavailable")
            print("❌ No signals found. Run generate_signals.py first.")
            return

        # Show top picks
        print("\n" + "="*60)
        print("TOP TRADING SIGNALS")
        print("="*60)
        cols = ['symbol', 'risk_score', 'return_forecast', 'signal_strength', 'recommendation']
        print(signals.sort_values('rank').head(10)[cols].to_string(index=False))

        # Save + health metrics
        generator.save_signals(signals)
        metrics = generator.generate_performance_metrics()

        print("\n" + "="*60)
        print("SIGNAL METRICS")
        print("="*60)
        for k, v in metrics.items():
            print(f"{k}: {v}")

        print("\n✅ Signal generation complete!")

    except Exception as e:
        logger.error(f"Signal generation failed: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
