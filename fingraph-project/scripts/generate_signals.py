#!/usr/bin/env python
"""
Daily Signal Generation Service for FinGraph
Generates trading signals for dashboard consumption (no label leakage)
"""

import sys
from pathlib import Path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

import torch
import pandas as pd
from datetime import datetime
import logging
from typing import Dict
import yaml

from src.models.gnn_model import FinancialGNN
from src.pipeline.graph_builder import TemporalGraphBuilder
from src.core.data_manager import UnifiedDataManager           # << unified name
from src.core.feature_engine import UnifiedFeatureEngine       # << unified name

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SignalGenerator:
    """Generate daily trading signals from trained model"""

    def __init__(self, config_path: str = 'config/pipeline_config.yaml'):
        with open(project_root / config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.data_manager = UnifiedDataManager(self.config)
        self.feature_engine = UnifiedFeatureEngine(self.config)
        self.graph_builder = TemporalGraphBuilder(self.config)

        # Load latest model
        self.model, self.ckpt_meta = self._load_latest_model()

    def _load_latest_model(self):
        """Load the most recent trained model & align input dims safely."""
        model_files = list((project_root / 'data/models').glob('*.pt'))
        if not model_files:
            raise FileNotFoundError("No trained model found!")

        latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
        logger.info(f"Loading model: {latest_model.name}")

        ckpt = torch.load(latest_model, map_location='cpu', weights_only=False)
        meta = ckpt.get('metadata', {})  # we saved this in your trainer
        train_feature_names = meta.get('feature_names')
        n_in = meta.get('num_features')

        if not train_feature_names or not n_in:
            raise RuntimeError("Checkpoint missing feature metadata; retrain with metadata saving enabled.")

        # Initialize model with training-time input size
        model = FinancialGNN(
            num_node_features=n_in,
            hidden_dim=self.config['model']['hidden_dim'],
            num_heads=self.config['model']['num_heads'],
            num_layers=self.config['model']['num_layers'],
            dropout=self.config['model']['dropout'],
            node_level=True
        )
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()
        return model, meta

    def generate_current_signals(self) -> pd.DataFrame:
        """Generate signals for the most recent valid feature date (prediction path, no labels)."""
        logger.info("Generating current signals...")

        # 1) Load full price history
        pkg = self.data_manager.load_all_data(use_cache=False)
        prices = pkg['prices'].copy()

        # Normalize timezone
        if hasattr(prices.index, 'tz') and prices.index.tz is not None:
            prices.index = prices.index.tz_localize(None)

        if prices.empty:
            logger.error("No price data available")
            return pd.DataFrame()

        # 2) Build prediction-only features for the latest trading day
        asof = prices.index.max()
        from pandas.tseries.offsets import BDay
        today = pd.Timestamp.now().normalize()
        days_stale = (today - asof.normalize()).days
        if days_stale > 2:  # tolerate weekend/holiday
            logger.warning(f"Signals are {days_stale} days old (as-of {asof.date()}). Market may be closed or data delayed.")

        features_pred = self.feature_engine.create_features_for_prediction(
            prices,
            end_date=asof
        )
        if features_pred.empty:
            logger.error("Prediction feature creation returned empty.")
            return pd.DataFrame()

        # 3) Reorder/align columns to exactly match training schema (no leakage cols)
        # Training feature order = metadata saved at training time
        train_feature_names = self.ckpt_meta['feature_names']  # list[str], includes ONLY model inputs at train time
        have_cols = set(features_pred.columns)

        cols_missing = [c for c in train_feature_names if c not in have_cols]
        cols_extra   = [c for c in features_pred.columns if c not in train_feature_names + ['date','symbol']]

        if cols_extra:
            # Drop any accidental new cols
            features_pred = features_pred.drop(columns=cols_extra)

        # Add any missing expected cols as zeros (defensive)
        for c in cols_missing:
            logger.warning(f"Prediction features missing expected column '{c}'. Filling with zeros.")
            features_pred[c] = 0.0

        # Final column order: date, symbol, then training feature order
        features_pred = features_pred[['date', 'symbol'] + train_feature_names]

        logger.info(f"Generating signals for date: {asof}")

        # 4) Build graph for prediction (no labels)
        graph = self.graph_builder.build_graph_for_prediction(features_pred, asof)
        if graph is None:
            logger.error(f"Failed to build graph for date {asof}")
            return pd.DataFrame()

        # 5) Inference
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(graph.x, graph.edge_index, graph.edge_attr)

        # 6) Assemble signal dataframe
        signals = pd.DataFrame({
            'date': [asof] * len(graph.symbols),
            'symbol': graph.symbols,
            'risk_score': outputs['risk'].squeeze().cpu().numpy(),
            'return_forecast': outputs['return'].squeeze().cpu().numpy(),
            'volatility_forecast': outputs['volatility'].squeeze().cpu().numpy()
        })

        # 7) Ranking & recommendations (safe math)
        signals['signal_strength'] = signals['return_forecast'] / (signals['risk_score'].astype(float) + 1e-2)
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
        signals_dir = project_root / 'data/signals'
        signals_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        csv_path = signals_dir / f'signals_{timestamp}.csv'
        signals.to_csv(csv_path, index=False)

        latest_csv = signals_dir / 'latest_signals.csv'
        signals.to_csv(latest_csv, index=False)

        # JSON
        sj = signals.copy()
        sj['date'] = sj['date'].astype(str)
        (signals_dir / f'signals_{timestamp}.json').write_text(sj.to_json(orient='records'))
        (signals_dir / 'latest_signals.json').write_text(sj.to_json(orient='records'))

        logger.info(f"Signals saved to {csv_path}")
        return csv_path

    def generate_performance_metrics(self) -> Dict:
        """Lightweight summary for dashboard health checks"""
        signals_dir = project_root / 'data/signals'
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
