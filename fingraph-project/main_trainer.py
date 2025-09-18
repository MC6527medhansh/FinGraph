"""
Main Training Pipeline - Complete End-to-End Orchestrator
This is the single entry point that runs everything correctly
"""

import argparse
import sys
import os
from pathlib import Path
import logging
import json
import yaml
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Optional, Any

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

# Import all our production components
from data_pipeline_quant import QuantDataPipeline, TemporalSample
from integrity_checker import QuantIntegrityChecker
from gnn_trainer import QuantGNNTrainer, TrainingConfig
from quant_backtester import RealisticBacktester, TradingConfig
from inference_engine import RealTimeInference

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class FinGraphPipeline:
    """
    Complete end-to-end pipeline for FinGraph.
    
    This orchestrates:
    1. Data collection with integrity
    2. Feature engineering without lookahead
    3. Graph construction with temporal dynamics
    4. GNN training with validation
    5. Backtesting with realistic costs
    6. Model deployment
    """
    
    def __init__(self, config_path: str = "configs/experiment.yaml"):
        """
        Initialize pipeline with configuration.
        
        Args:
            config_path: Path to experiment configuration
        """
        self.config = self._load_config(config_path)
        self.results = {}
        self.artifacts_dir = Path("artifacts") / datetime.now().strftime("%Y%m%d_%H%M%S")
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.data_pipeline = None
        self.integrity_checker = None
        self.trainer = None
        self.backtester = None
        self.inference_engine = None
        
    def _load_config(self, config_path: str) -> Dict:
        """Load experiment configuration"""
        config_file = Path(config_path)
        
        if config_file.exists():
            with open(config_file, 'r') as f:
                if config_file.suffix == '.yaml':
                    return yaml.safe_load(f)
                else:
                    return json.load(f)
        else:
            # Default configuration
            logger.warning(f"Config file {config_path} not found. Using defaults.")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Get default configuration"""
        return {
            'data': {
                'symbols': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 
                           'META', 'NVDA', 'NFLX', 'CRM', 'ADBE'],
                'start_date': '2019-01-01',
                'end_date': '2024-01-01',
                'min_history_days': 252,
                'label_horizon': 21,
                'sample_frequency': 5
            },
            'model': {
                'hidden_dim': 128,
                'num_heads': 8,
                'num_layers': 3,
                'dropout': 0.2,
                'learning_rate': 0.001,
                'batch_size': 32,
                'num_epochs': 100,
                'early_stopping_patience': 10
            },
            'backtest': {
                'initial_capital': 1000000,
                'position_size': 0.02,
                'commission_bps': 10,
                'slippage_bps': 5,
                'stop_loss': 0.05,
                'take_profit': 0.10
            },
            'validation': {
                'strict_mode': True,
                'bootstrap_samples': 1000,
                'confidence_level': 0.95
            }
        }
    
    def run(self, 
            skip_data_download: bool = False,
            skip_training: bool = False,
            skip_backtest: bool = False,
            skip_validation: bool = False) -> Dict[str, Any]:
        """
        Run complete pipeline.
        
        Args:
            skip_data_download: Use cached data if available
            skip_training: Skip model training
            skip_backtest: Skip backtesting
            skip_validation: Skip integrity validation
            
        Returns:
            Complete results dictionary
        """
        logger.info("="*80)
        logger.info("FINGRAPH PRODUCTION PIPELINE")
        logger.info("="*80)
        logger.info(f"Artifacts directory: {self.artifacts_dir}")
        
        try:
            # Step 1: Data Collection and Preparation
            if not skip_data_download:
                logger.info("\n" + "="*50)
                logger.info("STEP 1: DATA COLLECTION")
                logger.info("="*50)
                
                self.data_pipeline = QuantDataPipeline(
                    cache_dir=str(self.artifacts_dir / "data"),
                    min_history_days=self.config['data']['min_history_days'],
                    label_horizon=self.config['data']['label_horizon'],
                    validation_mode=self.config['validation']['strict_mode']
                )
                
                # Load market data
                market_data = self.data_pipeline.load_market_data(
                    symbols=self.config['data']['symbols'],
                    start_date=self.config['data']['start_date'],
                    end_date=self.config['data']['end_date'],
                    validate=True
                )
                
                # Create temporal dataset
                samples = self.data_pipeline.create_temporal_dataset(
                    data=market_data,
                    symbols=self.config['data']['symbols'],
                    sample_freq=self.config['data']['sample_frequency']
                )
                
                # Create train/val/test splits
                splits = self.data_pipeline.create_train_val_test_splits(
                    samples=samples,
                    train_pct=0.6,
                    val_pct=0.2,
                    gap_days=5
                )
                
                # Save processed data
                self.data_pipeline.save_dataset(
                    samples, 
                    str(self.artifacts_dir / "data" / "temporal_samples.json")
                )
                
                logger.info(f"✓ Data prepared: {len(samples)} samples")
                logger.info(f"  Train: {len(splits['train'])}")
                logger.info(f"  Val: {len(splits['val'])}")
                logger.info(f"  Test: {len(splits['test'])}")
                
                self.results['data'] = {
                    'num_samples': len(samples),
                    'train_size': len(splits['train']),
                    'val_size': len(splits['val']),
                    'test_size': len(splits['test']),
                    'date_range': f"{samples[0].timestamp} to {samples[-1].timestamp}"
                }
            else:
                logger.info("Skipping data download - using cached data")
                # Load cached data
                # Implementation depends on your caching strategy
            
            # Step 2: Integrity Validation
            if not skip_validation:
                logger.info("\n" + "="*50)
                logger.info("STEP 2: INTEGRITY VALIDATION")
                logger.info("="*50)
                
                self.integrity_checker = QuantIntegrityChecker(
                    strict_mode=self.config['validation']['strict_mode']
                )
                
                # Validate temporal integrity
                temporal_passed, temporal_details = self.integrity_checker.validate_temporal_integrity(samples)
                logger.info(f"Temporal integrity: {'PASSED' if temporal_passed else 'FAILED'}")
                
                # Validate splits
                split_passed, split_details = self.integrity_checker.validate_train_test_splits(
                    splits['train'], splits['val'], splits['test']
                )
                logger.info(f"Split validation: {'PASSED' if split_passed else 'FAILED'}")
                
                # Check for information leakage
                if len(samples) > 100:
                    features = np.vstack([s.features for s in samples[:100]])
                    labels = np.array([s.forward_return for s in samples[:100]])
                    timestamps = [s.timestamp for s in samples[:100]]
                    
                    no_leakage, leakage_details = self.integrity_checker.detect_information_leakage(
                        features, labels, timestamps
                    )
                    logger.info(f"Leakage detection: {'PASSED' if no_leakage else 'FAILED'}")
                
                if not (temporal_passed and split_passed):
                    if self.config['validation']['strict_mode']:
                        raise ValueError("Integrity validation failed in strict mode")
                    else:
                        logger.warning("Continuing despite validation failures")
                
                self.results['validation'] = {
                    'temporal_integrity': temporal_passed,
                    'split_validation': split_passed,
                    'no_leakage': no_leakage if 'no_leakage' in locals() else None
                }
            
            # Step 3: Model Training
            if not skip_training:
                logger.info("\n" + "="*50)
                logger.info("STEP 3: MODEL TRAINING")
                logger.info("="*50)
                
                # Configure training
                training_config = TrainingConfig(
                    batch_size=self.config['model']['batch_size'],
                    learning_rate=self.config['model']['learning_rate'],
                    num_epochs=self.config['model']['num_epochs'],
                    early_stopping_patience=self.config['model']['early_stopping_patience'],
                    hidden_dim=self.config['model']['hidden_dim'],
                    num_heads=self.config['model']['num_heads'],
                    num_layers=self.config['model']['num_layers'],
                    dropout=self.config['model']['dropout']
                )
                
                self.trainer = QuantGNNTrainer(config=training_config)
                
                # Train model
                num_features = len(splits['train'][0].features)
                training_results = self.trainer.train(
                    train_samples=splits['train'],
                    val_samples=splits['val'],
                    num_features=num_features
                )
                
                # Evaluate on test set
                test_graphs = self.trainer.prepare_graph_data(
                    splits['test'],
                    max([s.timestamp for s in splits['test']])
                )
                test_metrics = self.trainer.evaluate(test_graphs)
                
                # Save production model
                model_path = self.trainer.save_production_model(
                    version="1.0.0",
                    metadata={
                        'training_results': training_results,
                        'test_metrics': test_metrics,
                        'config': self.config
                    }
                )
                
                logger.info(f"✓ Model trained and saved: {model_path}")
                logger.info(f"  Best val loss: {training_results['best_val_loss']:.4f}")
                logger.info(f"  Test correlation: {test_metrics['correlation']:.3f}")
                
                self.results['training'] = {
                    'best_val_loss': training_results['best_val_loss'],
                    'test_metrics': test_metrics,
                    'model_params': training_results['model_params'],
                    'model_path': str(model_path)
                }
            
            # Step 4: Backtesting
            if not skip_backtest:
                logger.info("\n" + "="*50)
                logger.info("STEP 4: REALISTIC BACKTESTING")
                logger.info("="*50)
                
                # Generate predictions for backtesting
                test_predictions = self._generate_backtest_predictions(
                    splits['test'],
                    self.trainer.model
                )
                
                # Configure backtesting
                backtest_config = TradingConfig(
                    initial_capital=self.config['backtest']['initial_capital'],
                    position_size=self.config['backtest']['position_size'],
                    commission_bps=self.config['backtest']['commission_bps'],
                    slippage_bps=self.config['backtest']['slippage_bps'],
                    stop_loss=self.config['backtest']['stop_loss'],
                    take_profit=self.config['backtest']['take_profit']
                )
                
                self.backtester = RealisticBacktester(config=backtest_config)
                
                # Prepare market data for backtesting
                backtest_market_data = self._prepare_backtest_data(market_data, splits['test'])
                
                # Run backtest
                backtest_start = min([s.timestamp for s in splits['test']])
                backtest_end = max([s.timestamp for s in splits['test']])
                
                backtest_results = self.backtester.backtest(
                    predictions=test_predictions,
                    market_data=backtest_market_data,
                    start_date=backtest_start,
                    end_date=backtest_end
                )
                
                # Validate backtest results
                if not skip_validation:
                    bt_passed, bt_details = self.integrity_checker.validate_backtest_integrity(
                        backtest_results, backtest_market_data
                    )
                    logger.info(f"Backtest validation: {'PASSED' if bt_passed else 'FAILED'}")
                
                # Save backtest results
                self.backtester.save_results(
                    backtest_results,
                    str(self.artifacts_dir / "backtest_results.json")
                )
                
                logger.info(f"✓ Backtest complete")
                logger.info(f"  Total return: {backtest_results['total_return']:.1%}")
                logger.info(f"  Sharpe ratio: {backtest_results['sharpe_ratio']:.2f}")
                logger.info(f"  Max drawdown: {backtest_results['max_drawdown']:.1%}")
                
                # Compare with baseline
                baseline = backtest_results.get('baseline_comparison', {})
                if baseline:
                    outperformance = backtest_results['sharpe_ratio'] - baseline.get('sharpe_ratio', 0)
                    logger.info(f"  Outperformance vs buy-hold: {outperformance:.2f} Sharpe")
                
                self.results['backtest'] = backtest_results
            
            # Step 5: Deploy for Inference
            logger.info("\n" + "="*50)
            logger.info("STEP 5: INFERENCE DEPLOYMENT")
            logger.info("="*50)
            
            # Initialize inference engine
            self.inference_engine = RealTimeInference(
                model_path=self.results.get('training', {}).get('model_path', 'models/production/latest_model.pt')
            )
            
            # Test inference
            from inference_engine import PredictionRequest
            test_request = PredictionRequest(
                symbols=['AAPL', 'MSFT'],
                lookback_days=60,
                use_cache=False,
                include_confidence=True
            )
            
            test_response = self.inference_engine.predict(test_request)
            
            logger.info(f"✓ Inference engine deployed")
            logger.info(f"  Latency: {test_response.latency_ms:.1f}ms")
            logger.info(f"  Market regime: {test_response.market_regime}")
            
            self.results['inference'] = {
                'deployed': True,
                'test_latency_ms': test_response.latency_ms,
                'model_version': test_response.model_version
            }
            
            # Final summary
            self._print_final_summary()
            
            # Save all results
            self._save_all_results()
            
            return self.results
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            raise
    
    def _generate_backtest_predictions(self, 
                                      test_samples: List[TemporalSample],
                                      model) -> pd.DataFrame:
        """Generate predictions for backtesting"""
        predictions = []
        
        # Group samples by date
        from collections import defaultdict
        samples_by_date = defaultdict(list)
        for sample in test_samples:
            samples_by_date[sample.timestamp].append(sample)
        
        for date, date_samples in samples_by_date.items():
            # Build graph for this date
            graph = self.trainer.graph_builder.build_temporal_graph(
                test_samples, date, self.trainer.config.graph_lookback_window
            )
            
            # Run inference
            with torch.no_grad():
                graph = graph.to(self.trainer.device)
                outputs = model(graph.x, graph.edge_index, graph.edge_attr, graph.batch)
                
                risk_scores = outputs['risk'].squeeze().cpu().numpy()
                returns = outputs['return'].squeeze().cpu().numpy()
            
            # Create predictions for each symbol
            for i, sample in enumerate(date_samples):
                if i < len(risk_scores):
                    predictions.append({
                        'date': date,
                        'symbol': sample.symbol,
                        'signal': float(returns[i] - risk_scores[i]),  # Alpha signal
                        'predicted_risk': float(risk_scores[i]),
                        'predicted_return': float(returns[i])
                    })
        
        return pd.DataFrame(predictions)
    
    def _prepare_backtest_data(self, 
                              market_data: pd.DataFrame,
                              test_samples: List[TemporalSample]) -> pd.DataFrame:
        """Prepare market data for backtesting"""
        # Get date range from test samples
        test_dates = [s.timestamp for s in test_samples]
        min_date = min(test_dates)
        max_date = max(test_dates) + timedelta(days=30)  # Extra for position closing
        
        # Filter market data
        backtest_data = market_data[
            (market_data['timestamp'] >= min_date) &
            (market_data['timestamp'] <= max_date)
        ].copy()
        
        # Rename columns for backtester
        backtest_data = backtest_data.rename(columns={
            'timestamp': 'date',
            'Symbol': 'symbol',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        })
        
        return backtest_data
    
    def _print_final_summary(self):
        """Print final summary of results"""
        logger.info("\n" + "="*80)
        logger.info("PIPELINE SUMMARY")
        logger.info("="*80)
        
        # Data summary
        if 'data' in self.results:
            logger.info("\nDATA:")
            logger.info(f"  Samples: {self.results['data']['num_samples']}")
            logger.info(f"  Date range: {self.results['data']['date_range']}")
        
        # Validation summary
        if 'validation' in self.results:
            logger.info("\nVALIDATION:")
            for check, passed in self.results['validation'].items():
                status = "✓" if passed else "✗"
                logger.info(f"  {check}: {status}")
        
        # Training summary
        if 'training' in self.results:
            logger.info("\nTRAINING:")
            logger.info(f"  Model parameters: {self.results['training']['model_params']:,}")
            logger.info(f"  Test correlation: {self.results['training']['test_metrics']['correlation']:.3f}")
        
        # Backtest summary
        if 'backtest' in self.results:
            logger.info("\nBACKTEST:")
            logger.info(f"  Sharpe ratio: {self.results['backtest']['sharpe_ratio']:.2f}")
            logger.info(f"  Total return: {self.results['backtest']['total_return']:.1%}")
            logger.info(f"  Max drawdown: {self.results['backtest']['max_drawdown']:.1%}")
            logger.info(f"  Win rate: {self.results['backtest']['win_rate']:.1%}")
        
        logger.info("\n" + "="*80)
    
    def _save_all_results(self):
        """Save all pipeline results"""
        # Save results summary
        results_path = self.artifacts_dir / "pipeline_results.json"
        
        # Convert numpy/torch types for JSON serialization
        def convert_types(obj):
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (datetime, pd.Timestamp)):
                return obj.isoformat()
            elif torch is not None and isinstance(obj, torch.Tensor):
                return obj.cpu().numpy().tolist()
            return obj
        
        serializable_results = json.loads(
            json.dumps(self.results, default=convert_types)
        )
        
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Results saved to {results_path}")
        
        # Save configuration used
        config_path = self.artifacts_dir / "config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
        
        logger.info(f"Configuration saved to {config_path}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="FinGraph Production Training Pipeline")
    
    parser.add_argument('--config', type=str, default='configs/experiment.yaml',
                       help='Path to experiment configuration')
    parser.add_argument('--skip-data', action='store_true',
                       help='Skip data download, use cached')
    parser.add_argument('--skip-training', action='store_true',
                       help='Skip model training')
    parser.add_argument('--skip-backtest', action='store_true',
                       help='Skip backtesting')
    parser.add_argument('--skip-validation', action='store_true',
                       help='Skip integrity validation (not recommended)')
    parser.add_argument('--symbols', nargs='+',
                       help='Override symbols from config')
    parser.add_argument('--start-date', type=str,
                       help='Override start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str,
                       help='Override end date (YYYY-MM-DD)')
    
    args = parser.parse_args()
    
    # Create logs directory
    Path("logs").mkdir(exist_ok=True)
    
    # Initialize pipeline
    pipeline = FinGraphPipeline(config_path=args.config)
    
    # Override config if arguments provided
    if args.symbols:
        pipeline.config['data']['symbols'] = args.symbols
    if args.start_date:
        pipeline.config['data']['start_date'] = args.start_date
    if args.end_date:
        pipeline.config['data']['end_date'] = args.end_date
    
    # Run pipeline
    try:
        results = pipeline.run(
            skip_data_download=args.skip_data,
            skip_training=args.skip_training,
            skip_backtest=args.skip_backtest,
            skip_validation=args.skip_validation
        )
        
        logger.info("\n🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        return 0
        
    except Exception as e:
        logger.error(f"\n❌ PIPELINE FAILED: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())