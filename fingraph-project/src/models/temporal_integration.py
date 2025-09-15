"""
FinGraph Temporal Integration - COMPLETE INTEGRATION

This integrates the temporal fix with your existing FinGraph components
"""

import sys
import os
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

# Import your existing FinGraph components
from src.features.graph_data_loader import GraphDataLoader
from src.features.graph_constructor import FinGraphConstructor
from src.features.graph_analyzer import FinGraphAnalyzer

# Import the temporal system
from src.models.temporal_risk_predictor import TemporalRiskPredictor

import pandas as pd
import numpy as np
import torch
import logging
import json
from datetime import datetime
from typing import Optional, Dict, Any

PREDICTIONS_DIR = Path(__file__).resolve().parents[2] / 'data' / 'temporal_integration'
DEFAULT_PREDICTIONS_FILE = PREDICTIONS_DIR / 'predictions.csv'
DEFAULT_SUMMARY_FILE = PREDICTIONS_DIR / 'dashboard_summary.json'
DEFAULT_METADATA_FILE = PREDICTIONS_DIR / 'metadata.json'

logger = logging.getLogger(__name__)

class FinGraphTemporalIntegrator:
    """
    Integrates temporal risk prediction with existing FinGraph infrastructure
    """

    def __init__(
        self,
        ensure_fresh_data: bool = False,
        max_data_age_hours: Optional[int] = 24,
    ):
        """Create a new integrator instance.

        Args:
            ensure_fresh_data: When ``True``, the loader refreshes raw data if it is
                missing or outdated.
            max_data_age_hours: Maximum allowed age (in hours) for raw data files
                before triggering a refresh.
        """

        # Initialize existing FinGraph components
        self.graph_loader = GraphDataLoader()
        self.graph_constructor = FinGraphConstructor()

        self.ensure_fresh_data = ensure_fresh_data
        self.max_data_age_hours = max_data_age_hours

        # Initialize temporal predictor
        self.temporal_predictor = TemporalRiskPredictor()

        # Storage for results
        self.loaded_data = None
        self.enhanced_graph = None
        self.temporal_results = None
        self.integration_results: Dict[str, Any] = {}
    
    def load_existing_fingraph_data(
        self,
        refresh_if_missing: Optional[bool] = None,
        max_age_hours: Optional[int] = None,
    ):
        """Load data using existing FinGraph infrastructure."""
        logger.info("📂 Loading data using existing FinGraph components...")

        try:
            refresh = self.ensure_fresh_data if refresh_if_missing is None else refresh_if_missing
            max_age = self.max_data_age_hours if max_age_hours is None else max_age_hours

            # Use existing graph data loader
            self.loaded_data = self.graph_loader.load_latest_data(
                refresh_if_missing=refresh,
                max_age_hours=max_age,
            )

            logger.info("✅ Successfully loaded FinGraph data:")
            logger.info(f"  📊 Stock data: {len(self.loaded_data['stock_data'])} records")
            logger.info(f"  🏢 Company info: {len(self.loaded_data['company_info'])} companies")
            logger.info(f"  🏛️ Economic data: {self.loaded_data['economic_data'].shape}")
            logger.info(f"  🔗 Relationships: {len(self.loaded_data['relationship_data'])} connections")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load existing FinGraph data: {str(e)}")
            logger.info("💡 Make sure you have run data collection first:")
            logger.info("  python scripts/collect_data.py")
            logger.info("💡 Or run the pipeline with data refresh enabled: --refresh-data")
            return False
    
    def run_temporal_analysis(self):
        """Run temporal risk prediction on the loaded data"""
        logger.info("⚡ Running temporal risk analysis...")
        
        if self.loaded_data is None:
            logger.error("❌ No data loaded. Run load_existing_fingraph_data() first.")
            return False
        
        try:
            # Set the stock data from loaded FinGraph data
            self.temporal_predictor.stock_data = self.loaded_data['stock_data']
            
            # Run temporal analysis pipeline
            steps = [
                ("Creating temporal dataset", self.temporal_predictor.create_temporal_dataset),
                ("Creating temporal splits", self.temporal_predictor.create_train_test_splits),
                ("Training temporal models", self.temporal_predictor.train_models),
                ("Evaluating models", self.temporal_predictor.evaluate_models)
            ]
            
            for step_name, step_func in steps:
                logger.info(f"  🔧 {step_name}...")
                if not step_func():
                    logger.error(f"❌ Failed at: {step_name}")
                    return False
            
            # Store temporal results
            self.temporal_results = self.temporal_predictor.results
            
            logger.info("✅ Temporal analysis completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Temporal analysis failed: {str(e)}")
            return False
    
    def build_enhanced_graph(self):
        """Build enhanced graph combining original structure with temporal insights"""
        logger.info("🏗️ Building enhanced graph with temporal features...")
        
        if self.loaded_data is None:
            logger.error("❌ No loaded data available")
            return False
        
        try:
            # Build original graph using existing constructor
            original_graph = self.graph_constructor.build_graph(
                self.loaded_data['stock_data'],
                self.loaded_data['company_info'],
                self.loaded_data['economic_data'],
                self.loaded_data['relationship_data']
            )
            
            logger.info(f"✅ Original graph: {original_graph.num_nodes} nodes, {original_graph.num_edges} edges")
            
            # Enhance with temporal risk features
            if hasattr(self.temporal_predictor, 'temporal_df') and self.temporal_predictor.temporal_df is not None:
                enhanced_features = self._add_temporal_features_to_graph(
                    original_graph, 
                    self.temporal_predictor.temporal_df
                )
                original_graph.x = enhanced_features
                logger.info("✅ Added temporal risk features to graph nodes")
            
            self.enhanced_graph = original_graph
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to build enhanced graph: {str(e)}")
            return False
    
    def _add_temporal_features_to_graph(self, graph, temporal_df):
        """Add temporal risk features to graph node features"""
        
        # Get original node features
        original_features = graph.x.numpy()
        
        # Calculate company risk statistics
        company_risk_stats = temporal_df.groupby('symbol').agg({
            'risk_score': ['mean', 'std', 'max'],
            'volatility': 'mean',
            'high_risk': 'mean'
        }).reset_index()
        
        # Flatten column names
        company_risk_stats.columns = [
            'symbol', 'risk_mean', 'risk_std', 'risk_max', 
            'volatility_mean', 'high_risk_pct'
        ]
        
        # Add temporal features to each node
        enhanced_features = []
        node_mapping = graph.node_mapping
        
        for i, original_node_features in enumerate(original_features):
            # Find the node name for this index
            node_name = None
            for name, idx in node_mapping.items():
                if idx == i:
                    node_name = name
                    break
            
            # Add temporal risk features
            if node_name and node_name.startswith('company_'):
                # Extract symbol from node name
                symbol = node_name.replace('company_', '')
                
                # Find risk stats for this company
                company_stats = company_risk_stats[company_risk_stats['symbol'] == symbol]
                
                if not company_stats.empty:
                    temporal_features = [
                        company_stats.iloc[0]['risk_mean'],
                        company_stats.iloc[0]['risk_std'],
                        company_stats.iloc[0]['risk_max'],
                        company_stats.iloc[0]['volatility_mean'],
                        company_stats.iloc[0]['high_risk_pct']
                    ]
                else:
                    # Default values for companies without temporal data
                    temporal_features = [0.5, 0.2, 0.7, 0.25, 0.3]
            else:
                # Default values for non-company nodes (sectors, economic indicators)
                temporal_features = [0.5, 0.2, 0.7, 0.25, 0.3]
            
            # Combine original features with temporal features
            enhanced_node_features = np.concatenate([original_node_features, temporal_features])
            enhanced_features.append(enhanced_node_features)
        
        return torch.tensor(enhanced_features, dtype=torch.float32)
    
    def analyze_enhanced_graph(self):
        """Analyze the enhanced graph structure"""
        logger.info("📊 Analyzing enhanced graph...")
        
        if self.enhanced_graph is None:
            logger.error("❌ No enhanced graph available")
            return False
        
        try:
            # Use existing graph analyzer
            analyzer = FinGraphAnalyzer(self.enhanced_graph)
            analysis_results = analyzer.analyze_graph_structure()
            
            # Generate analysis report
            report = analyzer.generate_analysis_report()
            
            # Store analysis results
            self.integration_results['graph_analysis'] = analysis_results
            self.integration_results['analysis_report'] = report
            
            logger.info("✅ Graph analysis completed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Graph analysis failed: {str(e)}")
            return False
    
    def generate_risk_predictions(self):
        """Generate current risk predictions for all companies"""
        logger.info("🔮 Generating current risk predictions...")
        
        if self.temporal_predictor.temporal_df is None:
            logger.error("❌ No temporal data available")
            return False
        
        try:
            # Get latest risk scores for each company
            latest_predictions = []
            
            for symbol in self.temporal_predictor.temporal_df['symbol'].unique():
                company_data = self.temporal_predictor.temporal_df[
                    self.temporal_predictor.temporal_df['symbol'] == symbol
                ]
                
                if not company_data.empty:
                    # Get latest risk information
                    latest = company_data.sort_values('date').iloc[-1]
                    
                    prediction = {
                        'symbol': symbol,
                        'risk_score': float(latest['risk_score']),
                        'volatility': float(latest['volatility']),
                        'high_risk_flag': bool(latest['high_risk']),
                        'risk_level': self._categorize_risk(float(latest['risk_score'])),
                        'prediction_date': datetime.now().strftime('%Y-%m-%d'),
                        'data_date': latest['date'].strftime('%Y-%m-%d'),
                        'last_updated': datetime.now().isoformat()
                    }

                    recent_metrics = self._compute_recent_market_metrics(symbol)
                    prediction.update(recent_metrics)
                    
                    latest_predictions.append(prediction)
            
            # Convert to DataFrame and sort by risk
            predictions_df = pd.DataFrame(latest_predictions)
            predictions_df = predictions_df.sort_values('risk_score', ascending=False)
            
            self.integration_results['current_predictions'] = predictions_df
            
            logger.info(f"✅ Generated predictions for {len(predictions_df)} companies")
            logger.info(f"🚨 High risk companies: {(predictions_df['risk_level'] == 'High').sum()}")

            return True

        except Exception as e:
            logger.error(f"❌ Failed to generate predictions: {str(e)}")
            return False

    def _compute_recent_market_metrics(self, symbol: str) -> Dict[str, float]:
        """Compute supporting market metrics for stored predictions."""

        if self.loaded_data is None:
            return {
                'momentum_5d': 0.0,
                'momentum_20d': 0.0,
                'rsi': 50.0,
                'max_drawdown': 0.0
            }

        stock_df = self.loaded_data.get('stock_data')
        if stock_df is None or stock_df.empty or 'Symbol' not in stock_df.columns or 'Close' not in stock_df.columns:
            return {
                'momentum_5d': 0.0,
                'momentum_20d': 0.0,
                'rsi': 50.0,
                'max_drawdown': 0.0
            }

        symbol_data = stock_df[stock_df['Symbol'] == symbol].copy()
        if symbol_data.empty:
            return {
                'momentum_5d': 0.0,
                'momentum_20d': 0.0,
                'rsi': 50.0,
                'max_drawdown': 0.0
            }

        symbol_data = symbol_data.sort_index()
        closes = symbol_data['Close'].astype(float)
        returns = closes.pct_change().dropna()

        momentum_5d = float(returns.tail(5).mean()) if len(returns) >= 5 else 0.0
        momentum_20d = float(returns.tail(20).mean()) if len(returns) >= 20 else momentum_5d

        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = ((cumulative - running_max) / running_max).min() if not cumulative.empty else 0.0

        rsi = self._calculate_rsi(closes)
        rsi_value = float(rsi.iloc[-1]) if not rsi.empty else 50.0

        return {
            'momentum_5d': momentum_5d,
            'momentum_20d': momentum_20d,
            'rsi': rsi_value,
            'max_drawdown': float(abs(drawdown)) if drawdown is not None else 0.0
        }

    def _calculate_rsi(self, prices) -> pd.Series:
        """Helper to compute RSI for stored predictions."""

        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(method='bfill').fillna(50)
    
    def _categorize_risk(self, risk_score):
        """Categorize risk score into levels"""
        if risk_score >= 0.7:
            return 'High'
        elif risk_score >= 0.4:
            return 'Medium'
        else:
            return 'Low'
    
    def create_dashboard_summary(self):
        """Create summary data for dashboard/reporting"""
        logger.info("📋 Creating dashboard summary...")
        
        try:
            summary = {
                'timestamp': datetime.now().isoformat(),
                'data_summary': self._create_data_summary(),
                'model_performance': self._create_model_summary(),
                'risk_overview': self._create_risk_overview(),
                'top_risks': self._create_top_risks(),
                'graph_stats': self._create_graph_summary()
            }
            
            self.integration_results['dashboard_summary'] = summary
            
            logger.info("✅ Dashboard summary created")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to create dashboard summary: {str(e)}")
            return False
    
    def _create_data_summary(self):
        """Create data summary for dashboard"""
        if self.loaded_data is None:
            return {}
        
        return {
            'total_companies': len(self.loaded_data['stock_data']['Symbol'].unique()),
            'total_records': len(self.loaded_data['stock_data']),
            'date_range': {
                'start': str(self.loaded_data['stock_data'].index.min().date()),
                'end': str(self.loaded_data['stock_data'].index.max().date())
            },
            'temporal_samples': len(self.temporal_predictor.temporal_df) if self.temporal_predictor.temporal_df is not None else 0
        }
    
    def _create_model_summary(self):
        """Create model performance summary"""
        if self.temporal_results is None:
            return {}
        
        model_performance = {}
        for model_name, results in self.temporal_results.items():
            if isinstance(results, dict) and 'mse' in results:
                model_performance[model_name] = {
                    'mse': results['mse'],
                    'rmse': np.sqrt(results['mse'])
                }
        
        return model_performance
    
    def _create_risk_overview(self):
        """Create risk level overview"""
        if 'current_predictions' not in self.integration_results:
            return {}
        
        predictions = self.integration_results['current_predictions']
        
        return {
            'total_companies': len(predictions),
            'high_risk_count': (predictions['risk_level'] == 'High').sum(),
            'medium_risk_count': (predictions['risk_level'] == 'Medium').sum(),
            'low_risk_count': (predictions['risk_level'] == 'Low').sum(),
            'average_risk_score': predictions['risk_score'].mean(),
            'max_risk_score': predictions['risk_score'].max()
        }
    
    def _create_top_risks(self):
        """Create top risk companies list"""
        if 'current_predictions' not in self.integration_results:
            return []
        
        predictions = self.integration_results['current_predictions']
        top_10 = predictions.head(10)
        
        return top_10[['symbol', 'risk_score', 'risk_level', 'volatility']].to_dict('records')
    
    def _create_graph_summary(self):
        """Create graph structure summary"""
        if 'graph_analysis' not in self.integration_results:
            return {}
        
        analysis = self.integration_results['graph_analysis']
        
        return {
            'total_nodes': analysis['basic_stats']['num_nodes'],
            'total_edges': analysis['basic_stats']['num_edges'],
            'graph_density': analysis['basic_stats']['density'],
            'is_connected': analysis['connectivity']['is_connected'],
            'num_communities': analysis['communities']['num_communities']
        }
    
    def save_results(self):
        """Save all integration results"""
        logger.info("💾 Saving integration results...")
        
        try:
            PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

            # Save predictions
            if 'current_predictions' in self.integration_results:
                predictions_df = self.integration_results['current_predictions']
                timestamped_file = PREDICTIONS_DIR / f'risk_predictions_{timestamp}.csv'
                predictions_df.to_csv(timestamped_file, index=False)
                predictions_df.to_csv(DEFAULT_PREDICTIONS_FILE, index=False)
                logger.info(f"📊 Saved predictions: {timestamped_file.name} and {DEFAULT_PREDICTIONS_FILE.name}")

            # Save dashboard summary
            if 'dashboard_summary' in self.integration_results:
                summary = self.integration_results['dashboard_summary']
                timestamped_summary = PREDICTIONS_DIR / f'dashboard_summary_{timestamp}.json'
                with timestamped_summary.open('w', encoding='utf-8') as f:
                    json.dump(summary, f, indent=2, default=str)
                with DEFAULT_SUMMARY_FILE.open('w', encoding='utf-8') as f:
                    json.dump(summary, f, indent=2, default=str)
                logger.info(f"📋 Saved dashboard summaries to {timestamped_summary.name} and {DEFAULT_SUMMARY_FILE.name}")

                metadata = {
                    'generated_at': datetime.now().isoformat(),
                    'prediction_file': DEFAULT_PREDICTIONS_FILE.name,
                    'summary_file': DEFAULT_SUMMARY_FILE.name,
                    'companies': summary.get('risk_overview', {}).get('total_companies')
                }
                with DEFAULT_METADATA_FILE.open('w', encoding='utf-8') as f:
                    json.dump(metadata, f, indent=2)
                logger.info(f"🗂️ Updated metadata: {DEFAULT_METADATA_FILE.name}")

            # Save analysis report
            if 'analysis_report' in self.integration_results:
                report_file = PREDICTIONS_DIR / f'graph_analysis_report_{timestamp}.txt'
                with report_file.open('w', encoding='utf-8') as f:
                    f.write(self.integration_results['analysis_report'])
                logger.info(f"📄 Saved analysis report: {report_file.name}")

            # Save enhanced graph
            if self.enhanced_graph is not None:
                graph_file = PREDICTIONS_DIR / f'enhanced_graph_{timestamp}.pt'
                torch.save(self.enhanced_graph, graph_file)
                logger.info(f"🌐 Saved enhanced graph: {graph_file.name}")

            logger.info(f"✅ All results saved to {PREDICTIONS_DIR}/")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to save results: {str(e)}")
            return False
    
    def run_complete_integration(self):
        """Run the complete integration pipeline"""
        logger.info("🚀 Starting FinGraph Temporal Integration")
        logger.info("="*60)
        
        integration_steps = [
            ("Loading existing FinGraph data", self.load_existing_fingraph_data),
            ("Running temporal risk analysis", self.run_temporal_analysis),
            ("Building enhanced graph", self.build_enhanced_graph),
            ("Analyzing enhanced graph", self.analyze_enhanced_graph),
            ("Generating risk predictions", self.generate_risk_predictions),
            ("Creating dashboard summary", self.create_dashboard_summary),
            ("Saving results", self.save_results)
        ]
        
        success_count = 0
        for step_name, step_func in integration_steps:
            logger.info(f"\n📋 Step {success_count + 1}: {step_name}...")
            if step_func():
                success_count += 1
                logger.info(f"✅ Completed: {step_name}")
            else:
                logger.error(f"❌ Failed: {step_name}")
                break
        
        if success_count == len(integration_steps):
            logger.info("\n🎉 Integration completed successfully!")
            self._print_integration_summary()
            return True
        else:
            logger.error(f"\n❌ Integration failed at step {success_count + 1}")
            return False
    
    def _print_integration_summary(self):
        """Print summary of integration results"""
        logger.info("\n📊 INTEGRATION SUMMARY")
        logger.info("="*50)
        
        if 'dashboard_summary' in self.integration_results:
            summary = self.integration_results['dashboard_summary']
            
            # Data summary
            data_sum = summary.get('data_summary', {})
            logger.info(f"📈 Data: {data_sum.get('total_companies', 0)} companies, {data_sum.get('temporal_samples', 0)} temporal samples")
            
            # Model performance
            model_perf = summary.get('model_performance', {})
            if model_perf:
                best_model = min(model_perf.keys(), key=lambda x: model_perf[x]['mse'])
                logger.info(f"🏆 Best model: {best_model} (MSE: {model_perf[best_model]['mse']:.4f})")
            
            # Risk overview
            risk_overview = summary.get('risk_overview', {})
            logger.info(f"🚨 Risk levels: {risk_overview.get('high_risk_count', 0)} High, {risk_overview.get('medium_risk_count', 0)} Medium, {risk_overview.get('low_risk_count', 0)} Low")
            
            # Graph stats
            graph_stats = summary.get('graph_stats', {})
            logger.info(f"🌐 Enhanced graph: {graph_stats.get('total_nodes', 0)} nodes, {graph_stats.get('total_edges', 0)} edges")
        
        logger.info("\n💡 Next Steps:")
        logger.info("  1. Review results in data/temporal_integration/")
        logger.info("  2. Analyze top risk companies")
        logger.info("  3. Deploy real-time monitoring")
        logger.info("  4. Create Streamlit dashboard")

def main():
    """Main execution function"""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Create and run integrator
        integrator = FinGraphTemporalIntegrator()
        success = integrator.run_complete_integration()
        
        if success:
            logger.info("\n🎯 SUCCESS! FinGraph enhanced with temporal risk prediction")
            logger.info("🚀 Ready for Week 4: Deployment and Presentation")
        else:
            logger.error("\n❌ Integration failed - check the logs above")
        
        return integrator
        
    except Exception as e:
        logger.error(f"❌ Integration error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    integrator = main()