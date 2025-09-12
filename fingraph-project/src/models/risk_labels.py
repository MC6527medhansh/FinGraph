"""
Risk Label Generator for FinGraph
Creates risk labels from financial data for supervised learning
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class RiskLabelGenerator:
    """
    Generates risk labels from financial data
    
    Risk Types:
    - Volatility Risk: Based on price volatility
    - Drawdown Risk: Based on maximum drawdown
    - VaR Risk: Value at Risk calculation
    - Composite Risk: Combined risk score
    """
    
    def __init__(self, lookforward_days: int = 30):
        """
        Initialize risk label generator
        
        Args:
            lookforward_days: Days to look forward for risk calculation
        """
        self.lookforward_days = lookforward_days
        self.risk_thresholds = {
            'low_risk': 0.3,
            'medium_risk': 0.6,
            'high_risk': 1.0
        }
    
    def generate_risk_labels(self, stock_data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate comprehensive risk labels for all companies
        
        Args:
            stock_data: DataFrame with stock price data
            
        Returns:
            DataFrame with risk labels for each company and date
        """
        logger.info("📊 Generating risk labels from stock data...")
        
        all_risk_labels = []
        companies = stock_data['Symbol'].unique()
        
        for company in companies:
            logger.info(f"Processing risk labels for {company}...")
            company_data = stock_data[stock_data['Symbol'] == company].copy()
            company_labels = self._calculate_company_risk(company_data, company)
            all_risk_labels.append(company_labels)
        
        # Combine all company risk labels
        combined_labels = pd.concat(all_risk_labels, ignore_index=True)
        
        logger.info(f"✅ Generated risk labels for {len(companies)} companies")
        logger.info(f"   Total labels: {len(combined_labels)}")
        
        return combined_labels
    
    def _calculate_company_risk(self, company_data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Calculate risk labels for a single company"""
        company_data = company_data.sort_index()
        risk_labels = []
        
        # Need enough data for lookforward calculation
        min_required_days = self.lookforward_days + 10
        if len(company_data) < min_required_days:
            logger.warning(f"Insufficient data for {symbol}: {len(company_data)} days")
            return pd.DataFrame()
        
        # Calculate for each date (except last lookforward_days)
        for i in range(len(company_data) - self.lookforward_days):
            current_date = company_data.index[i]
            current_price = company_data.iloc[i]['Close']
            
            # Get future data for risk calculation
            future_data = company_data.iloc[i:i+self.lookforward_days+1]
            
            # Calculate different risk measures
            volatility_risk = self._calculate_volatility_risk(future_data)
            drawdown_risk = self._calculate_drawdown_risk(future_data)
            var_risk = self._calculate_var_risk(future_data)
            
            # Composite risk score (weighted average)
            composite_risk = (
                0.4 * volatility_risk + 
                0.4 * drawdown_risk + 
                0.2 * var_risk
            )
            
            # Risk classification
            risk_class = self._classify_risk(composite_risk)
            
            risk_labels.append({
                'date': current_date,
                'symbol': symbol,
                'current_price': current_price,
                'volatility_risk': volatility_risk,
                'drawdown_risk': drawdown_risk,
                'var_risk': var_risk,
                'composite_risk': composite_risk,
                'risk_class': risk_class,
                'risk_binary': 1 if composite_risk > self.risk_thresholds['medium_risk'] else 0
            })
        
        return pd.DataFrame(risk_labels)
    
    def _calculate_volatility_risk(self, price_data: pd.DataFrame) -> float:
        """Calculate volatility-based risk"""
        if len(price_data) < 2:
            return 0.0
        
        # Calculate returns
        returns = price_data['Close'].pct_change().dropna()
        
        if len(returns) == 0:
            return 0.0
        
        # Annualized volatility
        volatility = returns.std() * np.sqrt(252)
        
        # Normalize to 0-1 scale (high volatility = high risk)
        # Typical stock volatility ranges from 0.1 to 1.0+
        normalized_vol = min(volatility / 0.8, 1.0)
        
        return normalized_vol
    
    def _calculate_drawdown_risk(self, price_data: pd.DataFrame) -> float:
        """Calculate maximum drawdown risk"""
        if len(price_data) < 2:
            return 0.0
        
        prices = price_data['Close']
        
        # Calculate running maximum
        running_max = prices.expanding().max()
        
        # Calculate drawdown
        drawdown = (prices - running_max) / running_max
        
        # Maximum drawdown (most negative value)
        max_drawdown = abs(drawdown.min())
        
        # Normalize to 0-1 scale
        # Typical max drawdowns range from 0% to 50%+
        normalized_dd = min(max_drawdown / 0.3, 1.0)
        
        return normalized_dd
    
    def _calculate_var_risk(self, price_data: pd.DataFrame) -> float:
        """Calculate Value at Risk (5% VaR)"""
        if len(price_data) < 2:
            return 0.0
        
        # Calculate returns
        returns = price_data['Close'].pct_change().dropna()
        
        if len(returns) == 0:
            return 0.0
        
        # 5% VaR (5th percentile of returns)
        var_5 = np.percentile(returns, 5)
        
        # Convert to risk measure (negative returns = higher risk)
        var_risk = abs(min(var_5, 0))
        
        # Normalize to 0-1 scale
        # Typical 5% VaR ranges from 0% to 15%+
        normalized_var = min(var_risk / 0.1, 1.0)
        
        return normalized_var
    
    def _classify_risk(self, composite_risk: float) -> str:
        """Classify risk into categories"""
        if composite_risk <= self.risk_thresholds['low_risk']:
            return 'low'
        elif composite_risk <= self.risk_thresholds['medium_risk']:
            return 'medium'
        else:
            return 'high'
    
    def get_risk_statistics(self, risk_labels: pd.DataFrame) -> Dict:
        """Get statistics about generated risk labels"""
        if risk_labels.empty:
            return {}
        
        stats = {
            'total_labels': len(risk_labels),
            'companies': risk_labels['symbol'].nunique(),
            'date_range': {
                'start': risk_labels['date'].min(),
                'end': risk_labels['date'].max()
            },
            'risk_distribution': risk_labels['risk_class'].value_counts().to_dict(),
            'average_risks': {
                'volatility': risk_labels['volatility_risk'].mean(),
                'drawdown': risk_labels['drawdown_risk'].mean(),
                'var': risk_labels['var_risk'].mean(),
                'composite': risk_labels['composite_risk'].mean()
            },
            'high_risk_percentage': (risk_labels['risk_binary'].sum() / len(risk_labels)) * 100
        }
        
        return stats
    
    def create_temporal_splits(self, risk_labels: pd.DataFrame, 
                             train_ratio: float = 0.7,
                             val_ratio: float = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Create temporal train/validation/test splits
        
        Args:
            risk_labels: DataFrame with risk labels
            train_ratio: Proportion for training
            val_ratio: Proportion for validation
            
        Returns:
            Tuple of (train, validation, test) DataFrames
        """
        if risk_labels.empty:
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        
        # Sort by date
        risk_labels_sorted = risk_labels.sort_values('date')
        
        # Calculate split indices
        n_total = len(risk_labels_sorted)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        # Split data temporally (no data leakage)
        train_data = risk_labels_sorted.iloc[:n_train]
        val_data = risk_labels_sorted.iloc[n_train:n_train+n_val]
        test_data = risk_labels_sorted.iloc[n_train+n_val:]
        
        logger.info(f"📊 Temporal splits created:")
        logger.info(f"   Train: {len(train_data)} samples ({train_data['date'].min()} to {train_data['date'].max()})")
        logger.info(f"   Val: {len(val_data)} samples ({val_data['date'].min()} to {val_data['date'].max()})")
        logger.info(f"   Test: {len(test_data)} samples ({test_data['date'].min()} to {test_data['date'].max()})")
        
        return train_data, val_data, test_data

# Test function
def test_risk_label_generator():
    """Test risk label generation"""
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
    
    from src.features.graph_data_loader import GraphDataLoader
    
    # Load data
    loader = GraphDataLoader()
    data = loader.load_latest_data()
    
    if data['stock_data'] is None or data['stock_data'].empty:
        print("❌ No stock data available for risk label generation")
        return
    
    # Generate risk labels
    risk_gen = RiskLabelGenerator(lookforward_days=20)  # Shorter for testing
    risk_labels = risk_gen.generate_risk_labels(data['stock_data'])
    
    if risk_labels.empty:
        print("❌ No risk labels generated")
        return
    
    # Get statistics
    stats = risk_gen.get_risk_statistics(risk_labels)
    
    print("✅ Risk Label Generation Test Results:")
    print(f"   Total labels: {stats['total_labels']}")
    print(f"   Companies: {stats['companies']}")
    print(f"   Risk distribution: {stats['risk_distribution']}")
    print(f"   High risk %: {stats['high_risk_percentage']:.1f}%")
    print(f"   Average composite risk: {stats['average_risks']['composite']:.3f}")
    
    # Test temporal splits
    train, val, test = risk_gen.create_temporal_splits(risk_labels)
    print(f"\n📊 Temporal splits:")
    print(f"   Train: {len(train)} samples")
    print(f"   Validation: {len(val)} samples") 
    print(f"   Test: {len(test)} samples")
    
    # Save sample risk labels
    os.makedirs('data/processed', exist_ok=True)
    risk_labels.to_csv('data/processed/risk_labels_sample.csv', index=False)
    print(f"\n💾 Saved sample risk labels to data/processed/risk_labels_sample.csv")
    
    return risk_labels, stats

if __name__ == "__main__":
    test_risk_label_generator()