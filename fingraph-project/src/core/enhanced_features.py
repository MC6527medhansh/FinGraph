"""
Enhanced Feature Engineering with Stock-Specific Differentiation
Adds features that preserve individual stock characteristics
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class EnhancedFeatureEngine:
    """
    Feature engine that creates differentiating features for stocks.
    Includes cross-sectional, sector-relative, and idiosyncratic components.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.sector_map = {
            'AAPL': 'tech', 'MSFT': 'tech', 'GOOGL': 'tech', 
            'NVDA': 'tech', 'META': 'tech',
            'AMZN': 'consumer', 'TSLA': 'consumer',
            'JPM': 'finance', 'V': 'finance',
            'JNJ': 'healthcare'
        }
        
    def add_differentiating_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Add features that differentiate stocks while preserving temporal integrity.
        """
        logger.info("Adding stock-differentiating features")
        
        enhanced = features_df.copy()
        
        # Group by date for cross-sectional features
        for date in enhanced['date'].unique():
            date_mask = enhanced['date'] == date
            date_data = enhanced[date_mask]
            
            # 1. Cross-sectional ranking features (these differentiate by design)
            for col in ['return_20d', 'volatility_20d', 'volume_ratio_20d']:
                if col in date_data.columns:
                    ranks = date_data[col].rank(pct=True)
                    enhanced.loc[date_mask, f'{col}_rank'] = ranks
            
            # 2. Relative to market features
            for col in ['return_20d', 'volatility_20d']:
                if col in date_data.columns:
                    market_mean = date_data[col].mean()
                    market_std = date_data[col].std()
                    if market_std > 0:
                        enhanced.loc[date_mask, f'{col}_zscore'] = (date_data[col] - market_mean) / market_std
                    else:
                        enhanced.loc[date_mask, f'{col}_zscore'] = 0
            
            # 3. Sector-relative features
            for symbol in date_data['symbol'].unique():
                sector = self.sector_map.get(symbol, 'other')
                sector_stocks = [s for s, sec in self.sector_map.items() if sec == sector]
                sector_mask = date_data['symbol'].isin(sector_stocks)
                
                if sector_mask.sum() > 1:
                    sector_data = date_data[sector_mask]
                    symbol_mask = (enhanced['date'] == date) & (enhanced['symbol'] == symbol)
                    
                    # Relative to sector performance
                    enhanced.loc[symbol_mask, 'sector_relative_return'] = (
                        date_data[date_data['symbol'] == symbol]['return_20d'].values[0] - 
                        sector_data['return_20d'].mean()
                    )
        
        # 4. Stock-specific features (intrinsic to each stock)
        for symbol in enhanced['symbol'].unique():
            symbol_mask = enhanced['symbol'] == symbol
            symbol_data = enhanced[symbol_mask].sort_values('date')
            
            # Rolling volatility of volatility (vol of vol)
            if len(symbol_data) > 60:
                vols = symbol_data['volatility_20d'].rolling(20).std()
                enhanced.loc[symbol_mask, 'vol_of_vol'] = vols
            
            # Momentum consistency (how stable is the trend)
            if 'return_5d' in symbol_data.columns and 'return_20d' in symbol_data.columns:
                momentum_consistency = symbol_data['return_5d'].rolling(10).corr(symbol_data['return_20d'])
                enhanced.loc[symbol_mask, 'momentum_consistency'] = momentum_consistency
        
        # 5. Add one-hot encoded symbol identity (forces differentiation)
        symbol_dummies = pd.get_dummies(enhanced['symbol'], prefix='stock')
        for col in symbol_dummies.columns:
            enhanced[col] = symbol_dummies[col].values
        
        # Fill any NaN values
        enhanced = enhanced.fillna(0)
        
        logger.info(f"Enhanced features from {len(features_df.columns)} to {len(enhanced.columns)} columns")
        
        return enhanced