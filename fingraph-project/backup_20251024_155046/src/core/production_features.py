"""
Production Feature Engineering - Creates features WITHOUT needing future data
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class ProductionFeatureEngine:
    """
    Feature engine for live trading - no forward-looking data required
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.lookback_days = 60  # How much history we need
        
    def create_current_features(self, prices: pd.DataFrame, target_date: Optional[pd.Timestamp] = None) -> pd.DataFrame:
        """
        Create features for prediction (no labels needed)
        Args:
            prices: Historical prices up to current date
            target_date: Date to create features for (default: most recent)
        """
        
        if prices.empty:
            return pd.DataFrame()
        
        # Use most recent date if not specified
        if target_date is None:
            target_date = prices.index.max()
            
        logger.info(f"Creating features for prediction date: {target_date}")
        
        # Get symbols
        symbols = prices.columns.tolist() if isinstance(prices, pd.DataFrame) else prices['symbol'].unique()
        
        features_list = []
        
        for symbol in symbols:
            # Get price series for this symbol
            if isinstance(prices, pd.DataFrame):
                symbol_prices = prices[symbol].dropna()
            else:
                symbol_prices = prices[prices['symbol'] == symbol]['close']
            
            # Need at least lookback_days of history
            if len(symbol_prices) < self.lookback_days:
                continue
                
            # Get history up to target date
            history = symbol_prices[symbol_prices.index <= target_date].tail(self.lookback_days)
            
            if len(history) < 20:  # Minimum for indicators
                continue
            
            # Calculate SAME features as training, but only backward-looking
            features = {
                'date': target_date,
                'symbol': symbol,
                
                # Price-based features
                'return_5d': (history.iloc[-1] / history.iloc[-5] - 1) if len(history) >= 5 else 0,
                'return_20d': (history.iloc[-1] / history.iloc[-20] - 1) if len(history) >= 20 else 0,
                'return_60d': (history.iloc[-1] / history.iloc[-60] - 1) if len(history) >= 60 else 0,
                
                # Volatility features
                'volatility_5d': history.tail(5).pct_change().std() if len(history) >= 5 else 0,
                'volatility_20d': history.tail(20).pct_change().std() if len(history) >= 20 else 0,
                
                # Technical indicators
                'rsi': self._calculate_rsi(history.tail(14)),
                'macd_signal': self._calculate_macd_signal(history),
                
                # Volume patterns (if available)
                'volume_ratio_5d': 1.0,  # Placeholder
                'volume_ratio_20d': 1.0,  # Placeholder
                
                # Price levels
                'distance_from_high_20d': (history.iloc[-1] / history.tail(20).max() - 1),
                'distance_from_low_20d': (history.iloc[-1] / history.tail(20).min() - 1),
                
                # Momentum
                'momentum_5d': history.iloc[-1] - history.iloc[-5] if len(history) >= 5 else 0,
                'momentum_20d': history.iloc[-1] - history.iloc[-20] if len(history) >= 20 else 0,
                
                # Moving averages
                'ma_5': history.tail(5).mean(),
                'ma_20': history.tail(20).mean(),
                'ma_cross': history.tail(5).mean() / history.tail(20).mean() - 1,
                
                # Trend strength
                'trend_strength': self._calculate_trend_strength(history.tail(20)),
                
                # Recent performance
                'return_1d': (history.iloc[-1] / history.iloc[-2] - 1) if len(history) >= 2 else 0,
                'high_low_ratio': (history.tail(5).max() / history.tail(5).min() - 1) if len(history) >= 5 else 0,
                
                # Normalized price
                'normalized_price': (history.iloc[-1] - history.mean()) / history.std() if history.std() > 0 else 0,
            }
            
            features_list.append(features)
        
        features_df = pd.DataFrame(features_list)
        
        # Fill NaN values
        features_df = features_df.fillna(0)
        
        # CRITICAL: No forward-looking labels for production!
        # We're predicting these, not using them as features
        
        logger.info(f"Created features for {len(features_df)} symbols on {target_date}")
        
        return features_df
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI"""
        if len(prices) < period:
            return 50.0
            
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).mean()
        loss = (-delta.where(delta < 0, 0)).mean()
        
        if loss == 0:
            return 100.0
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd_signal(self, prices: pd.Series) -> float:
        """Calculate MACD signal"""
        if len(prices) < 26:
            return 0.0
            
        exp1 = prices.ewm(span=12, adjust=False).mean()
        exp2 = prices.ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        
        return (macd.iloc[-1] - signal.iloc[-1]) / prices.iloc[-1] if prices.iloc[-1] > 0 else 0
    
    def _calculate_trend_strength(self, prices: pd.Series) -> float:
        """Calculate trend strength using linear regression slope"""
        if len(prices) < 2:
            return 0.0
            
        x = np.arange(len(prices))
        y = prices.values
        
        if np.std(y) == 0:
            return 0.0
            
        slope = np.polyfit(x, y, 1)[0]
        return slope / prices.mean() if prices.mean() > 0 else 0
    
    