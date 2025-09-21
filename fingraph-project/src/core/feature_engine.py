"""
Unified Feature Engineering - Single source for ALL feature calculations
This replaces all existing feature extractors
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class UnifiedFeatureEngine:
    """
    Single feature engineering class for entire pipeline.
    Guarantees temporal integrity - NO LOOKAHEAD BIAS.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize with configuration
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.windows = config['features']['windows']
        self.label_horizon = config['features']['label_horizon']
        
        # Define feature names for consistency
        self.feature_names = self._define_feature_names()
        
    def _define_feature_names(self) -> List[str]:
        """Define all feature names"""
        names = []
        
        # Price features
        for window in self.windows:
            names.extend([
                f'return_{window}d',
                f'volatility_{window}d',
                f'volume_ratio_{window}d'
            ])
        
        # Technical indicators
        names.extend([
            'rsi_14',
            'macd_signal',
            'bb_position',
            'close_to_high',
            'close_to_low'
        ])
        
        # Market regime
        names.extend([
            'trend_strength',
            'volatility_regime',
            'volume_regime'
        ])
        
        return names
    
    def create_features(self, 
                       data: pd.DataFrame,
                       point_in_time: bool = True) -> pd.DataFrame:
        """
        Create all features from price data.
        
        Args:
            data: Price data from DataManager
            point_in_time: If True, ensure no lookahead bias
            
        Returns:
            DataFrame with features
        """
        logger.info("Creating features with temporal integrity")
        
        features = []
        
        # Process each symbol separately
        for symbol in data['symbol'].unique():
            symbol_data = data[data['symbol'] == symbol].copy()
            symbol_features = self._create_symbol_features(symbol_data, symbol)
            if len(symbol_features) > 0:
                features.append(symbol_features)
        
        # Combine all features
        if not features:
            return pd.DataFrame()
        
        all_features = pd.concat(features, ignore_index=True)
        logger.info(f"Created {len(all_features)} feature vectors")

        # ============================
        # ✱ NEW: Cross-sectional labels (correct place to do it)
        # ============================
        ### ✱ NEW — compute cross-sectional z-scores/percentiles by date, AFTER concatenation
        if 'forward_return' in all_features.columns:
            all_features['forward_return_cs_z'] = all_features.groupby('date')['forward_return']\
                .transform(lambda s: (s - s.mean()) / (s.std() + 1e-8))

        if 'forward_volatility' in all_features.columns:
            all_features['forward_volatility_cs_z'] = all_features.groupby('date')['forward_volatility']\
                .transform(lambda s: (s - s.mean()) / (s.std() + 1e-8))

            # percentile risk (0..1): higher vol → higher risk
            all_features['risk_score_cs_pct'] = all_features.groupby('date')['forward_volatility']\
                .transform(lambda s: s.rank(pct=True))

        # If you want to *train directly* on cross-sectional targets, uncomment:
        # all_features['forward_return']     = all_features['forward_return_cs_z']
        # all_features['forward_volatility'] = all_features['forward_volatility_cs_z']
        # all_features['risk_score']         = all_features['risk_score_cs_pct']
        # ============================

        return all_features
    
    def _create_symbol_features(self, 
                               data: pd.DataFrame,
                               symbol: str) -> pd.DataFrame:
        """Create features for a single symbol"""
        
        # Sort by date
        data = data.sort_index()

        # ✱ NEW — add return_1d for anyone who needs it (not used for labels below)
        ### ✱ NEW
        data['return_1d'] = data['close'].pct_change()

        # Need minimum history
        if len(data) < self.config['data']['min_history_days']:
            logger.warning(f"Insufficient data for {symbol}: {len(data)} days")
            return pd.DataFrame()
        
        feature_list = []
        
        # Calculate features for each valid date
        valid_start = self.config['data']['min_history_days']
        valid_end = len(data) - self.label_horizon
        
        for i in range(valid_start, valid_end, 5):  # Sample every 5 days
            date = data.index[i]
            
            # Get historical data up to this point (NO FUTURE DATA)
            historical = data.iloc[:i]
            
            # Calculate point-in-time features
            features = self._calculate_point_in_time_features(historical, date)
            
            if features is not None:
                # Calculate forward labels (for training) using ONLY the future window for THIS symbol
                future_data = data.iloc[i:i+self.label_horizon]

                # ✱ CHANGED — call the corrected forward-label function (no return_1d dependency)
                ### ✱ CHANGED
                labels = self._calculate_forward_labels(future_data)
                
                if labels is not None:
                    # Combine features and labels
                    sample = {
                        'date': date,
                        'symbol': symbol,
                        **features,
                        **labels
                    }
                    feature_list.append(sample)
        
        return pd.DataFrame(feature_list)
    
    def _calculate_point_in_time_features(self, 
                                         data: pd.DataFrame,
                                         date: datetime) -> Optional[Dict]:
        """
        Calculate features using only historical data up to date.
        This ensures no lookahead bias.
        """
        if len(data) < 20:  # Minimum required
            return None
        
        features = {}
        
        try:
            # Price returns
            close_prices = data['close'].values
            returns = pd.Series(close_prices).pct_change().dropna()
            
            for window in self.windows:
                if len(returns) >= window:
                    features[f'return_{window}d'] = returns.iloc[-window:].mean()
                    features[f'volatility_{window}d'] = returns.iloc[-window:].std() * np.sqrt(252)
                else:
                    features[f'return_{window}d'] = 0.0
                    features[f'volatility_{window}d'] = 0.2
            
            # Volume features
            volumes = data['volume'].values
            for window in self.windows:
                if len(volumes) >= window:
                    recent_vol = np.mean(volumes[-5:])
                    avg_vol = np.mean(volumes[-window:])
                    features[f'volume_ratio_{window}d'] = recent_vol / (avg_vol + 1e-8)
                else:
                    features[f'volume_ratio_{window}d'] = 1.0
            
            # Technical indicators
            features['rsi_14'] = self._calculate_rsi(close_prices) / 100
            
            # Bollinger Bands position
            if len(close_prices) >= 20:
                sma_20 = np.mean(close_prices[-20:])
                std_20 = np.std(close_prices[-20:])
                current = close_prices[-1]
                bb_upper = sma_20 + 2 * std_20
                bb_lower = sma_20 - 2 * std_20
                features['bb_position'] = (current - bb_lower) / (bb_upper - bb_lower + 1e-8)
            else:
                features['bb_position'] = 0.5
            
            # MACD (simplified)
            if len(close_prices) >= 26:
                ema_12 = pd.Series(close_prices).ewm(span=12).mean().iloc[-1]
                ema_26 = pd.Series(close_prices).ewm(span=26).mean().iloc[-1]
                features['macd_signal'] = (ema_12 - ema_26) / (close_prices[-1] + 1e-8)
            else:
                features['macd_signal'] = 0.0
            
            # Price position in daily range
            high = data['high'].iloc[-1]
            low = data['low'].iloc[-1]
            close = data['close'].iloc[-1]
            
            features['close_to_high'] = (close - low) / (high - low + 1e-8)
            features['close_to_low'] = (high - close) / (high - low + 1e-8)
            
            # Market regime indicators
            if len(returns) >= 60:
                features['trend_strength'] = returns.iloc[-20:].mean() / (returns.iloc[-60:].std() + 1e-8)
                features['volatility_regime'] = returns.iloc[-20:].std() / (returns.iloc[-60:].std() + 1e-8)
            else:
                features['trend_strength'] = 0.0
                features['volatility_regime'] = 1.0
            
            if len(volumes) >= 60:
                features['volume_regime'] = np.mean(volumes[-20:]) / (np.mean(volumes[-60:]) + 1e-8)
            else:
                features['volume_regime'] = 1.0
            
            # Validate all features are numeric
            for key, value in features.items():
                if np.isnan(value) or np.isinf(value):
                    features[key] = 0.0
            
            return features
            
        except Exception as e:
            logger.error(f"Error calculating features: {e}")
            return None
    
    # ============================
    # ✱ CHANGED: Correct forward labels (no return_1d dependency; use future window)
    # ============================
    def _calculate_forward_labels(self, future_data: pd.DataFrame) -> Optional[Dict]:
        """Compute forward labels for ONE symbol using only its future window."""
        if len(future_data) < 5:
            return None
        
        labels = {}
        
        try:
            # Future returns (total over horizon)
            fwd_return = (future_data['close'].iloc[-1] / future_data['close'].iloc[0]) - 1
            labels['forward_return'] = float(fwd_return)
            
            # Future volatility (annualized) within the horizon
            fwd_rets = future_data['close'].pct_change().dropna()
            fwd_vol = float(fwd_rets.std() * np.sqrt(252)) if len(fwd_rets) > 1 else 0.0
            labels['forward_volatility'] = fwd_vol
            
            # Future max drawdown
            if len(fwd_rets) > 0:
                cumulative = (1 + fwd_rets).cumprod()
                running_max = cumulative.cummax()
                drawdown = (cumulative - running_max) / (running_max + 1e-12)
                fwd_mdd = float(abs(drawdown.min()))
            else:
                fwd_mdd = 0.0
            labels['forward_max_drawdown'] = fwd_mdd
            
            # Risk score 0..1: high vol/drawdown → high risk
            vol_score = min(fwd_vol / 0.3, 1.0)
            dd_score  = min(fwd_mdd / 0.1, 1.0)
            risk = max(0.0, min(1.0, 0.7 * vol_score + 0.3 * dd_score))
            labels['risk_score'] = float(risk)
            
            return labels
            
        except Exception as e:
            logger.error(f"Error calculating labels: {e}")
            return None
    
    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """Calculate RSI indicator"""
        if len(prices) < period + 1:
            return 50.0
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / (avg_loss + 1e-12)
        rsi = 100 - (100 / (1 + rs))
        return rsi
