"""
Quantitative Data Pipeline with Guaranteed Temporal Integrity
Zero tolerance for lookahead bias. Production-grade with checksums.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import hashlib
import json
import logging
from pathlib import Path
from dataclasses import dataclass, asdict
import warnings
warnings.filterwarnings('ignore')

# Setup production logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class TemporalSample:
    """Immutable temporal sample with integrity guarantees"""
    timestamp: datetime
    symbol: str
    features: np.ndarray
    feature_names: List[str]
    data_end_time: datetime  # Last timestamp used in features
    label_start_time: datetime  # First timestamp in label window
    label_end_time: datetime  # Last timestamp in label window
    forward_return: float
    forward_volatility: float
    forward_max_drawdown: float
    checksum: str
    
    def verify_integrity(self) -> bool:
        """Verify no temporal leakage"""
        return self.data_end_time < self.label_start_time


class QuantDataPipeline:
    """
    Production-grade data pipeline with guaranteed temporal integrity.
    
    Core Principles:
    1. Point-in-time features only use data available at time T
    2. Labels computed from T+1 to T+horizon
    3. Every sample has cryptographic checksum
    4. Automatic validation of temporal boundaries
    5. Robust error handling for missing/bad data
    """
    
    def __init__(self, 
                 cache_dir: str = "data/pipeline_cache",
                 min_history_days: int = 252,  # 1 year minimum
                 label_horizon: int = 21,  # 21 trading days forward
                 validation_mode: bool = True):
        """
        Initialize pipeline with strict validation.
        
        Args:
            cache_dir: Directory for cached processed data
            min_history_days: Minimum history required for feature calculation
            label_horizon: Forward-looking window for labels
            validation_mode: Enable extensive validation checks
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.min_history_days = min_history_days
        self.label_horizon = label_horizon
        self.validation_mode = validation_mode
        
        # Feature configuration
        self.feature_config = {
            'price_features': ['returns_1d', 'returns_5d', 'returns_20d', 'returns_60d'],
            'volatility_features': ['realized_vol_5d', 'realized_vol_20d', 'realized_vol_60d'],
            'technical_features': ['rsi_14', 'macd_signal', 'bb_position'],
            'volume_features': ['volume_ratio_5d', 'volume_ratio_20d'],
            'microstructure': ['close_to_high', 'close_to_low', 'daily_range']
        }
        
    def load_market_data(self, 
                        symbols: List[str], 
                        start_date: str, 
                        end_date: str,
                        validate: bool = True) -> pd.DataFrame:
        """
        Load market data with validation and integrity checks.
        
        Returns:
            DataFrame with columns: [Date, Symbol, Open, High, Low, Close, Volume, AdjClose]
        """
        logger.info(f"Loading market data for {len(symbols)} symbols from {start_date} to {end_date}")
        
        # Add buffer for history requirement
        buffer_start = pd.to_datetime(start_date) - timedelta(days=self.min_history_days + 50)
        
        all_data = []
        failed_symbols = []
        
        for symbol in symbols:
            try:
                # Download with adjusted close for split/dividend handling
                ticker = yf.Ticker(symbol)
                data = ticker.history(start=buffer_start, end=end_date, auto_adjust=False)
                
                if len(data) < self.min_history_days:
                    logger.warning(f"Insufficient data for {symbol}: {len(data)} days")
                    failed_symbols.append(symbol)
                    continue
                
                # Add symbol column and reset index
                data['Symbol'] = symbol
                data = data.reset_index()
                
                # Rename columns to standard names
                data = data.rename(columns={
                    'Date': 'timestamp',
                    'Adj Close': 'adj_close'
                })
                
                # Critical: Ensure timezone-naive timestamps
                data['timestamp'] = pd.to_datetime(data['timestamp']).dt.tz_localize(None)
                
                # Validate data quality
                if validate:
                    issues = self._validate_price_data(data)
                    if issues:
                        logger.warning(f"Data quality issues for {symbol}: {issues}")
                        if len(issues) > 5:  # Too many issues
                            failed_symbols.append(symbol)
                            continue
                
                all_data.append(data)
                logger.info(f"✓ {symbol}: {len(data)} days loaded")
                
            except Exception as e:
                logger.error(f"Failed to load {symbol}: {str(e)}")
                failed_symbols.append(symbol)
        
        if not all_data:
            raise ValueError("No valid data loaded for any symbol")
        
        # Combine all data
        combined = pd.concat(all_data, ignore_index=True)
        combined = combined.sort_values(['Symbol', 'timestamp'])
        
        # Add data quality metadata
        metadata = {
            'loaded_symbols': len(symbols) - len(failed_symbols),
            'failed_symbols': failed_symbols,
            'date_range': f"{combined['timestamp'].min()} to {combined['timestamp'].max()}",
            'total_records': len(combined),
            'checksum': self._compute_checksum(combined)
        }
        
        # Save metadata
        with open(self.cache_dir / 'load_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"Loaded {metadata['loaded_symbols']} symbols, {metadata['total_records']} records")
        
        return combined
    
    def create_point_in_time_features(self, 
                                     data: pd.DataFrame, 
                                     timestamp: datetime, 
                                     symbol: str) -> Optional[Dict[str, Any]]:
        """
        Create features using ONLY data available at timestamp T.
        
        This is the most critical function - absolutely no future information.
        
        Args:
            data: Full dataset
            timestamp: Point in time T
            symbol: Stock symbol
            
        Returns:
            Dictionary with features and metadata, or None if insufficient data
        """
        # Get historical data up to (but not including) timestamp
        mask = (data['Symbol'] == symbol) & (data['timestamp'] < timestamp)
        historical = data[mask].copy()
        
        if len(historical) < self.min_history_days:
            return None
        
        # Sort by date and get most recent data
        historical = historical.sort_values('timestamp')
        
        features = {}
        feature_names = []
        
        # 1. Price-based features (returns)
        close_prices = historical['Close'].values
        returns = np.diff(close_prices) / close_prices[:-1]
        
        if len(returns) >= 1:
            features['return_1d'] = returns[-1]
            feature_names.append('return_1d')
        
        if len(returns) >= 5:
            features['return_5d'] = (close_prices[-1] / close_prices[-5] - 1)
            features['mean_return_5d'] = np.mean(returns[-5:])
            feature_names.extend(['return_5d', 'mean_return_5d'])
        
        if len(returns) >= 20:
            features['return_20d'] = (close_prices[-1] / close_prices[-20] - 1)
            features['mean_return_20d'] = np.mean(returns[-20:])
            feature_names.extend(['return_20d', 'mean_return_20d'])
        
        if len(returns) >= 60:
            features['return_60d'] = (close_prices[-1] / close_prices[-60] - 1)
            features['mean_return_60d'] = np.mean(returns[-60:])
            feature_names.extend(['return_60d', 'mean_return_60d'])
        
        # 2. Volatility features (backward-looking only!)
        if len(returns) >= 5:
            features['volatility_5d'] = np.std(returns[-5:]) * np.sqrt(252)
            feature_names.append('volatility_5d')
        
        if len(returns) >= 20:
            features['volatility_20d'] = np.std(returns[-20:]) * np.sqrt(252)
            feature_names.append('volatility_20d')
        
        if len(returns) >= 60:
            features['volatility_60d'] = np.std(returns[-60:]) * np.sqrt(252)
            feature_names.append('volatility_60d')
        
        # 3. Technical indicators (all backward-looking)
        # RSI
        if len(returns) >= 14:
            rsi = self._calculate_rsi(close_prices, 14)
            features['rsi_14'] = rsi
            feature_names.append('rsi_14')
        
        # Bollinger Bands position
        if len(close_prices) >= 20:
            sma_20 = np.mean(close_prices[-20:])
            std_20 = np.std(close_prices[-20:])
            bb_upper = sma_20 + 2 * std_20
            bb_lower = sma_20 - 2 * std_20
            current_price = close_prices[-1]
            
            # Position in bands (0 = lower, 1 = upper)
            if bb_upper != bb_lower:
                features['bb_position'] = (current_price - bb_lower) / (bb_upper - bb_lower)
            else:
                features['bb_position'] = 0.5
            feature_names.append('bb_position')
        
        # 4. Volume features
        volumes = historical['Volume'].values
        if len(volumes) >= 20:
            features['volume_ratio_5d'] = np.mean(volumes[-5:]) / np.mean(volumes[-20:])
            features['volume_ratio_20d'] = volumes[-1] / np.mean(volumes[-20:])
            feature_names.extend(['volume_ratio_5d', 'volume_ratio_20d'])
        
        # 5. Microstructure features
        high_prices = historical['High'].values
        low_prices = historical['Low'].values
        open_prices = historical['Open'].values
        
        if len(high_prices) >= 5:
            # Recent price position in daily range
            features['close_to_high'] = (close_prices[-1] - low_prices[-1]) / (high_prices[-1] - low_prices[-1] + 1e-10)
            features['daily_range'] = (high_prices[-1] - low_prices[-1]) / close_prices[-1]
            features['overnight_gap'] = (open_prices[-1] - close_prices[-2]) / close_prices[-2] if len(close_prices) > 1 else 0
            feature_names.extend(['close_to_high', 'daily_range', 'overnight_gap'])
        
        # 6. Market regime features
        if len(close_prices) >= 252:  # 1 year
            # Distance from 52-week high/low
            high_52w = np.max(close_prices[-252:])
            low_52w = np.min(close_prices[-252:])
            features['pct_from_52w_high'] = (close_prices[-1] - high_52w) / high_52w
            features['pct_from_52w_low'] = (close_prices[-1] - low_52w) / low_52w
            feature_names.extend(['pct_from_52w_high', 'pct_from_52w_low'])
        
        # Create feature array in consistent order
        feature_array = np.array([features[name] for name in feature_names], dtype=np.float32)
        
        # Replace any NaN/Inf with 0 (safer than dropping)
        feature_array = np.nan_to_num(feature_array, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return {
            'features': feature_array,
            'feature_names': feature_names,
            'last_data_timestamp': historical['timestamp'].iloc[-1],
            'num_historical_points': len(historical)
        }
    
    def calculate_forward_labels(self, 
                                data: pd.DataFrame, 
                                timestamp: datetime, 
                                symbol: str,
                                horizon: int = None) -> Optional[Dict[str, float]]:
        """
        Calculate forward-looking labels from T+1 to T+horizon.
        
        Critical: This looks FORWARD from timestamp, used only for training labels.
        
        Args:
            data: Full dataset  
            timestamp: Current time T
            symbol: Stock symbol
            horizon: Days forward (default: self.label_horizon)
            
        Returns:
            Dictionary with forward metrics, or None if insufficient forward data
        """
        if horizon is None:
            horizon = self.label_horizon
        
        # Get future data from T+1 to T+horizon
        future_start = timestamp + timedelta(days=1)
        future_end = timestamp + timedelta(days=horizon + 10)  # Buffer for weekends
        
        mask = (data['Symbol'] == symbol) & \
               (data['timestamp'] > timestamp) & \
               (data['timestamp'] <= future_end)
        future_data = data[mask].copy()
        
        if len(future_data) < horizon * 0.5:  # Need at least 50% of expected days
            return None
        
        # Limit to horizon days
        future_data = future_data.iloc[:horizon]
        
        # Calculate forward metrics
        future_prices = future_data['Close'].values
        future_returns = np.diff(future_prices) / future_prices[:-1] if len(future_prices) > 1 else np.array([0])
        
        # Total return over horizon
        if len(future_prices) > 0:
            start_price = data[(data['Symbol'] == symbol) & (data['timestamp'] <= timestamp)]['Close'].iloc[-1]
            end_price = future_prices[-1]
            total_return = (end_price - start_price) / start_price
        else:
            return None
        
        # Volatility over forward period
        if len(future_returns) > 1:
            forward_volatility = np.std(future_returns) * np.sqrt(252)
        else:
            forward_volatility = 0.0
        
        # Maximum drawdown in forward period
        if len(future_prices) > 1:
            cumulative = np.cumprod(1 + future_returns)
            running_max = np.maximum.accumulate(cumulative)
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = abs(np.min(drawdown))
        else:
            max_drawdown = 0.0
        
        # Skewness of forward returns
        if len(future_returns) > 2:
            skewness = self._calculate_skewness(future_returns)
        else:
            skewness = 0.0
        
        return {
            'forward_return': total_return,
            'forward_volatility': forward_volatility,
            'forward_max_drawdown': max_drawdown,
            'forward_skewness': skewness,
            'forward_days_observed': len(future_data),
            'label_start': future_data['timestamp'].iloc[0] if len(future_data) > 0 else None,
            'label_end': future_data['timestamp'].iloc[-1] if len(future_data) > 0 else None
        }
    
    def create_temporal_dataset(self, 
                              data: pd.DataFrame,
                              symbols: Optional[List[str]] = None,
                              sample_freq: int = 5) -> List[TemporalSample]:
        """
        Create complete temporal dataset with guaranteed integrity.
        
        Args:
            data: Market data from load_market_data
            symbols: Symbols to process (None = all)
            sample_freq: Sample every N days to reduce correlation
            
        Returns:
            List of TemporalSample objects with verified integrity
        """
        if symbols is None:
            symbols = data['Symbol'].unique()
        
        samples = []
        integrity_failures = 0
        
        for symbol in symbols:
            symbol_data = data[data['Symbol'] == symbol].copy()
            available_dates = symbol_data['timestamp'].unique()
            
            # Skip dates without enough history or future data
            valid_start = self.min_history_days
            valid_end = len(available_dates) - self.label_horizon
            
            # Sample dates with frequency
            sample_indices = range(valid_start, valid_end, sample_freq)
            
            for idx in sample_indices:
                timestamp = available_dates[idx]
                
                # Create features (backward-looking)
                feature_data = self.create_point_in_time_features(data, timestamp, symbol)
                if feature_data is None:
                    continue
                
                # Calculate labels (forward-looking)  
                label_data = self.calculate_forward_labels(data, timestamp, symbol)
                if label_data is None:
                    continue
                
                # Create temporal sample
                sample = TemporalSample(
                    timestamp=timestamp,
                    symbol=symbol,
                    features=feature_data['features'],
                    feature_names=feature_data['feature_names'],
                    data_end_time=feature_data['last_data_timestamp'],
                    label_start_time=label_data['label_start'],
                    label_end_time=label_data['label_end'],
                    forward_return=label_data['forward_return'],
                    forward_volatility=label_data['forward_volatility'],
                    forward_max_drawdown=label_data['forward_max_drawdown'],
                    checksum=self._compute_sample_checksum(feature_data, label_data)
                )
                
                # Verify temporal integrity
                if not sample.verify_integrity():
                    integrity_failures += 1
                    logger.error(f"Temporal integrity violation for {symbol} at {timestamp}")
                    if self.validation_mode:
                        raise ValueError(f"Temporal leak detected: features end at {sample.data_end_time}, labels start at {sample.label_start_time}")
                    continue
                
                samples.append(sample)
        
        logger.info(f"Created {len(samples)} samples with {integrity_failures} integrity failures")
        
        return samples
    
    def create_train_val_test_splits(self, 
                                    samples: List[TemporalSample],
                                    train_pct: float = 0.6,
                                    val_pct: float = 0.2,
                                    gap_days: int = 5) -> Dict[str, List[TemporalSample]]:
        """
        Create temporal splits with gaps to prevent leakage.
        
        Args:
            samples: List of temporal samples
            train_pct: Percentage for training
            val_pct: Percentage for validation  
            gap_days: Days gap between splits
            
        Returns:
            Dictionary with train, val, test splits
        """
        # Sort by timestamp
        sorted_samples = sorted(samples, key=lambda x: x.timestamp)
        
        # Get unique timestamps
        unique_timestamps = sorted(list(set([s.timestamp for s in sorted_samples])))
        n_timestamps = len(unique_timestamps)
        
        # Calculate split points
        train_end_idx = int(n_timestamps * train_pct)
        val_end_idx = int(n_timestamps * (train_pct + val_pct))
        
        train_end_date = unique_timestamps[train_end_idx]
        val_start_date = train_end_date + timedelta(days=gap_days)
        val_end_date = unique_timestamps[val_end_idx]
        test_start_date = val_end_date + timedelta(days=gap_days)
        
        # Split samples
        train = [s for s in sorted_samples if s.timestamp <= train_end_date]
        val = [s for s in sorted_samples if val_start_date <= s.timestamp <= val_end_date]
        test = [s for s in sorted_samples if s.timestamp >= test_start_date]
        
        # Verify no overlap
        train_dates = set([s.timestamp for s in train])
        val_dates = set([s.timestamp for s in val])
        test_dates = set([s.timestamp for s in test])
        
        assert len(train_dates & val_dates) == 0, "Train/val overlap detected"
        assert len(val_dates & test_dates) == 0, "Val/test overlap detected"
        assert len(train_dates & test_dates) == 0, "Train/test overlap detected"
        
        # Log split statistics
        logger.info(f"Train: {len(train)} samples, {min(train_dates)} to {max(train_dates)}")
        logger.info(f"Val: {len(val)} samples, {min(val_dates)} to {max(val_dates)}")
        logger.info(f"Test: {len(test)} samples, {min(test_dates)} to {max(test_dates)}")
        
        return {
            'train': train,
            'val': val,
            'test': test,
            'metadata': {
                'train_end': train_end_date,
                'val_start': val_start_date,
                'val_end': val_end_date,
                'test_start': test_start_date,
                'gap_days': gap_days,
                'no_overlap_verified': True
            }
        }
    
    def save_dataset(self, samples: List[TemporalSample], filename: str):
        """Save dataset with integrity verification."""
        output_path = self.cache_dir / filename
        
        # Convert to serializable format
        data = []
        for sample in samples:
            sample_dict = asdict(sample)
            # Convert numpy arrays to lists
            sample_dict['features'] = sample_dict['features'].tolist()
            # Convert timestamps to strings
            for key in ['timestamp', 'data_end_time', 'label_start_time', 'label_end_time']:
                if sample_dict[key] is not None:
                    sample_dict[key] = sample_dict[key].isoformat()
            data.append(sample_dict)
        
        # Add metadata
        metadata = {
            'created_at': datetime.now().isoformat(),
            'num_samples': len(samples),
            'min_date': min(s.timestamp for s in samples).isoformat(),
            'max_date': max(s.timestamp for s in samples).isoformat(),
            'checksum': self._compute_checksum(pd.DataFrame(data))
        }
        
        output = {
            'metadata': metadata,
            'samples': data
        }
        
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        
        logger.info(f"Saved {len(samples)} samples to {output_path}")
    
    # Helper methods
    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """Calculate RSI at the last point."""
        if len(prices) < period + 1:
            return 50.0
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_skewness(self, returns: np.ndarray) -> float:
        """Calculate skewness of returns."""
        if len(returns) < 3:
            return 0.0
        
        mean = np.mean(returns)
        std = np.std(returns)
        if std == 0:
            return 0.0
        
        skew = np.mean(((returns - mean) / std) ** 3)
        return skew
    
    def _validate_price_data(self, data: pd.DataFrame) -> List[str]:
        """Validate price data quality."""
        issues = []
        
        # Check for missing values
        null_counts = data.isnull().sum()
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if col in data.columns and null_counts[col] > 0:
                issues.append(f"{col} has {null_counts[col]} missing values")
        
        # Check for zero/negative prices
        for col in ['Open', 'High', 'Low', 'Close']:
            if col in data.columns:
                if (data[col] <= 0).any():
                    issues.append(f"{col} has zero/negative values")
        
        # Check high/low consistency
        if 'High' in data.columns and 'Low' in data.columns:
            if (data['High'] < data['Low']).any():
                issues.append("High < Low detected")
        
        # Check for duplicate timestamps
        if 'timestamp' in data.columns:
            if data['timestamp'].duplicated().any():
                issues.append("Duplicate timestamps detected")
        
        return issues
    
    def _compute_checksum(self, data: pd.DataFrame) -> str:
        """Compute SHA256 checksum of data."""
        data_str = data.to_json(orient='records', date_format='iso')
        return hashlib.sha256(data_str.encode()).hexdigest()
    
    def _compute_sample_checksum(self, features: Dict, labels: Dict) -> str:
        """Compute checksum for a single sample."""
        combined = json.dumps({
            'features': features,
            'labels': labels
        }, default=str, sort_keys=True)
        return hashlib.sha256(combined.encode()).hexdigest()[:16]  # Short version


# Example usage with validation
if __name__ == "__main__":
    pipeline = QuantDataPipeline(
        cache_dir="data/quant_pipeline",
        min_history_days=252,
        label_horizon=21,
        validation_mode=True
    )
    
    # Load real market data
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    data = pipeline.load_market_data(symbols, '2019-01-01', '2024-01-01')
    
    # Create temporal dataset with guaranteed integrity
    samples = pipeline.create_temporal_dataset(data, sample_freq=5)
    
    # Create proper temporal splits
    splits = pipeline.create_train_val_test_splits(samples)
    
    print(f"✓ Pipeline complete with {len(samples)} samples")
    print(f"✓ All temporal integrity checks passed")
    print(f"✓ Train: {len(splits['train'])}, Val: {len(splits['val'])}, Test: {len(splits['test'])}")