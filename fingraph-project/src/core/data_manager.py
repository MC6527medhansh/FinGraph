"""
Unified Data Manager - Single source of truth for ALL data operations
This replaces all existing data loaders
"""

import pandas as pd
import numpy as np
import yfinance as yf
from pathlib import Path
import pickle
import hashlib
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class DataCache:
    """Smart caching system with versioning"""
    
    def __init__(self, cache_dir: str = "data/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.cache_dir / "cache_metadata.json"
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict:
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_metadata(self):
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2, default=str)
    
    def get_cache_key(self, params: Dict) -> str:
        """Generate cache key from parameters"""
        param_str = json.dumps(params, sort_keys=True)
        return hashlib.md5(param_str.encode()).hexdigest()
    
    def get(self, key: str, max_age_hours: int = 24) -> Optional[Any]:
        """Get from cache if not expired"""
        cache_file = self.cache_dir / f"{key}.pkl"
        
        if not cache_file.exists():
            return None
        
        # Check age
        if key in self.metadata:
            cached_time = datetime.fromisoformat(self.metadata[key]['timestamp'])
            age = datetime.now() - cached_time
            
            if age.total_seconds() > max_age_hours * 3600:
                logger.info(f"Cache expired for {key}")
                return None
        
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    
    def set(self, key: str, data: Any, metadata: Dict = None):
        """Save to cache with metadata"""
        cache_file = self.cache_dir / f"{key}.pkl"
        
        with open(cache_file, 'wb') as f:
            pickle.dump(data, f)
        
        self.metadata[key] = {
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {}
        }
        self._save_metadata()
        
        logger.info(f"Cached data with key {key}")


class DataManager:
    """
    Single source of truth for ALL data operations in FinGraph.
    This replaces all existing data loaders and collectors.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize with configuration
        
        Args:
            config: Configuration dictionary from pipeline_config.yaml
        """
        self.config = config
        self.cache = DataCache(config['data']['cache_dir'])
        self.symbols = config['data']['symbols']
        self.start_date = config['data']['start_date']
        end_date = config['data'].get('end_date')
        if end_date is None or end_date == 'null' or end_date == '':
            self.end_date = datetime.now().strftime('%Y-%m-%d')
        else:
            self.end_date = end_date
        
        # Data storage
        self.price_data = None
        self.feature_data = None
        self.processed_data = None
        
    def load_all_data(self, use_cache: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Load all required data for the pipeline.
        This is the ONLY method other components should call.
        
        Returns:
            Dictionary containing:
            - 'prices': Raw price data
            - 'features': Calculated features
            - 'metadata': Data statistics
        """
        logger.info("Loading all data through Unified Data Manager")
        
        # Try cache first
        if use_cache:
            cache_key = self.cache.get_cache_key({
                'symbols': self.symbols,
                'start_date': self.start_date,
                'end_date': self.end_date,
                'version': '2.0'
            })
            
            cached_data = self.cache.get(cache_key, max_age_hours=self.config['data']['cache_ttl_hours'])
            if cached_data is not None:
                logger.info("Using cached data")
                return cached_data
        
        # Download fresh data
        logger.info("Downloading fresh data")
        self.price_data = self._download_price_data()
        
        # Validate data quality
        self._validate_data_quality()
        
        # Calculate basic statistics
        metadata = self._calculate_metadata()
        
        # Prepare return package
        data_package = {
            'prices': self.price_data,
            'metadata': metadata,
            'config': self.config
        }
        
        # Cache the package
        if use_cache:
            self.cache.set(cache_key, data_package, metadata)
        
        return data_package
    
    def _download_price_data(self) -> pd.DataFrame:
        """Download price data for all symbols - fixed version"""
        all_data = []
        failed_symbols = []
        
        for symbol in self.symbols:
            try:
                logger.info(f"Downloading {symbol}")
                
                # Download single symbol to avoid MultiIndex issues
                ticker = yf.Ticker(symbol)
                
                # Get historical data
                data = ticker.history(
                    start=self.start_date,
                    end=self.end_date,
                    auto_adjust=True,
                    actions=False
                )
                
                if len(data) > self.config['data']['min_history_days']:
                    # Reset index to get Date as column
                    data = data.reset_index()
                    
                    # Add symbol column
                    data['symbol'] = symbol
                    
                    # Standardize column names
                    data.columns = [col.lower().replace(' ', '_') for col in data.columns]
                    
                    all_data.append(data)
                else:
                    logger.warning(f"Insufficient data for {symbol}: {len(data)} days")
                    failed_symbols.append(symbol)
                        
            except Exception as e:
                logger.error(f"Failed to download {symbol}: {e}")
                failed_symbols.append(symbol)
        
        if not all_data:
            raise ValueError("No data downloaded successfully")
        
        # Combine all data
        combined = pd.concat(all_data, ignore_index=True)
        
        # Ensure datetime index
        if 'date' in combined.columns:
            combined['date'] = pd.to_datetime(combined['date'])
            combined = combined.set_index('date')
        
        # Validate required columns exist
        required_cols = ['open', 'high', 'low', 'close', 'volume', 'symbol']
        missing_cols = [col for col in required_cols if col not in combined.columns]
        if missing_cols:
            logger.warning(f"Missing columns: {missing_cols}")
        
        logger.info(f"Downloaded data for {len(self.symbols) - len(failed_symbols)} symbols")
        if failed_symbols:
            logger.warning(f"Failed symbols: {failed_symbols}")
        
        return combined
    
    def _validate_data_quality(self):
        """Validate data quality and fix issues"""
        if self.price_data is None:
            raise ValueError("No price data to validate")
        
        # Check for missing values
        missing_pct = self.price_data.isnull().sum() / len(self.price_data)
        
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in self.price_data.columns:
                if missing_pct[col] > self.config['data']['max_missing_pct']:
                    logger.warning(f"Column {col} has {missing_pct[col]:.1%} missing")
        
        # Forward fill missing values
        self.price_data = self.price_data.fillna(method='ffill').fillna(method='bfill')
        
        # Remove any remaining rows with missing critical data
        self.price_data = self.price_data.dropna(subset=['close'])
        
        # Validate price consistency
        if 'high' in self.price_data.columns and 'low' in self.price_data.columns:
            invalid = self.price_data['high'] < self.price_data['low']
            if invalid.any():
                logger.warning(f"Found {invalid.sum()} rows with high < low")
                self.price_data = self.price_data[~invalid]
        
        logger.info(f"Data validation complete: {len(self.price_data)} rows")
    
    def _calculate_metadata(self) -> Dict:
        """Calculate data statistics"""
        return {
            'total_rows': len(self.price_data),
            'symbols': list(self.price_data['symbol'].unique()),
            'date_range': {
                'start': str(self.price_data.index.min()),
                'end': str(self.price_data.index.max())
            },
            'columns': list(self.price_data.columns),
            'missing_pct': self.price_data.isnull().sum().to_dict(),
            'downloaded_at': datetime.now().isoformat()
        }
    
    def get_symbol_data(self, symbol: str) -> pd.DataFrame:
        """Get data for a specific symbol"""
        if self.price_data is None:
            self.load_all_data()
        
        return self.price_data[self.price_data['symbol'] == symbol].copy()
    
    def get_date_range_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Get data for a specific date range"""
        if self.price_data is None:
            self.load_all_data()
        
        mask = (self.price_data.index >= start_date) & (self.price_data.index <= end_date)
        return self.price_data[mask].copy()
    
    def save_processed_data(self, data: pd.DataFrame, name: str):
        """Save processed data with versioning"""
        save_dir = Path(self.config['paths']['data_dir']) / 'processed'
        save_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = save_dir / f"{name}_{timestamp}.parquet"
        
        data.to_parquet(filename, engine='pyarrow')
        logger.info(f"Saved processed data to {filename}")
        
        # Also save as 'latest' for easy access
        latest_file = save_dir / f"{name}_latest.parquet"
        data.to_parquet(latest_file, engine='pyarrow')
        
        return filename