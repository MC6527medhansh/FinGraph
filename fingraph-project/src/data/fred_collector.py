"""
FRED Economic Data Collector for FinGraph
Collects macroeconomic indicators from Federal Reserve Economic Data
"""

import pandas as pd
import numpy as np
from fredapi import Fred
import logging
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import yaml

logger = logging.getLogger(__name__)

class FREDCollector:
    """
    Collects economic data from Federal Reserve Economic Data (FRED)
    
    Features:
    - Major economic indicators (GDP, inflation, rates)
    - Market stress indicators (VIX, yield spreads)
    - Labor market data (unemployment, job growth)
    - Financial sector health metrics
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize FRED collector
        
        Args:
            api_key: FRED API key (if None, loads from config.yaml)
        """
        if api_key is None:
            api_key = self._load_api_key()
        
        self.fred = Fred(api_key=api_key)
        
        # Economic indicators we'll collect
        self.economic_series = {
            # Interest Rates
            'fed_funds_rate': 'FEDFUNDS',
            'treasury_10y': 'GS10',
            'treasury_2y': 'GS2',
            'treasury_3m': 'GS3M',
            
            # Economic Growth
            'gdp_growth': 'GDPC1',
            'unemployment_rate': 'UNRATE',
            'industrial_production': 'INDPRO',
            'consumer_confidence': 'UMCSENT',
            
            # Inflation
            'cpi_all': 'CPIAUCSL',
            'cpi_core': 'CPILFESL',
            'ppi': 'PPIACO',
            'inflation_expectation': 'T5YIE',
            
            # Market Indicators
            'vix': 'VIXCLS',
            'dollar_index': 'DTWEXBGS',
            'crude_oil': 'DCOILWTICO',
            
            # Financial Sector
            'credit_spread': 'BAMLC0A0CM',
            'mortgage_30y': 'MORTGAGE30US',
            'bank_lending_standards': 'DRTSCLCC'
        }
    
    def _load_api_key(self) -> str:
        """Load FRED API key from config file"""
        try:
            with open('config.yaml', 'r') as f:
                config = yaml.safe_load(f)
            return config['apis']['fred']['api_key']
        except:
            raise ValueError("FRED API key not found. Add it to config.yaml or pass directly")
    
    def get_economic_indicator(self, 
                             series_id: str, 
                             start_date: str, 
                             end_date: str) -> Optional[pd.DataFrame]:
        """
        Download single economic indicator
        
        Args:
            series_id: FRED series ID (e.g., 'FEDFUNDS')
            start_date: Start date 'YYYY-MM-DD'
            end_date: End date 'YYYY-MM-DD'
            
        Returns:
            DataFrame with date index and values
        """
        try:
            data = self.fred.get_series(series_id, start=start_date, end=end_date)
            
            if data.empty:
                logger.warning(f"No data found for {series_id}")
                return None
            
            # Convert to DataFrame
            df = data.to_frame(name='value')
            df['series_id'] = series_id
            df.index.name = 'date'
            
            logger.info(f"✅ Downloaded {len(df)} records for {series_id}")
            return df
            
        except Exception as e:
            logger.error(f"❌ Failed to download {series_id}: {str(e)}")
            return None
    
    def get_all_economic_data(self, 
                            start_date: str, 
                            end_date: str) -> pd.DataFrame:
        """
        Download all economic indicators
        
        Returns:
            DataFrame with all economic data
        """
        all_data = []
        
        logger.info(f"📊 Downloading {len(self.economic_series)} economic indicators...")
        
        for name, series_id in self.economic_series.items():
            logger.info(f"Downloading {name} ({series_id})...")
            
            data = self.get_economic_indicator(series_id, start_date, end_date)
            if data is not None:
                data['indicator_name'] = name
                all_data.append(data)
        
        if all_data:
            combined_data = pd.concat(all_data, ignore_index=False)
            
            # Create pivot table for easier analysis
            pivot_data = combined_data.pivot_table(
                index='date', 
                columns='indicator_name', 
                values='value'
            )
            
            logger.info(f"✅ Downloaded {len(self.economic_series)} economic indicators")
            logger.info(f"Date range: {pivot_data.index.min()} to {pivot_data.index.max()}")
            
            return pivot_data
        else:
            logger.error("❌ No economic data was downloaded")
            return pd.DataFrame()
    
    def calculate_derived_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate derived economic indicators
        
        Args:
            data: DataFrame with economic indicators
            
        Returns:
            DataFrame with additional derived indicators
        """
        if data.empty:
            return data
        
        # Yield curve slopes
        if 'treasury_10y' in data.columns and 'treasury_2y' in data.columns:
            data['yield_curve_2_10'] = data['treasury_10y'] - data['treasury_2y']
        
        if 'treasury_10y' in data.columns and 'treasury_3m' in data.columns:
            data['yield_curve_3m_10y'] = data['treasury_10y'] - data['treasury_3m']
        
        # Real interest rates (nominal - inflation)
        if 'fed_funds_rate' in data.columns and 'cpi_all' in data.columns:
            # Calculate CPI year-over-year change
            cpi_yoy = data['cpi_all'].pct_change(periods=12) * 100
            data['real_fed_funds'] = data['fed_funds_rate'] - cpi_yoy
        
        # Economic stress indicators
        if 'unemployment_rate' in data.columns:
            # Unemployment rate change (momentum)
            data['unemployment_change'] = data['unemployment_rate'].diff()
        
        if 'vix' in data.columns:
            # VIX relative to its moving average (fear gauge)
            data['vix_ma20'] = data['vix'].rolling(window=20).mean()
            data['vix_relative'] = data['vix'] / data['vix_ma20']
        
        logger.info(f"✅ Added {data.shape[1] - len(self.economic_series)} derived indicators")
        return data
    
    def get_series_info(self, series_id: str) -> Dict:
        """Get metadata about a FRED series"""
        try:
            info = self.fred.get_series_info(series_id)
            return {
                'id': series_id,
                'title': info['title'],
                'units': info['units'],
                'frequency': info['frequency'],
                'last_updated': info['last_updated']
            }
        except Exception as e:
            logger.error(f"Failed to get info for {series_id}: {str(e)}")
            return {}

# Test function
def test_fred_collector():
    """Test FRED collector with sample data"""
    
    # Note: You need to add your FRED API key to config.yaml first
    try:
        collector = FREDCollector()
        
        # Test single indicator
        print("🧪 Testing single indicator download...")
        fed_data = collector.get_economic_indicator('FEDFUNDS', '2024-01-01', '2024-09-01')
        if fed_data is not None:
            print(f"✅ Fed funds data: {len(fed_data)} records")
        
        # Test batch download (subset for faster testing)
        print("\n🧪 Testing batch economic data...")
        
        # Temporarily reduce series for testing
        original_series = collector.economic_series.copy()
        collector.economic_series = {
            'fed_funds_rate': 'FEDFUNDS',
            'unemployment_rate': 'UNRATE',
            'vix': 'VIXCLS'
        }
        
        econ_data = collector.get_all_economic_data('2024-01-01', '2024-09-01')
        if not econ_data.empty:
            print(f"✅ Economic data: {econ_data.shape}")
            print(f"Indicators: {list(econ_data.columns)}")
        
        # Test derived indicators
        derived_data = collector.calculate_derived_indicators(econ_data)
        print(f"✅ With derived indicators: {derived_data.shape}")
        
        # Restore original series
        collector.economic_series = original_series
        
    except Exception as e:
        print(f"❌ FRED test failed: {str(e)}")
        print("Make sure you've added your FRED API key to config.yaml")

if __name__ == "__main__":
    test_fred_collector()