"""
Yahoo Finance Data Collector for FinGraph
Handles stock prices, financial statements, and company statistics
"""

import yfinance as yf
import pandas as pd
import numpy as np
import time
import logging
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class YahooFinanceCollector:
    """
    Collects financial data from Yahoo Finance API
    
    Features:
    - Stock price data with technical indicators
    - Company financial statements  
    - Key statistics and ratios
    - Batch processing with rate limiting
    - Data validation and error handling
    """
    
    def __init__(self, rate_limit: float = 0.5):
        """
        Initialize collector with rate limiting
        
        Args:
            rate_limit: Seconds between API calls (default 0.5 = 2 calls/second)
        """
        self.rate_limit = rate_limit
        self.failed_downloads = []
        
    def get_stock_data(self, 
                      symbol: str, 
                      start_date: str, 
                      end_date: str) -> Optional[pd.DataFrame]:
        """
        Download stock price data for a single symbol
        
        Args:
            symbol: Stock ticker (e.g., 'AAPL')
            start_date: Start date 'YYYY-MM-DD'
            end_date: End date 'YYYY-MM-DD'
            
        Returns:
            DataFrame with OHLCV data + technical indicators
        """
        try:
            # Download data
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_date, end=end_date)
            
            if data.empty:
                logger.warning(f"No data found for {symbol}")
                return None
                
            # Add technical indicators
            data = self._add_technical_indicators(data)
            
            # Add symbol column
            data['Symbol'] = symbol
            
            logger.info(f"✅ Downloaded {len(data)} days for {symbol}")
            return data
            
        except Exception as e:
            logger.error(f"❌ Failed to download {symbol}: {str(e)}")
            self.failed_downloads.append(symbol)
            return None
            
    def _add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to price data"""
        
        # RSI (Relative Strength Index)
        data['RSI'] = self._calculate_rsi(data['Close'])
        
        # Moving averages
        data['SMA_20'] = data['Close'].rolling(window=20).mean()
        data['SMA_50'] = data['Close'].rolling(window=50).mean()
        
        # Bollinger Bands
        data['BB_upper'], data['BB_lower'] = self._calculate_bollinger_bands(data['Close'])
        
        # Volatility (20-day rolling)
        data['Volatility'] = data['Close'].pct_change().rolling(window=20).std() * np.sqrt(252)
        
        # Price momentum
        data['Returns_1d'] = data['Close'].pct_change()
        data['Returns_5d'] = data['Close'].pct_change(periods=5)
        data['Returns_20d'] = data['Close'].pct_change(periods=20)
        
        return data
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_bollinger_bands(self, prices: pd.Series, window: int = 20) -> Tuple[pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        sma = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()
        upper = sma + (std * 2)
        lower = sma - (std * 2)
        return upper, lower
    
    def get_company_info(self, symbol: str) -> Optional[Dict]:
        """
        Get company fundamental data and statistics
        
        Returns:
            Dictionary with company info, financials, and key stats
        """
        try:
            ticker = yf.Ticker(symbol)
            
            # Get basic info
            info = ticker.info
            
            # Extract key metrics
            company_data = {
                'symbol': symbol,
                'company_name': info.get('longName', 'Unknown'),
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown'),
                'country': info.get('country', 'Unknown'),
                'market_cap': info.get('marketCap', 0),
                'enterprise_value': info.get('enterpriseValue', 0),
                'pe_ratio': info.get('trailingPE', None),
                'forward_pe': info.get('forwardPE', None),
                'peg_ratio': info.get('pegRatio', None),
                'price_to_book': info.get('priceToBook', None),
                'price_to_sales': info.get('priceToSalesTrailing12Months', None),
                'debt_to_equity': info.get('debtToEquity', None),
                'roe': info.get('returnOnEquity', None),
                'roa': info.get('returnOnAssets', None),
                'profit_margin': info.get('profitMargins', None),
                'beta': info.get('beta', None),
                'dividend_yield': info.get('dividendYield', None),
                'revenue_growth': info.get('revenueGrowth', None),
                'earnings_growth': info.get('earningsGrowth', None),
                'current_ratio': info.get('currentRatio', None),
                'quick_ratio': info.get('quickRatio', None),
                'cash_per_share': info.get('totalCashPerShare', None),
                'book_value': info.get('bookValue', None),
                'employees': info.get('fullTimeEmployees', None),
                'website': info.get('website', ''),
                'description': info.get('longBusinessSummary', '')[:500]  # Truncate
            }
            
            logger.info(f"✅ Got company info for {symbol}")
            return company_data
            
        except Exception as e:
            logger.error(f"❌ Failed to get info for {symbol}: {str(e)}")
            return None
    
    def batch_download_stocks(self, 
                            symbols: List[str], 
                            start_date: str, 
                            end_date: str) -> pd.DataFrame:
        """
        Download stock data for multiple symbols with rate limiting
        
        Args:
            symbols: List of stock tickers
            start_date: Start date 'YYYY-MM-DD'  
            end_date: End date 'YYYY-MM-DD'
            
        Returns:
            Combined DataFrame with all stock data
        """
        all_data = []
        
        logger.info(f"📥 Starting batch download for {len(symbols)} symbols...")
        
        for i, symbol in enumerate(symbols):
            # Rate limiting
            if i > 0:
                time.sleep(self.rate_limit)
            
            # Progress update
            if i % 10 == 0:
                logger.info(f"Progress: {i}/{len(symbols)} symbols processed")
            
            # Download data
            data = self.get_stock_data(symbol, start_date, end_date)
            if data is not None:
                all_data.append(data)
        
        if all_data:
            combined_data = pd.concat(all_data, ignore_index=True)
            logger.info(f"✅ Successfully downloaded data for {len(all_data)} symbols")
            logger.info(f"❌ Failed downloads: {len(self.failed_downloads)} symbols")
            return combined_data
        else:
            logger.error("❌ No data was successfully downloaded")
            return pd.DataFrame()
    
    def batch_download_company_info(self, symbols: List[str]) -> pd.DataFrame:
        """Download company information for multiple symbols"""
        all_info = []
        
        logger.info(f"📊 Downloading company info for {len(symbols)} symbols...")
        
        for i, symbol in enumerate(symbols):
            if i > 0:
                time.sleep(self.rate_limit)
                
            if i % 10 == 0:
                logger.info(f"Progress: {i}/{len(symbols)} companies processed")
            
            info = self.get_company_info(symbol)
            if info:
                all_info.append(info)
        
        if all_info:
            df = pd.DataFrame(all_info)
            logger.info(f"✅ Got company info for {len(all_info)} companies")
            return df
        else:
            logger.error("❌ No company info was downloaded")
            return pd.DataFrame()

# Quick test function
def test_yahoo_collector():
    """Test the Yahoo Finance collector with sample data"""
    collector = YahooFinanceCollector()
    
    # Test single stock
    print("🧪 Testing single stock download...")
    aapl_data = collector.get_stock_data('AAPL', '2024-01-01', '2024-09-01')
    if aapl_data is not None:
        print(f"✅ AAPL data: {len(aapl_data)} days")
        print(f"Columns: {list(aapl_data.columns)}")
    
    # Test company info
    print("\n🧪 Testing company info...")
    aapl_info = collector.get_company_info('AAPL')
    if aapl_info:
        print(f"✅ AAPL info: {aapl_info['company_name']}, {aapl_info['sector']}")
    
    # Test batch download (small sample)
    print("\n🧪 Testing batch download...")
    test_symbols = ['AAPL', 'MSFT', 'GOOGL']
    batch_data = collector.batch_download_stocks(test_symbols, '2024-08-01', '2024-09-01')
    if not batch_data.empty:
        print(f"✅ Batch data: {len(batch_data)} total records")
        print(f"Symbols: {batch_data['Symbol'].unique()}")

if __name__ == "__main__":
    test_yahoo_collector()