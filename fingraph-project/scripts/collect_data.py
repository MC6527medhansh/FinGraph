"""
Master Data Collection Script for FinGraph
Orchestrates data collection from all sources
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data.yahoo_collector import YahooFinanceCollector
from src.data.fred_collector import FREDCollector
from src.data.sec_collector import SECCollector
from src.data.data_validator import DataValidator

import pandas as pd
import logging
from datetime import datetime, timedelta
import yaml

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FinGraphDataCollector:
    """
    Master data collector that orchestrates all data sources
    """
    
    def __init__(self, config_path: str = 'config.yaml'):
        """Initialize with configuration"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize collectors
        self.yahoo = YahooFinanceCollector()
        
        try:
            self.fred = FREDCollector()
        except:
            logger.warning("FRED collector failed - check API key in config.yaml")
            self.fred = None
            
        self.sec = SECCollector()
        self.validator = DataValidator()
    
    def collect_all_data(self, symbols: list = None):
        """
        Collect data from all sources
        
        Args:
            symbols: List of stock symbols (if None, uses default set)
        """
        if symbols is None:
            # Default set of large cap stocks for testing
            symbols = [
                'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA',
                'META', 'NVDA', 'NFLX', 'CRM', 'ADBE'
            ]
        
        start_date = self.config['data']['start_date']
        end_date = self.config['data']['end_date']
        
        logger.info(f"🚀 Starting data collection for {len(symbols)} symbols")
        logger.info(f"📅 Date range: {start_date} to {end_date}")
        
        # 1. Collect stock price data
        logger.info("📊 Collecting stock price data...")
        stock_data = self.yahoo.batch_download_stocks(symbols, start_date, end_date)
        
        # 2. Collect company information
        logger.info("🏢 Collecting company information...")
        company_info = self.yahoo.batch_download_company_info(symbols)
        
        # 3. Collect economic data
        economic_data = pd.DataFrame()
        if self.fred:
            logger.info("🏛️ Collecting economic data...")
            economic_data = self.fred.get_all_economic_data(start_date, end_date)
            if not economic_data.empty:
                economic_data = self.fred.calculate_derived_indicators(economic_data)
        
        # 4. Collect relationship data
        logger.info("🔗 Collecting relationship data...")
        relationship_data = self.sec.batch_get_relationships(symbols[:5])  # Limit for demo
        
        # 5. Validate all data
        logger.info("✅ Validating data quality...")
        validation_report = self.validator.generate_validation_report(
            stock_data, economic_data, relationship_data
        )
        
        # 6. Save data
        self._save_data(stock_data, company_info, economic_data, relationship_data, validation_report)
        
        logger.info("🎉 Data collection completed successfully!")
        return {
            'stock_data': stock_data,
            'company_info': company_info,
            'economic_data': economic_data,
            'relationship_data': relationship_data,
            'validation_report': validation_report
        }
    
    def _save_data(self, stock_data, company_info, economic_data, relationship_data, validation_report):
        """Save collected data to files"""
        
        # Create data directories if they don't exist
        os.makedirs('data/raw', exist_ok=True)
        os.makedirs('data/processed', exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save raw data
        if not stock_data.empty:
            stock_data.to_csv(f'data/raw/stock_data_{timestamp}.csv')
            logger.info(f"💾 Saved stock data: {len(stock_data)} records")
        
        if not company_info.empty:
            company_info.to_csv(f'data/raw/company_info_{timestamp}.csv', index=False)
            logger.info(f"💾 Saved company info: {len(company_info)} companies")
        
        if not economic_data.empty:
            economic_data.to_csv(f'data/raw/economic_data_{timestamp}.csv')
            logger.info(f"💾 Saved economic data: {economic_data.shape}")
        
        if not relationship_data.empty:
            relationship_data.to_csv(f'data/raw/relationship_data_{timestamp}.csv', index=False)
            logger.info(f"💾 Saved relationship data: {len(relationship_data)} relationships")
        
        # Save validation report
        with open(f'data/raw/validation_report_{timestamp}.txt', 'w') as f:
            f.write(validation_report)
        logger.info("💾 Saved validation report")

def main():
    """Main execution function"""
    try:
        collector = FinGraphDataCollector()
        
        # Test with small set of symbols first
        test_symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
        
        results = collector.collect_all_data(test_symbols)
        
        print("\n" + "="*50)
        print("DATA COLLECTION SUMMARY")
        print("="*50)
        print(f"Stock data: {len(results['stock_data'])} records")
        print(f"Company info: {len(results['company_info'])} companies")
        print(f"Economic data: {results['economic_data'].shape}")
        print(f"Relationships: {len(results['relationship_data'])} relationships")
        print("\nValidation Report:")
        print(results['validation_report'])
        
    except Exception as e:
        logger.error(f"❌ Data collection failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()