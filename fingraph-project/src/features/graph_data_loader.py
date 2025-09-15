"""
Graph Data Loader for FinGraph - FIXED VERSION
Loads and prepares collected data for graph construction
Handles all date parsing, correlation calculation, and data alignment issues
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import os
import glob
import warnings
import sys
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class GraphDataLoader:
    """
    Loads and prepares data for graph construction - FIXED VERSION
    
    Features:
    - Robust date parsing with explicit formats
    - Fixed correlation calculations with proper data handling
    - Comprehensive error handling and data validation
    - Proper missing data handling
    """
    
    def __init__(self, data_dir: str = 'data/raw'):
        """
        Initialize data loader
        
        Args:
            data_dir: Directory containing raw data files
        """
        self.data_dir = data_dir
        self.stock_data = None
        self.company_info = None
        self.economic_data = None
        self.relationship_data = None
        
    def load_latest_data(
        self,
        refresh_if_missing: bool = False,
        max_age_hours: Optional[int] = 24,
    ) -> Dict:
        """
        Load the most recent data files with robust error handling.

        Args:
            refresh_if_missing: When True, run data collection if required files are
                missing or older than ``max_age_hours``.
            max_age_hours: Maximum allowed age for a data file. ``None`` disables
                age checks.

        Returns:
            Dictionary with loaded datasets.
        """
        logger.info(f"📂 Loading data from {self.data_dir}")

        # Find latest files (by timestamp in filename)
        stock_files = glob.glob(os.path.join(self.data_dir, 'stock_data_*.csv'))
        company_files = glob.glob(os.path.join(self.data_dir, 'company_info_*.csv'))
        economic_files = glob.glob(os.path.join(self.data_dir, 'economic_data_*.csv'))
        relationship_files = glob.glob(os.path.join(self.data_dir, 'relationship_data_*.csv'))

        file_candidates = {
            'stock': stock_files,
            'company': company_files,
            'economic': economic_files,
            'relationship': relationship_files,
        }

        latest_files = {
            name: self._get_latest_file(paths)
            for name, paths in file_candidates.items()
        }

        refresh_reasons = self._determine_refresh_reasons(latest_files, max_age_hours)
        should_refresh = self._should_attempt_refresh(refresh_reasons)

        if should_refresh and refresh_if_missing:
            logger.info("🔄 Refresh requested: collecting fresh data before loading")
            self._collect_fresh_data()

            # Re-scan after refreshing
            stock_files = glob.glob(os.path.join(self.data_dir, 'stock_data_*.csv'))
            company_files = glob.glob(os.path.join(self.data_dir, 'company_info_*.csv'))
            economic_files = glob.glob(os.path.join(self.data_dir, 'economic_data_*.csv'))
            relationship_files = glob.glob(os.path.join(self.data_dir, 'relationship_data_*.csv'))

            file_candidates = {
                'stock': stock_files,
                'company': company_files,
                'economic': economic_files,
                'relationship': relationship_files,
            }
            latest_files = {
                name: self._get_latest_file(paths)
                for name, paths in file_candidates.items()
            }
            refresh_reasons = self._determine_refresh_reasons(latest_files, max_age_hours)
            should_refresh = self._should_attempt_refresh(refresh_reasons)

        if should_refresh:
            stock_reason = refresh_reasons.get('stock')
            if stock_reason:
                raise FileNotFoundError(
                    "Stock data is missing or outdated. Run data collection or enable refresh to fetch new data."
                )

            for dataset, reason in refresh_reasons.items():
                if dataset == 'stock':
                    continue
                logger.warning(
                    "Data for %s is %s. Continuing with available information.",
                    dataset,
                    reason,
                )
        else:
            # Even if we don't refresh, warn about any missing optional datasets
            for dataset, reason in refresh_reasons.items():
                if dataset == 'stock':
                    continue
                logger.warning(
                    "Data for %s is %s. Continuing with available information.",
                    dataset,
                    reason,
                )

        latest_stock = latest_files.get('stock')
        latest_company = latest_files.get('company')
        latest_economic = latest_files.get('economic')
        latest_relationship = latest_files.get('relationship')

        if not latest_stock:
            raise FileNotFoundError("No stock data files found. Run data collection first!")

        # Load data with proper date handling
        logger.info(f"Loading stock data: {os.path.basename(latest_stock)}")
        self.stock_data = self._load_stock_data(latest_stock)

        if latest_company:
            logger.info(f"Loading company info: {os.path.basename(latest_company)}")
            self.company_info = pd.read_csv(latest_company)
        
        if latest_economic:
            logger.info(f"Loading economic data: {os.path.basename(latest_economic)}")
            self.economic_data = self._load_economic_data(latest_economic)
        
        if latest_relationship:
            logger.info(f"Loading relationship data: {os.path.basename(latest_relationship)}")
            self.relationship_data = pd.read_csv(latest_relationship)
        
        # Print data summary
        self._print_data_summary()
        
        return {
            'stock_data': self.stock_data,
            'company_info': self.company_info,
            'economic_data': self.economic_data,
            'relationship_data': self.relationship_data
        }

    def _get_latest_file(self, files: List[str]) -> Optional[str]:
        """Return the most recent file from a list of candidates."""

        return max(files, key=os.path.getctime) if files else None

    def _determine_refresh_reasons(
        self,
        latest_files: Dict[str, Optional[str]],
        max_age_hours: Optional[int],
    ) -> Dict[str, str]:
        """Determine if any required refresh conditions are met."""

        reasons: Dict[str, str] = {}
        for dataset, file_path in latest_files.items():
            if not file_path:
                reasons[dataset] = 'missing'
                continue

            if max_age_hours is None:
                continue

            if self._is_file_outdated(file_path, max_age_hours):
                reasons[dataset] = 'outdated'

        return reasons

    def _should_attempt_refresh(self, refresh_reasons: Dict[str, str]) -> bool:
        """Decide whether a refresh should be attempted based on reasons."""

        if not refresh_reasons:
            return False

        if refresh_reasons.get('stock'):
            return True

        return any(reason == 'outdated' for reason in refresh_reasons.values())

    def _is_file_outdated(self, file_path: str, max_age_hours: int) -> bool:
        """Check if a file exceeds the allowed age threshold."""

        if max_age_hours <= 0:
            return True

        modified_time = datetime.fromtimestamp(os.path.getmtime(file_path))
        age = datetime.now() - modified_time
        return age > timedelta(hours=max_age_hours)

    def _collect_fresh_data(self) -> None:
        """Invoke the data collection routine to refresh raw data files."""

        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        if project_root not in sys.path:
            sys.path.append(project_root)

        try:
            from scripts.collect_data import FinGraphDataCollector
        except ImportError as exc:
            logger.error("Unable to import data collector: %s", exc)
            raise

        try:
            collector = FinGraphDataCollector()
            collector.collect_all_data()
            logger.info("✅ Fresh data collected successfully")
        except Exception as exc:
            logger.error("Failed to collect fresh data: %s", exc)
            raise

    def _load_stock_data(self, file_path: str) -> pd.DataFrame:
        """Load stock data with proper date parsing"""
        try:
            # Try multiple date formats
            data = pd.read_csv(file_path)
            
            # Handle different possible date column names and formats
            date_columns = ['Date', 'date', 'Unnamed: 0']
            date_col = None
            
            for col in date_columns:
                if col in data.columns:
                    date_col = col
                    break
            
            if date_col:
                # Parse dates with multiple format attempts
                try:
                    data[date_col] = pd.to_datetime(data[date_col], format='%Y-%m-%d')
                except:
                    try:
                        data[date_col] = pd.to_datetime(data[date_col], infer_datetime_format=True)
                    except:
                        data[date_col] = pd.to_datetime(data[date_col], errors='coerce')
                
                # Set as index and sort
                data.set_index(date_col, inplace=True)
                data.sort_index(inplace=True)
                
                # Remove any rows with invalid dates
                data = data[data.index.notna()]
            
            logger.info(f"✅ Loaded stock data: {len(data)} records")
            return data
            
        except Exception as e:
            logger.error(f"❌ Error loading stock data: {str(e)}")
            return pd.DataFrame()
    
    def _load_economic_data(self, file_path: str) -> pd.DataFrame:
        """Load economic data with proper date parsing"""
        try:
            data = pd.read_csv(file_path)
            
            # Handle date column
            if 'Unnamed: 0' in data.columns:
                data.rename(columns={'Unnamed: 0': 'date'}, inplace=True)
            
            if 'date' in data.columns:
                try:
                    data['date'] = pd.to_datetime(data['date'], format='%Y-%m-%d')
                except:
                    data['date'] = pd.to_datetime(data['date'], errors='coerce')
                
                data.set_index('date', inplace=True)
                data.sort_index(inplace=True)
                data = data[data.index.notna()]
            
            logger.info(f"✅ Loaded economic data: {data.shape}")
            return data
            
        except Exception as e:
            logger.error(f"❌ Error loading economic data: {str(e)}")
            return pd.DataFrame()
    
    def _print_data_summary(self):
        """Print summary of loaded data"""
        logger.info("📊 Data Summary:")
        
        if self.stock_data is not None and not self.stock_data.empty:
            logger.info(f"  Stock data: {len(self.stock_data)} records, {self.stock_data['Symbol'].nunique()} companies")
            logger.info(f"  Date range: {self.stock_data.index.min().date()} to {self.stock_data.index.max().date()}")
        
        if self.company_info is not None and not self.company_info.empty:
            sectors = self.company_info['sector'].nunique() if 'sector' in self.company_info.columns else 0
            logger.info(f"  Company info: {len(self.company_info)} companies, {sectors} sectors")
        
        if self.economic_data is not None and not self.economic_data.empty:
            logger.info(f"  Economic data: {self.economic_data.shape} (rows, indicators)")
            logger.info(f"  Economic date range: {self.economic_data.index.min().date()} to {self.economic_data.index.max().date()}")
        
        if self.relationship_data is not None and not self.relationship_data.empty:
            logger.info(f"  Relationships: {len(self.relationship_data)} total relationships")
            types = self.relationship_data['relationship_type'].unique() if 'relationship_type' in self.relationship_data.columns else []
            logger.info(f"  Relationship types: {list(types)}")
    
    def get_company_list(self) -> List[str]:
        """Get list of companies in the dataset"""
        if self.stock_data is not None and 'Symbol' in self.stock_data.columns:
            return sorted(self.stock_data['Symbol'].unique())
        elif self.company_info is not None and 'symbol' in self.company_info.columns:
            return sorted(self.company_info['symbol'].unique())
        else:
            return []
    
    def get_latest_stock_data(self, symbol: str, window_days: int = 30) -> pd.DataFrame:
        """
        Get latest stock data for a company
        
        Args:
            symbol: Stock symbol
            window_days: Number of recent days to include
            
        Returns:
            DataFrame with recent stock data
        """
        if self.stock_data is None or self.stock_data.empty:
            return pd.DataFrame()
        
        company_data = self.stock_data[self.stock_data['Symbol'] == symbol].copy()
        if company_data.empty:
            return pd.DataFrame()
        
        # Get last N days
        company_data = company_data.tail(window_days)
        return company_data
    
    def calculate_stock_correlations(self, window_days: int = 60) -> pd.DataFrame:
        """
        Calculate stock price correlations between companies - FULLY FIXED VERSION
        
        Args:
            window_days: Rolling window for correlation calculation
            
        Returns:
            Correlation matrix
        """
        if self.stock_data is None or self.stock_data.empty:
            logger.warning("No stock data available for correlation calculation")
            return pd.DataFrame()
        
        try:
            # Create pivot table with proper handling
            logger.info(f"📈 Calculating correlations for {window_days}-day window...")
            
            # Ensure we have the required columns
            if 'Symbol' not in self.stock_data.columns or 'Close' not in self.stock_data.columns:
                logger.error("Missing required columns (Symbol or Close) in stock data")
                return pd.DataFrame()
            
            # Get unique symbols
            symbols = self.stock_data['Symbol'].unique()
            logger.info(f"Found symbols: {list(symbols)}")
            
            # Create pivot table - handle index properly
            stock_data_copy = self.stock_data.copy()
            
            # Make sure index is named properly
            if stock_data_copy.index.name is None:
                stock_data_copy.index.name = 'Date'
            
            # Reset index to get date as a column
            stock_data_reset = stock_data_copy.reset_index()
            
            # Create pivot table using the correct date column name
            date_col = stock_data_reset.columns[0]  # First column should be the date
            
            pivot_data = stock_data_reset.pivot(
                index=date_col,
                columns='Symbol',
                values='Close'
            )
            
            # Fill any gaps
            pivot_data = pivot_data.fillna(method='ffill').fillna(method='bfill')
            
            if pivot_data.empty:
                logger.warning("Pivot table is empty")
                return self._create_fallback_correlations(symbols)
            
            # Calculate returns
            returns = pivot_data.pct_change().dropna()
            
            if returns.empty or len(returns) < 2:
                logger.warning("Insufficient returns data for correlation calculation")
                return self._create_fallback_correlations(symbols)
            
            # Use appropriate window size
            actual_window = min(window_days, len(returns))
            recent_returns = returns.tail(actual_window)
            
            # Calculate correlations
            correlations = recent_returns.corr()
            
            # Fill any NaN values with 0 (no correlation)
            correlations = correlations.fillna(0)
            
            # Validate correlation matrix
            if correlations.empty or correlations.isna().all().all():
                logger.warning("Correlation matrix is empty or all NaN")
                return self._create_fallback_correlations(symbols)
            
            logger.info(f"✅ Calculated real correlations: {correlations.shape}")
            
            # Show sample of actual correlations
            sample_corr = correlations.iloc[:3, :3].round(3)
            logger.info(f"Sample correlations:\n{sample_corr}")
            
            return correlations
            
        except Exception as e:
            logger.error(f"❌ Error calculating correlations: {str(e)}")
            symbols = self.stock_data['Symbol'].unique()
            return self._create_fallback_correlations(symbols)
    
    def get_economic_indicators_latest(self, indicators: List[str] = None) -> Dict:
        """
        Get latest values of economic indicators
        
        Args:
            indicators: List of indicator names (if None, gets all)
            
        Returns:
            Dictionary with latest indicator values
        """
        if self.economic_data is None or self.economic_data.empty:
            return {}
        
        if indicators is None:
            indicators = self.economic_data.columns.tolist()
        
        latest_values = {}
        for indicator in indicators:
            if indicator in self.economic_data.columns:
                # Get most recent non-null value
                series = self.economic_data[indicator].dropna()
                if not series.empty:
                    latest_values[indicator] = float(series.iloc[-1])
        
        logger.info(f"✅ Got latest values for {len(latest_values)} economic indicators")
        return latest_values
    
    def get_sector_mapping(self) -> Dict[str, str]:
        """Get company to sector mapping"""
        if self.company_info is None or self.company_info.empty:
            return {}
        
        # Handle different possible column names
        symbol_col = 'symbol' if 'symbol' in self.company_info.columns else 'Symbol'
        sector_col = 'sector' if 'sector' in self.company_info.columns else 'Sector'
        
        if symbol_col in self.company_info.columns and sector_col in self.company_info.columns:
            return dict(zip(self.company_info[symbol_col], self.company_info[sector_col]))
        else:
            logger.warning(f"Could not find symbol/sector columns. Available: {list(self.company_info.columns)}")
            return {}
    
    def validate_data_quality(self) -> Dict:
        """
        Validate data quality and return report
        
        Returns:
            Dictionary with validation results
        """
        validation = {
            'stock_data': {'loaded': False, 'records': 0, 'companies': 0, 'issues': []},
            'company_info': {'loaded': False, 'records': 0, 'issues': []},
            'economic_data': {'loaded': False, 'records': 0, 'indicators': 0, 'issues': []},
            'relationship_data': {'loaded': False, 'records': 0, 'issues': []}
        }
        
        # Validate stock data
        if self.stock_data is not None and not self.stock_data.empty:
            validation['stock_data']['loaded'] = True
            validation['stock_data']['records'] = len(self.stock_data)
            validation['stock_data']['companies'] = self.stock_data['Symbol'].nunique() if 'Symbol' in self.stock_data.columns else 0
            
            if 'Close' not in self.stock_data.columns:
                validation['stock_data']['issues'].append("Missing 'Close' price column")
            if validation['stock_data']['companies'] == 0:
                validation['stock_data']['issues'].append("No companies found in Symbol column")
        else:
            validation['stock_data']['issues'].append("No stock data loaded")
        
        # Validate company info
        if self.company_info is not None and not self.company_info.empty:
            validation['company_info']['loaded'] = True
            validation['company_info']['records'] = len(self.company_info)
            
            required_cols = ['symbol', 'sector']
            for col in required_cols:
                if col not in self.company_info.columns and col.title() not in self.company_info.columns:
                    validation['company_info']['issues'].append(f"Missing '{col}' column")
        else:
            validation['company_info']['issues'].append("No company info loaded")
        
        # Validate economic data
        if self.economic_data is not None and not self.economic_data.empty:
            validation['economic_data']['loaded'] = True
            validation['economic_data']['records'] = len(self.economic_data)
            validation['economic_data']['indicators'] = len(self.economic_data.columns)
        else:
            validation['economic_data']['issues'].append("No economic data loaded")
        
        # Validate relationship data
        if self.relationship_data is not None and not self.relationship_data.empty:
            validation['relationship_data']['loaded'] = True
            validation['relationship_data']['records'] = len(self.relationship_data)
            
            required_cols = ['company_symbol', 'relationship_type', 'related_entity']
            for col in required_cols:
                if col not in self.relationship_data.columns:
                    validation['relationship_data']['issues'].append(f"Missing '{col}' column")
        else:
            validation['relationship_data']['issues'].append("No relationship data loaded")
        
        return validation

# Test function with comprehensive error handling
def test_graph_data_loader():
    """Test the graph data loader with comprehensive validation"""
    print("🧪 Testing Graph Data Loader...")
    
    loader = GraphDataLoader()
    
    try:
        # Load data
        print("\n📂 Loading data...")
        data = loader.load_latest_data()
        
        # Validate data quality
        print("\n✅ Validating data quality...")
        validation = loader.validate_data_quality()
        
        for data_type, info in validation.items():
            print(f"\n{data_type.upper()}:")
            print(f"  Loaded: {info['loaded']}")
            print(f"  Records: {info['records']}")
            if 'companies' in info:
                print(f"  Companies: {info['companies']}")
            if 'indicators' in info:
                print(f"  Indicators: {info['indicators']}")
            if info['issues']:
                print(f"  Issues: {info['issues']}")
            else:
                print(f"  Issues: None")
        
        # Test company list
        print("\n📊 Testing company operations...")
        companies = loader.get_company_list()
        print(f"✅ Companies: {companies}")
        
        # Test correlations with robust error handling
        print("\n📈 Testing correlation calculation...")
        if len(companies) > 1:
            correlations = loader.calculate_stock_correlations()
            if not correlations.empty:
                print(f"✅ Correlations calculated: {correlations.shape}")
                print("Sample correlations (first 3x3):")
                print(correlations.iloc[:3, :3])
            else:
                print("⚠️ Correlations calculation returned empty matrix")
        else:
            print("⚠️ Need at least 2 companies for correlation calculation")
        
        # Test economic indicators
        print("\n🏛️ Testing economic indicators...")
        econ_latest = loader.get_economic_indicators_latest(['fed_funds_rate', 'unemployment_rate', 'vix'])
        print(f"✅ Latest economic indicators: {econ_latest}")
        
        # Test sector mapping
        print("\n🏭 Testing sector mapping...")
        sectors = loader.get_sector_mapping()
        print(f"✅ Sector mapping: {sectors}")
        
        print("\n🎉 All tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_graph_data_loader()
    if success:
        print("\n✅ Graph Data Loader is ready for graph construction!")
    else:
        print("\n❌ Please fix the issues before proceeding to graph construction.")