"""
Data Validation Framework for FinGraph
Ensures data quality and consistency across all sources
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class DataValidator:
    """
    Validates financial data for quality and consistency
    
    Features:
    - Missing data detection and reporting
    - Outlier identification
    - Data type validation
    - Time series consistency checks
    - Cross-source validation
    """
    
    def __init__(self):
        self.validation_results = {}
    
    def validate_stock_data(self, data: pd.DataFrame) -> Dict:
        """
        Validate stock price data
        
        Args:
            data: DataFrame with stock price data
            
        Returns:
            Dictionary with validation results
        """
        results = {
            'total_records': len(data),
            'symbols': data['Symbol'].nunique() if 'Symbol' in data.columns else 0,
            'date_range': None,
            'missing_data': {},
            'outliers': {},
            'errors': []
        }
        
        if data.empty:
            results['errors'].append("Empty dataset")
            return results
        
        # Check date range
        if data.index.name == 'Date' or 'Date' in data.columns:
            date_col = data.index if data.index.name == 'Date' else data['Date']
            results['date_range'] = {
                'start': str(date_col.min()),
                'end': str(date_col.max()),
                'total_days': (date_col.max() - date_col.min()).days
            }
        
        # Check for missing data
        price_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in price_columns:
            if col in data.columns:
                missing_count = data[col].isna().sum()
                missing_pct = (missing_count / len(data)) * 100
                results['missing_data'][col] = {
                    'count': int(missing_count),
                    'percentage': round(missing_pct, 2)
                }
        
        # Check for outliers in price data
        if 'Close' in data.columns:
            # Check for extreme price movements (>50% in one day)
            if 'Returns_1d' in data.columns:
                extreme_moves = data[abs(data['Returns_1d']) > 0.5]
                results['outliers']['extreme_price_moves'] = len(extreme_moves)
            
            # Check for zero or negative prices
            invalid_prices = data[data['Close'] <= 0]
            results['outliers']['invalid_prices'] = len(invalid_prices)
        
        # Check for volume anomalies
        if 'Volume' in data.columns:
            # Volume should not be negative
            negative_volume = data[data['Volume'] < 0]
            results['outliers']['negative_volume'] = len(negative_volume)
            
            # Check for extremely high volume (>10x median)
            median_volume = data['Volume'].median()
            high_volume = data[data['Volume'] > median_volume * 10]
            results['outliers']['high_volume_days'] = len(high_volume)
        
        logger.info(f"Stock data validation: {results['total_records']} records, {len(results['errors'])} errors")
        return results
    
    def validate_economic_data(self, data: pd.DataFrame) -> Dict:
        """
        Validate economic indicator data
        
        Args:
            data: DataFrame with economic indicators
            
        Returns:
            Dictionary with validation results
        """
        results = {
            'total_records': len(data),
            'indicators': list(data.columns) if not data.empty else [],
            'date_range': None,
            'missing_data': {},
            'anomalies': {},
            'errors': []
        }
        
        if data.empty:
            results['errors'].append("Empty economic dataset")
            return results
        
        # Check date range
        if hasattr(data.index, 'min'):
            results['date_range'] = {
                'start': str(data.index.min()),
                'end': str(data.index.max()),
                'frequency': self._detect_frequency(data.index)
            }
        
        # Check missing data for each indicator
        for col in data.columns:
            missing_count = data[col].isna().sum()
            missing_pct = (missing_count / len(data)) * 100
            results['missing_data'][col] = {
                'count': int(missing_count),
                'percentage': round(missing_pct, 2)
            }
        
        # Check for economic anomalies
        # Negative interest rates (unusual but possible)
        rate_columns = ['fed_funds_rate', 'treasury_10y', 'treasury_2y']
        for col in rate_columns:
            if col in data.columns:
                negative_rates = data[data[col] < 0]
                if len(negative_rates) > 0:
                    results['anomalies'][f'{col}_negative'] = len(negative_rates)
        
        # Unemployment rate checks
        if 'unemployment_rate' in data.columns:
            # Very high unemployment (>20%)
            high_unemployment = data[data['unemployment_rate'] > 20]
            results['anomalies']['high_unemployment'] = len(high_unemployment)
            
            # Very low unemployment (<2%)
            low_unemployment = data[data['unemployment_rate'] < 2]
            results['anomalies']['low_unemployment'] = len(low_unemployment)
        
        # VIX checks (volatility index)
        if 'vix' in data.columns:
            # Extremely high VIX (>50, indicates panic)
            high_vix = data[data['vix'] > 50]
            results['anomalies']['high_vix_days'] = len(high_vix)
        
        logger.info(f"Economic data validation: {len(data.columns)} indicators, {len(results['errors'])} errors")
        return results
    
    def validate_relationships(self, data: pd.DataFrame) -> Dict:
        """
        Validate company relationship data
        
        Args:
            data: DataFrame with relationship data
            
        Returns:
            Dictionary with validation results
        """
        results = {
            'total_relationships': len(data),
            'companies': data['company_symbol'].nunique() if 'company_symbol' in data.columns else 0,
            'relationship_types': [],
            'missing_data': {},
            'errors': []
        }
        
        if data.empty:
            results['errors'].append("Empty relationships dataset")
            return results
        
        # Check relationship types
        if 'relationship_type' in data.columns:
            results['relationship_types'] = data['relationship_type'].value_counts().to_dict()
        
        # Check for missing data
        required_columns = ['company_symbol', 'relationship_type', 'related_entity']
        for col in required_columns:
            if col in data.columns:
                missing_count = data[col].isna().sum()
                results['missing_data'][col] = int(missing_count)
            else:
                results['errors'].append(f"Missing required column: {col}")
        
        # Check for self-relationships (company related to itself)
        if 'company_symbol' in data.columns and 'related_entity' in data.columns:
            # This is tricky because related_entity might be company names, not symbols
            # For now, just check if they're exactly equal
            self_relations = data[data['company_symbol'] == data['related_entity']]
            if len(self_relations) > 0:
                results['anomalies'] = {'self_relationships': len(self_relations)}
        
        logger.info(f"Relationship validation: {results['total_relationships']} relationships, {len(results['errors'])} errors")
        return results
    
    def _detect_frequency(self, date_index) -> str:
        """Detect the frequency of a time series"""
        if len(date_index) < 2:
            return "unknown"
        
        # Calculate typical gap between dates
        gaps = date_index[1:] - date_index[:-1]
        median_gap = gaps.median()
        
        if median_gap.days <= 1:
            return "daily"
        elif median_gap.days <= 7:
            return "weekly"
        elif median_gap.days <= 31:
            return "monthly"
        elif median_gap.days <= 95:
            return "quarterly"
        else:
            return "annual"
    
    def generate_validation_report(self, 
                                 stock_data: pd.DataFrame = None,
                                 economic_data: pd.DataFrame = None,
                                 relationship_data: pd.DataFrame = None) -> str:
        """
        Generate comprehensive validation report
        
        Returns:
            Formatted validation report string
        """
        report = ["=" * 50]
        report.append("FINOGRAPH DATA VALIDATION REPORT")
        report.append("=" * 50)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Stock data validation
        if stock_data is not None:
            stock_results = self.validate_stock_data(stock_data)
            report.append("📊 STOCK DATA VALIDATION")
            report.append("-" * 30)
            report.append(f"Total records: {stock_results['total_records']:,}")
            report.append(f"Companies: {stock_results['symbols']}")
            
            if stock_results['date_range']:
                report.append(f"Date range: {stock_results['date_range']['start']} to {stock_results['date_range']['end']}")
            
            # Missing data summary
            if stock_results['missing_data']:
                report.append("\nMissing Data:")
                for col, info in stock_results['missing_data'].items():
                    report.append(f"  {col}: {info['count']} records ({info['percentage']}%)")
            
            # Outliers summary
            if stock_results['outliers']:
                report.append("\nOutliers Detected:")
                for outlier_type, count in stock_results['outliers'].items():
                    if count > 0:
                        report.append(f"  {outlier_type}: {count} records")
            
            if stock_results['errors']:
                report.append(f"\n❌ Errors: {len(stock_results['errors'])}")
                for error in stock_results['errors']:
                    report.append(f"  - {error}")
            else:
                report.append("\n✅ No critical errors found")
            
            report.append("")
        
        # Economic data validation
        if economic_data is not None:
            econ_results = self.validate_economic_data(economic_data)
            report.append("🏛️ ECONOMIC DATA VALIDATION")
            report.append("-" * 30)
            report.append(f"Total records: {econ_results['total_records']:,}")
            report.append(f"Indicators: {len(econ_results['indicators'])}")
            
            if econ_results['date_range']:
                report.append(f"Date range: {econ_results['date_range']['start']} to {econ_results['date_range']['end']}")
                report.append(f"Frequency: {econ_results['date_range']['frequency']}")
            
            # High missing data indicators
            high_missing = {k: v for k, v in econ_results['missing_data'].items() if v['percentage'] > 10}
            if high_missing:
                report.append("\nIndicators with >10% missing data:")
                for indicator, info in high_missing.items():
                    report.append(f"  {indicator}: {info['percentage']}%")
            
            if econ_results['errors']:
                report.append(f"\n❌ Errors: {len(econ_results['errors'])}")
            else:
                report.append("\n✅ No critical errors found")
            
            report.append("")
        
        # Relationship data validation
        if relationship_data is not None:
            rel_results = self.validate_relationships(relationship_data)
            report.append("🏢 RELATIONSHIP DATA VALIDATION")
            report.append("-" * 30)
            report.append(f"Total relationships: {rel_results['total_relationships']:,}")
            report.append(f"Companies: {rel_results['companies']}")
            
            if rel_results['relationship_types']:
                report.append("\nRelationship Types:")
                for rel_type, count in rel_results['relationship_types'].items():
                    report.append(f"  {rel_type}: {count}")
            
            if rel_results['errors']:
                report.append(f"\n❌ Errors: {len(rel_results['errors'])}")
            else:
                report.append("\n✅ No critical errors found")
        
        report.append("=" * 50)
        return "\n".join(report)

# Test function
def test_data_validator():
    """Test data validation framework"""
    validator = DataValidator()
    
    # Create sample data for testing
    print("🧪 Creating sample data for validation testing...")
    
    # Sample stock data
    dates = pd.date_range('2024-01-01', '2024-09-01', freq='D')
    stock_data = pd.DataFrame({
        'Open': np.random.uniform(100, 200, len(dates)),
        'High': np.random.uniform(200, 220, len(dates)),
        'Low': np.random.uniform(80, 100, len(dates)),
        'Close': np.random.uniform(100, 200, len(dates)),
        'Volume': np.random.randint(1000000, 10000000, len(dates)),
        'Symbol': 'TEST'
    }, index=dates)
    
    # Add some missing data and outliers for testing
    stock_data.loc[stock_data.index[10:15], 'Close'] = np.nan  # Missing data
    stock_data.loc[stock_data.index[20], 'Close'] = -5  # Invalid price
    
    # Sample economic data
    econ_data = pd.DataFrame({
        'fed_funds_rate': np.random.uniform(0, 5, len(dates)),
        'unemployment_rate': np.random.uniform(3, 8, len(dates)),
        'vix': np.random.uniform(10, 30, len(dates))
    }, index=dates)
    
    # Add some anomalies
    econ_data.loc[econ_data.index[30], 'unemployment_rate'] = 25  # Very high
    econ_data.loc[econ_data.index[40], 'vix'] = 75  # Panic level
    
    # Sample relationship data
    rel_data = pd.DataFrame({
        'company_symbol': ['AAPL', 'MSFT', 'GOOGL'],
        'relationship_type': ['supplier', 'partner', 'customer'],
        'related_entity': ['Samsung', 'Intel', 'Advertisers']
    })
    
    # Test validation
    print("\n🧪 Testing stock data validation...")
    stock_results = validator.validate_stock_data(stock_data)
    print(f"✅ Stock validation completed: {stock_results['total_records']} records")
    
    print("\n🧪 Testing economic data validation...")
    econ_results = validator.validate_economic_data(econ_data)
    print(f"✅ Economic validation completed: {len(econ_results['indicators'])} indicators")
    
    print("\n🧪 Testing relationship validation...")
    rel_results = validator.validate_relationships(rel_data)
    print(f"✅ Relationship validation completed: {rel_results['total_relationships']} relationships")
    
    # Generate full report
    print("\n📋 Generating validation report...")
    report = validator.generate_validation_report(stock_data, econ_data, rel_data)
    print(report)

if __name__ == "__main__":
    test_data_validator()