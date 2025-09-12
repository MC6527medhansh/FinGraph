"""
Debug Script for Temporal Issue

This will help to see exactly what's going wrong with the standalone test
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import logging

# Setup detailed logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def debug_data_structure():
    """Debug the data structure differences"""
    print("🔍 DEBUGGING DATA STRUCTURE DIFFERENCES")
    print("="*60)
    
    # 1. Download fresh Yahoo Finance data (like standalone test)
    print("\n📊 Fresh Yahoo Finance Data:")
    print("-"*30)
    
    try:
        fresh_data = yf.download('AAPL', start='2022-01-01', end='2024-09-01', progress=False)
        fresh_data['Symbol'] = 'AAPL'
        
        print(f"Shape: {fresh_data.shape}")
        print(f"Columns: {list(fresh_data.columns)}")
        print(f"Index type: {type(fresh_data.index)}")
        print(f"Index name: {fresh_data.index.name}")
        print(f"Date range: {fresh_data.index.min()} to {fresh_data.index.max()}")
        print(f"Sample data:")
        print(fresh_data.head(3))
        print(f"Data types:")
        print(fresh_data.dtypes)
        
    except Exception as e:
        print(f"❌ Error downloading fresh data: {e}")
        return False
    
    # 2. Load existing FinGraph data (like integration test)  
    print("\n📊 Existing FinGraph Data:")
    print("-"*30)
    
    try:
        from src.features.graph_data_loader import GraphDataLoader
        loader = GraphDataLoader()
        existing_data = loader.load_latest_data()
        stock_data = existing_data['stock_data']
        
        print(f"Shape: {stock_data.shape}")
        print(f"Columns: {list(stock_data.columns)}")
        print(f"Index type: {type(stock_data.index)}")
        print(f"Index name: {stock_data.index.name}")
        print(f"Date range: {stock_data.index.min()} to {stock_data.index.max()}")
        print(f"Sample data:")
        print(stock_data.head(3))
        print(f"Data types:")
        print(stock_data.dtypes)
        
    except Exception as e:
        print(f"❌ Error loading existing data: {e}")
        return False
    
    # 3. Compare the structures
    print("\n🔍 STRUCTURE COMPARISON:")
    print("-"*30)
    
    fresh_cols = set(fresh_data.columns)
    existing_cols = set(stock_data.columns)
    
    print(f"Fresh data columns: {sorted(fresh_cols)}")
    print(f"Existing data columns: {sorted(existing_cols)}")
    print(f"Columns only in fresh: {sorted(fresh_cols - existing_cols)}")
    print(f"Columns only in existing: {sorted(existing_cols - fresh_cols)}")
    print(f"Common columns: {sorted(fresh_cols & existing_cols)}")
    
    return fresh_data, stock_data

def debug_feature_extraction(fresh_data, stock_data):
    """Debug the feature extraction process"""
    print("\n🔧 DEBUGGING FEATURE EXTRACTION")
    print("="*50)
    
    from src.models.temporal_risk_predictor import TemporalFeatureExtractor
    extractor = TemporalFeatureExtractor()
    
    # Test on fresh data
    print("\n📊 Testing Fresh Data:")
    print("-"*25)
    
    # Get a sample date from fresh data
    fresh_dates = sorted(fresh_data.index.unique())
    test_date = fresh_dates[30]  # Use 30th date to ensure history
    
    print(f"Test date: {test_date}")
    print(f"Historical data available: {len(fresh_data[fresh_data.index < test_date])}")
    
    try:
        fresh_features = extractor.extract_daily_features(fresh_data, test_date, 'AAPL')
        print(f"Fresh features result: {fresh_features}")
        if fresh_features is not None:
            print(f"Feature shape: {fresh_features.shape}")
            print(f"Feature values: {fresh_features}")
        else:
            print("❌ Fresh features extraction failed")
            
            # Debug step by step
            print("\n🔍 Step-by-step debugging:")
            historical = fresh_data[
                (fresh_data['Symbol'] == 'AAPL') & 
                (fresh_data.index < test_date)
            ].copy()
            print(f"Historical data shape: {historical.shape}")
            print(f"Required columns present:")
            required_cols = ['Close', 'Volume', 'Open', 'High', 'Low']
            for col in required_cols:
                present = col in historical.columns
                print(f"  {col}: {present}")
                if present:
                    print(f"    Sample values: {historical[col].head(3).tolist()}")
                    print(f"    Any NaN?: {historical[col].isna().any()}")
            
    except Exception as e:
        print(f"❌ Fresh feature extraction error: {e}")
        import traceback
        traceback.print_exc()
    
    # Test on existing data
    print("\n📊 Testing Existing Data:")
    print("-"*25)
    
    # Get a sample date from existing data  
    existing_dates = sorted(stock_data.index.unique())
    if len(existing_dates) > 30:
        test_date_existing = existing_dates[30]
        
        print(f"Test date: {test_date_existing}")
        print(f"Historical data available: {len(stock_data[(stock_data['Symbol'] == 'AAPL') & (stock_data.index < test_date_existing)])}")
        
        try:
            existing_features = extractor.extract_daily_features(stock_data, test_date_existing, 'AAPL')
            print(f"Existing features result: {existing_features}")
            if existing_features is not None:
                print(f"Feature shape: {existing_features.shape}")
                print(f"Feature values: {existing_features}")
            else:
                print("❌ Existing features extraction failed")
        except Exception as e:
            print(f"❌ Existing feature extraction error: {e}")

def debug_risk_calculation(fresh_data, stock_data):
    """Debug the risk calculation process"""
    print("\n📈 DEBUGGING RISK CALCULATION")
    print("="*40)
    
    from src.models.temporal_risk_predictor import RiskLabelCalculator
    calculator = RiskLabelCalculator()
    
    # Test on fresh data
    print("\n📊 Testing Fresh Data Risk Calculation:")
    print("-"*40)
    
    fresh_dates = sorted(fresh_data.index.unique())
    if len(fresh_dates) > 50:
        test_date = fresh_dates[30]
        
        print(f"Test date: {test_date}")
        print(f"Looking for future data from: {test_date}")
        
        future_data = fresh_data[
            (fresh_data['Symbol'] == 'AAPL') & 
            (fresh_data.index > test_date) &
            (fresh_data.index <= test_date + timedelta(days=25))
        ]
        print(f"Future data available: {len(future_data)} records")
        
        if len(future_data) > 0:
            print(f"Future date range: {future_data.index.min()} to {future_data.index.max()}")
        
        try:
            risk_result = calculator.calculate_forward_risk(fresh_data, test_date, 'AAPL', 21)
            print(f"Fresh risk result: {risk_result}")
        except Exception as e:
            print(f"❌ Fresh risk calculation error: {e}")
            import traceback
            traceback.print_exc()

def main():
    """Main debug function"""
    print("🐛 TEMPORAL ISSUE DEBUGGER")
    print("="*60)
    
    # Step 1: Compare data structures
    result = debug_data_structure()
    if not result:
        print("❌ Could not load data for comparison")
        return
    
    fresh_data, stock_data = result
    
    # Step 2: Debug feature extraction
    debug_feature_extraction(fresh_data, stock_data)
    
    # Step 3: Debug risk calculation  
    debug_risk_calculation(fresh_data, stock_data)
    
    print("\n🎯 DEBUG COMPLETE")
    print("Check the output above to identify the issue")

if __name__ == "__main__":
    main()