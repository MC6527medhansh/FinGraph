#!/usr/bin/env python
"""
Monitor and validate signal quality
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json


def analyze_signal_distribution():
    """Analyze the distribution and quality of signals"""
    
    signals_path = Path('data/signals/latest_signals.csv')
    if not signals_path.exists():
        print("❌ No signals found. Run generate_signals.py first.")
        return None
    
    signals = pd.read_csv(signals_path)
    
    if signals.empty:
        print("❌ Signals file is empty")
        return None
    
    print("\n" + "="*60)
    print("SIGNAL DISTRIBUTION ANALYSIS")
    print("="*60)
    
    # Date of signals
    print(f"\nSignal Date: {signals['date'].iloc[0]}")
    print(f"Total Stocks: {len(signals)}")
    
    # Risk distribution
    print(f"\nRisk Scores:")
    print(f"  Mean: {signals['risk_score'].mean():.4f}")
    print(f"  Std:  {signals['risk_score'].std():.4f}")
    print(f"  Min:  {signals['risk_score'].min():.4f}")
    print(f"  Max:  {signals['risk_score'].max():.4f}")
    
    # Return forecast distribution
    print(f"\nReturn Forecasts:")
    print(f"  Mean: {signals['return_forecast'].mean():.4f}")
    print(f"  Std:  {signals['return_forecast'].std():.4f}")
    print(f"  Min:  {signals['return_forecast'].min():.4f}")
    print(f"  Max:  {signals['return_forecast'].max():.4f}")
    
    # Volatility forecast
    print(f"\nVolatility Forecasts:")
    print(f"  Mean: {signals['volatility_forecast'].mean():.4f}")
    print(f"  Std:  {signals['volatility_forecast'].std():.4f}")
    
    # Recommendations breakdown
    print(f"\nRecommendations:")
    rec_counts = signals['recommendation'].value_counts()
    for rec, count in rec_counts.items():
        print(f"  {rec}: {count}")
    
    # Check for diversity
    unique_risks = signals['risk_score'].nunique()
    unique_returns = signals['return_forecast'].nunique()
    
    print(f"\nSignal Diversity:")
    print(f"  Unique risk scores: {unique_risks}/{len(signals)}")
    print(f"  Unique return forecasts: {unique_returns}/{len(signals)}")
    
    # Quality checks
    print(f"\nQuality Checks:")
    if unique_risks < len(signals) * 0.5:
        print("  ⚠️ WARNING: Low diversity in risk scores")
    else:
        print("  ✅ Good diversity in risk scores")
        
    if unique_returns < len(signals) * 0.5:
        print("  ⚠️ WARNING: Low diversity in return forecasts")
    else:
        print("  ✅ Good diversity in return forecasts")
    
    if signals['risk_score'].std() < 0.01:
        print("  ⚠️ WARNING: Risk scores have very low variance")
    else:
        print("  ✅ Risk scores show reasonable variance")
    
    # Top recommendations
    print(f"\nTop 3 Recommendations:")
    top_3 = signals.nsmallest(3, 'rank')[['symbol', 'recommendation', 'signal_strength']]
    for idx, row in top_3.iterrows():
        print(f"  {row['symbol']}: {row['recommendation']} (strength: {row['signal_strength']:.4f})")
    
    return signals


if __name__ == "__main__":
    analyze_signal_distribution()