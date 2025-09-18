# FinGraph Production System

## 🚨 CRITICAL: This is REAL Quantitative Trading Infrastructure

This is not a toy project. This system:
- ✅ Uses REAL Graph Neural Networks with message passing
- ✅ Guarantees ZERO lookahead bias
- ✅ Includes realistic transaction costs (15bps total)
- ✅ Validates temporal integrity at every step
- ✅ Bootstraps confidence intervals for all metrics
- ✅ Compares against proper baselines

## 📋 System Requirements

```bash
# Core dependencies
python >= 3.8
pytorch >= 2.0
torch-geometric >= 2.3
yfinance >= 0.2
pandas >= 1.5
numpy >= 1.21
scikit-learn >= 1.2
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Complete Pipeline
```bash
# Run everything with default configuration
python main_trainer.py

# Run with custom symbols
python main_trainer.py --symbols AAPL MSFT GOOGL AMZN TSLA

# Run with custom date range
python main_trainer.py --start-date 2020-01-01 --end-date 2024-01-01

# Skip certain steps (for debugging)
python main_trainer.py --skip-validation  # NOT RECOMMENDED
python main_trainer.py --skip-backtest    # Skip backtesting
```

### 3. Check Results
```bash
# Results are saved in timestamped directory
ls artifacts/YYYYMMDD_HHMMSS/

# Key files:
# - pipeline_results.json: Complete metrics
# - backtest_results.json: Trading performance
# - models/production/latest_model.pt: Trained model
```

## 🏗️ Architecture Overview

### Data Pipeline (`data_pipeline_quant.py`)
- **Purpose**: Load market data with ZERO lookahead bias
- **Key Features**:
  - Point-in-time feature engineering
  - Forward-looking labels (T+1 to T+21)
  - Cryptographic checksums for data integrity
  - Temporal sample validation

### Integrity Checker (`integrity_checker.py`)
- **Purpose**: Validate everything is correct
- **Validates**:
  - No temporal leakage
  - Proper train/val/test separation
  - Statistical validity of predictions
  - Model not memorizing

### GNN Trainer (`gnn_trainer.py`)
- **Purpose**: Train REAL Graph Neural Networks
- **Architecture**:
  - Graph Attention Networks (GAT)
  - Multi-head attention (8 heads)
  - 3 layers of message passing
  - Multi-task learning (risk, volatility, returns)

### Backtester (`quant_backtester.py`)
- **Purpose**: Realistic trading simulation
- **Includes**:
  - 10bps commission + 5bps slippage
  - Stop loss and take profit
  - Position sizing and risk management
  - Bootstrap confidence intervals

### Inference Engine (`inference_engine.py`)
- **Purpose**: Real-time predictions
- **Features**:
  - Live market data ingestion
  - Dynamic graph construction
  - Caching and optimization
  - Confidence intervals via dropout

## 📊 Performance Metrics

### What to Expect (Realistic)
```
Sharpe Ratio: 0.8 - 1.5 (Good)
Max Drawdown: 15-25% (Acceptable)
Win Rate: 45-55% (Normal)
Correlation: 0.15-0.30 (Significant)
```

### Red Flags (Something Wrong)
```
Sharpe Ratio > 3.0: Likely overfitting or lookahead
Max Drawdown < 5%: Unrealistic, check for errors
Win Rate > 70%: Probably has future information
Correlation > 0.6: Model memorizing data
```

## 🔍 Validation Checklist

### Before Deployment
- [ ] Temporal integrity passed
- [ ] No information leakage detected
- [ ] Train/val/test properly separated
- [ ] Transaction costs applied
- [ ] Baseline comparison done
- [ ] Bootstrap confidence intervals calculated
- [ ] Backtest on out-of-sample period

### Daily Monitoring
- [ ] Check inference latency < 1s
- [ ] Verify cache hit rate > 50%
- [ ] Monitor prediction distribution
- [ ] Track realized vs predicted performance
- [ ] Review risk exposure

## 🛠️ Configuration

### Edit `configs/experiment.yaml`

```yaml
# Key parameters to tune
data:
  symbols: [...]           # Your universe
  label_horizon: 21        # Prediction horizon
  
model:
  hidden_dim: 128         # Model capacity
  num_layers: 3           # Graph depth
  dropout: 0.2            # Regularization
  
backtest:
  commission_bps: 10      # Your broker's fees
  slippage_bps: 5         # Market impact
  position_size: 0.02     # Risk per trade
```

## 🚨 Common Issues and Solutions

### Issue: "Temporal integrity violation"
**Solution**: Features are using future data. Check feature engineering.

### Issue: "Correlation too high between features and labels"
**Solution**: Information leakage. Review data pipeline.

### Issue: "Sharpe ratio unrealistic (>3)"
**Solution**: Overfitting or lookahead bias. Increase validation.

### Issue: "No edges in graph"
**Solution**: Correlation threshold too high. Lower to 0.2.

### Issue: "Out of memory"
**Solution**: Reduce batch_size or num_layers.

## 📈 Production Deployment

### 1. Train Production Model
```bash
python main_trainer.py --config configs/production.yaml
```

### 2. Run Inference API
```bash
uvicorn inference_engine:app --host 0.0.0.0 --port 8000
```

### 3. Test API
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"symbols": ["AAPL", "MSFT"], "lookback_days": 60}'
```

## 📊 Interpreting Results

### Risk Scores
- `0.0 - 0.3`: Low risk
- `0.3 - 0.7`: Medium risk  
- `0.7 - 1.0`: High risk

### Trading Signals
- `signal = expected_return - risk_score`
- Positive signal: Consider long position
- Negative signal: Consider short position
- Near zero: No position

## 🔬 Advanced Usage

### Custom Feature Engineering
```python
# Add your features in data_pipeline_quant.py
def create_point_in_time_features():
    # YOUR FEATURES HERE
    # Remember: NO FUTURE INFORMATION
```

### Custom Graph Construction
```python
# Modify in gnn_trainer.py
def build_temporal_graph():
    # Add sector relationships
    # Add economic indicators
    # Add custom edges
```

### Ensemble Models
```python
# Combine multiple models
predictions = [
    gnn_model.predict(data),
    rf_model.predict(data),
    lstm_model.predict(data)
]
final = np.mean(predictions, axis=0)
```

## 🎯 Performance Optimization

### Speed Improvements
1. Enable caching: `use_cache: true`
2. Reduce graph size: Increase `correlation_threshold`
3. Use GPU: `device: cuda`
4. Batch predictions: Use `batch_predict`

### Accuracy Improvements
1. More data: Increase `lookback_days`
2. More features: Add technical indicators
3. Larger model: Increase `hidden_dim`
4. More layers: Increase `num_layers`

## 📝 Citation

If you use this system in research:
```bibtex
@software{fingraph2024,
  title={FinGraph: Production Graph Neural Networks for Financial Risk},
  author={Your Team},
  year={2024},
  url={https://github.com/yourrepo/fingraph}
}
```

## ⚠️ Legal Disclaimer

**IMPORTANT**: This system is for research and educational purposes. 
- Past performance does not indicate future results
- Trading involves substantial risk of loss
- Always validate results before deploying capital
- Comply with all regulatory requirements

## 🔒 Security Notes

1. **Never commit API keys**: Use environment variables
2. **Validate all inputs**: Prevent injection attacks
3. **Rate limit API**: Prevent abuse
4. **Monitor for anomalies**: Set up alerts
5. **Regular audits**: Check for data/model drift

## 📞 Support

For issues:
1. Check validation reports in `artifacts/*/validation_*.json`
2. Review logs in `logs/training.log`
3. Verify data integrity with checksums
4. Run integrity checker separately

## 🎓 Key Takeaways

1. **Temporal Integrity is Everything**: One lookahead bias ruins everything
2. **Transaction Costs Matter**: 15bps can turn profit into loss
3. **Real GNNs Use Message Passing**: Not just MLPs
4. **Bootstrap Everything**: Point estimates lie
5. **Compare with Baselines**: Always benchmark against buy-and-hold

## ✅ Final Checklist

Before going live with real money:
- [ ] Backtested on 5+ years of data
- [ ] Out-of-sample test period profitable
- [ ] Sharpe ratio > 0.5 after costs
- [ ] Maximum drawdown < 30%
- [ ] Passed all integrity checks
- [ ] Inference latency < 1 second
- [ ] Risk management rules defined
- [ ] Stop losses implemented
- [ ] Position sizing appropriate
- [ ] Legal/compliance approved

---