import torch
import torch_geometric
import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt

print("🔥 Environment Test Results:")
print(f"✅ Python: {torch.__version__}")
print(f"✅ PyTorch: {torch.__version__}")
print(f"✅ PyTorch Geometric: {torch_geometric.__version__}")
print(f"✅ CUDA Available: {torch.cuda.is_available()}")
print(f"✅ Pandas: {pd.__version__}")

# Quick test: Download Apple stock data
try:
    aapl = yf.download("AAPL", start="2024-01-01", end="2024-02-01")
    print(f"✅ Yahoo Finance: Downloaded {len(aapl)} days of AAPL data")
except:
    print("❌ Yahoo Finance: Error - check internet connection")

print("\n🚀 Environment setup complete!")