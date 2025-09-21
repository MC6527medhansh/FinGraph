#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Debug negative correlations in model predictions
THIS IS CRITICAL - Model currently predicts opposite of reality
"""

import sys
from pathlib import Path

# Ensure project root on path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  # optional, but kept for parity
from torch_geometric.loader import DataLoader
from collections.abc import Iterable
import yaml

# Import components
from src.models.gnn_model import FinancialGNN
from src.pipeline.graph_builder import TemporalGraphBuilder


# -------------------------------
# Utility helpers (robust casting)
# -------------------------------
def to_1d_list(x):
    """
    Robustly convert x (tensor/np array/list/scalar/None) to a 1-D Python list.
    - Handles 0-D scalars and (B,1)/(1,B) shapes.
    - Detaches tensors and moves to CPU.
    - Returns [] for None.
    """
    if x is None:
        return []
    # Torch tensor
    if 'torch' in str(type(x)):
        try:
            x = x.detach().cpu().numpy()
        except Exception:
            # If it's already numpy-like or not a tensor, continue below
            pass
    # Numpy array or scalar
    if isinstance(x, np.ndarray):
        return np.ravel(x).tolist()
    # Python iterables (lists/tuples, etc.), but not strings/bytes
    if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
        try:
            return np.ravel(np.array(list(x), dtype=object)).tolist()
        except Exception:
            return list(x)
    # Fallback scalar
    return [x]


def safe_corrcoef(x, y):
    """
    Safe Pearson correlation for 1D arrays; returns np.nan if not computable.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 2 or y.size < 2:
        return np.nan
    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def debug_correlations():
    print("=" * 60)
    print("🔍 DEBUGGING NEGATIVE CORRELATIONS")
    print("=" * 60)

    # -----------------------
    # Locate and load model
    # -----------------------
    model_dir = Path('data/models')
    model_files = sorted(model_dir.glob('*.pt'), key=lambda p: p.stat().st_mtime)
    if not model_files:
        print("❌ No model checkpoint found in data/models/*.pt")
        return
    latest_model = model_files[-1]

    # PyTorch >= 2.6 default is weights_only=True (safer). We stored a dict, so allow full unpickling here.
    try:
        checkpoint = torch.load(latest_model, map_location='cpu', weights_only=False)
    except TypeError:
        # For older PyTorch that doesn't accept weights_only kwarg
        checkpoint = torch.load(latest_model, map_location='cpu')
    except Exception as e:
        print(f"❌ Failed to load checkpoint {latest_model.name}: {e}")
        return

    # -----------------------
    # Load features parquet
    # -----------------------
    feat_path = Path('data/processed/features_latest.parquet')
    if not feat_path.exists():
        print(f"❌ Missing features parquet: {feat_path}")
        return

    features = pd.read_parquet(feat_path)

    print("\n1. CHECKING LABEL DISTRIBUTION:")
    print("-" * 40)
    for col in ['risk_score', 'forward_return', 'forward_volatility']:
        if col in features.columns:
            print(f"{col.replace('_', ' ').title()} range: "
                  f"[{features[col].min():.3f}, {features[col].max():.3f}]")
            print(f"{col.replace('_', ' ').title()} mean: {features[col].mean():.3f}")
        else:
            print(f"⚠️ Column missing in features: {col}")

    print("\n2. CHECKING FEATURE-LABEL RELATIONSHIPS:")
    print("-" * 40)
    # Guard against missing columns
    if 'volatility_20d' in features.columns and 'risk_score' in features.columns:
        vol_risk_corr = features['volatility_20d'].corr(features['risk_score'])
        print(f"Volatility → Risk correlation: {vol_risk_corr:.3f} (should be positive)")
    else:
        vol_risk_corr = np.nan
        print("⚠️ Cannot compute Volatility → Risk correlation (missing columns).")

    if 'return_20d' in features.columns and 'risk_score' in features.columns:
        return_risk_corr = features['return_20d'].corr(features['risk_score'])
        print(f"Return → Risk correlation: {return_risk_corr:.3f} (should be negative)")
    else:
        return_risk_corr = np.nan
        print("⚠️ Cannot compute Return → Risk correlation (missing columns).")

    print("\n3. VERIFYING LABEL CALCULATION:")
    print("-" * 40)
    # Sample and recompute expected risk (if columns exist)
    needed = {'symbol', 'date', 'volatility_20d', 'forward_volatility', 'risk_score', 'forward_return', 'forward_max_drawdown'}
    if needed.issubset(features.columns):
        sample = features.sample(min(5, len(features)), random_state=42)
        for idx, row in sample.iterrows():
            print(f"\nSymbol: {row['symbol']}, Date: {row['date']}")
            print(f"  Historical volatility: {row['volatility_20d']:.3f}")
            print(f"  Forward volatility: {row['forward_volatility']:.3f}")
            print(f"  Risk score: {row['risk_score']:.3f}")
            print(f"  Forward return: {row['forward_return']:.3f}")

            expected_risk = min(row['forward_volatility'] / 0.5, 1.0) * 0.6 + \
                            min(row['forward_max_drawdown'] / 0.2, 1.0) * 0.4
            print(f"  Expected risk (recalculated): {expected_risk:.3f}")

            if abs(expected_risk - row['risk_score']) > 0.1:
                print("  ⚠️ MISMATCH! Risk calculation might be wrong")
    else:
        print("⚠️ Skipping per-row verification (missing columns for recomputation).")

    print("\n4. CHECKING MODEL PREDICTIONS:")
    print("-" * 40)

    # -----------------------
    # Build graphs from config
    # -----------------------
    cfg_path = Path('config/pipeline_config.yaml')
    if not cfg_path.exists():
        print(f"❌ Missing config: {cfg_path}")
        return
    with open(cfg_path, 'r') as f:
        config = yaml.safe_load(f)

    graph_builder = TemporalGraphBuilder(config)
    graphs = graph_builder.create_temporal_graphs(features, max_graphs=10)

    if not graphs:
        print("❌ Graph builder returned no graphs.")
        return

    # -----------------------
    # Init model and load weights
    # -----------------------
    model = FinancialGNN(
        num_node_features=graphs[0].x.shape[1],
        hidden_dim=config['model']['hidden_dim'],
        num_heads=config['model']['num_heads'],
        num_layers=config['model']['num_layers'],
        dropout=config['model']['dropout']
    )

    # Accept either raw state_dict or checkpoint dict with 'model_state_dict'
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif isinstance(checkpoint, dict):
        # Best effort: try to load the whole dict as weights
        try:
            model.load_state_dict(checkpoint)
        except Exception as e:
            print(f"❌ Could not load state_dict from checkpoint keys: {list(checkpoint.keys())[:5]}... Error: {e}")
            return
    else:
        print("❌ Unexpected checkpoint type; expected dict.")
        return

    model.eval()

    # -----------------------
    # Predict on a few graphs
    # -----------------------
    test_loader = DataLoader(graphs[-5:], batch_size=1, shuffle=False)

    predictions = []
    actuals = []

    with torch.no_grad():
        for batch in test_loader:
            outputs = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

            # Accept dict or tensor output
            out = outputs.get('risk') if isinstance(outputs, dict) else outputs

            pred_list = to_1d_list(out)
            # Labels may be in y_risk or y; try y_risk first
            act = getattr(batch, 'y_risk', None)
            act_list = to_1d_list(act)
            if not act_list:  # fallback
                act_list = to_1d_list(getattr(batch, 'y', None))

            # Align lengths (per-graph vs per-node cases)
            n = min(len(pred_list), len(act_list))
            if n == 0:
                print("⚠️ Skipping a batch with no comparable predictions/labels.")
                continue

            pred_list = pred_list[:n]
            act_list = act_list[:n]

            predictions.extend(pred_list)
            actuals.extend(act_list)

            print(f"\nBatch predictions:")
            print(f"  Predicted risk: {pred_list[:3]} ... (n={len(pred_list)})")
            print(f"  Actual risk:    {act_list[:3]} ... (n={len(act_list)})")

    # -----------------------
    # Visualization & metrics
    # -----------------------
    if len(predictions) < 2 or len(actuals) < 2:
        print("\n⚠️ Not enough data to compute correlation or plot meaningfully.")
        return

    predictions_arr = np.asarray(predictions, dtype=float)
    actuals_arr = np.asarray(actuals, dtype=float)

    # Clean NaNs/infs
    valid = np.isfinite(predictions_arr) & np.isfinite(actuals_arr)
    predictions_arr = predictions_arr[valid]
    actuals_arr = actuals_arr[valid]

    if len(predictions_arr) < 2:
        print("\n⚠️ Not enough finite pairs after cleaning for correlation/plots.")
        return

    std_pred = float(np.std(predictions_arr))
    std_act = float(np.std(actuals_arr))
    corr = safe_corrcoef(actuals_arr, predictions_arr)

    # ---- Plotting (safe) ----
    plt.figure(figsize=(10, 4))

    # Scatter
    plt.subplot(1, 2, 1)
    plt.scatter(actuals_arr, predictions_arr, alpha=0.5)
    # Reference y=x on [0,1] domain if applicable
    try:
        lo = float(np.nanmin([actuals_arr.min(), predictions_arr.min(), 0.0]))
        hi = float(np.nanmax([actuals_arr.max(), predictions_arr.max(), 1.0]))
    except Exception:
        lo, hi = 0.0, 1.0
    plt.plot([lo, hi], [lo, hi], 'r--', label='Perfect prediction')
    plt.xlabel('Actual Risk')
    plt.ylabel('Predicted Risk')
    plt.title(f'Risk Predictions (corr={corr:.3f} | n={len(actuals_arr)})')
    plt.legend()

    # Histograms
    plt.subplot(1, 2, 2)
    plt.hist(predictions_arr, bins=20, alpha=0.5, label='Predictions')
    plt.hist(actuals_arr, bins=20, alpha=0.5, label='Actuals')
    plt.xlabel('Risk Score')
    plt.ylabel('Frequency')
    plt.title('Distribution of Risk Scores')
    plt.legend()

    plt.tight_layout()
    out_png = 'debug_correlations.png'
    plt.savefig(out_png)
    print(f"\n✅ Saved visualization to {out_png}")

    # -----------------------
    # Diagnostics
    # -----------------------
    print("\n" + "=" * 60)
    print("🔬 DIAGNOSIS:")
    if np.isnan(corr):
        print("❌ Cannot compute correlation (zero variance or insufficient data).")
    else:
        print(f"Correlation (Pred vs Actual): {corr:.3f}")

    if std_pred < 1e-3:
        print("❌ PROBLEM: Model predictions have near-zero variance.")
        print("   FIX: Check loss scaling, target scaling/normalization, dropout, or learning rate.")

    if std_act < 1e-3:
        print("⚠️ Actual risk has near-zero variance in this sample (check label window, symbol mix, or filtering).")

    if not np.isnan(vol_risk_corr) and vol_risk_corr < 0:
        print("❌ PROBLEM: Volatility-Risk correlation is negative!")
        print("   FIX: Check risk_score calculation in feature engineering.")

    # Inversion hint
    if not np.isnan(corr) and corr < 0:
        print("❌ PROBLEM: Predictions are inverted vs. labels.")
        print("   LIKELY CAUSE: Label sign/definition or loss target misalignment.")
        print("   FIX: Verify _calculate_forward_labels and loss target direction.")

    print("=" * 60)


if __name__ == "__main__":
    debug_correlations()
