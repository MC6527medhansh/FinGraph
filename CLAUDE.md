# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FinGraph is a production Graph Neural Network (GNN) system for financial risk prediction and trading signal generation. It uses Graph Attention Networks (GAT) to model relationships between stocks and predict risk scores, returns, and volatility.

All code lives inside `fingraph-project/`. Scripts must be run from that directory.

## Common Commands

All commands should be run from `fingraph-project/`:

```bash
cd fingraph-project

# Activate virtual environment (use .venv or venv, both exist)
source .venv/bin/activate

# Run the full training pipeline
python scripts/run_pipeline.py --config config/pipeline_config.yaml

# Generate daily trading signals (requires a trained model in data/models/)
python scripts/generate_signals.py

# Run backtest on a trained model
python scripts/run_backtest.py

# Run system health check
python scripts/health_check.py

# Run the Streamlit dashboard locally
streamlit run src/dashboard/app.py

# Debug signal correlations
python scripts/debug_correlations.py
```

## Docker (Two-Service Architecture)

The system uses two separate Docker images with split requirements to minimize image sizes:

- `Dockerfile.dashboard` — Streamlit UI only (`dashboard-requirements.txt`: streamlit, pandas, plotly, yfinance)
- `Dockerfile.worker` — GNN inference worker (`worker-requirements.txt`: torch, torch-geometric, scikit-learn)

```bash
docker build -f Dockerfile.dashboard -t fingraph-dashboard .
docker build -f Dockerfile.worker -t fingraph-worker .
```

## Architecture

The pipeline flows through five stages:

1. **Data** (`src/core/data_manager.py`) — `UnifiedDataManager` fetches price data via yfinance with MD5-keyed disk caching (24h TTL). Raw data → `data/cache/`.

2. **Features** (`src/core/feature_engine.py`) — `UnifiedFeatureEngine` computes technical indicators and rolling statistics with **zero lookahead bias** (point-in-time only). Labels (`forward_return`, `risk_score`, `forward_volatility`) are kept strictly separate from input features.

3. **Graph Construction** (`src/pipeline/graph_builder.py`) — `TemporalGraphBuilder` builds one `torch_geometric.data.Data` graph per trading date. Nodes = stocks; edges = correlation-based (threshold: 0.3). Edge attributes: `[correlation, abs_correlation, positive_flag]`.

4. **Model** (`src/models/gnn_model.py`) — `FinancialGNN` with 2 GAT layers, skip connections, and three output heads: `risk_head` (sigmoid), `return_head`, and `volatility_head`. Makes **node-level** (per-stock) predictions, not graph-level.

   There is also a legacy `RealFinancialGNN` in `src/models/gnn_trainer.py` with 3 GAT layers. `FinancialGNN` in `gnn_model.py` is the active production model.

5. **Backtest** (`src/backtesting/backtester.py`) — `FinGraphBacktester` simulates t+1 execution with 10bps commission + 5bps slippage, weekly rebalancing, stop-loss, and max holding period of 20 days.

**Signal → Dashboard flow:** `generate_signals.py` writes `data/signals/latest_signals.csv` + `data/health/latest_health.json`. The Streamlit app reads these files (with 60s/300s `st.cache_data` TTL) and can trigger re-generation via a button.

## Key Configuration

All pipeline parameters live in `config/pipeline_config.yaml`:
- `data.symbols` — stock universe (default: AAPL, MSFT, GOOGL, AMZN, TSLA, META, NVDA, JPM, V, JNJ)
- `features.label_horizon` — forward return window in days (default: 21)
- `model.hidden_dim`, `model.num_layers` — GNN capacity
- `graph.correlation_threshold` — edge creation threshold (lower = more edges)
- `validation.train_pct / val_pct / test_pct` — temporal split ratios with 5-day gaps between splits
- `backtesting.commission_pct`, `backtesting.slippage_pct` — transaction cost model

## Data & Model Artifacts

Gitignored paths that must exist locally:
- `fingraph-project/data/models/*.pt` — trained model checkpoints (loaded by `generate_signals.py` by recency)
- `fingraph-project/data/processed/features_latest.parquet` — precomputed features
- `fingraph-project/data/signals/latest_signals.csv` — signals consumed by dashboard
- `fingraph-project/data/cache/` — yfinance disk cache

Models are versioned via `scripts/model_manager.py` which maintains `data/models/model_registry.json` with SHA-256 hashes.

## Critical Constraints

- **No lookahead bias**: Features must only use data available at prediction time. Label columns (`forward_return`, `forward_volatility`, `forward_max_drawdown`, `risk_score`, and their z-scored variants) are explicitly excluded from node feature tensors in `graph_builder.py`.
- **Temporal splits**: Train/val/test are chronological with 5-day gaps — never shuffle time series data.
- **Node-level predictions**: The active `FinancialGNN` produces per-stock outputs, not pooled graph-level outputs.
- **Signal**: `signal = expected_return - risk_score`. Positive = long candidate, negative = avoid.
