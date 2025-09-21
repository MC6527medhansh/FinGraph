#!/usr/bin/env python
"""
Diagnose whether the model is producing NODE-LEVEL variation.
- Verifies import path
- Loads latest checkpoint + latest features
- Builds one temporal graph
- Prints feature & prediction dispersion across nodes
"""

import sys
from pathlib import Path

# --- Make "src" importable no matter where you run this from ---
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]           # .../fingraph-project
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import pandas as pd
import yaml

from src.models.gnn_model import FinancialGNN
from src.pipeline.graph_builder import TemporalGraphBuilder

def main():
    # --- Locate artifacts ---
    model_files = sorted((PROJECT_ROOT / "data" / "models").glob("*.pt"))
    if not model_files:
        print("❌ No model checkpoint found in data/models. Train first: scripts/run_pipeline.py --mode train --save")
        sys.exit(1)
    checkpoint_path = max(model_files, key=lambda p: p.stat().st_mtime)

    features_candidates = sorted((PROJECT_ROOT / "data" / "processed").glob("features_*.parquet"))
    latest_features = PROJECT_ROOT / "data" / "processed" / "features_latest.parquet"
    if latest_features.exists():
        features_path = latest_features
    elif features_candidates:
        features_path = max(features_candidates, key=lambda p: p.stat().st_mtime)
    else:
        print("❌ No features parquet found in data/processed. Run pipeline first.")
        sys.exit(1)

    config_path = PROJECT_ROOT / "config" / "pipeline_config.yaml"
    if not config_path.exists():
        print("❌ Missing config/pipeline_config.yaml")
        sys.exit(1)

    # --- Load data/config/model ---
    print(f"✅ Using checkpoint: {checkpoint_path.name}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    print(f"✅ Using features:  {features_path.name}")
    features = pd.read_parquet(features_path)

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # --- Build a single graph for inspection ---
    graph_builder = TemporalGraphBuilder(config)
    graphs = graph_builder.create_temporal_graphs(features, max_graphs=1)
    if not graphs:
        print("❌ Graph builder returned no graphs.")
        sys.exit(1)
    graph = graphs[0]

    # --- Init model ---
    model = FinancialGNN(
        num_node_features=graph.x.shape[1],
        hidden_dim=config["model"]["hidden_dim"],
        num_heads=config["model"]["num_heads"],
        num_layers=config["model"]["num_layers"],
        dropout=config["model"]["dropout"],
        node_level=True  # must be True for node-level predictions
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # --- Diagnostics ---
    print("\n===== NODE FEATURE DIAGNOSTICS =====")
    print(f"Nodes: {graph.x.shape[0]}, Features per node: {graph.x.shape[1]}")

    # variance across nodes (per feature), then mean of those variances
    feat_var_across_nodes = graph.x.var(dim=0, unbiased=False).mean().item()
    print(f"Feature variance across nodes (mean): {feat_var_across_nodes:.6f}")

    # correlation across nodes: corrcoef expects rows=variables -> transpose
    try:
        corr = torch.corrcoef(graph.x.T)  # features x features
        mean_abs_corr = corr.abs()[~torch.eye(corr.shape[0], dtype=bool)].mean().item()
        print(f"Mean absolute inter-feature correlation: {mean_abs_corr:.6f}")
    except Exception as e:
        print(f"(warn) corrcoef failed: {e}")

    with torch.no_grad():
        # Quick peek at pre-GNN projection dispersion
        x_proj = model.input_projection(graph.x)
        proj_var = x_proj.var(dim=0, unbiased=False).mean().item()
        print(f"After input projection - mean feature variance: {proj_var:.6f}")

        # Forward pass
        outputs = model(graph.x, graph.edge_index, getattr(graph, "edge_attr", None))

        print("\n===== PREDICTION DIAGNOSTICS (NODE-LEVEL EXPECTED) =====")
        for key in ["risk", "return", "volatility"]:
            if key not in outputs:
                print(f"{key:10s}: (missing from model outputs)")
                continue
            vals = outputs[key].detach().cpu().view(-1)
            uniq = torch.unique(vals)
            print(f"{key:10s}: shape={tuple(outputs[key].shape)} "
                  f"unique={uniq.numel():d}/{graph.x.shape[0]} "
                  f"min={vals.min():.6f} max={vals.max():.6f} std={vals.std(unbiased=False):.6f}")

            # Print first 10 for a quick visual
            head_n = min(10, vals.numel())
            print(f"  head({head_n}): {vals[:head_n].tolist()}")

        # Heuristic verdicts
        risk_vals = outputs["risk"].detach().cpu().view(-1) if "risk" in outputs else None
        ret_vals  = outputs["return"].detach().cpu().view(-1) if "return" in outputs else None
        vol_vals  = outputs["volatility"].detach().cpu().view(-1) if "volatility" in outputs else None

        print("\n===== VERDICT =====")
        flags = []
        if risk_vals is not None and torch.unique(risk_vals).numel() <= 3:
            flags.append("Risk predictions have very low uniqueness across nodes (likely graph-level supervision/broadcast).")
        if ret_vals is not None and torch.unique(ret_vals).numel() <= 3:
            flags.append("Return predictions have very low uniqueness across nodes.")
        if vol_vals is not None and torch.unique(vol_vals).numel() <= 3:
            flags.append("Volatility predictions have very low uniqueness across nodes.")
        if not flags:
            print("✅ Predictions show node-level dispersion.")
        else:
            for f in flags:
                print(f"⚠️  {f}")
            print("➡️  Likely cause: training used graph-level targets; enable node-level labels and loss.")

if __name__ == "__main__":
    main()
