# Diagnose training-vs-inference feature schema and leakage

import sys
from pathlib import Path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

import pandas as pd
import yaml

from src.core.feature_engine import UnifiedFeatureEngine
from src.pipeline.graph_builder import TemporalGraphBuilder

def main():
    # Load config
    with open(project_root / "config/pipeline_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # 1) What the TRAINING graphs actually fed to the model?
    train_df = pd.read_parquet(project_root / "data/processed/features_latest.parquet")

    # Use the SAME exclusion logic as graph_builder currently uses
    TRAIN_EXCLUDE = {
        'date', 'symbol',
        'forward_return', 'forward_volatility', 'forward_max_drawdown',
        'risk_score',
        # >>> SHOULD HAVE BEEN EXCLUDED BUT WASN'T:
        # 'forward_return_cs_z', 'forward_volatility_cs_z', 'risk_score_cs_pct'
    }
    train_feature_cols = [c for c in train_df.columns if c not in TRAIN_EXCLUDE]
    print(f"TRAIN graph inputs ({len(train_feature_cols)}): {train_feature_cols}")

    # 2) What the LIVE prediction engine produces?
    engine = UnifiedFeatureEngine(config)

    # Synthesize “today” prediction features from last saved training set’s last date
    # (This is only to inspect column set/ordering, not to run the model.)
    last_date = pd.to_datetime(train_df['date']).max()
    live_like = (
        train_df.loc[train_df['date'] == last_date, ['date', 'symbol'] + engine._define_feature_names()]
        .dropna(axis=1, how='all')
        .drop_duplicates(['date', 'symbol'])
    )

    pred_exclude = {'date', 'symbol'}
    pred_feature_cols = [c for c in live_like.columns if c not in pred_exclude]

    print(f"PRED feature columns ({len(pred_feature_cols)}): {pred_feature_cols}")

    # 3) Diff
    t_set, p_set = set(train_feature_cols), set(pred_feature_cols)
    only_in_train = sorted(t_set - p_set)
    only_in_pred  = sorted(p_set - t_set)

    print("\n=== DIFF (train - pred) ===")
    print("Only in TRAIN:", only_in_train)
    print("Only in PRED :", only_in_pred)

    # 4) Leakage check (label-derived columns in inputs)
    leakage_cols = [c for c in train_feature_cols if c.endswith('_cs_z') or c == 'risk_score_cs_pct']
    if leakage_cols:
        print("\n⚠️  LEAKAGE DETECTED in TRAIN INPUTS:", leakage_cols)
        print("These must be excluded from node features and you must retrain.")
    else:
        print("\n✅ No label-derived columns detected in train inputs.")

if __name__ == "__main__":
    main()
