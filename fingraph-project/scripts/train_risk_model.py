"""Utility for training and persisting the FinGraph risk model."""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import yfinance as yf

from src.utils import dump_artifact

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
LOGGER = logging.getLogger(__name__)

FEATURE_COLUMNS = [
    "volatility",
    "momentum_5d",
    "momentum_20d",
    "max_drawdown",
    "rsi",
    "volume_ratio",
]


def _calculate_rsi(close_prices: pd.Series, window: int = 14) -> float:
    """Compute relative strength index value."""
    delta = close_prices.diff()
    gain = delta.where(delta > 0, 0.0).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window=window).mean()

    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    if rsi.empty:
        return 50.0
    value = rsi.iloc[-1]
    if np.isnan(value):
        return 50.0
    return float(value)


def _compute_metrics(history: pd.DataFrame) -> Optional[Dict[str, float]]:
    """Compute predictive features from trailing price history."""
    if len(history) < 10:
        return None

    close_prices = history["Close"].dropna()
    if close_prices.empty:
        return None

    returns = close_prices.pct_change().dropna()
    if returns.empty:
        return None

    volatility = float(returns.std() * np.sqrt(252))
    volatility = float(np.clip(volatility, 1e-4, 2.0))

    momentum_5d = float(returns.tail(5).mean()) if len(returns) >= 5 else 0.0
    momentum_20d = float(returns.tail(20).mean()) if len(returns) >= 20 else momentum_5d

    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = float(((cumulative - running_max) / running_max).min()) if not cumulative.empty else 0.0
    max_drawdown = float(abs(drawdown))

    rsi = _calculate_rsi(close_prices)

    volume = history["Volume"].ffill()
    avg_volume = volume.rolling(20).mean()
    if avg_volume.isna().all():
        volume_ratio = 1.0
    else:
        volume_ratio = float((volume.iloc[-1] / avg_volume.iloc[-1]) if avg_volume.iloc[-1] else 1.0)
    volume_ratio = float(np.clip(volume_ratio, 0.1, 10.0))

    metrics = {
        "volatility": volatility,
        "momentum_5d": momentum_5d,
        "momentum_20d": momentum_20d,
        "max_drawdown": max_drawdown,
        "rsi": rsi,
        "volume_ratio": volume_ratio,
    }
    return metrics


def _compute_target(metrics: Dict[str, float]) -> float:
    """Derive deterministic risk score using heuristic weights."""
    volatility = metrics["volatility"]
    momentum_20d = metrics["momentum_20d"]
    max_drawdown = metrics["max_drawdown"]
    rsi = metrics["rsi"]
    volume_ratio = metrics["volume_ratio"]

    vol_score = min(volatility / 0.8, 1.0)
    momentum_score = max(0.0, min(1.0, 0.5 - momentum_20d * 10.0))
    drawdown_score = min(max_drawdown / 0.3, 1.0)
    rsi_risk = 0.7 if rsi <= 30 or rsi >= 70 else 0.3
    volume_spike_risk = min(volume_ratio / 3.0, 1.0) * 0.2

    risk_score = (
        vol_score * 0.35
        + momentum_score * 0.25
        + drawdown_score * 0.25
        + rsi_risk * 0.10
        + volume_spike_risk * 0.05
    )
    return float(np.clip(risk_score, 0.05, 0.95))


def _generate_synthetic_history(symbols: List[str], start: datetime, end: datetime) -> pd.DataFrame:
    """Generate synthetic but realistic OHLCV data when market data is unavailable."""
    rng = np.random.default_rng(42)
    dates = pd.date_range(start=start, end=end, freq="B")
    frames: List[pd.DataFrame] = []

    for symbol in symbols:
        base_price = rng.uniform(50, 200)
        daily_returns = rng.normal(0, 0.015, size=len(dates))
        price_path = base_price * np.exp(np.cumsum(daily_returns))

        close = price_path
        open_ = close * (1 + rng.normal(0, 0.003, size=len(dates)))
        high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.01, size=len(dates))))
        low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.01, size=len(dates))))
        low = np.clip(low, a_min=1.0, a_max=None)
        volume = rng.integers(1_000_000, 5_000_000, size=len(dates))

        frame = pd.DataFrame(
            {
                "Open": open_,
                "High": high,
                "Low": low,
                "Close": close,
                "Volume": volume,
                "Symbol": symbol,
            },
            index=dates,
        )
        frames.append(frame)

    return pd.concat(frames)


def fetch_market_history(symbols: List[str], start: datetime, end: datetime) -> pd.DataFrame:
    """Download historical data for provided symbols."""
    frames: List[pd.DataFrame] = []
    for symbol in symbols:
        LOGGER.info("Downloading %s", symbol)
        try:
            data = yf.download(symbol, start=start, end=end, progress=False, auto_adjust=False)
        except Exception as exc:  # pragma: no cover - network issues handled gracefully
            LOGGER.warning("Failed to download %s: %s", symbol, exc)
            continue

        if data.empty:
            LOGGER.warning("No data returned for %s", symbol)
            continue

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]

        required_cols = {"Open", "High", "Low", "Close", "Volume"}
        if not required_cols.issubset(data.columns):
            LOGGER.warning("Skipping %s due to missing columns", symbol)
            continue

        data = data.sort_index()
        data["Symbol"] = symbol
        frames.append(data)

    if not frames:
        LOGGER.warning("Falling back to synthetic market data.")
        return _generate_synthetic_history(symbols, start, end)

    return pd.concat(frames)


def build_training_dataset(
    history: pd.DataFrame,
    lookback_days: int = 60,
) -> pd.DataFrame:
    """Construct supervised dataset from trailing market history."""
    samples: List[Dict[str, float]] = []
    symbols = history["Symbol"].unique()

    for symbol in symbols:
        symbol_history = history[history["Symbol"] == symbol]
        for idx in range(lookback_days, len(symbol_history)):
            window = symbol_history.iloc[idx - lookback_days : idx]
            metrics = _compute_metrics(window)
            if not metrics:
                continue

            target = _compute_target(metrics)
            sample = {
                "symbol": symbol,
                "as_of": symbol_history.index[idx],
                "target": target,
            }
            sample.update(metrics)
            samples.append(sample)

    if not samples:
        raise RuntimeError("Unable to create any training samples.")

    return pd.DataFrame(samples)


def _split_dataset(dataset: pd.DataFrame, train_ratio: float = 0.8) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split dataset into train and validation arrays preserving temporal order."""
    dataset = dataset.sort_values("as_of")
    split_idx = max(1, int(len(dataset) * train_ratio))
    split_idx = min(split_idx, len(dataset) - 1)

    X = dataset[FEATURE_COLUMNS].to_numpy(dtype=float)
    y = dataset["target"].to_numpy(dtype=float)

    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    return X_train, y_train, X_val, y_val


def train_model(dataset: pd.DataFrame) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    """Train a linear regression model using numpy."""
    X_train, y_train, X_val, y_val = _split_dataset(dataset)

    if X_train.size == 0:
        X_train, y_train = X_val, y_val
    if X_val.size == 0:
        X_val, y_val = X_train, y_train

    feature_means = X_train.mean(axis=0)
    feature_stds = X_train.std(axis=0)
    feature_stds[feature_stds == 0] = 1.0

    X_train_scaled = (X_train - feature_means) / feature_stds
    X_val_scaled = (X_val - feature_means) / feature_stds

    # Append bias term for closed-form solution
    ones_train = np.ones((X_train_scaled.shape[0], 1))
    design_matrix = np.hstack([X_train_scaled, ones_train])
    weights, *_ = np.linalg.lstsq(design_matrix, y_train, rcond=None)

    coef = weights[:-1]
    intercept = weights[-1]

    predictions = X_val_scaled @ coef + intercept
    mse = float(np.mean((predictions - y_val) ** 2))
    rmse = float(np.sqrt(mse))

    metrics = {"mse": mse, "rmse": rmse, "validation_samples": int(len(y_val))}
    LOGGER.info("Evaluation metrics: %s", metrics)

    model_state = {
        "coefficients": coef,
        "intercept": intercept,
        "feature_means": feature_means,
        "feature_stds": feature_stds,
    }

    return model_state, metrics


def persist_model(
    model_state: Dict[str, np.ndarray],
    metrics: Dict[str, float],
    symbols: List[str],
    lookback_days: int,
    output_path: Path,
) -> None:
    """Persist trained artefacts to disk using JSON-friendly types."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model_payload = {
        "coefficients": model_state["coefficients"].tolist(),
        "intercept": float(model_state["intercept"]),
        "feature_means": model_state["feature_means"].tolist(),
        "feature_stds": model_state["feature_stds"].tolist(),
    }

    metrics_payload = {}
    for key, value in metrics.items():
        if isinstance(value, (int, np.integer)):
            metrics_payload[key] = int(value)
        else:
            metrics_payload[key] = float(value)

    payload = {
        "model": model_payload,
        "feature_names": FEATURE_COLUMNS,
        "metrics": metrics_payload,
        "trained_at": datetime.utcnow().isoformat(),
        "training_symbols": symbols,
        "lookback_days": lookback_days,
    }
    dump_artifact(payload, output_path)
    LOGGER.info("Model saved to %s", output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train FinGraph risk model")
    parser.add_argument("--symbols", nargs="*", default=["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"], help="Symbols to use")
    parser.add_argument("--years", type=int, default=3, help="Number of years of history to download")
    parser.add_argument("--lookback", type=int, default=60, help="Lookback window for feature computation")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "models" / "temporal_risk_model.json",
        help="Path to save trained model",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    end_date = datetime.utcnow().date()
    start_date = end_date - timedelta(days=365 * args.years)

    history = fetch_market_history(args.symbols, start_date, end_date)
    dataset = build_training_dataset(history, lookback_days=args.lookback)

    LOGGER.info("Generated %s training samples", len(dataset))

    model_state, metrics = train_model(dataset)
    persist_model(model_state, metrics, args.symbols, args.lookback, args.output)


if __name__ == "__main__":
    main()
