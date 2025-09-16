"""Train a lightweight linear risk model and persist it as JSON.

The training routine intentionally favours reliability over complexity –
it only relies on public market data (via :mod:`yfinance`) and falls back
to synthetic samples when the download fails.  The resulting model stores
its parameters as JSON so it can be version controlled alongside the
project.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

# Ensure ``src`` can be imported when the script is executed from the
# repository root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import save_artifact  # noqa: E402  (import after sys.path tweak)

LOGGER = logging.getLogger("train_risk_model")

DEFAULT_LOOKBACK = 60
FEATURE_NAMES = [
    "volatility",
    "momentum_5d",
    "momentum_20d",
    "max_drawdown",
    "rsi",
    "volume_ratio",
]

MODEL_PATH = PROJECT_ROOT / "models" / "temporal_risk_model.json"
PERFORMANCE_PATH = PROJECT_ROOT / "models" / "performance.json"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the FinGraph risk model")
    parser.add_argument(
        "--symbols",
        nargs="*",
        default=["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"],
        help="List of ticker symbols to include in training",
    )
    parser.add_argument(
        "--years",
        type=int,
        default=3,
        help="Number of years of history to download for each symbol",
    )
    parser.add_argument(
        "--lookback",
        type=int,
        default=DEFAULT_LOOKBACK,
        help="Number of trailing days used to compute risk features",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=MODEL_PATH,
        help="Path where the JSON model artifact will be written",
    )
    parser.add_argument(
        "--performance-output",
        type=Path,
        default=PERFORMANCE_PATH,
        help="Optional metrics summary path (JSON)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
        help="Verbosity for console logging",
    )
    return parser.parse_args(argv)


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def download_history(symbol: str, years: int, lookback: int) -> Optional[pd.DataFrame]:
    """Retrieve OHLCV history for ``symbol``.

    The download uses :func:`yfinance.download`.  When the call fails
    (for example due to missing network access) ``None`` is returned so
    the caller can fall back to a synthetic time series.
    """

    end = datetime.utcnow()
    start = end - timedelta(days=int(years * 365 + lookback + 30))

    try:
        data = yf.download(symbol, start=start, end=end, progress=False, auto_adjust=False)
    except Exception as exc:  # pragma: no cover - network failures are environment specific
        LOGGER.warning("%s: failed to download history (%s)", symbol, exc)
        return None

    if data.empty:
        LOGGER.warning("%s: received empty dataset from yfinance", symbol)
        return None

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]

    required = {"Open", "High", "Low", "Close", "Volume"}
    missing = required.difference(data.columns)
    if missing:
        LOGGER.warning("%s: dataset missing columns: %s", symbol, sorted(missing))
        return None

    frame = data.loc[:, ["Open", "High", "Low", "Close", "Volume"]].copy()
    frame.index = pd.to_datetime(frame.index)
    frame.sort_index(inplace=True)
    frame = frame.ffill().dropna(subset=["Close"])
    return frame


def generate_synthetic_history(symbol: str, years: int, lookback: int, rng: np.random.Generator) -> pd.DataFrame:
    """Create a deterministic synthetic OHLCV series."""

    total_days = max(int(years * 252) + lookback + 30, lookback + 30)
    index = pd.bdate_range(end=datetime.utcnow(), periods=total_days)
    daily_returns = rng.normal(loc=0.0005, scale=0.02, size=len(index))

    prices = 100.0 * np.exp(np.cumsum(daily_returns))
    close = pd.Series(prices, index=index)
    open_price = close.shift(1).fillna(close)
    high = close * (1 + np.abs(rng.normal(0.001, 0.01, len(index))))
    low = close / (1 + np.abs(rng.normal(0.001, 0.01, len(index))))
    volume = rng.lognormal(mean=12, sigma=0.3, size=len(index))

    frame = pd.DataFrame(
        {
            "Open": open_price.values,
            "High": np.maximum.reduce([open_price.values, high.values, close.values]),
            "Low": np.minimum.reduce([low.values, close.values, open_price.values]),
            "Close": close.values,
            "Volume": volume,
        },
        index=index,
    )
    frame.index.name = "Date"
    LOGGER.info("%s: using synthetic price series with %d rows", symbol, len(frame))
    return frame


def calculate_rsi(close: pd.Series, window: int = 14) -> float:
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    value = rsi.iloc[-1] if not rsi.empty else 50.0
    if pd.isna(value):
        return 50.0
    return float(value)


def compute_window_metrics(window: pd.DataFrame) -> Optional[Dict[str, float]]:
    if len(window) < 5:
        return None

    close_prices = window["Close"].astype(float)
    returns = close_prices.pct_change().dropna()
    if returns.empty:
        return None

    volatility = float(np.clip(returns.std() * np.sqrt(252), 1e-4, 2.0))
    momentum_5d = float(returns.tail(5).mean()) if len(returns) >= 5 else 0.0
    momentum_20d = float(returns.tail(20).mean()) if len(returns) >= 20 else momentum_5d

    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = float(((cumulative - running_max) / running_max).min()) if not cumulative.empty else 0.0
    max_drawdown = float(abs(drawdown))

    rsi_value = calculate_rsi(close_prices)

    volume_series = window["Volume"].ffill()
    avg_volume = volume_series.rolling(20).mean().iloc[-1] if len(volume_series) >= 20 else volume_series.iloc[-1]
    if pd.isna(avg_volume) or avg_volume <= 0:
        volume_ratio = 1.0
    else:
        volume_ratio = float(np.clip(volume_series.iloc[-1] / avg_volume, 0.1, 10.0))

    return {
        "volatility": volatility,
        "momentum_5d": momentum_5d,
        "momentum_20d": momentum_20d,
        "max_drawdown": max_drawdown,
        "rsi": rsi_value,
        "volume_ratio": volume_ratio,
    }


def heuristic_risk_score(metrics: Dict[str, float]) -> float:
    volatility_score = min(metrics["volatility"] / 0.8, 1.0)
    momentum_score = max(0.0, min(1.0, 0.5 - metrics["momentum_20d"] * 10.0))
    drawdown_score = min(metrics["max_drawdown"] / 0.3, 1.0)
    rsi_risk = 0.3 if 30 < metrics["rsi"] < 70 else 0.7
    volume_spike_risk = min(metrics["volume_ratio"] / 3.0, 1.0) * 0.2

    score = (
        volatility_score * 0.35
        + momentum_score * 0.25
        + drawdown_score * 0.25
        + rsi_risk * 0.10
        + volume_spike_risk * 0.05
    )
    return float(np.clip(score, 0.0, 1.0))


def build_training_dataset(symbols: Iterable[str], years: int, lookback: int) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    rows = []

    for symbol in symbols:
        history = download_history(symbol, years, lookback)
        if history is None:
            history = generate_synthetic_history(symbol, years, lookback, rng)

        history = history.sort_index()
        for end in range(lookback, len(history)):
            window = history.iloc[end - lookback : end]
            metrics = compute_window_metrics(window)
            if not metrics:
                continue

            risk_score = heuristic_risk_score(metrics)
            rows.append(
                {
                    "symbol": symbol,
                    "date": history.index[end - 1],
                    **metrics,
                    "risk_score": risk_score,
                }
            )

    dataset = pd.DataFrame(rows)
    if not dataset.empty:
        dataset["date"] = pd.to_datetime(dataset["date"])
        dataset.sort_values(["symbol", "date"], inplace=True)
        dataset.reset_index(drop=True, inplace=True)
    return dataset


def _prepare_feature_matrix(
    frame: pd.DataFrame,
    means: Optional[pd.Series] = None,
    stds: Optional[pd.Series] = None,
) -> Tuple[np.ndarray, pd.Series, pd.Series]:
    values = frame[FEATURE_NAMES].astype(float)

    if means is None or stds is None:
        means = values.mean(axis=0)
        stds = values.std(axis=0, ddof=0).replace(0, 1.0)
    else:
        stds = stds.replace(0, 1.0)

    standardized = (values - means) / stds
    standardized = standardized.fillna(0.0)
    return standardized.to_numpy(dtype=float), means, stds


def _linear_regression(train_X: np.ndarray, train_y: np.ndarray) -> np.ndarray:
    augmented = np.hstack([train_X, np.ones((train_X.shape[0], 1), dtype=float)])
    coeffs, *_ = np.linalg.lstsq(augmented, train_y, rcond=None)
    return coeffs


def _predict(coefficients: np.ndarray, features: np.ndarray) -> np.ndarray:
    weights = coefficients[:-1]
    intercept = coefficients[-1]
    return features @ weights + intercept


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    if len(y_true) == 0:
        return {"mse": 0.0, "rmse": 0.0, "mae": 0.0, "r2": 0.0}

    y_pred = np.clip(y_pred, 0.0, 1.0)
    diff = y_true - y_pred
    mse = float(np.mean(diff**2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(diff)))
    if len(y_true) > 1:
        total_var = np.sum((y_true - np.mean(y_true)) ** 2)
        resid = np.sum(diff**2)
        r2 = 1 - resid / total_var if total_var > 0 else 0.0
    else:
        r2 = 0.0
    return {"mse": mse, "rmse": rmse, "mae": mae, "r2": float(r2)}


def train_model(dataset: pd.DataFrame, lookback: int, symbols: List[str]) -> Dict[str, Any]:
    if dataset.empty:
        raise RuntimeError("Dataset is empty; unable to train model")

    feature_frame = dataset[FEATURE_NAMES]
    target = dataset["risk_score"].astype(float).to_numpy()

    split_idx = int(len(dataset) * 0.8)
    if split_idx <= len(FEATURE_NAMES) or split_idx >= len(dataset):
        split_idx = max(len(dataset) - 1, 1)

    train_features = feature_frame.iloc[:split_idx]
    val_features = feature_frame.iloc[split_idx:]
    train_targets = target[:split_idx]
    val_targets = target[split_idx:]

    train_matrix, means, stds = _prepare_feature_matrix(train_features)
    coefficients = _linear_regression(train_matrix, train_targets)
    predictions_train = _predict(coefficients, train_matrix)

    val_matrix, _, _ = _prepare_feature_matrix(val_features, means, stds)
    predictions_val = _predict(coefficients, val_matrix)

    train_metrics = _metrics(train_targets, predictions_train)
    val_metrics = _metrics(val_targets, predictions_val)

    weights = coefficients[:-1].tolist()
    intercept = float(coefficients[-1])

    artifact = {
        "model": {
            "coefficients": [float(w) for w in weights],
            "intercept": intercept,
            "feature_means": [float(v) for v in means.tolist()],
            "feature_stds": [float(v) for v in stds.tolist()],
        },
        "feature_names": FEATURE_NAMES,
        "lookback_days": int(lookback),
        "trained_at": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "training_symbols": symbols,
        "metrics": {
            "mse": float(val_metrics["mse"]),
            "rmse": float(val_metrics["rmse"]),
            "mae": float(val_metrics["mae"]),
            "r2": float(val_metrics["r2"]),
            "training_samples": int(len(train_targets)),
            "validation_samples": int(len(val_targets)),
        },
    }

    performance_summary = {
        "generated_at": artifact["trained_at"],
        "metrics": {
            "Linear Regression": {
                "mse": float(val_metrics["mse"]),
                "rmse": float(val_metrics["rmse"]),
                "mae": float(val_metrics["mae"]),
                "r2": float(val_metrics["r2"]),
                "training_samples": int(len(train_targets)),
                "validation_samples": int(len(val_targets)),
            }
        },
    }

    return {"artifact": artifact, "performance": performance_summary}


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    configure_logging(args.log_level)

    LOGGER.info(
        "Training risk model with %d symbols, %d years of history, lookback=%d",
        len(args.symbols),
        args.years,
        args.lookback,
    )

    dataset = build_training_dataset(args.symbols, args.years, args.lookback)
    if dataset.empty:
        LOGGER.error("No training samples could be produced. Aborting.")
        return 1

    results = train_model(dataset, args.lookback, list(args.symbols))
    save_artifact(args.output, results["artifact"])
    LOGGER.info("Model artifact saved to %s", args.output)

    if args.performance_output:
        save_artifact(args.performance_output, results["performance"])
        LOGGER.info("Performance summary saved to %s", args.performance_output)

    LOGGER.info("Training complete")
    return 0


if __name__ == "__main__":  # pragma: no cover - manual execution entry-point
    raise SystemExit(main())
