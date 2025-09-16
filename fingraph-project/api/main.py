"""
FinGraph API Service - FIXED VERSION with Real-Time Data Generation
This version generates fresh risk predictions on each request
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
import yfinance as yf
import logging

from src.features.graph_data_loader import GraphDataLoader

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)
project_root_path = Path(project_root)

from src.utils import load_artifact  # noqa: E402  (import after path tweak)

PERFORMANCE_FILE = os.path.join(project_root, "models", "performance.json")
PREDICTIONS_DIR = os.path.join(project_root, "data", "temporal_integration")
DEFAULT_PREDICTIONS_FILE = os.path.join(PREDICTIONS_DIR, "predictions.csv")
DEFAULT_SUMMARY_FILE = os.path.join(PREDICTIONS_DIR, "dashboard_summary.json")
SUPPORTED_DATA_SOURCES = {"live", "stored", "auto"}

MODEL_PATH = project_root_path / "models" / "temporal_risk_model.json"

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_ARTIFACT: Optional[Dict[str, Any]] = None
try:
    MODEL_ARTIFACT = load_artifact(MODEL_PATH)
    logger.info("Loaded risk model artifact from %s", MODEL_PATH)
except FileNotFoundError:
    logger.warning("Risk model artifact not found at %s", MODEL_PATH)
except Exception as exc:  # pragma: no cover - defensive logging
    logger.error("Failed to load risk model artifact: %s", exc)

app = FastAPI(
    title="FinGraph API",
    description="Financial Risk Assessment API - Real-Time Analysis",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Response models (keep existing)
class RiskScore(BaseModel):
    symbol: str
    risk_score: float
    risk_level: str
    volatility: float

class HealthStatus(BaseModel):
    status: str
    data_available: bool
    last_update: Optional[str]
    companies_count: int
    symbol_source: str
    tracked_symbols: List[str] = Field(default_factory=list)
    available_symbols: List[str] = Field(default_factory=list)
    unavailable_symbols: Dict[str, str] = Field(default_factory=dict)

class RealTimeRiskCalculator:
    """Calculate real-time risk scores from live market data."""

    def __init__(self, data_dir: Optional[str] = None, model_artifact: Optional[Dict[str, Any]] = None):
        self.data_dir = self._resolve_data_dir(data_dir)
        self.cache: List[Dict[str, Any]] = []
        self.cache_duration = 300  # 5 minutes cache
        self.last_update: Optional[datetime] = None
        self.last_errors: Dict[str, str] = {}
        self.symbol_source = "default"
        self.symbols = self._load_symbols_from_dataset()

        self.lookback_days = int(model_artifact.get("lookback_days", 60)) if model_artifact else 60
        self.model_ready = False
        self.model_feature_order: List[str] = []
        self.model_coefficients: Optional[np.ndarray] = None
        self.model_intercept: float = 0.0
        self.model_feature_means: Optional[np.ndarray] = None
        self.model_feature_stds: Optional[np.ndarray] = None
        self.model_metadata: Dict[str, Any] = {}
        self.model_metrics: Dict[str, Any] = {}

        if model_artifact:
            self._load_model_parameters(model_artifact)

    def _load_model_parameters(self, artifact: Dict[str, Any]) -> None:
        model_state = artifact.get("model", {})
        feature_names = list(artifact.get("feature_names", []))
        coefficients = model_state.get("coefficients")
        intercept = model_state.get("intercept")
        means = model_state.get("feature_means")
        stds = model_state.get("feature_stds")

        if not (feature_names and coefficients and means and stds):
            logger.warning("Model artifact missing required fields; falling back to heuristic scoring")
            return

        if not (len(feature_names) == len(coefficients) == len(means) == len(stds)):
            logger.warning("Model artifact dimensions do not align; falling back to heuristic scoring")
            return

        self.model_feature_order = feature_names
        self.model_coefficients = np.array(coefficients, dtype=float)
        self.model_intercept = float(intercept)
        self.model_feature_means = np.array(means, dtype=float)
        self.model_feature_stds = np.array(stds, dtype=float)
        self.model_feature_stds[self.model_feature_stds == 0] = 1.0
        self.model_ready = True
        self.model_metadata = {
            "trained_at": artifact.get("trained_at"),
            "training_symbols": artifact.get("training_symbols", []),
            "lookback_days": artifact.get("lookback_days"),
            "feature_names": feature_names,
        }
        self.model_metrics = artifact.get("metrics", {}) or {}
        logger.info("Risk calculator configured with persisted model coefficients")

    def _resolve_data_dir(self, data_dir: Optional[str]) -> Path:
        if data_dir:
            return Path(data_dir)
        return Path(project_root) / "data" / "raw"

    def _default_symbols(self) -> List[str]:
        return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX', 'CRM', 'ADBE',
                'JPM', 'BAC', 'V', 'MA', 'DIS', 'ORCL', 'IBM', 'INTC', 'AMD', 'QCOM']

    def _load_symbols_from_dataset(self) -> List[str]:
        data_path = self.data_dir

        try:
            if data_path.exists():
                loader = GraphDataLoader(data_dir=str(data_path))
                loader.load_latest_data()
                symbols = loader.get_company_list()

                if symbols:
                    self.symbol_source = f"dataset:{data_path}"
                    logger.info(f"📈 Loaded {len(symbols)} symbols from dataset at {data_path}")
                    return symbols
                logger.warning(f"⚠️ Dataset at {data_path} did not yield any symbols")
            else:
                logger.warning(f"⚠️ Data directory {data_path} does not exist; using default symbols")
        except Exception as exc:
            logger.warning(f"⚠️ Could not load symbols from dataset ({exc}); using default symbols")

        self.symbol_source = "default"
        default_symbols = self._default_symbols()
        logger.info(f"📋 Using default symbol list with {len(default_symbols)} entries")
        return default_symbols
    
    def _should_refresh(self) -> bool:
        """Return ``True`` when cached values are stale."""
        if self.last_update is None:
            return True
        return (datetime.now() - self.last_update).seconds > self.cache_duration

    def _compute_metrics(self, price_data: pd.DataFrame) -> Optional[Dict[str, float]]:
        if price_data is None or len(price_data) < 5:
            return None

        frame = price_data.copy()
        if isinstance(frame.columns, pd.MultiIndex):
            frame.columns = [col[0] if isinstance(col, tuple) else col for col in frame.columns]

        required = {"Open", "High", "Low", "Close", "Volume"}
        if not required.issubset(frame.columns):
            missing = required.difference(frame.columns)
            logger.warning("Price data missing required columns: %s", sorted(missing))
            return None

        ordered = frame.loc[:, ["Open", "High", "Low", "Close", "Volume"]].sort_index()
        ordered = ordered.ffill().dropna(subset=["Close"])
        if ordered.empty:
            return None

        window = ordered.tail(max(self.lookback_days, 5))
        returns = window["Close"].pct_change().dropna()
        if returns.empty:
            return None

        volatility = float(np.clip(returns.std() * np.sqrt(252), 1e-4, 2.0))
        momentum_5d = float(returns.tail(5).mean()) if len(returns) >= 5 else 0.0
        momentum_20d = float(returns.tail(20).mean()) if len(returns) >= 20 else momentum_5d

        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = float(((cumulative - running_max) / running_max).min()) if not cumulative.empty else 0.0
        max_drawdown = float(abs(drawdown))

        delta = window["Close"].diff()
        gain = delta.where(delta > 0, 0.0).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        current_rsi = float(rsi.iloc[-1]) if not rsi.empty else 50.0
        if np.isnan(current_rsi):
            current_rsi = 50.0

        avg_volume = window["Volume"].rolling(20).mean()
        if avg_volume.empty or np.isnan(avg_volume.iloc[-1]) or avg_volume.iloc[-1] <= 0:
            volume_ratio = 1.0
        else:
            volume_ratio = float(np.clip(window["Volume"].iloc[-1] / avg_volume.iloc[-1], 0.1, 10.0))

        return {
            "volatility": volatility,
            "momentum_5d": momentum_5d,
            "momentum_20d": momentum_20d,
            "max_drawdown": max_drawdown,
            "rsi": current_rsi,
            "volume_ratio": volume_ratio,
        }

    def _predict_with_model(self, metrics: Dict[str, float]) -> Optional[float]:
        if (
            not self.model_ready
            or self.model_coefficients is None
            or self.model_feature_means is None
            or self.model_feature_stds is None
        ):
            return None

        try:
            values = np.array([float(metrics[name]) for name in self.model_feature_order], dtype=float)
        except KeyError as exc:  # pragma: no cover - defensive
            logger.debug("Missing feature for model prediction: %s", exc)
            return None

        standardized = (values - self.model_feature_means) / self.model_feature_stds
        standardized = np.nan_to_num(standardized, nan=0.0)
        prediction = float(np.dot(standardized, self.model_coefficients) + self.model_intercept)
        return float(np.clip(prediction, 0.0, 1.0))

    @staticmethod
    def _heuristic_score(metrics: Dict[str, float]) -> float:
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

    def calculate_risk_from_price_data(self, price_data: pd.DataFrame) -> Optional[Dict[str, float]]:
        metrics = self._compute_metrics(price_data)
        if metrics is None:
            return None

        prediction = self._predict_with_model(metrics)
        if prediction is None:
            prediction = self._heuristic_score(metrics)

        metrics["risk_score"] = float(prediction)
        return metrics
    
    def get_real_time_risk_scores(self):
        """Get real-time risk scores for all symbols"""
        
        # Check cache
        if not self._should_refresh() and self.cache:
            logger.info("📊 Returning cached risk data")
            return self.cache
        
        logger.info("🔄 Calculating fresh risk scores from live market data...")
        
        risk_data: List[Dict[str, Any]] = []
        errors: Dict[str, str] = {}
        end_date = datetime.now()
        history_span = max(self.lookback_days * 3, 120)
        start_date = end_date - timedelta(days=history_span)

        for symbol in self.symbols:
            try:
                # Download real-time data
                stock_data = yf.download(
                    symbol,
                    start=start_date.strftime('%Y-%m-%d'),
                    end=end_date.strftime('%Y-%m-%d'),
                    progress=False
                )

                if len(stock_data) > 5:
                    # Calculate risk metrics
                    metrics = self.calculate_risk_from_price_data(stock_data)

                    if metrics:
                        # Determine risk level
                        risk_score = metrics['risk_score']
                        if risk_score >= 0.7:
                            risk_level = 'High'
                        elif risk_score >= 0.4:
                            risk_level = 'Medium'
                        else:
                            risk_level = 'Low'
                        
                        risk_data.append({
                            'symbol': symbol,
                            'risk_score': round(risk_score, 4),
                            'risk_level': risk_level,
                            'volatility': round(metrics['volatility'], 4),
                            'momentum_5d': round(metrics['momentum_5d'], 4),
                            'momentum_20d': round(metrics['momentum_20d'], 4),
                            'rsi': round(metrics['rsi'], 2),
                            'max_drawdown': round(metrics['max_drawdown'], 4),
                            'last_updated': datetime.now().isoformat()
                        })
                        logger.info(f"✅ {symbol}: risk={risk_score:.3f}, level={risk_level}")
                    else:
                        message = "Unable to calculate risk metrics from downloaded data"
                        logger.warning(f"⚠️ {symbol}: {message}")
                        errors[symbol] = message
                else:
                    message = "Insufficient historical data returned for analysis"
                    logger.warning(f"⚠️ {symbol}: {message}")
                    errors[symbol] = message

            except Exception as e:
                message = f"Error retrieving market data: {e}"
                logger.error(f"❌ {symbol}: {message}")
                errors[symbol] = message

        # Update cache
        self.cache = risk_data
        self.last_update = datetime.now()
        self.last_errors = errors

        logger.info(f"📊 Calculated risk for {len(risk_data)} companies")
        return risk_data
    
# Global calculator instance
risk_calculator = RealTimeRiskCalculator(model_artifact=MODEL_ARTIFACT)


def stored_predictions_available(path: str = DEFAULT_PREDICTIONS_FILE) -> bool:
    """Return True when a persisted predictions snapshot exists."""

    return os.path.exists(path) and os.path.getsize(path) > 0


def categorize_risk_level(risk_score: float) -> str:
    """Categorize a numeric risk score into a discrete level."""

    if risk_score >= 0.7:
        return "High"
    if risk_score >= 0.4:
        return "Medium"
    return "Low"


def standardize_prediction_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure prediction data contains the columns needed by the API."""

    if df is None:
        return pd.DataFrame()

    standardized = df.copy()

    if 'symbol' in standardized.columns:
        standardized['symbol'] = standardized['symbol'].astype(str).str.upper()
    else:
        standardized['symbol'] = ''

    if 'risk_score' not in standardized.columns:
        logger.error("Predictions data missing 'risk_score' column")
        standardized['risk_score'] = np.nan

    standardized['risk_score'] = pd.to_numeric(standardized['risk_score'], errors='coerce')
    standardized = standardized[standardized['risk_score'].notnull()]

    if 'risk_level' not in standardized.columns:
        standardized['risk_level'] = standardized['risk_score'].apply(categorize_risk_level)

    numeric_columns = {
        'volatility': np.nan,
        'momentum_5d': np.nan,
        'momentum_20d': np.nan,
        'rsi': np.nan,
        'max_drawdown': np.nan,
    }

    for column, default in numeric_columns.items():
        if column in standardized.columns:
            standardized[column] = pd.to_numeric(standardized[column], errors='coerce')
        else:
            standardized[column] = default

    if 'last_updated' in standardized.columns:
        standardized['last_updated'] = standardized['last_updated'].astype(str)
    elif 'prediction_date' in standardized.columns:
        standardized['last_updated'] = standardized['prediction_date'].astype(str)
    else:
        standardized['last_updated'] = datetime.now().isoformat()

    return standardized


def load_stored_predictions(predictions_path: str = DEFAULT_PREDICTIONS_FILE) -> Optional[pd.DataFrame]:
    """Load predictions persisted by the temporal integrator."""

    if not os.path.exists(predictions_path):
        logger.info("Stored predictions file not found: %s", predictions_path)
        return None

    try:
        predictions_df = pd.read_csv(predictions_path)
        if predictions_df.empty:
            logger.warning("Stored predictions file is empty: %s", predictions_path)
            return None

        standardized_df = standardize_prediction_dataframe(predictions_df)
        if standardized_df.empty:
            logger.warning("Stored predictions did not contain usable rows")
            return None

        logger.info("Loaded %d stored predictions", len(standardized_df))
        return standardized_df

    except Exception as exc:
        logger.error("Failed to load stored predictions: %s", exc)
        return None


def load_dashboard_summary(summary_path: str = DEFAULT_SUMMARY_FILE) -> Optional[Dict[str, Any]]:
    """Load dashboard summary generated by the integrator."""

    if not os.path.exists(summary_path):
        return None

    try:
        with open(summary_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except Exception as exc:
        logger.error("Failed to load dashboard summary: %s", exc)
    return None


def get_risk_dataframe(source: str = "live", refresh: bool = False) -> Tuple[Optional[pd.DataFrame], str]:
    """Return a DataFrame of risk data for the requested source."""

    requested_source = (source or "live").lower()
    if requested_source not in SUPPORTED_DATA_SOURCES:
        logger.warning("Unknown risk data source '%s', defaulting to live", requested_source)
        requested_source = "live"

    if requested_source in {"stored", "auto"}:
        stored_df = load_stored_predictions()
        if stored_df is not None:
            return stored_df, "stored"
        if requested_source == "stored":
            return None, "stored"
        logger.info("Stored predictions unavailable; falling back to live heuristics")

    if refresh:
        risk_calculator.last_update = None

    live_records = risk_calculator.get_real_time_risk_scores()
    if not live_records:
        return None, "live"

    live_df = pd.DataFrame(live_records)
    live_df = standardize_prediction_dataframe(live_df)
    return live_df, "live"


def get_last_update_from_df(df: pd.DataFrame) -> Optional[str]:
    """Extract the latest timestamp from a predictions DataFrame."""

    if df is None or df.empty:
        return None

    if 'last_updated' in df.columns:
        series = df['last_updated'].dropna()
        if not series.empty:
            try:
                return pd.to_datetime(series).max().isoformat()
            except Exception:
                return str(series.iloc[-1])

    if 'prediction_date' in df.columns:
        series = df['prediction_date'].dropna()
        if not series.empty:
            try:
                return pd.to_datetime(series).max().isoformat()
            except Exception:
                return str(series.iloc[-1])

    return None


def row_to_risk_score(row: pd.Series) -> RiskScore:
    """Convert a DataFrame row into a RiskScore response model."""

    volatility = row.get('volatility', 0.0)
    if pd.isna(volatility):
        volatility = 0.0

    return RiskScore(
        symbol=str(row['symbol']),
        risk_score=float(row['risk_score']),
        risk_level=str(row['risk_level']),
        volatility=float(volatility)
    )


def load_saved_model_performance() -> Dict:
    """Load persisted model performance metrics from disk."""
    if not os.path.exists(PERFORMANCE_FILE):
        logger.warning(f"Performance metrics file not found at {PERFORMANCE_FILE}")
        return {}

    try:
        with open(PERFORMANCE_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if not isinstance(data, dict):
            logger.warning("Performance metrics file has unexpected format")
            return {}

        metrics = data.get('metrics') if isinstance(data, dict) else None

        if isinstance(metrics, dict):
            normalised_metrics = {}
            for model_name, values in metrics.items():
                if not isinstance(values, dict):
                    continue

                entry = {}
                mse_value = values.get('mse')
                if mse_value is not None:
                    entry['mse'] = float(mse_value)

                rmse_value = values.get('rmse')
                if rmse_value is not None:
                    entry['rmse'] = float(rmse_value)
                elif mse_value is not None:
                    entry['rmse'] = float(np.sqrt(mse_value))

                normalised_metrics[model_name] = entry

            data['metrics'] = normalised_metrics

        return data

    except json.JSONDecodeError as decode_error:
        logger.error(f"Invalid JSON in performance metrics file: {decode_error}")
    except Exception as exc:
        logger.error(f"Failed to load performance metrics: {exc}")

    return {}

@app.get("/", response_model=Dict)
async def root():
    """FinGraph API - Real-Time Financial Risk Assessment"""
    cached_symbols = [entry['symbol'] for entry in risk_calculator.cache] if risk_calculator.cache else []
    stored_available = stored_predictions_available()
    stored_last_update = None
    if stored_available:
        try:
            stored_last_update = datetime.fromtimestamp(os.path.getmtime(DEFAULT_PREDICTIONS_FILE)).isoformat()
        except OSError:
            stored_last_update = None

    model_info = {
        "loaded": bool(risk_calculator.model_ready),
        "lookback_days": risk_calculator.lookback_days,
        "trained_at": risk_calculator.model_metadata.get("trained_at") if risk_calculator.model_metadata else None,
        "training_symbols": risk_calculator.model_metadata.get("training_symbols", []) if risk_calculator.model_metadata else [],
        "feature_names": risk_calculator.model_metadata.get("feature_names", []) if risk_calculator.model_metadata else [],
        "metrics": risk_calculator.model_metrics,
        "artifact_path": str(MODEL_PATH),
    }

    return {
        "message": "FinGraph API - Real-Time Financial Risk Assessment",
        "version": "2.0.0",
        "status": "production",
        "features": {
            "real_time_data": True,
            "cache_duration": "5 minutes",
            "companies_tracked": len(risk_calculator.symbols),
            "metrics": ["risk_score", "volatility", "momentum", "rsi", "drawdown"],
            "stored_predictions_available": stored_available
        },
        "endpoints": {
            "health": "/health",
            "risk": "/risk",
            "risk_company": "/risk/{symbol}",
            "portfolio": "/portfolio",
            "alerts": "/alerts"
        },
        "data_sources": {
            "default": "live",
            "supported": sorted(SUPPORTED_DATA_SOURCES),
            "stored_last_update": stored_last_update
        },
        "last_update": risk_calculator.last_update.isoformat() if risk_calculator.last_update else None,
        "companies": {
            "source": risk_calculator.symbol_source,
            "tracked": list(risk_calculator.symbols),
            "available": cached_symbols,
            "unavailable": dict(risk_calculator.last_errors)
        },
        "model": model_info,
    }

@app.get("/health", response_model=HealthStatus)
async def health(source: str = Query("live", description="Data source: live, stored, or auto")):
    """Health check supporting live or stored prediction sources."""

    df, data_source = get_risk_dataframe(source)
    if df is None:
        if data_source == "stored":
            raise HTTPException(status_code=404, detail="Stored predictions not available")
        raise HTTPException(status_code=503, detail="Unable to fetch market data")

    available_symbols = df['symbol'].tolist()
    last_update = get_last_update_from_df(df) or datetime.now().isoformat()

    if data_source == "live":
        tracked = list(risk_calculator.symbols)
        unavailable = dict(risk_calculator.last_errors)
        symbol_source = risk_calculator.symbol_source
    else:
        tracked = available_symbols
        unavailable = {}
        symbol_source = "stored"

    return HealthStatus(
        status="healthy",
        data_available=not df.empty,
        last_update=last_update,
        companies_count=len(df),
        symbol_source=symbol_source,
        tracked_symbols=tracked,
        available_symbols=available_symbols,
        unavailable_symbols=unavailable
    )

@app.get("/risk", response_model=List[RiskScore])
async def get_all_risks(
    risk_level: Optional[str] = Query(None, description="Filter by risk level: Low, Medium, High"),
    sort_by: str = Query("risk_score", description="Sort by risk_score or symbol"),
    limit: int = Query(10, ge=1, le=100),
    refresh: bool = Query(False, description="Force refresh data"),
    source: str = Query("live", description="Data source: live, stored, or auto")
):
    """Get risk scores from either live calculations or stored predictions."""

    df, data_source = get_risk_dataframe(source, refresh)

    if df is None:
        if data_source == "stored":
            raise HTTPException(status_code=404, detail="Stored predictions not available")
        raise HTTPException(status_code=503, detail="Unable to fetch market data")

    if risk_level:
        df = df[df['risk_level'].str.lower() == risk_level.lower()]

    if sort_by not in df.columns:
        raise HTTPException(status_code=400, detail=f"Invalid sort column: {sort_by}")

    ascending = sort_by != "risk_score"
    df = df.sort_values(sort_by, ascending=ascending)

    df = df.head(limit)

    return [row_to_risk_score(row) for _, row in df.iterrows()]

@app.get("/risk/{symbol}", response_model=RiskScore)
async def get_company_risk(
    symbol: str,
    source: str = Query("live", description="Data source: live, stored, or auto")
):
    """Get risk for a specific company from live or stored data."""

    normalized_source = (source or "live").lower()
    if normalized_source not in SUPPORTED_DATA_SOURCES:
        normalized_source = "live"

    symbol = symbol.upper()
    df, data_source = get_risk_dataframe(normalized_source)

    if df is None:
        if data_source == "stored":
            raise HTTPException(status_code=404, detail="Stored predictions not available")
        raise HTTPException(status_code=503, detail="Unable to fetch market data")

    match = df[df['symbol'] == symbol]
    if not match.empty:
        return row_to_risk_score(match.iloc[0])

    if data_source == "stored" and normalized_source == "auto":
        live_df, _ = get_risk_dataframe("live")
        if live_df is not None:
            match = live_df[live_df['symbol'] == symbol]
            if not match.empty:
                return row_to_risk_score(match.iloc[0])
        data_source = "live"

    if data_source == "stored":
        raise HTTPException(status_code=404, detail=f"Company {symbol} not found in stored predictions")

    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=60)
        stock_data = yf.download(symbol, start=start_date, end=end_date, progress=False)

        if len(stock_data) > 5:
            metrics = risk_calculator.calculate_risk_from_price_data(stock_data)
            if metrics:
                risk_score = metrics['risk_score']
                risk_level = categorize_risk_level(risk_score)

                return RiskScore(
                    symbol=symbol,
                    risk_score=risk_score,
                    risk_level=risk_level,
                    volatility=metrics['volatility']
                )
    except Exception as exc:
        logger.error("On-demand risk calculation failed for %s: %s", symbol, exc)

    raise HTTPException(status_code=404, detail=f"Company {symbol} not found")

@app.get("/portfolio")
async def get_portfolio_summary(
    source: str = Query("live", description="Data source: live, stored, or auto")
):
    """Portfolio overview using live or stored predictions."""

    df, data_source = get_risk_dataframe(source)
    if df is None:
        if data_source == "stored":
            raise HTTPException(status_code=404, detail="Stored predictions not available")
        raise HTTPException(status_code=503, detail="Unable to fetch market data")

    timestamp = get_last_update_from_df(df) or datetime.now().isoformat()

    avg_risk = float(df['risk_score'].mean()) if not df.empty else 0.0
    if 'volatility' in df.columns and df['volatility'].notna().any():
        avg_volatility = float(df['volatility'].dropna().mean())
    else:
        avg_volatility = 0.0

    momentum_column = 'momentum_5d' if 'momentum_5d' in df.columns else 'risk_score'
    if df.empty:
        momentum_records = []
    else:
        top_n = min(3, len(df))
        momentum_subset = df.nlargest(top_n, momentum_column).copy()
        if 'momentum_5d' not in momentum_subset.columns:
            momentum_subset['momentum_5d'] = momentum_subset[momentum_column]
        momentum_records = momentum_subset[['symbol', 'momentum_5d']].to_dict('records')

    def _top_symbol(column: str, ascending: bool = False) -> str:
        if column not in df.columns:
            return 'N/A'
        metric = pd.to_numeric(df[column], errors='coerce')
        temp_df = df.copy()
        temp_df['_metric'] = metric
        temp_df = temp_df[temp_df['_metric'].notna()]
        if temp_df.empty:
            return 'N/A'
        temp_df = temp_df.sort_values('_metric', ascending=ascending)
        return str(temp_df.iloc[0]['symbol'])

    best_momentum_symbol = 'N/A'
    if 'momentum_20d' in df.columns:
        best_momentum_symbol = _top_symbol('momentum_20d', ascending=False)
    if best_momentum_symbol == 'N/A':
        best_momentum_symbol = _top_symbol('risk_score', ascending=False)

    market_summary = {
        "most_risky": _top_symbol('risk_score', ascending=False),
        "least_risky": _top_symbol('risk_score', ascending=True),
        "highest_volatility": _top_symbol('volatility', ascending=False),
        "best_momentum": best_momentum_symbol
    }

    if data_source == 'live':
        model_performance = load_saved_model_performance()
    else:
        summary = load_dashboard_summary() or {}
        model_performance = summary.get('model_performance', {})

    portfolio = {
        "timestamp": timestamp,
        "companies_analyzed": len(df),
        "risk_distribution": df['risk_level'].value_counts().to_dict(),
        "average_risk_score": avg_risk,
        "average_volatility": avg_volatility,
        "high_momentum_stocks": momentum_records,
        "model_performance": model_performance,
        "market_summary": market_summary,
        "data_source": data_source
    }

    return portfolio

@app.get("/alerts")
async def get_risk_alerts(
    threshold: float = Query(0.7, ge=0.0, le=1.0),
    source: str = Query("live", description="Data source: live, stored, or auto")
):
    """Risk alerts from live or stored predictions."""

    df, data_source = get_risk_dataframe(source)
    if df is None:
        if data_source == "stored":
            raise HTTPException(status_code=404, detail="Stored predictions not available")
        raise HTTPException(status_code=503, detail="Unable to fetch market data")

    high_risk = df[df['risk_score'] >= threshold]

    alerts = []
    default_timestamp = get_last_update_from_df(df) or datetime.now().isoformat()
    for _, row in high_risk.iterrows():
        volatility = row.get('volatility', 0.0)
        if pd.isna(volatility):
            volatility = 0.0

        alerts.append({
            "symbol": row['symbol'],
            "risk_score": float(row['risk_score']),
            "risk_level": row['risk_level'],
            "volatility": float(volatility),
            "momentum_5d": row.get('momentum_5d', float(row['risk_score'])),
            "rsi": row.get('rsi', 50),
            "message": f"{row['symbol']} risk score {row['risk_score']:.3f} exceeds threshold {threshold:.2f}",
            "timestamp": row.get('last_updated', default_timestamp)
        })

    if data_source == 'live':
        market_status = "open" if 9 <= datetime.now().hour < 16 else "closed"
    else:
        market_status = "offline"

    return {
        "threshold": threshold,
        "alert_count": len(alerts),
        "alerts": alerts,
        "generated_at": datetime.now().isoformat(),
        "market_status": market_status,
        "data_source": data_source
    }

def run_server():
    """Run the API server"""
    print("🚀 Starting FinGraph Real-Time API...")
    print(f"📊 Tracking {len(risk_calculator.symbols)} companies")
    print(f"⏱️ Cache duration: {risk_calculator.cache_duration} seconds")
    print("✅ Ready to serve real-time risk assessments")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    run_server()