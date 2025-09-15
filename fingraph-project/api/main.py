"""
FinGraph API Service - FIXED VERSION with Real-Time Data Generation
This version generates fresh risk predictions on each request
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import yfinance as yf
import logging

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)
project_root_path = Path(project_root)

from src.utils import load_artifact  # noqa: E402  # pylint: disable=wrong-import-position

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_PATH = project_root_path / "models" / "temporal_risk_model.json"
MODEL_ARTIFACT: Optional[Dict[str, Any]] = None
try:
    MODEL_ARTIFACT = load_artifact(MODEL_PATH)
    logger.info("Loaded risk model artifact from %s", MODEL_PATH)
except FileNotFoundError:
    logger.error("Risk model artifact not found at %s", MODEL_PATH)
except Exception as exc:  # pragma: no cover - defensive logging
    logger.error("Failed to load risk model artifact: %s", exc)

DEFAULT_SYMBOLS = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX', 'CRM', 'ADBE',
    'JPM', 'BAC', 'V', 'MA', 'DIS', 'ORCL', 'IBM', 'INTC', 'AMD', 'QCOM'
]


class MarketDataError(Exception):
    """Raised when market data retrieval or processing fails."""


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

def calculate_rsi(close_prices: pd.Series, window: int = 14) -> float:
    """Compute the relative strength index for the provided price series."""
    delta = close_prices.diff()
    gain = delta.where(delta > 0, 0.0).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    if rsi.empty:
        return 50.0
    value = rsi.iloc[-1]
    if pd.isna(value):
        return 50.0
    return float(value)


class RiskModelService:
    """Predict risk scores using the persisted regression model."""

    def __init__(self, model_artifact: Optional[Dict[str, Any]], symbols: Optional[List[str]] = None):
        if not model_artifact:
            raise RuntimeError("Risk model artifact could not be loaded.")

        model_state = model_artifact.get("model")
        if not model_state:
            raise RuntimeError("Risk model artifact missing model state.")

        self.symbols = symbols or DEFAULT_SYMBOLS
        self.cache: List[Dict[str, Any]] = []
        self.cache_duration = 300  # seconds
        self.last_update: Optional[datetime] = None

        self.feature_names: List[str] = list(model_artifact.get("feature_names", []))
        if not self.feature_names:
            raise RuntimeError("Model artifact missing feature names.")

        self.lookback_days = int(model_artifact.get("lookback_days", 60))
        self.training_info = {
            "trained_at": model_artifact.get("trained_at"),
            "training_symbols": model_artifact.get("training_symbols", []),
            "lookback_days": self.lookback_days,
        }
        self.model_metrics = model_artifact.get("metrics", {})

        self.coefficients = np.array(model_state.get("coefficients", []), dtype=float)
        self.intercept = float(model_state.get("intercept", 0.0))
        self.feature_means = np.array(
            model_state.get("feature_means", np.zeros(len(self.feature_names))),
            dtype=float,
        )
        self.feature_stds = np.array(
            model_state.get("feature_stds", np.ones(len(self.feature_names))),
            dtype=float,
        )

        if len(self.coefficients) != len(self.feature_names):
            raise RuntimeError("Model coefficients do not align with feature names.")

        self.feature_stds[self.feature_stds == 0] = 1.0

    def _should_refresh(self) -> bool:
        if self.last_update is None:
            return True
        return (datetime.now() - self.last_update).seconds > self.cache_duration

    def _download_history(self, symbol: str) -> pd.DataFrame:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=max(self.lookback_days * 3, 120))

        try:
            data = yf.download(symbol, start=start_date, end=end_date, progress=False)
        except Exception as exc:  # pragma: no cover - network failures
            raise MarketDataError(f"{symbol}: failed to download market data ({exc})") from exc

        if data.empty:
            raise MarketDataError(f"{symbol}: no market data available")

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]

        required_columns = {"Open", "High", "Low", "Close", "Volume"}
        if not required_columns.issubset(data.columns):
            missing = required_columns.difference(data.columns)
            raise MarketDataError(f"{symbol}: missing columns {sorted(missing)}")

        data = data.sort_index()
        data = data.ffill()
        data = data.dropna(subset=["Close"])

        if len(data) < self.lookback_days:
            raise MarketDataError(f"{symbol}: insufficient history ({len(data)} rows)")

        return data

    def _compute_metrics(self, history: pd.DataFrame) -> Dict[str, float]:
        window = history.tail(self.lookback_days)

        close_prices = window["Close"].dropna()
        if close_prices.empty:
            raise MarketDataError("Close price history unavailable")

        returns = close_prices.pct_change().dropna()
        if returns.empty:
            raise MarketDataError("Insufficient return history")

        volatility = float(np.clip(returns.std() * np.sqrt(252), 1e-4, 2.0))
        momentum_5d = float(returns.tail(5).mean()) if len(returns) >= 5 else 0.0
        momentum_20d = float(returns.tail(20).mean()) if len(returns) >= 20 else momentum_5d

        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = float(((cumulative - running_max) / running_max).min()) if not cumulative.empty else 0.0
        max_drawdown = float(abs(drawdown))

        rsi_value = calculate_rsi(close_prices)

        volume_series = window["Volume"].ffill()
        avg_volume = volume_series.rolling(20).mean()
        latest_avg = avg_volume.iloc[-1] if len(avg_volume) else np.nan
        if np.isnan(latest_avg) or latest_avg <= 0:
            volume_ratio = 1.0
        else:
            volume_ratio = float(volume_series.iloc[-1] / latest_avg)
        volume_ratio = float(np.clip(volume_ratio, 0.1, 10.0))

        return {
            "volatility": volatility,
            "momentum_5d": momentum_5d,
            "momentum_20d": momentum_20d,
            "max_drawdown": max_drawdown,
            "rsi": rsi_value,
            "volume_ratio": volume_ratio,
        }

    def _feature_vector(self, metrics: Dict[str, float]) -> np.ndarray:
        try:
            values = [float(metrics[name]) for name in self.feature_names]
        except KeyError as exc:
            raise MarketDataError(f"Missing feature value: {exc.args[0]}") from exc
        return np.array(values, dtype=float)

    def _predict_risk(self, feature_vector: np.ndarray) -> float:
        standardized = (feature_vector - self.feature_means) / self.feature_stds
        prediction = float(np.dot(standardized, self.coefficients) + self.intercept)
        return float(np.clip(prediction, 0.0, 1.0))

    @staticmethod
    def _label_risk(score: float) -> str:
        if score >= 0.7:
            return "High"
        if score >= 0.4:
            return "Medium"
        return "Low"

    def _score_symbol(self, symbol: str) -> Dict[str, Any]:
        history = self._download_history(symbol)
        metrics = self._compute_metrics(history)
        feature_vector = self._feature_vector(metrics)
        risk_score = self._predict_risk(feature_vector)

        return {
            "symbol": symbol,
            "risk_score": round(risk_score, 4),
            "risk_level": self._label_risk(risk_score),
            "volatility": round(metrics["volatility"], 4),
            "momentum_5d": round(metrics["momentum_5d"], 4),
            "momentum_20d": round(metrics["momentum_20d"], 4),
            "rsi": round(metrics["rsi"], 2),
            "max_drawdown": round(metrics["max_drawdown"], 4),
            "volume_ratio": round(metrics["volume_ratio"], 4),
            "last_updated": datetime.now().isoformat(),
        }

    def _collect_scores(self) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        errors: Dict[str, str] = {}

        for symbol in self.symbols:
            try:
                result = self._score_symbol(symbol)
            except MarketDataError as exc:
                errors[symbol] = str(exc)
                logger.warning("⚠️ %s", exc)
                continue

            results.append(result)

        if not results:
            details = "; ".join(f"{sym}: {msg}" for sym, msg in errors.items()) or "No market data available"
            raise MarketDataError(details)

        return results

    def get_real_time_risk_scores(self, force_refresh: bool = False) -> List[Dict[str, Any]]:
        if not force_refresh and self.cache and not self._should_refresh():
            logger.info("📊 Returning cached risk data")
            return self.cache

        try:
            results = self._collect_scores()
        except MarketDataError as exc:
            logger.error("❌ Failed to refresh risk scores: %s", exc)
            if self.cache:
                logger.warning("Serving last known risk data from %s", self.last_update)
                return self.cache
            raise

        self.cache = results
        self.last_update = datetime.now()
        logger.info("📊 Calculated risk for %s companies", len(results))
        return results

    def get_company_risk(self, symbol: str) -> Dict[str, Any]:
        symbol = symbol.upper()
        cached = next((item for item in self.cache if item["symbol"] == symbol), None)

        try:
            result = self._score_symbol(symbol)
        except MarketDataError as exc:
            if cached:
                logger.warning("Returning cached risk for %s due to error: %s", symbol, exc)
                return cached
            raise

        for idx, item in enumerate(self.cache):
            if item["symbol"] == symbol:
                self.cache[idx] = result
                break
        else:
            self.cache.append(result)

        return result

    def get_model_performance_data(self) -> Dict[str, Dict[str, Any]]:
        if not self.model_metrics:
            return {}

        return {
            "Linear Regression": {
                "mse": round(float(self.model_metrics.get("mse", 0.0)), 6),
                "rmse": round(float(self.model_metrics.get("rmse", 0.0)), 6),
                "validation_samples": int(self.model_metrics.get("validation_samples", 0)),
            }
        }

    def metadata(self) -> Dict[str, Any]:
        return self.training_info


try:
    risk_service = RiskModelService(MODEL_ARTIFACT, DEFAULT_SYMBOLS)
except Exception as exc:  # pragma: no cover - initialization guard
    logger.error("Unable to initialize risk model service: %s", exc)
    risk_service = None


def require_risk_service() -> RiskModelService:
    """Ensure the risk service is available for request handling."""
    if risk_service is None:
        raise HTTPException(status_code=500, detail="Risk model service unavailable")
    return risk_service

@app.get("/", response_model=Dict)
async def root():
    """FinGraph API - Real-Time Financial Risk Assessment"""
    service = require_risk_service()
    metadata = service.metadata()

    return {
        "message": "FinGraph API - Real-Time Financial Risk Assessment",
        "version": "2.0.0",
        "status": "production",
        "features": {
            "real_time_data": True,
            "cache_duration": f"{service.cache_duration // 60} minutes",
            "companies_tracked": len(service.symbols),
            "metrics": ["risk_score", "volatility", "momentum_5d", "momentum_20d", "rsi", "max_drawdown"]
        },
        "endpoints": {
            "health": "/health",
            "risk": "/risk",
            "risk_company": "/risk/{symbol}",
            "portfolio": "/portfolio",
            "alerts": "/alerts"
        },
        "model": metadata,
        "last_update": service.last_update.isoformat() if service.last_update else None
    }

@app.get("/health", response_model=HealthStatus)
async def health():
    """Health check with real-time status"""
    service = require_risk_service()
    try:
        risk_data = service.get_real_time_risk_scores()
    except MarketDataError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    return HealthStatus(
        status="healthy",
        data_available=len(risk_data) > 0,
        last_update=datetime.now().isoformat(),
        companies_count=len(risk_data)
    )

@app.get("/risk", response_model=List[RiskScore])
async def get_all_risks(
    risk_level: Optional[str] = Query(None, description="Filter by risk level: Low, Medium, High"),
    sort_by: str = Query("risk_score", description="Sort by risk_score or symbol"),
    limit: int = Query(10, ge=1, le=100),
    refresh: bool = Query(False, description="Force refresh data")
):
    """Get real-time risk scores for all companies"""

    service = require_risk_service()

    if refresh:
        service.last_update = None

    try:
        risk_data = service.get_real_time_risk_scores(force_refresh=refresh)
    except MarketDataError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    if not risk_data:
        raise HTTPException(status_code=503, detail="Unable to fetch market data")
    
    # Convert to DataFrame for easier filtering
    df = pd.DataFrame(risk_data)
    
    # Filter by risk level if specified
    if risk_level:
        df = df[df['risk_level'].str.lower() == risk_level.lower()]
    
    # Sort
    ascending = sort_by != "risk_score"
    df = df.sort_values(sort_by, ascending=ascending)
    
    # Limit
    df = df.head(limit)
    
    # Convert to response format
    return [
        RiskScore(
            symbol=row['symbol'],
            risk_score=row['risk_score'],
            risk_level=row['risk_level'],
            volatility=row['volatility']
        )
        for _, row in df.iterrows()
    ]

@app.get("/risk/{symbol}", response_model=RiskScore)
async def get_company_risk(symbol: str):
    """Get real-time risk for specific company"""
    service = require_risk_service()
    symbol = symbol.upper()

    try:
        risk_data = service.get_real_time_risk_scores()
    except MarketDataError as exc:
        if service.cache:
            risk_data = service.cache
        else:
            raise HTTPException(status_code=503, detail=str(exc))

    for company in risk_data:
        if company['symbol'] == symbol:
            return RiskScore(
                symbol=company['symbol'],
                risk_score=company['risk_score'],
                risk_level=company['risk_level'],
                volatility=company['volatility']
            )

    try:
        company = service.get_company_risk(symbol)
    except MarketDataError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    return RiskScore(
        symbol=company['symbol'],
        risk_score=company['risk_score'],
        risk_level=company['risk_level'],
        volatility=company['volatility']
    )

@app.get("/portfolio")
async def get_portfolio_summary():
    """Real-time portfolio overview"""
    service = require_risk_service()
    try:
        risk_data = service.get_real_time_risk_scores()
    except MarketDataError as exc:
        if service.cache:
            risk_data = service.cache
        else:
            raise HTTPException(status_code=503, detail=str(exc))

    if not risk_data:
        raise HTTPException(status_code=503, detail="Unable to fetch market data")

    df = pd.DataFrame(risk_data)
    
    portfolio = {
        "timestamp": datetime.now().isoformat(),
        "companies_analyzed": len(df),
        "risk_distribution": df['risk_level'].value_counts().to_dict(),
        "average_risk_score": float(df['risk_score'].mean()),
        "average_volatility": float(df['volatility'].mean()),
        "high_momentum_stocks": df.nlargest(3, 'momentum_5d')[['symbol', 'momentum_5d']].to_dict('records'),
        "model_performance": service.get_model_performance_data(),
        "market_summary": {
            "most_risky": df.nlargest(1, 'risk_score').iloc[0]['symbol'],
            "least_risky": df.nsmallest(1, 'risk_score').iloc[0]['symbol'],
            "highest_volatility": df.nlargest(1, 'volatility').iloc[0]['symbol'],
            "best_momentum": df.nlargest(1, 'momentum_20d').iloc[0]['symbol'] if 'momentum_20d' in df.columns else 'N/A'
        }
    }
    
    return portfolio

@app.get("/alerts")
async def get_risk_alerts(threshold: float = Query(0.7, ge=0.0, le=1.0)):
    """Real-time risk alerts"""
    service = require_risk_service()
    try:
        risk_data = service.get_real_time_risk_scores()
    except MarketDataError as exc:
        if service.cache:
            risk_data = service.cache
        else:
            raise HTTPException(status_code=503, detail=str(exc))

    if not risk_data:
        raise HTTPException(status_code=503, detail="Unable to fetch market data")

    df = pd.DataFrame(risk_data)
    high_risk = df[df['risk_score'] >= threshold]
    
    alerts = []
    for _, row in high_risk.iterrows():
        alerts.append({
            "symbol": row['symbol'],
            "risk_score": row['risk_score'],
            "risk_level": row['risk_level'],
            "volatility": row['volatility'],
            "momentum_5d": row.get('momentum_5d', 0),
            "rsi": row.get('rsi', 50),
            "message": f"{row['symbol']} risk score {row['risk_score']:.3f} exceeds threshold {threshold:.2f}",
            "timestamp": row['last_updated']
        })
    
    return {
        "threshold": threshold,
        "alert_count": len(alerts),
        "alerts": alerts,
        "generated_at": datetime.now().isoformat(),
        "market_status": "open" if datetime.now().hour >= 9 and datetime.now().hour < 16 else "closed"
    }

def run_server():
    """Run the API server"""
    print("🚀 Starting FinGraph Real-Time API...")
    if risk_service is None:
        raise RuntimeError("Risk model service unavailable")

    print(f"📊 Tracking {len(risk_service.symbols)} companies")
    print(f"⏱️ Cache duration: {risk_service.cache_duration} seconds")
    print("✅ Ready to serve real-time risk assessments")

    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    run_server()
