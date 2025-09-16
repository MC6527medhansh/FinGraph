import sys
from pathlib import Path
from typing import List

import pandas as pd
import pytest

# Ensure both the repository root and API package directory are importable before other imports
ROOT_DIR = Path(__file__).resolve().parents[2]
PROJECT_DIR = Path(__file__).resolve().parents[1]
for path in (ROOT_DIR, PROJECT_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.append(path_str)

from fastapi.testclient import TestClient

from api import main  # noqa: E402  pylint: disable=wrong-import-position


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """Provide a FastAPI test client with deterministic market data."""

    dates = pd.date_range("2023-01-01", periods=200, freq="D")
    base_prices = [100 + idx for idx in range(len(dates))]

    def _fake_download(symbol: str, *args, **kwargs) -> pd.DataFrame:
        offset = 0.1 if symbol == "AAPL" else 0.2
        volumes: List[int] = [1_000_000 + idx * (50 if symbol == "AAPL" else 75) for idx in range(len(dates))]

        data = {
            "Open": [price + 0.5 for price in base_prices],
            "High": [price + 1.5 for price in base_prices],
            "Low": [price - 1.5 for price in base_prices],
            "Close": [price + offset for price in base_prices],
            "Volume": volumes,
        }
        return pd.DataFrame(data, index=dates)

    monkeypatch.setattr(main.yf, "download", _fake_download)
    monkeypatch.setattr(main.risk_calculator, "symbols", ["AAPL", "MSFT"])
    monkeypatch.setattr(main.risk_calculator, "cache", [])
    monkeypatch.setattr(main.risk_calculator, "last_update", None)
    monkeypatch.setattr(main.risk_calculator, "last_errors", {})

    with TestClient(main.app) as test_client:
        yield test_client


def test_root_endpoint_returns_metadata(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200

    data = response.json()
    assert data["message"].startswith("FinGraph API")
    assert set(data["endpoints"]).issuperset({"health", "risk", "alerts"})
    assert sorted(data["companies"]["tracked"]) == ["AAPL", "MSFT"]


def test_health_endpoint_returns_status_and_companies(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "healthy"
    assert data["data_available"] is True
    assert sorted(data["tracked_symbols"]) == ["AAPL", "MSFT"]
    assert data["companies_count"] == len(data["available_symbols"]) > 0


def test_risk_endpoint_returns_risk_scores(client: TestClient) -> None:
    response = client.get("/risk", params={"limit": 5})
    assert response.status_code == 200

    payload = response.json()
    assert isinstance(payload, list) and payload

    for entry in payload:
        assert entry["symbol"] in {"AAPL", "MSFT"}
        assert 0.0 <= entry["risk_score"] <= 1.0
        assert entry["risk_level"] in {"Low", "Medium", "High"}
        assert entry["volatility"] >= 0.0


def test_risk_detail_endpoint_returns_single_company(client: TestClient) -> None:
    response = client.get("/risk/AAPL")
    assert response.status_code == 200

    result = response.json()
    assert result["symbol"] == "AAPL"
    assert 0.0 <= result["risk_score"] <= 1.0
    assert result["risk_level"] in {"Low", "Medium", "High"}
    assert result["volatility"] >= 0.0


def test_portfolio_endpoint_returns_summary(client: TestClient) -> None:
    response = client.get("/portfolio")
    assert response.status_code == 200

    summary = response.json()
    assert summary["companies_analyzed"] >= 1
    assert "risk_distribution" in summary and isinstance(summary["risk_distribution"], dict)
    assert "market_summary" in summary and isinstance(summary["market_summary"], dict)
    assert summary["data_source"] in {"live", "stored"}


def test_alerts_endpoint_returns_thresholded_alerts(client: TestClient) -> None:
    response = client.get("/alerts", params={"threshold": 0.0})
    assert response.status_code == 200

    alerts = response.json()
    assert alerts["threshold"] == 0.0
    assert alerts["alert_count"] == len(alerts["alerts"])
    for alert in alerts["alerts"]:
        assert alert["symbol"] in {"AAPL", "MSFT"}
        assert 0.0 <= alert["risk_score"] <= 1.0
        assert alert["risk_level"] in {"Low", "Medium", "High"}
        assert alert["volatility"] >= 0.0
