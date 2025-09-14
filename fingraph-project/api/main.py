"""
FinGraph API Service - Serves your existing temporal_integration results
"""

import os
import sys
import json
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import glob
import statistics
from collections import defaultdict

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

app = FastAPI(
    title="FinGraph API",
    description="Financial Risk Assessment API - Serves existing results",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Response models
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
    
class ModelPerformance(BaseModel):
    model_name: str
    mse: float
    rmse: float
    timestamp: str

class DataDriftCheck(BaseModel):
    status: str
    risk_drift_percentage: float
    volatility_drift_percentage: float
    check_timestamp: str
    threshold_exceeded: bool

class SystemMonitoring(BaseModel):
    api_status: str
    data_freshness_hours: float
    model_count: int
    last_prediction_time: Optional[str]
    system_uptime: str

class DataLoader:
    """Loads from YOUR existing temporal_integration results"""
    
    def __init__(self):
        self.data_dir = os.path.join(project_root, "data", "temporal_integration")
        self.risk_data = None
        self.summary_data = None
        self.last_loaded = None
    
    def load_results(self) -> bool:
        """Load your existing results"""
        try:
            if not os.path.exists(self.data_dir):
                return False
            
            files = os.listdir(self.data_dir)
            
            # Load risk predictions
            prediction_files = [f for f in files if f.startswith('risk_predictions_')]
            if prediction_files:
                latest_file = max(prediction_files)
                self.risk_data = pd.read_csv(os.path.join(self.data_dir, latest_file))
            
            # Load summary
            summary_files = [f for f in files if f.startswith('dashboard_summary_')]
            if summary_files:
                latest_file = max(summary_files)
                with open(os.path.join(self.data_dir, latest_file), 'r') as f:
                    self.summary_data = json.load(f)
            
            self.last_loaded = datetime.now()
            return True
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
        
    def get_model_performance_data(self):
        """Extract model performance from your summary data"""
        if not self.summary_data or 'model_performance' not in self.summary_data:
            return []
        
        performance_list = []
        for model_name, metrics in self.summary_data['model_performance'].items():
            performance_list.append({
                'model_name': model_name,
                'mse': float(metrics.get('mse', 0.0)),
                'rmse': float(metrics.get('rmse', 0.0)),
                'timestamp': self.last_loaded.isoformat() if self.last_loaded else datetime.now().isoformat()
            })
        return performance_list
    
    def check_data_drift_status(self):
        """Analyze your risk data for drift indicators"""
        if self.risk_data is None or len(self.risk_data) == 0:
            return {
                'status': 'no_data',
                'risk_drift_percentage': 0.0,
                'volatility_drift_percentage': 0.0,
                'check_timestamp': datetime.now().isoformat(),
                'threshold_exceeded': False
            }
        
        # Based on your actual data: AAPL=0.299, TSLA=0.863, avg=0.511
        current_risk_avg = float(self.risk_data['risk_score'].mean())
        expected_risk_avg = 0.511  # From your dashboard_summary
        risk_drift = abs(current_risk_avg - expected_risk_avg) / expected_risk_avg * 100
        
        # Based on your actual volatility data  
        current_vol_avg = float(self.risk_data['volatility'].mean())
        expected_vol_avg = 0.35  # Estimated from your data range
        vol_drift = abs(current_vol_avg - expected_vol_avg) / expected_vol_avg * 100
        
        drift_threshold = 15.0  # 15% drift threshold
        threshold_exceeded = risk_drift > drift_threshold or vol_drift > drift_threshold
        
        status = 'warning' if threshold_exceeded else 'stable'
        
        return {
            'status': status,
            'risk_drift_percentage': float(risk_drift),
            'volatility_drift_percentage': float(vol_drift),
            'check_timestamp': datetime.now().isoformat(),
            'threshold_exceeded': threshold_exceeded
        }
    
    def get_system_monitoring_data(self):
        """Get system health metrics"""
        if self.last_loaded:
            hours_since_load = (datetime.now() - self.last_loaded).total_seconds() / 3600
            data_freshness = float(hours_since_load)
        else:
            data_freshness = 999.0  # Very stale
        
        model_count = len(self.summary_data.get('model_performance', {})) if self.summary_data else 0
        
        # System uptime (simplified - when data was last loaded)
        uptime = f"{data_freshness:.1f} hours since data load"
        
        return {
            'api_status': 'healthy' if data_freshness < 48 else 'stale',
            'data_freshness_hours': data_freshness,
            'model_count': model_count,
            'last_prediction_time': self.last_loaded.isoformat() if self.last_loaded else None,
            'system_uptime': uptime
        }

# Global data loader
loader = DataLoader()

@app.get("/", response_model=Dict)
async def root():
    """FinGraph API - Production-Ready Financial Risk Assessment"""
    return {
        "message": "FinGraph API - Financial Risk Assessment with Production Monitoring",
        "version": "1.0.0",
        "status": "production-ready",
        "endpoints": {
            "core": {
                "health": "/health",
                "risk_all": "/risk",
                "risk_company": "/risk/{symbol}",
                "portfolio": "/portfolio",
                "alerts": "/alerts"
            },
            "monitoring": {
                "performance": "/monitoring/performance",
                "drift": "/monitoring/drift",
                "system": "/monitoring/system",
                "dashboard": "/monitoring/dashboard"
            }
        },
        "documentation": {
            "interactive": "/docs",
            "redoc": "/redoc"
        },
        "data_source": "FinGraph Temporal Integration Results"
    }

@app.get("/health", response_model=HealthStatus)
async def health():
    """Health check - uses YOUR results"""
    data_loaded = loader.load_results()
    
    return HealthStatus(
        status="healthy" if data_loaded else "no_data",
        data_available=data_loaded,
        last_update=loader.last_loaded.isoformat() if loader.last_loaded else None,
        companies_count=len(loader.risk_data) if loader.risk_data is not None else 0
    )

@app.get("/risk", response_model=List[RiskScore])
async def get_all_risks(
    risk_level: Optional[str] = Query(None, description="Filter by risk level: Low, Medium, High"),
    sort_by: str = Query("risk_score", description="Sort by risk_score or symbol"),
    limit: int = Query(10, ge=1, le=100)
):
    """Get all company risk scores from YOUR results"""
    if not loader.load_results() or loader.risk_data is None:
        raise HTTPException(status_code=503, detail="No risk data available. Run the pipeline first.")
    
    df = loader.risk_data.copy()
    
    # Filter by risk level
    if risk_level:
        df = df[df['risk_level'].str.lower() == risk_level.lower()]
    
    # Sort
    ascending = sort_by != "risk_score"  # Risk score descending, symbol ascending
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
    """Get risk for specific company from YOUR results"""
    if not loader.load_results() or loader.risk_data is None:
        raise HTTPException(status_code=503, detail="No risk data available. Run the pipeline first.")
    
    symbol = symbol.upper()
    company_data = loader.risk_data[loader.risk_data['symbol'] == symbol]
    
    if company_data.empty:
        raise HTTPException(status_code=404, detail=f"Company {symbol} not found in results")
    
    row = company_data.iloc[0]
    return RiskScore(
        symbol=row['symbol'],
        risk_score=row['risk_score'],
        risk_level=row['risk_level'],
        volatility=row['volatility']
    )

@app.get("/portfolio")
async def get_portfolio_summary():
    """Portfolio overview from YOUR results"""
    if not loader.load_results():
        raise HTTPException(status_code=503, detail="No data available. Run the pipeline first.")
    
    portfolio = {
        "timestamp": datetime.now().isoformat(),
        "companies_analyzed": 0,
        "risk_distribution": {},
        "average_risk_score": 0.0,
        "model_performance": {}
    }
    
    # Risk data summary
    if loader.risk_data is not None:
        portfolio["companies_analyzed"] = len(loader.risk_data)
        portfolio["risk_distribution"] = loader.risk_data['risk_level'].value_counts().to_dict()
        portfolio["average_risk_score"] = float(loader.risk_data['risk_score'].mean())
    
    # Model performance from summary
    if loader.summary_data and 'model_performance' in loader.summary_data:
        portfolio["model_performance"] = loader.summary_data['model_performance']
    
    return portfolio

@app.get("/alerts")
async def get_risk_alerts(threshold: float = Query(0.7, ge=0.0, le=1.0)):
    """Risk alerts from YOUR results"""
    if not loader.load_results() or loader.risk_data is None:
        raise HTTPException(status_code=503, detail="No risk data available")
    
    high_risk = loader.risk_data[loader.risk_data['risk_score'] >= threshold]
    
    alerts = []
    for _, row in high_risk.iterrows():
        alerts.append({
            "symbol": row['symbol'],
            "risk_score": row['risk_score'],
            "risk_level": row['risk_level'],
            "message": f"{row['symbol']} risk score {row['risk_score']:.3f} exceeds threshold {threshold:.2f}"
        })
    
    return {
        "threshold": threshold,
        "alert_count": len(alerts),
        "alerts": alerts,
        "generated_at": datetime.now().isoformat()
    }

@app.get("/monitoring/performance", response_model=List[ModelPerformance])
async def get_model_performance():
    """Get model performance metrics from your temporal integration results"""
    if not loader.load_results():
        raise HTTPException(status_code=503, detail="No model performance data available. Run temporal integration first.")
    
    performance_data = loader.get_model_performance_data()
    
    if not performance_data:
        raise HTTPException(status_code=404, detail="No performance metrics found in results")
    
    return [
        ModelPerformance(
            model_name=item['model_name'],
            mse=item['mse'],
            rmse=item['rmse'],
            timestamp=item['timestamp']
        )
        for item in performance_data
    ]

@app.get("/monitoring/drift", response_model=DataDriftCheck)
async def check_data_drift():
    """Check for data drift in risk predictions"""
    if not loader.load_results():
        raise HTTPException(status_code=503, detail="No data available for drift analysis")
    
    drift_data = loader.check_data_drift_status()
    
    return DataDriftCheck(
        status=drift_data['status'],
        risk_drift_percentage=drift_data['risk_drift_percentage'],
        volatility_drift_percentage=drift_data['volatility_drift_percentage'],
        check_timestamp=drift_data['check_timestamp'],
        threshold_exceeded=drift_data['threshold_exceeded']
    )

@app.get("/monitoring/system", response_model=SystemMonitoring)
async def get_system_monitoring():
    """Get overall system health and monitoring data"""
    if not loader.load_results():
        raise HTTPException(status_code=503, detail="System monitoring unavailable")
    
    system_data = loader.get_system_monitoring_data()
    
    return SystemMonitoring(
        api_status=system_data['api_status'],
        data_freshness_hours=system_data['data_freshness_hours'],
        model_count=system_data['model_count'],
        last_prediction_time=system_data['last_prediction_time'],
        system_uptime=system_data['system_uptime']
    )

@app.get("/monitoring/dashboard")
async def get_monitoring_dashboard():
    """Complete monitoring dashboard data for operations team"""
    if not loader.load_results():
        raise HTTPException(status_code=503, detail="Monitoring dashboard unavailable")
    
    # Gather all monitoring data
    performance = loader.get_model_performance_data()
    drift = loader.check_data_drift_status()
    system = loader.get_system_monitoring_data()
    
    return {
        "timestamp": datetime.now().isoformat(),
        "overall_status": "healthy" if system['api_status'] == 'healthy' and drift['status'] == 'stable' else "warning",
        "model_performance": performance,
        "data_drift": drift,
        "system_health": system,
        "summary": {
            "total_models": len(performance),
            "best_model": min(performance, key=lambda x: x['mse'])['model_name'] if performance else None,
            "drift_detected": drift['threshold_exceeded'],
            "system_operational": system['api_status'] == 'healthy'
        }
    }

def run_server():
    """Run the API server"""
    print("🚀 Starting FinGraph API...")
    print(f"📂 Looking for data in: {loader.data_dir}")
    
    # Try to load data on startup
    if loader.load_results():
        print(f"✅ Found data: {len(loader.risk_data)} companies")
    else:
        print("⚠️ No data found. Run temporal integration first.")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    run_server()