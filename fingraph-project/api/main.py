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

# Global data loader
loader = DataLoader()

@app.get("/", response_model=Dict)
async def root():
    """API root"""
    return {
        "message": "FinGraph API - Financial Risk Assessment",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "risk_all": "/risk",
            "risk_company": "/risk/{symbol}",
            "portfolio": "/portfolio"
        }
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