# FinGraph API Reference

## Base URL
http://localhost:8000

## Authentication
None required for current deployment

## Core Endpoints

### GET /
**Description**: API information and endpoint directory  
**Response**: JSON with endpoint map  
**Example**: 
```bash
curl http://localhost:8000/
GET /health
Description: System health check
Response: HealthStatus model
Example:
bashcurl http://localhost:8000/health
GET /risk
Description: Risk scores for all companies
Query Parameters:

risk_level (optional): Filter by "Low", "Medium", "High"
sort_by (optional): "risk_score" or "symbol"
limit (optional): Max results (1-100)

Example:
bashcurl "http://localhost:8000/risk?risk_level=High&limit=5"
GET /risk/{symbol}
Description: Risk score for specific company
Path Parameters: symbol - Stock ticker (e.g., AAPL)
Example:
bashcurl http://localhost:8000/risk/TSLA
GET /portfolio
Description: Portfolio overview with risk distribution
Response: Portfolio summary with model performance
Example:
bashcurl http://localhost:8000/portfolio
GET /alerts
Description: Risk alerts above threshold
Query Parameters: threshold (0.0-1.0, default 0.7)
Example:
bashcurl "http://localhost:8000/alerts?threshold=0.6"
Monitoring Endpoints
GET /monitoring/performance
Description: Model performance metrics
Response: Array of ModelPerformance objects
Example:
bashcurl http://localhost:8000/monitoring/performance
GET /monitoring/drift
Description: Data drift detection status
Response: DataDriftCheck object
Example:
bashcurl http://localhost:8000/monitoring/drift
GET /monitoring/system
Description: System health monitoring
Response: SystemMonitoring object
Example:
bashcurl http://localhost:8000/monitoring/system
GET /monitoring/dashboard
Description: Complete monitoring dashboard data
Response: Comprehensive monitoring JSON
Example:
bashcurl http://localhost:8000/monitoring/dashboard
Response Models
RiskScore
json{
  "symbol": "TSLA",
  "risk_score": 0.863,
  "risk_level": "High", 
  "volatility": 0.730
}
ModelPerformance
json{
  "model_name": "Simple GNN",
  "mse": 0.0223,
  "rmse": 0.1495,
  "timestamp": "2025-09-14T10:41:11.138749"
}
Error Codes

200: Success
404: Company/data not found
503: Service unavailable (no data)

Rate Limits
None currently enforced
Interactive Documentation
Visit http://localhost:8000/docs for Swagger UI