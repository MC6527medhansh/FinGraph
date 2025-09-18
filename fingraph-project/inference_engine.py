"""
Production Inference Engine for Real-time Risk Predictions
Serves live predictions with proper feature engineering and graph construction
"""

import torch
import numpy as np
import pandas as pd
import yfinance as yf
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import json
import logging
import pickle
from dataclasses import dataclass
import hashlib
import time
from concurrent.futures import ThreadPoolExecutor
import redis
import asyncio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)


@dataclass
class PredictionRequest:
    """Request for risk prediction"""
    symbols: List[str]
    lookback_days: int = 60
    use_cache: bool = True
    include_confidence: bool = True


class PredictionResponse(BaseModel):
    """Response with predictions"""
    timestamp: str
    predictions: Dict[str, Dict[str, float]]
    confidence_intervals: Optional[Dict[str, Dict[str, Tuple[float, float]]]]
    market_regime: str
    model_version: str
    latency_ms: float


class ModelCache:
    """In-memory cache for models and features"""
    
    def __init__(self, ttl_seconds: int = 300):
        self.ttl = ttl_seconds
        self.cache = {}
        self.timestamps = {}
        
    def get(self, key: str) -> Optional[Any]:
        """Get from cache if not expired"""
        if key in self.cache:
            if time.time() - self.timestamps[key] < self.ttl:
                return self.cache[key]
            else:
                del self.cache[key]
                del self.timestamps[key]
        return None
    
    def set(self, key: str, value: Any):
        """Set cache value"""
        self.cache[key] = value
        self.timestamps[key] = time.time()
    
    def clear_expired(self):
        """Clear expired entries"""
        current_time = time.time()
        expired_keys = [k for k, t in self.timestamps.items() 
                       if current_time - t > self.ttl]
        for key in expired_keys:
            del self.cache[key]
            del self.timestamps[key]


class RealTimeInference:
    """
    Production inference engine for real-time predictions.
    
    Features:
    1. Live market data ingestion
    2. Real-time feature calculation
    3. Dynamic graph construction
    4. Model ensemble predictions
    5. Confidence intervals
    6. Caching and optimization
    """
    
    def __init__(self, 
                 model_path: str = "models/production/latest_model.pt",
                 config_path: str = "configs/inference_config.json"):
        """
        Initialize inference engine.
        
        Args:
            model_path: Path to trained model
            config_path: Path to inference configuration
        """
        self.model_path = Path(model_path)
        self.config = self._load_config(config_path)
        self.model = None
        self.graph_builder = None
        self.feature_extractor = None
        self.cache = ModelCache(ttl_seconds=self.config.get('cache_ttl', 300))
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        self._load_model()
        
        # Initialize components
        self._initialize_components()
        
        # Market regime detector
        self.regime_thresholds = {
            'bull': 0.02,  # > 2% average return
            'bear': -0.02,  # < -2% average return
            'volatile': 0.03,  # > 3% volatility
        }
        
    def _load_config(self, config_path: str) -> Dict:
        """Load inference configuration"""
        config_file = Path(config_path)
        if config_file.exists():
            with open(config_file, 'r') as f:
                return json.load(f)
        else:
            # Default configuration
            return {
                'cache_ttl': 300,
                'max_batch_size': 50,
                'feature_lookback': 60,
                'correlation_threshold': 0.3,
                'confidence_n_samples': 100,
                'timeout_seconds': 30
            }
    
    def _load_model(self):
        """Load trained model with validation"""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        # Load model package
        model_package = torch.load(self.model_path, map_location=self.device)
        
        # Validate checksum
        if 'checksum' in model_package:
            logger.info(f"Model checksum: {model_package['checksum']}")
        
        # Initialize model architecture
        from gnn_trainer import RealFinancialGNN  # Import from previous artifact
        
        model_config = model_package['model_config']
        self.model = RealFinancialGNN(
            num_node_features=model_config['num_node_features'],
            hidden_dim=model_config['hidden_dim']
        ).to(self.device)
        
        # Load weights
        self.model.load_state_dict(model_package['model_state_dict'])
        self.model.eval()
        
        # Store metadata
        self.model_version = model_package.get('version', 'unknown')
        self.model_metadata = model_package.get('metadata', {})
        
        logger.info(f"Loaded model version: {self.model_version}")
    
    def _initialize_components(self):
        """Initialize feature extractor and graph builder"""
        # Feature extractor
        self.feature_extractor = FeatureExtractor(
            lookback_days=self.config['feature_lookback']
        )
        
        # Graph builder
        self.graph_builder = LiveGraphBuilder(
            correlation_threshold=self.config['correlation_threshold']
        )
    
    def predict(self, request: PredictionRequest) -> PredictionResponse:
        """
        Generate real-time predictions.
        
        Args:
            request: Prediction request with symbols
            
        Returns:
            Prediction response with risk scores
        """
        start_time = time.time()
        
        # Check cache
        cache_key = self._get_cache_key(request)
        if request.use_cache:
            cached_result = self.cache.get(cache_key)
            if cached_result:
                logger.info("Returning cached prediction")
                return cached_result
        
        # Fetch live market data
        market_data = self._fetch_live_data(request.symbols, request.lookback_days)
        
        # Calculate features for each symbol
        features = self._calculate_live_features(market_data, request.symbols)
        
        # Build live graph
        graph = self.graph_builder.build_live_graph(features, market_data)
        
        # Run inference
        predictions = self._run_inference(graph)
        
        # Calculate confidence intervals if requested
        confidence_intervals = None
        if request.include_confidence:
            confidence_intervals = self._calculate_confidence_intervals(
                graph, n_samples=self.config['confidence_n_samples']
            )
        
        # Detect market regime
        market_regime = self._detect_market_regime(market_data)
        
        # Prepare response
        latency_ms = (time.time() - start_time) * 1000
        
        response = PredictionResponse(
            timestamp=datetime.now().isoformat(),
            predictions=predictions,
            confidence_intervals=confidence_intervals,
            market_regime=market_regime,
            model_version=self.model_version,
            latency_ms=latency_ms
        )
        
        # Cache result
        if request.use_cache:
            self.cache.set(cache_key, response)
        
        logger.info(f"Inference completed in {latency_ms:.1f}ms")
        
        return response
    
    def _fetch_live_data(self, symbols: List[str], lookback_days: int) -> pd.DataFrame:
        """Fetch live market data with parallel downloads"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days + 10)  # Buffer for weekends
        
        all_data = []
        
        # Use thread pool for parallel downloads
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = []
            for symbol in symbols:
                future = executor.submit(self._download_symbol, symbol, start_date, end_date)
                futures.append((symbol, future))
            
            for symbol, future in futures:
                try:
                    data = future.result(timeout=self.config['timeout_seconds'])
                    if data is not None:
                        all_data.append(data)
                except Exception as e:
                    logger.error(f"Failed to fetch {symbol}: {e}")
        
        if not all_data:
            raise ValueError("No market data available")
        
        combined = pd.concat(all_data, ignore_index=True)
        return combined
    
    def _download_symbol(self, symbol: str, start_date: datetime, end_date: datetime) -> Optional[pd.DataFrame]:
        """Download single symbol data"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_date, end=end_date)
            
            if len(data) < 20:  # Minimum required data points
                return None
            
            data['Symbol'] = symbol
            data = data.reset_index()
            data['Date'] = pd.to_datetime(data['Date']).dt.tz_localize(None)
            
            return data
            
        except Exception as e:
            logger.warning(f"Download failed for {symbol}: {e}")
            return None
    
    def _calculate_live_features(self, 
                                market_data: pd.DataFrame,
                                symbols: List[str]) -> Dict[str, np.ndarray]:
        """Calculate features for live prediction"""
        features = {}
        
        for symbol in symbols:
            symbol_data = market_data[market_data['Symbol'] == symbol].copy()
            
            if len(symbol_data) < 20:
                continue
            
            # Calculate technical features
            symbol_features = self.feature_extractor.extract(symbol_data)
            
            if symbol_features is not None:
                features[symbol] = symbol_features
        
        return features
    
    def _run_inference(self, graph) -> Dict[str, Dict[str, float]]:
        """Run model inference on graph"""
        with torch.no_grad():
            # Move graph to device
            graph = graph.to(self.device)
            
            # Forward pass
            outputs = self.model(
                graph.x,
                graph.edge_index,
                graph.edge_attr,
                graph.batch
            )
            
            # Extract predictions
            risk_scores = outputs['risk'].squeeze().cpu().numpy()
            volatility = outputs['volatility'].squeeze().cpu().numpy()
            returns = outputs['return'].squeeze().cpu().numpy()
        
        # Map back to symbols
        predictions = {}
        
        if hasattr(graph, 'symbol_mapping'):
            for symbol, node_idx in graph.symbol_mapping.items():
                if node_idx < len(risk_scores):
                    predictions[symbol] = {
                        'risk_score': float(risk_scores[node_idx]),
                        'volatility': float(volatility[node_idx]),
                        'expected_return': float(returns[node_idx]),
                        'risk_level': self._categorize_risk(risk_scores[node_idx])
                    }
        else:
            # Fallback if no mapping
            predictions['portfolio'] = {
                'risk_score': float(np.mean(risk_scores)),
                'volatility': float(np.mean(volatility)),
                'expected_return': float(np.mean(returns)),
                'risk_level': self._categorize_risk(np.mean(risk_scores))
            }
        
        return predictions
    
    def _calculate_confidence_intervals(self,
                                       graph,
                                       n_samples: int = 100) -> Dict[str, Dict[str, Tuple[float, float]]]:
        """Calculate confidence intervals using dropout uncertainty"""
        # Enable dropout for uncertainty estimation
        self.model.train()
        
        all_predictions = {
            'risk': [],
            'volatility': [],
            'return': []
        }
        
        # Multiple forward passes with dropout
        for _ in range(n_samples):
            with torch.no_grad():
                outputs = self.model(
                    graph.x.to(self.device),
                    graph.edge_index.to(self.device),
                    graph.edge_attr.to(self.device) if graph.edge_attr is not None else None,
                    graph.batch.to(self.device) if graph.batch is not None else None
                )
                
                all_predictions['risk'].append(outputs['risk'].squeeze().cpu().numpy())
                all_predictions['volatility'].append(outputs['volatility'].squeeze().cpu().numpy())
                all_predictions['return'].append(outputs['return'].squeeze().cpu().numpy())
        
        # Calculate confidence intervals
        confidence_intervals = {}
        
        for metric in ['risk', 'volatility', 'return']:
            predictions = np.array(all_predictions[metric])
            lower = np.percentile(predictions, 2.5, axis=0)
            upper = np.percentile(predictions, 97.5, axis=0)
            
            if hasattr(graph, 'symbol_mapping'):
                for symbol, node_idx in graph.symbol_mapping.items():
                    if symbol not in confidence_intervals:
                        confidence_intervals[symbol] = {}
                    
                    if node_idx < len(lower):
                        confidence_intervals[symbol][f'{metric}_ci'] = (
                            float(lower[node_idx]),
                            float(upper[node_idx])
                        )
        
        # Set back to eval mode
        self.model.eval()
        
        return confidence_intervals
    
    def _detect_market_regime(self, market_data: pd.DataFrame) -> str:
        """Detect current market regime"""
        # Calculate recent market statistics
        recent_data = market_data.tail(20)
        
        if len(recent_data) < 5:
            return "unknown"
        
        # Calculate average return and volatility
        returns = recent_data.groupby('Symbol')['Close'].pct_change().dropna()
        avg_return = returns.mean()
        volatility = returns.std() * np.sqrt(252)
        
        # Classify regime
        if volatility > self.regime_thresholds['volatile']:
            return "volatile"
        elif avg_return > self.regime_thresholds['bull']:
            return "bull"
        elif avg_return < self.regime_thresholds['bear']:
            return "bear"
        else:
            return "neutral"
    
    def _categorize_risk(self, risk_score: float) -> str:
        """Categorize risk score into levels"""
        if risk_score >= 0.7:
            return "high"
        elif risk_score >= 0.4:
            return "medium"
        else:
            return "low"
    
    def _get_cache_key(self, request: PredictionRequest) -> str:
        """Generate cache key for request"""
        key_parts = [
            ','.join(sorted(request.symbols)),
            str(request.lookback_days),
            datetime.now().strftime('%Y%m%d_%H')  # Cache for 1 hour
        ]
        key_str = '_'.join(key_parts)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def batch_predict(self, requests: List[PredictionRequest]) -> List[PredictionResponse]:
        """Process batch of prediction requests"""
        results = []
        
        # Group requests by similar parameters for efficiency
        grouped = {}
        for request in requests:
            key = (request.lookback_days, request.use_cache, request.include_confidence)
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(request)
        
        # Process each group
        for params, group_requests in grouped.items():
            # Combine symbols from all requests in group
            all_symbols = list(set(sum([r.symbols for r in group_requests], [])))
            
            # Create combined request
            combined_request = PredictionRequest(
                symbols=all_symbols,
                lookback_days=params[0],
                use_cache=params[1],
                include_confidence=params[2]
            )
            
            # Get predictions for all symbols
            combined_response = self.predict(combined_request)
            
            # Split results back to individual requests
            for request in group_requests:
                individual_predictions = {
                    s: combined_response.predictions[s]
                    for s in request.symbols
                    if s in combined_response.predictions
                }
                
                individual_confidence = None
                if combined_response.confidence_intervals:
                    individual_confidence = {
                        s: combined_response.confidence_intervals[s]
                        for s in request.symbols
                        if s in combined_response.confidence_intervals
                    }
                
                results.append(PredictionResponse(
                    timestamp=combined_response.timestamp,
                    predictions=individual_predictions,
                    confidence_intervals=individual_confidence,
                    market_regime=combined_response.market_regime,
                    model_version=combined_response.model_version,
                    latency_ms=combined_response.latency_ms
                ))
        
        return results
    
    def health_check(self) -> Dict[str, Any]:
        """Check system health"""
        health = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'model_version': self.model_version,
            'device': str(self.device),
            'cache_size': len(self.cache.cache),
            'issues': []
        }
        
        # Check model
        if self.model is None:
            health['status'] = 'unhealthy'
            health['issues'].append('Model not loaded')
        
        # Check cache
        self.cache.clear_expired()
        
        # Test inference
        try:
            test_request = PredictionRequest(
                symbols=['AAPL'],
                lookback_days=30,
                use_cache=False,
                include_confidence=False
            )
            _ = self.predict(test_request)
        except Exception as e:
            health['status'] = 'degraded'
            health['issues'].append(f'Inference test failed: {str(e)}')
        
        return health


class FeatureExtractor:
    """Extract features from market data"""
    
    def __init__(self, lookback_days: int = 60):
        self.lookback_days = lookback_days
    
    def extract(self, data: pd.DataFrame) -> Optional[np.ndarray]:
        """Extract features from symbol data"""
        if len(data) < 20:
            return None
        
        features = []
        
        # Price features
        close_prices = data['Close'].values
        returns = np.diff(close_prices) / close_prices[:-1]
        
        features.extend([
            np.mean(returns[-5:]) if len(returns) >= 5 else 0,  # 5-day return
            np.mean(returns[-20:]) if len(returns) >= 20 else 0,  # 20-day return
            np.std(returns[-20:]) * np.sqrt(252) if len(returns) >= 20 else 0,  # Volatility
        ])
        
        # Technical indicators
        # RSI
        rsi = self._calculate_rsi(close_prices)
        features.append(rsi / 100)
        
        # Bollinger position
        if len(close_prices) >= 20:
            sma = np.mean(close_prices[-20:])
            std = np.std(close_prices[-20:])
            bb_position = (close_prices[-1] - (sma - 2*std)) / (4*std) if std > 0 else 0.5
            features.append(bb_position)
        else:
            features.append(0.5)
        
        # Volume features
        volumes = data['Volume'].values
        if len(volumes) >= 20:
            volume_ratio = volumes[-1] / np.mean(volumes[-20:]) if np.mean(volumes[-20:]) > 0 else 1
            features.append(np.log(volume_ratio + 1))
        else:
            features.append(0)
        
        return np.array(features, dtype=np.float32)
    
    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """Calculate RSI"""
        if len(prices) < period + 1:
            return 50.0
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi


class LiveGraphBuilder:
    """Build graph from live market data"""
    
    def __init__(self, correlation_threshold: float = 0.3):
        self.correlation_threshold = correlation_threshold
    
    def build_live_graph(self, 
                        features: Dict[str, np.ndarray],
                        market_data: pd.DataFrame):
        """Build graph for live inference"""
        import torch
        from torch_geometric.data import Data
        
        # Create node features
        symbols = list(features.keys())
        node_features = np.vstack([features[s] for s in symbols])
        
        # Calculate correlations for edges
        edge_list = []
        edge_features = []
        
        # Calculate recent correlations
        for i, symbol1 in enumerate(symbols):
            for j, symbol2 in enumerate(symbols):
                if i >= j:
                    continue
                
                # Get recent returns for correlation
                data1 = market_data[market_data['Symbol'] == symbol1]['Close'].pct_change().dropna()
                data2 = market_data[market_data['Symbol'] == symbol2]['Close'].pct_change().dropna()
                
                if len(data1) > 10 and len(data2) > 10:
                    # Align data
                    min_len = min(len(data1), len(data2))
                    corr = np.corrcoef(data1.iloc[-min_len:], data2.iloc[-min_len:])[0, 1]
                    
                    if abs(corr) > self.correlation_threshold:
                        edge_list.append([i, j])
                        edge_list.append([j, i])
                        edge_features.append([corr, abs(corr), 1 if corr > 0 else 0])
                        edge_features.append([corr, abs(corr), 1 if corr > 0 else 0])
        
        # Create graph
        x = torch.FloatTensor(node_features)
        
        if edge_list:
            edge_index = torch.LongTensor(edge_list).t().contiguous()
            edge_attr = torch.FloatTensor(edge_features)
        else:
            # Minimal connectivity
            edge_index = torch.LongTensor([[0, 1], [1, 0]]).t() if len(symbols) > 1 else torch.LongTensor([[], []])
            edge_attr = torch.FloatTensor([[0, 0, 0], [0, 0, 0]]) if len(symbols) > 1 else torch.FloatTensor([])
        
        graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        graph.symbol_mapping = {symbol: i for i, symbol in enumerate(symbols)}
        
        return graph


# FastAPI application
app = FastAPI(title="FinGraph Inference API", version="1.0.0")
inference_engine = None


@app.on_event("startup")
async def startup_event():
    """Initialize inference engine on startup"""
    global inference_engine
    inference_engine = RealTimeInference()
    logger.info("Inference engine initialized")


@app.get("/health")
async def health():
    """Health check endpoint"""
    return inference_engine.health_check()


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Generate predictions endpoint"""
    try:
        response = inference_engine.predict(request)
        return response
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch_predict")
async def batch_predict(requests: List[PredictionRequest]):
    """Batch prediction endpoint"""
    try:
        responses = inference_engine.batch_predict(requests)
        return responses
    except Exception as e:
        logger.error(f"Batch prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    # Test inference engine
    engine = RealTimeInference()
    
    test_request = PredictionRequest(
        symbols=['AAPL', 'MSFT', 'GOOGL'],
        lookback_days=60,
        use_cache=True,
        include_confidence=True
    )
    
    response = engine.predict(test_request)
    print(f"Predictions: {response.predictions}")
    print(f"Market regime: {response.market_regime}")
    print(f"Latency: {response.latency_ms:.1f}ms")