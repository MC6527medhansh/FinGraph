"""
Dynamic Graph Construction from Real Market Data
Builds time-varying graphs with real relationships
"""

import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import networkx as nx
from scipy.stats import spearmanr
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DynamicGraphBuilder:
    """
    Builds dynamic financial graphs that change over time.
    
    Key features:
    1. Time-varying edges based on rolling correlations
    2. Sector relationships from actual industry classifications
    3. Economic indicator connections based on sensitivity analysis
    4. Dynamic node features that update with market conditions
    """
    
    def __init__(
        self,
        correlation_window: int = 60,
        correlation_threshold: float = 0.3,
        min_common_dates: int = 30,
        include_sectors: bool = True,
        include_economic: bool = True
    ):
        """
        Initialize the dynamic graph builder.
        
        Args:
            correlation_window: Days to calculate rolling correlation
            correlation_threshold: Minimum correlation for edge creation
            min_common_dates: Minimum overlapping dates for correlation
            include_sectors: Whether to add sector aggregation nodes
            include_economic: Whether to add economic indicator nodes
        """
        self.correlation_window = correlation_window
        self.correlation_threshold = correlation_threshold
        self.min_common_dates = min_common_dates
        self.include_sectors = include_sectors
        self.include_economic = include_economic
        
        # Track node mappings
        self.node_to_idx = {}
        self.idx_to_node = {}
        self.node_types = {}  # 0: company, 1: sector, 2: economic
        
        # Store correlation history for analysis
        self.correlation_history = []
        
    def build_temporal_graph(
        self,
        market_data: pd.DataFrame,
        company_info: pd.DataFrame,
        economic_data: Optional[pd.DataFrame],
        target_date: pd.Timestamp,
        lookback_days: int = 252
    ) -> Data:
        """
        Build a graph for a specific point in time.
        
        Args:
            market_data: DataFrame with columns [Date, Symbol, Close, Volume, ...]
            company_info: DataFrame with [Symbol, Sector, Industry, MarketCap, ...]
            economic_data: DataFrame with economic indicators (optional)
            target_date: The date to build the graph for
            lookback_days: How many days of history to use
            
        Returns:
            PyTorch Geometric Data object representing the graph
        """
        logger.info(f"Building graph for {target_date}")
        
        # Reset node mappings for this graph
        self.node_to_idx = {}
        self.idx_to_node = {}
        self.node_types = {}
        
        # Filter data to lookback window
        start_date = target_date - timedelta(days=lookback_days)
        historical_data = market_data[
            (market_data.index > start_date) & 
            (market_data.index <= target_date)
        ].copy()
        
        if len(historical_data) == 0:
            raise ValueError(f"No data available for {target_date}")
        
        # Get unique symbols
        symbols = historical_data['Symbol'].unique()
        logger.info(f"Found {len(symbols)} companies with data")
        
        # Build node features and mappings
        node_features = []
        
        # 1. Company nodes
        for symbol in symbols:
            node_idx = len(self.node_to_idx)
            self.node_to_idx[f"company_{symbol}"] = node_idx
            self.idx_to_node[node_idx] = f"company_{symbol}"
            self.node_types[node_idx] = 0  # Company type
            
            # Extract features for this company
            features = self._extract_company_features(
                historical_data[historical_data['Symbol'] == symbol],
                company_info[company_info['symbol'] == symbol] if not company_info.empty else None
            )
            node_features.append(features)
        
        # 2. Sector nodes (if enabled)
        if self.include_sectors and not company_info.empty:
            sectors = company_info['sector'].dropna().unique()
            for sector in sectors:
                if pd.notna(sector) and sector != 'Unknown':
                    node_idx = len(self.node_to_idx)
                    self.node_to_idx[f"sector_{sector}"] = node_idx
                    self.idx_to_node[node_idx] = f"sector_{sector}"
                    self.node_types[node_idx] = 1  # Sector type
                    
                    # Extract sector-level features
                    features = self._extract_sector_features(
                        historical_data,
                        company_info[company_info['sector'] == sector]['symbol'].tolist()
                    )
                    node_features.append(features)
        
        # 3. Economic indicator nodes (if enabled)
        if self.include_economic and economic_data is not None:
            economic_indicators = ['VIX', 'DXY', 'TNX', 'GOLD']  # Key indicators
            for indicator in economic_indicators:
                if indicator in economic_data.columns:
                    node_idx = len(self.node_to_idx)
                    self.node_to_idx[f"economic_{indicator}"] = node_idx
                    self.idx_to_node[node_idx] = f"economic_{indicator}"
                    self.node_types[node_idx] = 2  # Economic type
                    
                    # Extract economic features
                    features = self._extract_economic_features(
                        economic_data[indicator].loc[:target_date]
                    )
                    node_features.append(features)
        
        # Build edges
        edge_list, edge_features = self._build_edges(
            historical_data, company_info, economic_data, target_date
        )
        
        # Convert to tensors
        x = torch.FloatTensor(np.array(node_features))
        edge_index = torch.LongTensor(edge_list).t()
        edge_attr = torch.FloatTensor(edge_features)
        node_type = torch.LongTensor([self.node_types[i] for i in range(len(node_features))])
        
        # Create PyTorch Geometric Data object
        graph = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            node_type=node_type,
            num_nodes=len(node_features)
        )
        
        # Add metadata
        graph.date = target_date
        graph.node_mapping = self.node_to_idx.copy()
        graph.symbols = symbols.tolist()
        
        logger.info(f"Graph built: {graph.num_nodes} nodes, {graph.num_edges} edges")
        
        return graph
    
    def _extract_company_features(
        self,
        company_data: pd.DataFrame,
        company_info: Optional[pd.Series]
    ) -> np.ndarray:
        """
        Extract point-in-time features for a company node.
        All features use only historical data (no look-ahead).
        """
        features = []
        
        # Price-based features
        if 'Close' in company_data.columns:
            prices = company_data['Close'].fillna(method='ffill')
            returns = prices.pct_change().dropna()
            
            # Momentum features
            features.extend([
                returns.iloc[-1] if len(returns) > 0 else 0,  # 1-day return
                returns.iloc[-5:].mean() if len(returns) >= 5 else 0,  # 5-day return
                returns.iloc[-20:].mean() if len(returns) >= 20 else 0,  # 20-day return
                returns.iloc[-60:].mean() if len(returns) >= 60 else 0,  # 60-day return
            ])
            
            # Volatility features
            features.extend([
                returns.iloc[-20:].std() * np.sqrt(252) if len(returns) >= 20 else 0.2,
                returns.iloc[-60:].std() * np.sqrt(252) if len(returns) >= 60 else 0.2,
            ])
            
            # Price levels
            if len(prices) >= 20:
                sma_20 = prices.iloc[-20:].mean()
                sma_60 = prices.iloc[-60:].mean() if len(prices) >= 60 else sma_20
                current_price = prices.iloc[-1]
                
                features.extend([
                    (current_price / sma_20 - 1) if sma_20 > 0 else 0,
                    (current_price / sma_60 - 1) if sma_60 > 0 else 0,
                ])
            else:
                features.extend([0, 0])
            
            # RSI
            if len(returns) >= 14:
                delta = prices.diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
                features.append(rsi.iloc[-1] / 100 if not pd.isna(rsi.iloc[-1]) else 0.5)
            else:
                features.append(0.5)
        
        # Volume features
        if 'Volume' in company_data.columns:
            volumes = company_data['Volume'].fillna(0)
            if len(volumes) >= 20:
                vol_ratio = volumes.iloc[-1] / volumes.iloc[-20:].mean() if volumes.iloc[-20:].mean() > 0 else 1
                features.append(np.log1p(vol_ratio))
            else:
                features.append(0)
        
        # Fundamental features (if available)
        if company_info is not None and not company_info.empty:
            # Market cap (log scale)
            market_cap = company_info.get('market_cap', 1e9)
            features.append(np.log10(max(market_cap, 1e6)))
            
            # Valuation ratios
            features.extend([
                company_info.get('pe_ratio', 15) / 100,  # Normalize
                company_info.get('price_to_book', 1) / 10,
                company_info.get('debt_to_equity', 1) / 100,
            ])
        else:
            features.extend([9, 0.15, 0.1, 0.01])  # Default values
        
        return np.array(features, dtype=np.float32)
    
    def _extract_sector_features(
        self,
        market_data: pd.DataFrame,
        sector_symbols: List[str]
    ) -> np.ndarray:
        """
        Extract aggregated features for a sector node.
        """
        features = []
        
        # Filter to sector companies
        sector_data = market_data[market_data['Symbol'].isin(sector_symbols)]
        
        if not sector_data.empty:
            # Aggregate returns
            sector_returns = sector_data.groupby('Date')['Close'].mean().pct_change()
            
            features.extend([
                sector_returns.iloc[-1] if len(sector_returns) > 0 else 0,
                sector_returns.iloc[-5:].mean() if len(sector_returns) >= 5 else 0,
                sector_returns.iloc[-20:].mean() if len(sector_returns) >= 20 else 0,
            ])
            
            # Sector volatility
            features.append(
                sector_returns.iloc[-20:].std() * np.sqrt(252) if len(sector_returns) >= 20 else 0.15
            )
            
            # Number of companies in sector
            features.append(len(sector_symbols) / 100)  # Normalize
            
            # Sector dispersion (how different companies are within sector)
            if len(sector_symbols) > 1:
                company_returns = []
                for symbol in sector_symbols:
                    sym_data = sector_data[sector_data['Symbol'] == symbol]['Close']
                    if len(sym_data) > 1:
                        company_returns.append(sym_data.pct_change().iloc[-1])
                
                if company_returns:
                    features.append(np.std(company_returns))
                else:
                    features.append(0.05)
            else:
                features.append(0.05)
        else:
            features = [0, 0, 0, 0.15, 0.01, 0.05]
        
        # Pad to match company feature size
        while len(features) < 14:
            features.append(0)
        
        return np.array(features[:14], dtype=np.float32)
    
    def _extract_economic_features(
        self,
        indicator_data: pd.Series
    ) -> np.ndarray:
        """
        Extract features for economic indicator nodes.
        """
        features = []
        
        if len(indicator_data) > 0:
            # Current level
            current = indicator_data.iloc[-1]
            features.append(current / 100 if current < 100 else np.log1p(current))
            
            # Changes
            if len(indicator_data) >= 2:
                features.append(indicator_data.pct_change().iloc[-1])
            else:
                features.append(0)
            
            if len(indicator_data) >= 5:
                features.append(indicator_data.pct_change(5).iloc[-1])
            else:
                features.append(0)
            
            if len(indicator_data) >= 20:
                features.append(indicator_data.pct_change(20).iloc[-1])
            else:
                features.append(0)
            
            # Moving averages
            if len(indicator_data) >= 20:
                ma_20 = indicator_data.iloc[-20:].mean()
                features.append((current / ma_20 - 1) if ma_20 != 0 else 0)
            else:
                features.append(0)
            
            # Volatility
            if len(indicator_data) >= 20:
                features.append(indicator_data.pct_change().iloc[-20:].std())
            else:
                features.append(0.01)
        else:
            features = [0, 0, 0, 0, 0, 0.01]
        
        # Pad to match company feature size
        while len(features) < 14:
            features.append(0)
        
        return np.array(features[:14], dtype=np.float32)
    
    def _build_edges(
        self,
        market_data: pd.DataFrame,
        company_info: pd.DataFrame,
        economic_data: Optional[pd.DataFrame],
        target_date: pd.Timestamp
    ) -> Tuple[List[List[int]], List[List[float]]]:
        """
        Build edges based on various relationships.
        """
        edge_list = []
        edge_features = []
        
        # 1. Correlation edges between companies
        correlation_edges = self._build_correlation_edges(market_data)
        edge_list.extend(correlation_edges[0])
        edge_features.extend(correlation_edges[1])
        
        # 2. Sector membership edges
        if self.include_sectors and not company_info.empty:
            sector_edges = self._build_sector_edges(company_info)
            edge_list.extend(sector_edges[0])
            edge_features.extend(sector_edges[1])
        
        # 3. Economic impact edges
        if self.include_economic and economic_data is not None:
            economic_edges = self._build_economic_edges(market_data, economic_data)
            edge_list.extend(economic_edges[0])
            edge_features.extend(economic_edges[1])
        
        return edge_list, edge_features
    
    def _build_correlation_edges(
        self,
        market_data: pd.DataFrame
    ) -> Tuple[List[List[int]], List[List[float]]]:
        """
        Build edges based on return correlations.
        """
        edge_list = []
        edge_features = []
        
        # Pivot to get price matrix
        prices = market_data.pivot_table(
            index='Date',
            columns='Symbol',
            values='Close'
        ).fillna(method='ffill')
        
        # Calculate returns
        returns = prices.pct_change().dropna()
        
        if len(returns) < self.min_common_dates:
            return edge_list, edge_features
        
        # Use recent window for correlation
        recent_returns = returns.iloc[-self.correlation_window:]
        
        # Calculate correlation matrix
        corr_matrix = recent_returns.corr(method='pearson')
        
        # Also calculate rank correlation for robustness
        rank_corr_matrix = recent_returns.corr(method='spearman')
        
        # Build edges based on correlation threshold
        symbols = list(prices.columns)
        
        for i, sym1 in enumerate(symbols):
            for j, sym2 in enumerate(symbols):
                if i < j:  # Avoid self-loops and duplicates
                    pearson_corr = corr_matrix.loc[sym1, sym2]
                    spearman_corr = rank_corr_matrix.loc[sym1, sym2]
                    
                    # Use average of Pearson and Spearman
                    avg_corr = (pearson_corr + spearman_corr) / 2
                    
                    if abs(avg_corr) > self.correlation_threshold:
                        # Get node indices
                        node1_idx = self.node_to_idx.get(f"company_{sym1}")
                        node2_idx = self.node_to_idx.get(f"company_{sym2}")
                        
                        if node1_idx is not None and node2_idx is not None:
                            # Add bidirectional edge
                            edge_list.append([node1_idx, node2_idx])
                            edge_list.append([node2_idx, node1_idx])
                            
                            # Edge features: [correlation, abs_correlation, is_positive]
                            edge_feat = [
                                avg_corr,
                                abs(avg_corr),
                                1.0 if avg_corr > 0 else 0.0
                            ]
                            edge_features.append(edge_feat)
                            edge_features.append(edge_feat)  # Same features for reverse edge
        
        # Store correlation matrix for analysis
        self.correlation_history.append({
            'date': returns.index[-1],
            'correlation_matrix': corr_matrix,
            'num_edges': len(edge_list) // 2  # Divide by 2 because edges are bidirectional
        })
        
        return edge_list, edge_features
    
    def _build_sector_edges(
        self,
        company_info: pd.DataFrame
    ) -> Tuple[List[List[int]], List[List[float]]]:
        """
        Build edges connecting companies to their sectors.
        """
        edge_list = []
        edge_features = []
        
        for _, company in company_info.iterrows():
            symbol = company['symbol']
            sector = company.get('sector', 'Unknown')
            
            if pd.notna(sector) and sector != 'Unknown':
                company_idx = self.node_to_idx.get(f"company_{symbol}")
                sector_idx = self.node_to_idx.get(f"sector_{sector}")
                
                if company_idx is not None and sector_idx is not None:
                    # Bidirectional edge between company and sector
                    edge_list.append([company_idx, sector_idx])
                    edge_list.append([sector_idx, company_idx])
                    
                    # Edge features: [membership_strength, is_primary, weight]
                    edge_feat = [1.0, 1.0, 0.8]  # Strong membership
                    edge_features.append(edge_feat)
                    edge_features.append(edge_feat)
        
        return edge_list, edge_features
    
    def _build_economic_edges(
        self,
        market_data: pd.DataFrame,
        economic_data: pd.DataFrame
    ) -> Tuple[List[List[int]], List[List[float]]]:
        """
        Build edges based on economic sensitivity.
        """
        edge_list = []
        edge_features = []
        
        # Calculate sensitivities (simplified - in practice, use regression)
        sensitivity_map = {
            'VIX': {  # Volatility affects all
                'all': 0.5,
            },
            'TNX': {  # Interest rates affect rate-sensitive sectors
                'Financials': 0.8,
                'Real Estate': 0.9,
                'Utilities': 0.7,
            },
            'DXY': {  # Dollar affects exporters
                'Technology': 0.6,
                'Industrials': 0.5,
            },
            'GOLD': {  # Gold correlation with certain sectors
                'Materials': 0.7,
                'Mining': 0.8,
            }
        }
        
        for indicator, sensitivities in sensitivity_map.items():
            econ_idx = self.node_to_idx.get(f"economic_{indicator}")
            if econ_idx is None:
                continue
            
            for target, sensitivity in sensitivities.items():
                if target == 'all':
                    # Connect to all company nodes
                    for node_name, node_idx in self.node_to_idx.items():
                        if node_name.startswith('company_'):
                            edge_list.append([econ_idx, node_idx])
                            edge_features.append([sensitivity, abs(sensitivity), 1.0])
                else:
                    # Connect to specific sector
                    sector_idx = self.node_to_idx.get(f"sector_{target}")
                    if sector_idx is not None:
                        edge_list.append([econ_idx, sector_idx])
                        edge_features.append([sensitivity, abs(sensitivity), 1.0])
        
        return edge_list, edge_features
    
    def build_graph_sequence(
        self,
        market_data: pd.DataFrame,
        company_info: pd.DataFrame,
        economic_data: Optional[pd.DataFrame],
        dates: List[pd.Timestamp],
        lookback_days: int = 252
    ) -> List[Data]:
        """
        Build a sequence of graphs over time for temporal modeling.
        """
        graphs = []
        
        for date in dates:
            try:
                graph = self.build_temporal_graph(
                    market_data,
                    company_info,
                    economic_data,
                    date,
                    lookback_days
                )
                graphs.append(graph)
            except Exception as e:
                logger.warning(f"Failed to build graph for {date}: {e}")
                continue
        
        return graphs
    
    def analyze_graph_evolution(
        self,
        graphs: List[Data]
    ) -> pd.DataFrame:
        """
        Analyze how the graph structure changes over time.
        """
        analysis = []
        
        for graph in graphs:
            stats = {
                'date': graph.date,
                'num_nodes': graph.num_nodes,
                'num_edges': graph.num_edges,
                'density': graph.num_edges / (graph.num_nodes * (graph.num_nodes - 1)),
                'num_companies': (graph.node_type == 0).sum().item(),
                'num_sectors': (graph.node_type == 1).sum().item(),
                'num_economic': (graph.node_type == 2).sum().item(),
            }
            analysis.append(stats)
        
        return pd.DataFrame(analysis)