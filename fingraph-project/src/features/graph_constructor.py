"""
Graph Constructor for FinGraph
Builds graph structure with nodes, edges, and features
"""

import pandas as pd
import numpy as np
import torch
import torch_geometric
from torch_geometric.data import Data
import networkx as nx
import logging
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

class FinGraphConstructor:
    """
    Constructs financial graph with companies, relationships, and economic factors
    
    Graph Structure:
    - Company nodes: Individual stocks with financial features
    - Sector nodes: Industry groupings
    - Economic nodes: Macro indicators that affect markets
    - Relationship edges: Business connections, correlations
    - Economic edges: How macro factors influence companies/sectors
    """
    
    def __init__(self):
        self.node_mapping = {}  # Maps node names to indices
        self.node_features = {}  # Node feature data
        self.edge_list = []     # Edge connections
        self.edge_features = [] # Edge feature data
        self.node_types = {}    # Node type mapping
        
    def build_graph(self, 
                   stock_data: pd.DataFrame,
                   company_info: pd.DataFrame,
                   economic_data: pd.DataFrame,
                   relationship_data: pd.DataFrame) -> Data:
        """
        Build complete financial graph
        
        Args:
            stock_data: Stock price and technical data
            company_info: Company fundamental information
            economic_data: Economic indicators
            relationship_data: Business relationships
            
        Returns:
            PyTorch Geometric Data object
        """
        logger.info("🏗️ Building financial graph structure...")
        
        # Step 1: Create nodes
        self._create_company_nodes(stock_data, company_info)
        self._create_sector_nodes(company_info)
        self._create_economic_nodes(economic_data)
        
        # Step 2: Create edges
        self._create_relationship_edges(relationship_data)
        self._create_correlation_edges(stock_data)
        self._create_sector_membership_edges(company_info)
        self._create_economic_impact_edges(stock_data, economic_data)
        
        # Step 3: Build PyTorch Geometric graph
        graph = self._build_pytorch_geometric_graph()
        
        logger.info(f"✅ Graph constructed: {graph.num_nodes} nodes, {graph.num_edges} edges")
        return graph
    
    def _create_company_nodes(self, stock_data: pd.DataFrame, company_info: pd.DataFrame):
        """Create company nodes with financial features"""
        logger.info("📊 Creating company nodes...")
        
        companies = sorted(stock_data['Symbol'].unique())
        
        for i, symbol in enumerate(companies):
            node_id = f"company_{symbol}"
            self.node_mapping[node_id] = len(self.node_mapping)
            self.node_types[self.node_mapping[node_id]] = 'company'
            
            # Get company data
            company_stock = stock_data[stock_data['Symbol'] == symbol]
            company_fundamentals = company_info[company_info['symbol'] == symbol]
            
            # Calculate financial features
            features = self._calculate_company_features(company_stock, company_fundamentals)
            self.node_features[self.node_mapping[node_id]] = features
        
        logger.info(f"✅ Created {len(companies)} company nodes")
    
    def _calculate_company_features(self, stock_data: pd.DataFrame, fundamental_data: pd.DataFrame) -> np.ndarray:
        """Calculate comprehensive company features"""
        features = []
        
        if not stock_data.empty:
            # Price-based features (recent 30 days)
            recent_data = stock_data.tail(30)
            
            # Price momentum features
            features.extend([
                recent_data['Returns_1d'].mean(),  # Average daily return
                recent_data['Returns_5d'].mean(),  # Average 5-day return
                recent_data['Returns_20d'].mean(), # Average 20-day return
                recent_data['Volatility'].mean(),  # Average volatility
                recent_data['RSI'].iloc[-1] if not recent_data['RSI'].isna().all() else 50,  # Latest RSI
            ])
            
            # Volume features
            features.extend([
                recent_data['Volume'].mean(),      # Average volume
                recent_data['Volume'].std(),       # Volume volatility
            ])
            
            # Technical indicator features
            latest_close = recent_data['Close'].iloc[-1]
            latest_sma20 = recent_data['SMA_20'].iloc[-1] if not recent_data['SMA_20'].isna().all() else latest_close
            latest_sma50 = recent_data['SMA_50'].iloc[-1] if not recent_data['SMA_50'].isna().all() else latest_close
            
            features.extend([
                latest_close / latest_sma20 - 1,  # Price vs 20-day MA
                latest_close / latest_sma50 - 1,  # Price vs 50-day MA
                latest_sma20 / latest_sma50 - 1,  # MA cross signal
            ])
        else:
            # Default values if no stock data
            features.extend([0.0] * 10)
        
        # Fundamental features
        if not fundamental_data.empty:
            fund_row = fundamental_data.iloc[0]
            
            # Valuation ratios
            features.extend([
                self._safe_log(fund_row.get('market_cap', 1e9)),  # Log market cap
                fund_row.get('pe_ratio', 15.0) if pd.notna(fund_row.get('pe_ratio')) else 15.0,
                fund_row.get('price_to_book', 2.0) if pd.notna(fund_row.get('price_to_book')) else 2.0,
                fund_row.get('price_to_sales', 3.0) if pd.notna(fund_row.get('price_to_sales')) else 3.0,
            ])
            
            # Financial health ratios
            features.extend([
                fund_row.get('debt_to_equity', 50.0) if pd.notna(fund_row.get('debt_to_equity')) else 50.0,
                fund_row.get('current_ratio', 1.5) if pd.notna(fund_row.get('current_ratio')) else 1.5,
                fund_row.get('roe', 0.15) if pd.notna(fund_row.get('roe')) else 0.15,
                fund_row.get('roa', 0.08) if pd.notna(fund_row.get('roa')) else 0.08,
            ])
            
            # Growth and profitability
            features.extend([
                fund_row.get('revenue_growth', 0.1) if pd.notna(fund_row.get('revenue_growth')) else 0.1,
                fund_row.get('earnings_growth', 0.1) if pd.notna(fund_row.get('earnings_growth')) else 0.1,
                fund_row.get('profit_margin', 0.1) if pd.notna(fund_row.get('profit_margin')) else 0.1,
                fund_row.get('beta', 1.0) if pd.notna(fund_row.get('beta')) else 1.0,
            ])
        else:
            # Default fundamental values
            features.extend([20.0, 15.0, 2.0, 3.0, 50.0, 1.5, 0.15, 0.08, 0.1, 0.1, 0.1, 1.0])
        
        return np.array(features, dtype=np.float32)
    
    def _safe_log(self, value: float) -> float:
        """Safe logarithm that handles edge cases"""
        return np.log(max(value, 1e6))  # Minimum 1M market cap
    
    def _create_sector_nodes(self, company_info: pd.DataFrame):
        """Create sector nodes"""
        logger.info("🏭 Creating sector nodes...")
        
        if company_info.empty:
            return
        
        sectors = company_info['sector'].dropna().unique()
        
        for sector in sectors:
            if sector and sector != 'Unknown':
                node_id = f"sector_{sector.replace(' ', '_')}"
                self.node_mapping[node_id] = len(self.node_mapping)
                self.node_types[self.node_mapping[node_id]] = 'sector'
                
                # Calculate sector aggregate features
                sector_companies = company_info[company_info['sector'] == sector]
                features = self._calculate_sector_features(sector_companies)
                self.node_features[self.node_mapping[node_id]] = features
        
        logger.info(f"✅ Created {len(sectors)} sector nodes")
    
    def _calculate_sector_features(self, sector_companies: pd.DataFrame) -> np.ndarray:
        """Calculate sector-level features"""
        features = []
        
        # Sector size and composition
        features.extend([
            len(sector_companies),  # Number of companies in sector
            sector_companies['market_cap'].sum() if 'market_cap' in sector_companies.columns else 1e12,
        ])
        
        # Average sector metrics
        numeric_cols = ['pe_ratio', 'price_to_book', 'debt_to_equity', 'roe', 'beta']
        for col in numeric_cols:
            if col in sector_companies.columns:
                avg_val = sector_companies[col].mean()
                features.append(avg_val if pd.notna(avg_val) else 0.0)
            else:
                features.append(0.0)
        
        # Pad to match company feature size
        while len(features) < 22:  # Match company feature length
            features.append(0.0)
        
        return np.array(features, dtype=np.float32)
    
    def _create_economic_nodes(self, economic_data: pd.DataFrame):
        """Create economic indicator nodes"""
        logger.info("🏛️ Creating economic indicator nodes...")
        
        if economic_data.empty:
            return
        
        # Key economic indicators to include as nodes
        key_indicators = [
            'fed_funds_rate', 'treasury_10y', 'unemployment_rate', 
            'vix', 'cpi_all', 'gdp_growth'
        ]
        
        for indicator in key_indicators:
            if indicator in economic_data.columns:
                node_id = f"economic_{indicator}"
                self.node_mapping[node_id] = len(self.node_mapping)
                self.node_types[self.node_mapping[node_id]] = 'economic'
                
                # Calculate economic indicator features
                features = self._calculate_economic_features(economic_data, indicator)
                self.node_features[self.node_mapping[node_id]] = features
        
        logger.info(f"✅ Created {len(key_indicators)} economic nodes")
    
    def _calculate_economic_features(self, economic_data: pd.DataFrame, indicator: str) -> np.ndarray:
        """Calculate economic indicator features"""
        series = economic_data[indicator].dropna()
        
        if series.empty:
            return np.zeros(22, dtype=np.float32)  # Match feature size
        
        features = []
        
        # Current level and recent changes
        current_value = series.iloc[-1]
        features.extend([
            current_value,
            series.pct_change().iloc[-1] if len(series) > 1 else 0.0,  # Recent change
            series.pct_change(periods=30).iloc[-1] if len(series) > 30 else 0.0,  # 30-period change
        ])
        
        # Statistical features
        recent_series = series.tail(60)  # Last 60 observations
        features.extend([
            recent_series.mean(),
            recent_series.std(),
            recent_series.min(),
            recent_series.max(),
            (current_value - recent_series.mean()) / (recent_series.std() + 1e-8),  # Z-score
        ])
        
        # Trend features
        if len(recent_series) > 10:
            # Simple linear trend
            x = np.arange(len(recent_series))
            y = recent_series.values
            trend = np.polyfit(x, y, 1)[0]  # Slope
            features.append(trend)
        else:
            features.append(0.0)
        
        # Pad to match feature size
        while len(features) < 22:
            features.append(0.0)
        
        return np.array(features, dtype=np.float32)
    
    def _create_relationship_edges(self, relationship_data: pd.DataFrame):
        """Create edges from business relationships"""
        logger.info("🔗 Creating relationship edges...")
        
        if relationship_data.empty:
            return
        
        relationship_weights = {
            'subsidiaries': 0.9,  # Strong connection
            'suppliers': 0.7,     # Important business connection
            'customers': 0.7,     # Important business connection
            'partners': 0.6,      # Moderate connection
            'competitors': 0.4    # Weak connection (competitive relationship)
        }
        
        for _, row in relationship_data.iterrows():
            company_symbol = row['company_symbol']
            related_entity = row['related_entity']
            rel_type = row['relationship_type']
            
            # Create edge from company to related entity (if entity is also a company)
            company_node = f"company_{company_symbol}"
            
            # Check if related entity is one of our companies
            related_company_node = None
            for symbol in relationship_data['company_symbol'].unique():
                if symbol.upper() in related_entity.upper() or related_entity.upper() in symbol.upper():
                    related_company_node = f"company_{symbol}"
                    break
            
            if company_node in self.node_mapping and related_company_node in self.node_mapping:
                # Company-to-company relationship
                source = self.node_mapping[company_node]
                target = self.node_mapping[related_company_node]
                weight = relationship_weights.get(rel_type, 0.5)
                
                self.edge_list.append([source, target])
                self.edge_features.append([weight, 1.0, 0.0])  # [strength, is_business_rel, is_correlation]
        
        logger.info(f"✅ Created {len(self.edge_list)} relationship edges")
    
    def _create_correlation_edges(self, stock_data: pd.DataFrame):
        """Create edges based on stock price correlations"""
        logger.info("📈 Creating correlation edges...")
        
        if stock_data.empty:
            return
        
        # Calculate correlation matrix
        price_data = stock_data.pivot_table(
            index=stock_data.index,
            columns='Symbol',
            values='Close'
        )
        
        returns = price_data.pct_change().dropna()
        correlations = returns.corr()
        
        # Create edges for strong correlations
        companies = correlations.index.tolist()
        
        for i, company1 in enumerate(companies):
            for j, company2 in enumerate(companies):
                if i < j:  # Avoid duplicate edges
                    correlation = correlations.loc[company1, company2]
                    
                    if abs(correlation) > 0.3:  # Only strong correlations
                        node1 = f"company_{company1}"
                        node2 = f"company_{company2}"
                        
                        if node1 in self.node_mapping and node2 in self.node_mapping:
                            source = self.node_mapping[node1]
                            target = self.node_mapping[node2]
                            
                            self.edge_list.append([source, target])
                            # [strength, is_business_rel, is_correlation]
                            self.edge_features.append([abs(correlation), 0.0, 1.0])
        
        logger.info(f"✅ Added correlation edges (total edges: {len(self.edge_list)})")
    
    def _create_sector_membership_edges(self, company_info: pd.DataFrame):
        """Create edges connecting companies to their sectors"""
        logger.info("🏭 Creating sector membership edges...")
        
        if company_info.empty:
            return
        
        for _, row in company_info.iterrows():
            symbol = row['symbol']
            sector = row['sector']
            
            if pd.notna(sector) and sector != 'Unknown':
                company_node = f"company_{symbol}"
                sector_node = f"sector_{sector.replace(' ', '_')}"
                
                if company_node in self.node_mapping and sector_node in self.node_mapping:
                    source = self.node_mapping[company_node]
                    target = self.node_mapping[sector_node]
                    
                    self.edge_list.append([source, target])
                    # [strength, is_business_rel, is_correlation]
                    self.edge_features.append([1.0, 0.0, 0.0])  # Full membership
        
        logger.info(f"✅ Added sector membership edges (total edges: {len(self.edge_list)})")
    
    def _create_economic_impact_edges(self, stock_data: pd.DataFrame, economic_data: pd.DataFrame):
        """Create edges from economic indicators to companies/sectors"""
        logger.info("🏛️ Creating economic impact edges...")
        
        if stock_data.empty or economic_data.empty:
            return
        
        # Economic indicators that affect different sectors differently
        economic_impacts = {
            'fed_funds_rate': ['Technology', 'Real Estate', 'Utilities'],  # Interest sensitive
            'unemployment_rate': ['Consumer Discretionary', 'Retail', 'Services'],
            'vix': ['all'],  # Market volatility affects everyone
            'treasury_10y': ['Financial Services', 'Banks', 'Insurance'],
            'cpi_all': ['Consumer Staples', 'Energy', 'Materials']  # Inflation sensitive
        }
        
        for indicator, affected_sectors in economic_impacts.items():
            economic_node = f"economic_{indicator}"
            
            if economic_node in self.node_mapping:
                econ_idx = self.node_mapping[economic_node]
                
                # Connect to all companies if 'all' sectors
                if 'all' in affected_sectors:
                    for node_name, node_idx in self.node_mapping.items():
                        if node_name.startswith('company_'):
                            self.edge_list.append([econ_idx, node_idx])
                            self.edge_features.append([0.3, 0.0, 0.0])  # Economic impact
                else:
                    # Connect to specific sectors
                    for sector in affected_sectors:
                        sector_node = f"sector_{sector.replace(' ', '_')}"
                        if sector_node in self.node_mapping:
                            target_idx = self.node_mapping[sector_node]
                            self.edge_list.append([econ_idx, target_idx])
                            self.edge_features.append([0.5, 0.0, 0.0])  # Stronger sector impact
        
        logger.info(f"✅ Added economic impact edges (total edges: {len(self.edge_list)})")
    
    def _build_pytorch_geometric_graph(self) -> Data:
        """Build final PyTorch Geometric graph"""
        logger.info("⚡ Building PyTorch Geometric graph...")
        
        # Prepare node features matrix
        num_nodes = len(self.node_mapping)
        feature_dim = 22  # Standardized feature dimension
        
        node_feature_matrix = np.zeros((num_nodes, feature_dim))
        for node_idx, features in self.node_features.items():
            if len(features) < feature_dim:
                # Pad features if needed
                padded_features = np.zeros(feature_dim)
                padded_features[:len(features)] = features
                node_feature_matrix[node_idx] = padded_features
            else:
                node_feature_matrix[node_idx] = features[:feature_dim]
        
        # Prepare edge indices and features
        if self.edge_list:
            edge_index = torch.tensor(self.edge_list, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(self.edge_features, dtype=torch.float)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, 3), dtype=torch.float)
        
        # Create node type tensor
        node_type_list = []
        type_mapping = {'company': 0, 'sector': 1, 'economic': 2}
        for i in range(num_nodes):
            node_type = self.node_types.get(i, 'company')
            node_type_list.append(type_mapping[node_type])
        
        # Create PyTorch Geometric Data object
        graph = Data(
            x=torch.tensor(node_feature_matrix, dtype=torch.float),
            edge_index=edge_index,
            edge_attr=edge_attr,
            node_type=torch.tensor(node_type_list, dtype=torch.long),
            num_nodes=num_nodes
        )
        
        # Add metadata
        graph.node_mapping = self.node_mapping
        graph.node_types = self.node_types
        
        logger.info(f"✅ PyTorch Geometric graph ready!")
        logger.info(f"   Nodes: {graph.num_nodes} ({feature_dim} features each)")
        logger.info(f"   Edges: {graph.num_edges} (3 features each)")
        
        return graph
    
    def visualize_graph(self, graph: Data, figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Visualize the financial graph
        
        Args:
            graph: PyTorch Geometric graph
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        logger.info("🎨 Creating graph visualization...")
        
        # Convert to NetworkX for visualization
        G = nx.Graph()
        
        # Add nodes with types
        node_colors = []
        node_sizes = []
        node_labels = {}
        
        color_map = {'company': 'lightblue', 'sector': 'lightgreen', 'economic': 'salmon'}
        size_map = {'company': 300, 'sector': 500, 'economic': 400}
        
        for node_name, node_idx in self.node_mapping.items():
            node_type = self.node_types[node_idx]
            G.add_node(node_idx)
            node_colors.append(color_map[node_type])
            node_sizes.append(size_map[node_type])
            
            # Create readable labels
            if node_type == 'company':
                label = node_name.replace('company_', '')
            elif node_type == 'sector':
                label = node_name.replace('sector_', '').replace('_', ' ')[:10]
            else:
                label = node_name.replace('economic_', '')[:8]
            node_labels[node_idx] = label
        
        # Add edges
        edge_list = graph.edge_index.t().numpy()
        for edge in edge_list:
            G.add_edge(edge[0], edge[1])
        
        # Create visualization
        fig, ax = plt.subplots(figsize=figsize)
        
        # Use spring layout for better visualization
        pos = nx.spring_layout(G, k=2, iterations=50)
        
        # Draw the graph
        nx.draw(G, pos, 
                node_color=node_colors,
                node_size=node_sizes,
                with_labels=True,
                labels=node_labels,
                font_size=8,
                font_weight='bold',
                edge_color='gray',
                alpha=0.7,
                ax=ax)
        
        # Add legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue', 
                      markersize=10, label='Companies'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgreen', 
                      markersize=12, label='Sectors'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='salmon', 
                      markersize=11, label='Economic Indicators')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        ax.set_title('FinGraph: Financial Network Structure', fontsize=16, fontweight='bold')
        ax.axis('off')
        
        plt.tight_layout()
        logger.info("✅ Graph visualization created")
        
        return fig

# Test function
def test_graph_constructor():
    """Test graph construction"""
    from graph_data_loader import GraphDataLoader
    
    # Load data
    loader = GraphDataLoader()
    try:
        data = loader.load_latest_data()
        
        # Build graph
        constructor = FinGraphConstructor()
        graph = constructor.build_graph(
            data['stock_data'],
            data['company_info'], 
            data['economic_data'],
            data['relationship_data']
        )
        
        print(f"✅ Graph created successfully!")
        print(f"   Nodes: {graph.num_nodes}")
        print(f"   Edges: {graph.num_edges}")
        print(f"   Node features: {graph.x.shape}")
        print(f"   Edge features: {graph.edge_attr.shape}")
        
        # Create visualization
        fig = constructor.visualize_graph(graph)
        plt.savefig('data/processed/financial_graph.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return graph
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        return None

if __name__ == "__main__":
    test_graph_constructor()