"""
Graph Analyzer for FinGraph - FIXED VERSION
Analyzes graph structure and properties for insights
"""

import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import degree, to_networkx
import networkx as nx
import logging
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

class FinGraphAnalyzer:
    """
    Analyzes financial graph structure and properties
    
    Features:
    - Graph connectivity analysis
    - Node centrality calculations
    - Community detection
    - Feature distribution analysis
    - Risk propagation pathway identification
    """
    
    def __init__(self, graph: Data):
        """
        Initialize analyzer with graph
        
        Args:
            graph: PyTorch Geometric graph
        """
        self.graph = graph
        self.nx_graph = to_networkx(graph, to_undirected=True)
        self.analysis_results = {}
    
    def analyze_graph_structure(self) -> Dict:
        """
        Comprehensive graph structure analysis
        
        Returns:
            Dictionary with analysis results
        """
        logger.info("📊 Analyzing graph structure...")
        
        results = {}
        
        # Basic graph statistics
        results['basic_stats'] = self._calculate_basic_stats()
        
        # Node centrality measures
        results['centrality'] = self._calculate_centrality_measures()
        
        # Connectivity analysis
        results['connectivity'] = self._analyze_connectivity()
        
        # Community detection
        results['communities'] = self._detect_communities()
        
        # Feature analysis
        results['features'] = self._analyze_node_features()
        
        # Edge analysis
        results['edges'] = self._analyze_edge_properties()
        
        self.analysis_results = results
        logger.info("✅ Graph analysis completed")
        
        return results
    
    def _calculate_basic_stats(self) -> Dict:
        """Calculate basic graph statistics"""
        num_nodes = self.graph.num_nodes
        num_edges = self.graph.num_edges
        
        # Node type distribution
        node_types = self.graph.node_type.numpy()
        type_counts = {
            'company': int(np.sum(node_types == 0)),
            'sector': int(np.sum(node_types == 1)),
            'economic': int(np.sum(node_types == 2))
        }
        
        # Degree statistics - FIXED: Convert to int
        degrees = degree(self.graph.edge_index[0], num_nodes=num_nodes).numpy().astype(int)
        
        # Density
        max_edges = num_nodes * (num_nodes - 1) / 2
        density = num_edges / max_edges if max_edges > 0 else 0
        
        return {
            'num_nodes': num_nodes,
            'num_edges': num_edges,
            'node_types': type_counts,
            'density': density,
            'avg_degree': float(degrees.mean()),
            'max_degree': int(degrees.max()),
            'min_degree': int(degrees.min()),
            'degree_std': float(degrees.std())
        }
    
    def _calculate_centrality_measures(self) -> Dict:
        """Calculate node centrality measures"""
        centrality = {}
        
        # Degree centrality
        degree_cent = nx.degree_centrality(self.nx_graph)
        centrality['degree'] = degree_cent
        
        # Betweenness centrality (important for risk propagation)
        betweenness_cent = nx.betweenness_centrality(self.nx_graph)
        centrality['betweenness'] = betweenness_cent
        
        # Eigenvector centrality (influence measure)
        try:
            eigenvector_cent = nx.eigenvector_centrality(self.nx_graph, max_iter=1000)
            centrality['eigenvector'] = eigenvector_cent
        except:
            centrality['eigenvector'] = {node: 0.0 for node in self.nx_graph.nodes()}
        
        # PageRank (Google's algorithm adapted for financial networks)
        pagerank_cent = nx.pagerank(self.nx_graph)
        centrality['pagerank'] = pagerank_cent
        
        return centrality
    
    def _analyze_connectivity(self) -> Dict:
        """Analyze graph connectivity"""
        connectivity = {}
        
        # Check if graph is connected
        connectivity['is_connected'] = nx.is_connected(self.nx_graph)
        
        # Number of connected components
        connectivity['num_components'] = nx.number_connected_components(self.nx_graph)
        
        # Largest component size
        if connectivity['num_components'] > 0:
            largest_cc = max(nx.connected_components(self.nx_graph), key=len)
            connectivity['largest_component_size'] = len(largest_cc)
            connectivity['largest_component_ratio'] = len(largest_cc) / self.graph.num_nodes
        else:
            connectivity['largest_component_size'] = 0
            connectivity['largest_component_ratio'] = 0.0
        
        # Average path length (if connected)
        if connectivity['is_connected']:
            connectivity['avg_path_length'] = nx.average_shortest_path_length(self.nx_graph)
        else:
            # Calculate for largest component
            if connectivity['largest_component_size'] > 1:
                largest_subgraph = self.nx_graph.subgraph(largest_cc)
                connectivity['avg_path_length'] = nx.average_shortest_path_length(largest_subgraph)
            else:
                connectivity['avg_path_length'] = 0.0
        
        # Clustering coefficient
        connectivity['avg_clustering'] = nx.average_clustering(self.nx_graph)
        
        return connectivity
    
    def _detect_communities(self) -> Dict:
        """Detect communities in the graph"""
        communities = {}
        
        try:
            # Use Louvain algorithm for community detection
            community_dict = nx.community.louvain_communities(self.nx_graph)
            
            communities['num_communities'] = len(community_dict)
            communities['community_sizes'] = [len(community) for community in community_dict]
            communities['modularity'] = nx.community.modularity(self.nx_graph, community_dict)
            
            # Identify which companies are in which communities
            communities['community_mapping'] = {}
            for i, community in enumerate(community_dict):
                communities['community_mapping'][f'community_{i}'] = list(community)
            
        except Exception as e:
            logger.warning(f"Community detection failed: {str(e)}")
            communities = {
                'num_communities': 1,
                'community_sizes': [self.graph.num_nodes],
                'modularity': 0.0,
                'community_mapping': {'community_0': list(range(self.graph.num_nodes))}
            }
        
        return communities
    
    def _analyze_node_features(self) -> Dict:
        """Analyze node feature distributions"""
        features = {}
        
        # Feature statistics by node type
        node_features = self.graph.x.numpy()
        node_types = self.graph.node_type.numpy()
        
        for node_type_idx, node_type_name in enumerate(['company', 'sector', 'economic']):
            mask = node_types == node_type_idx
            if np.any(mask):
                type_features = node_features[mask]
                features[f'{node_type_name}_features'] = {
                    'count': int(np.sum(mask)),
                    'mean': type_features.mean(axis=0).tolist(),
                    'std': type_features.std(axis=0).tolist(),
                    'min': type_features.min(axis=0).tolist(),
                    'max': type_features.max(axis=0).tolist()
                }
        
        # Overall feature statistics
        features['overall'] = {
            'feature_dim': node_features.shape[1],
            'mean': node_features.mean(axis=0).tolist(),
            'std': node_features.std(axis=0).tolist()
        }
        
        return features
    
    def _analyze_edge_properties(self) -> Dict:
        """Analyze edge properties"""
        edges = {}
        
        if self.graph.edge_attr is not None:
            edge_features = self.graph.edge_attr.numpy()
            
            # Edge feature statistics
            edges['num_edges'] = edge_features.shape[0]
            edges['edge_feature_dim'] = edge_features.shape[1]
            
            # Analyze different edge types (based on our edge features)
            # [strength, is_business_rel, is_correlation]
            if edge_features.shape[1] >= 3:
                strength = edge_features[:, 0]
                business_edges = edge_features[:, 1] > 0.5
                correlation_edges = edge_features[:, 2] > 0.5
                
                edges['strength_stats'] = {
                    'mean': float(strength.mean()),
                    'std': float(strength.std()),
                    'min': float(strength.min()),
                    'max': float(strength.max())
                }
                
                edges['edge_types'] = {
                    'business_relationships': int(np.sum(business_edges)),
                    'correlations': int(np.sum(correlation_edges)),
                    'other': int(len(edge_features) - np.sum(business_edges) - np.sum(correlation_edges))
                }
        
        return edges
    
    def identify_key_nodes(self, top_k: int = 5) -> Dict:
        """
        Identify most important nodes for risk assessment
        
        Args:
            top_k: Number of top nodes to return for each measure
            
        Returns:
            Dictionary with key nodes
        """
        logger.info(f"🎯 Identifying top {top_k} key nodes...")
        
        if 'centrality' not in self.analysis_results:
            self.analyze_graph_structure()
        
        centrality = self.analysis_results['centrality']
        key_nodes = {}
        
        # Get node mapping for readable names
        node_mapping = getattr(self.graph, 'node_mapping', {})
        reverse_mapping = {v: k for k, v in node_mapping.items()}
        
        for measure_name, measure_values in centrality.items():
            # Sort nodes by centrality measure
            sorted_nodes = sorted(measure_values.items(), key=lambda x: x[1], reverse=True)
            
            key_nodes[measure_name] = []
            for node_idx, centrality_value in sorted_nodes[:top_k]:
                node_name = reverse_mapping.get(node_idx, f"node_{node_idx}")
                key_nodes[measure_name].append({
                    'node_name': node_name,
                    'centrality_value': centrality_value,
                    'node_type': self._get_node_type_name(node_idx)
                })
        
        return key_nodes
    
    def _get_node_type_name(self, node_idx: int) -> str:
        """Get readable node type name"""
        if hasattr(self.graph, 'node_type'):
            type_idx = self.graph.node_type[node_idx].item()
            type_names = ['company', 'sector', 'economic']
            return type_names[type_idx] if type_idx < len(type_names) else 'unknown'
        return 'unknown'
    
    def generate_analysis_report(self) -> str:
        """
        Generate comprehensive analysis report
        
        Returns:
            Formatted analysis report string
        """
        if not self.analysis_results:
            self.analyze_graph_structure()
        
        report = []
        report.append("=" * 50)
        report.append("FINOGRAPH STRUCTURE ANALYSIS REPORT")
        report.append("=" * 50)
        report.append(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Basic statistics
        basic = self.analysis_results['basic_stats']
        report.append("📊 BASIC GRAPH STATISTICS")
        report.append("-" * 30)
        report.append(f"Nodes: {basic['num_nodes']:,}")
        report.append(f"Edges: {basic['num_edges']:,}")
        report.append(f"Density: {basic['density']:.4f}")
        report.append(f"Average degree: {basic['avg_degree']:.2f}")
        report.append("")
        
        # Node types
        report.append("Node type distribution:")
        for node_type, count in basic['node_types'].items():
            percentage = (count / basic['num_nodes']) * 100
            report.append(f"  {node_type}: {count} ({percentage:.1f}%)")
        report.append("")
        
        # Connectivity
        connectivity = self.analysis_results['connectivity']
        report.append("🔗 CONNECTIVITY ANALYSIS")
        report.append("-" * 30)
        report.append(f"Connected: {connectivity['is_connected']}")
        report.append(f"Components: {connectivity['num_components']}")
        report.append(f"Largest component: {connectivity['largest_component_ratio']:.1%}")
        report.append(f"Average path length: {connectivity['avg_path_length']:.2f}")
        report.append(f"Average clustering: {connectivity['avg_clustering']:.4f}")
        report.append("")
        
        # Communities
        communities = self.analysis_results['communities']
        report.append("👥 COMMUNITY STRUCTURE")
        report.append("-" * 30)
        report.append(f"Communities detected: {communities['num_communities']}")
        report.append(f"Modularity: {communities['modularity']:.4f}")
        report.append(f"Community sizes: {communities['community_sizes']}")
        report.append("")
        
        # Key nodes
        key_nodes = self.identify_key_nodes(top_k=3)
        report.append("🎯 KEY NODES FOR RISK ASSESSMENT")
        report.append("-" * 30)
        
        for measure, nodes in key_nodes.items():
            report.append(f"\nTop nodes by {measure}:")
            for node_info in nodes:
                name = node_info['node_name'].replace('company_', '').replace('sector_', '').replace('economic_', '')
                report.append(f"  {name} ({node_info['node_type']}): {node_info['centrality_value']:.4f}")
        
        report.append("")
        report.append("=" * 50)
        
        return "\n".join(report)
    
    def create_analysis_visualizations(self) -> List[plt.Figure]:
        """
        Create comprehensive analysis visualizations
        
        Returns:
            List of matplotlib figures
        """
        logger.info("📈 Creating analysis visualizations...")
        
        figures = []
        
        try:
            # 1. Degree distribution
            fig1 = self._plot_degree_distribution()
            figures.append(fig1)
        except Exception as e:
            logger.warning(f"Failed to create degree distribution plot: {str(e)}")
        
        try:
            # 2. Centrality comparison
            fig2 = self._plot_centrality_comparison()
            figures.append(fig2)
        except Exception as e:
            logger.warning(f"Failed to create centrality comparison plot: {str(e)}")
        
        try:
            # 3. Node feature distributions
            fig3 = self._plot_feature_distributions()
            figures.append(fig3)
        except Exception as e:
            logger.warning(f"Failed to create feature distribution plot: {str(e)}")
        
        try:
            # 4. Community structure
            fig4 = self._plot_community_structure()
            figures.append(fig4)
        except Exception as e:
            logger.warning(f"Failed to create community structure plot: {str(e)}")
        
        logger.info(f"✅ Created {len(figures)} analysis visualizations")
        return figures
    
    def _plot_degree_distribution(self) -> plt.Figure:
        """Plot degree distribution - FIXED VERSION"""
        # FIXED: Convert degrees to integers
        degrees = degree(self.graph.edge_index[0], num_nodes=self.graph.num_nodes).numpy().astype(int)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Histogram
        ax1.hist(degrees, bins=max(1, min(20, len(np.unique(degrees)))), alpha=0.7, color='skyblue', edgecolor='black')
        ax1.set_xlabel('Degree')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Degree Distribution')
        ax1.grid(True, alpha=0.3)
        
        # Log-log plot to check for power law
        try:
            degree_counts = np.bincount(degrees)
            non_zero_counts = degree_counts[degree_counts > 0]
            non_zero_degrees = np.arange(len(degree_counts))[degree_counts > 0]
            
            if len(non_zero_degrees) > 1:
                ax2.loglog(non_zero_degrees, non_zero_counts, 'o-', alpha=0.7)
                ax2.set_xlabel('Degree (log scale)')
                ax2.set_ylabel('Frequency (log scale)')
                ax2.set_title('Log-Log Degree Distribution')
                ax2.grid(True, alpha=0.3)
            else:
                ax2.text(0.5, 0.5, 'Insufficient data\nfor log-log plot', 
                        ha='center', va='center', transform=ax2.transAxes)
                ax2.set_title('Log-Log Degree Distribution')
        except Exception as e:
            ax2.text(0.5, 0.5, f'Log-log plot failed:\n{str(e)}', 
                    ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Log-Log Degree Distribution (Failed)')
        
        plt.tight_layout()
        return fig
    
    def _plot_centrality_comparison(self) -> plt.Figure:
        """Plot centrality measures comparison"""
        if 'centrality' not in self.analysis_results:
            self.analyze_graph_structure()
        
        centrality = self.analysis_results['centrality']
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        for i, (measure_name, measure_values) in enumerate(centrality.items()):
            if i < 4:  # Only plot first 4 measures
                values = list(measure_values.values())
                if values:  # Check if we have values
                    axes[i].hist(values, bins=max(1, min(10, len(values))), alpha=0.7, 
                               color='lightcoral', edgecolor='black')
                    axes[i].set_xlabel(f'{measure_name.title()} Centrality')
                    axes[i].set_ylabel('Frequency')
                    axes[i].set_title(f'{measure_name.title()} Centrality Distribution')
                    axes[i].grid(True, alpha=0.3)
                else:
                    axes[i].text(0.5, 0.5, 'No data', ha='center', va='center', 
                               transform=axes[i].transAxes)
                    axes[i].set_title(f'{measure_name.title()} Centrality Distribution')
        
        plt.tight_layout()
        return fig
    
    def _plot_feature_distributions(self) -> plt.Figure:
        """Plot node feature distributions by type"""
        node_features = self.graph.x.numpy()
        node_types = self.graph.node_type.numpy()
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        # Plot first 6 features
        type_names = ['Company', 'Sector', 'Economic']
        colors = ['lightblue', 'lightgreen', 'salmon']
        
        for feature_idx in range(min(6, node_features.shape[1])):
            ax = axes[feature_idx]
            
            for type_idx, (type_name, color) in enumerate(zip(type_names, colors)):
                mask = node_types == type_idx
                if np.any(mask):
                    type_features = node_features[mask, feature_idx]
                    if len(type_features) > 0:
                        ax.hist(type_features, alpha=0.6, label=type_name, color=color, 
                               bins=max(1, min(10, len(type_features))))
            
            ax.set_xlabel(f'Feature {feature_idx + 1}')
            ax.set_ylabel('Frequency')
            ax.set_title(f'Feature {feature_idx + 1} Distribution')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def _plot_community_structure(self) -> plt.Figure:
        """Plot community structure"""
        if 'communities' not in self.analysis_results:
            self.analyze_graph_structure()
        
        communities = self.analysis_results['communities']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Community size distribution
        sizes = communities['community_sizes']
        if sizes:
            ax1.bar(range(len(sizes)), sizes, color='gold', alpha=0.7, edgecolor='black')
            ax1.set_xlabel('Community')
            ax1.set_ylabel('Size')
            ax1.set_title('Community Size Distribution')
            ax1.grid(True, alpha=0.3)
        else:
            ax1.text(0.5, 0.5, 'No community data', ha='center', va='center', 
                    transform=ax1.transAxes)
            ax1.set_title('Community Size Distribution')
        
        # Community structure visualization
        try:
            if len(self.nx_graph.nodes()) > 0:
                pos = nx.spring_layout(self.nx_graph, k=1, iterations=50)
                
                # Color nodes by community
                community_mapping = communities['community_mapping']
                node_colors = []
                
                for node in self.nx_graph.nodes():
                    for community_name, community_nodes in community_mapping.items():
                        if node in community_nodes:
                            community_idx = int(community_name.split('_')[1])
                            node_colors.append(plt.cm.Set3(community_idx / max(1, len(community_mapping))))
                            break
                    else:
                        node_colors.append('gray')
                
                nx.draw(self.nx_graph, pos, 
                       node_color=node_colors,
                       node_size=100,
                       with_labels=False,
                       edge_color='gray',
                       alpha=0.7,
                       ax=ax2)
                
                ax2.set_title('Community Structure')
            else:
                ax2.text(0.5, 0.5, 'No nodes to visualize', 
                        ha='center', va='center', transform=ax2.transAxes)
                ax2.set_title('Community Structure')
            
        except Exception as e:
            ax2.text(0.5, 0.5, f'Visualization failed:\n{str(e)}', 
                    ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Community Structure (Failed)')
        
        plt.tight_layout()
        return fig

# Test function
def test_graph_analyzer():
    """Test graph analyzer"""
    from graph_data_loader import GraphDataLoader
    from graph_constructor import FinGraphConstructor
    
    try:
        # Load data and build graph
        loader = GraphDataLoader()
        data = loader.load_latest_data()
        
        constructor = FinGraphConstructor()
        graph = constructor.build_graph(
            data['stock_data'],
            data['company_info'],
            data['economic_data'],
            data['relationship_data']
        )
        
        # Analyze graph
        analyzer = FinGraphAnalyzer(graph)
        analysis = analyzer.analyze_graph_structure()
        
        # Generate report
        report = analyzer.generate_analysis_report()
        print(report)
        
        # Create directories if they don't exist
        import os
        os.makedirs('data/processed', exist_ok=True)
        
        # Save report
        with open('data/processed/graph_analysis_report.txt', 'w') as f:
            f.write(report)
        
        # Create visualizations
        figures = analyzer.create_analysis_visualizations()
        
        # Save visualizations
        for i, fig in enumerate(figures):
            fig.savefig(f'data/processed/graph_analysis_{i+1}.png', dpi=300, bbox_inches='tight')
            plt.close(fig)  # Close figure to free memory
        
        print(f"\n✅ Analysis complete! Saved {len(figures)} visualizations and report.")
        
        return analyzer
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    test_graph_analyzer()