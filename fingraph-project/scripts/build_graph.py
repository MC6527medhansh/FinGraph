"""
Master Graph Building Script for FinGraph
Orchestrates complete graph construction and analysis pipeline
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.features.graph_data_loader import GraphDataLoader
from src.features.graph_constructor import FinGraphConstructor
from src.features.graph_analyzer import FinGraphAnalyzer

import torch
import pandas as pd
import logging
from datetime import datetime
import pickle

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FinGraphPipeline:
    """
    Complete graph building and analysis pipeline
    """
    
    def __init__(self):
        self.loader = GraphDataLoader()
        self.constructor = FinGraphConstructor()
        self.graph = None
        self.analyzer = None
    
    def build_complete_graph(self, save_results: bool = True):
        """
        Build complete financial graph with analysis
        
        Args:
            save_results: Whether to save graph and analysis results
        """
        logger.info("🚀 Starting complete graph building pipeline...")
        
        # Step 1: Load data
        logger.info("📂 Step 1: Loading data...")
        data = self.loader.load_latest_data()
        
        # Step 2: Build graph
        logger.info("🏗️ Step 2: Constructing graph...")
        self.graph = self.constructor.build_graph(
            data['stock_data'],
            data['company_info'],
            data['economic_data'],
            data['relationship_data']
        )
        
        # Step 3: Analyze graph
        logger.info("📊 Step 3: Analyzing graph structure...")
        self.analyzer = FinGraphAnalyzer(self.graph)
        analysis_results = self.analyzer.analyze_graph_structure()
        
        # Step 4: Generate visualizations
        logger.info("🎨 Step 4: Creating visualizations...")
        graph_viz = self.constructor.visualize_graph(self.graph)
        analysis_figures = self.analyzer.create_analysis_visualizations()
        
        # Step 5: Generate reports
        logger.info("📋 Step 5: Generating reports...")
        analysis_report = self.analyzer.generate_analysis_report()
        key_nodes = self.analyzer.identify_key_nodes(top_k=5)
        
        # Step 6: Save results
        if save_results:
            logger.info("💾 Step 6: Saving results...")
            self._save_results(graph_viz, analysis_figures, analysis_report, key_nodes)
        
        # Summary
        self._print_summary(analysis_results, key_nodes)
        
        logger.info("🎉 Graph building pipeline completed successfully!")
        
        return {
            'graph': self.graph,
            'analysis': analysis_results,
            'key_nodes': key_nodes,
            'analyzer': self.analyzer
        }
    
    def _save_results(self, graph_viz, analysis_figures, analysis_report, key_nodes):
        """Save all results to files"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Create directories
        os.makedirs('data/processed', exist_ok=True)
        os.makedirs('data/graphs', exist_ok=True)
        
        # Save graph
        torch.save(self.graph, f'data/graphs/financial_graph_{timestamp}.pt')
        
        # Save graph as pickle for NetworkX compatibility
        with open(f'data/graphs/graph_constructor_{timestamp}.pkl', 'wb') as f:
            pickle.dump(self.constructor, f)
        
        # Save visualizations
        graph_viz.savefig(f'data/processed/financial_network_{timestamp}.png', 
                         dpi=300, bbox_inches='tight')
        
        for i, fig in enumerate(analysis_figures):
            fig.savefig(f'data/processed/analysis_{i+1}_{timestamp}.png', 
                       dpi=300, bbox_inches='tight')
        
        # Save reports
        with open(f'data/processed/graph_analysis_report_{timestamp}.txt', 'w') as f:
            f.write(analysis_report)
        
        # Save key nodes as JSON
        import json
        with open(f'data/processed/key_nodes_{timestamp}.json', 'w') as f:
            json.dump(key_nodes, f, indent=2)
        
        logger.info(f"💾 Saved graph, visualizations, and reports with timestamp {timestamp}")
    
    def _print_summary(self, analysis_results, key_nodes):
        """Print pipeline summary"""
        print("\n" + "="*60)
        print("🏆 FINGRAPH CONSTRUCTION SUMMARY")
        print("="*60)
        
        basic = analysis_results['basic_stats']
        connectivity = analysis_results['connectivity']
        
        print(f"📊 Graph Structure:")
        print(f"   • Nodes: {basic['num_nodes']:,} ({basic['node_types']})")
        print(f"   • Edges: {basic['num_edges']:,}")
        print(f"   • Density: {basic['density']:.4f}")
        print(f"   • Average degree: {basic['avg_degree']:.2f}")
        
        print(f"\n🔗 Connectivity:")
        print(f"   • Connected: {connectivity['is_connected']}")
        print(f"   • Components: {connectivity['num_components']}")
        print(f"   • Largest component: {connectivity['largest_component_ratio']:.1%}")
        print(f"   • Average clustering: {connectivity['avg_clustering']:.4f}")
        
        print(f"\n🎯 Most Important Nodes (by PageRank):")
        if 'pagerank' in key_nodes:
            for node_info in key_nodes['pagerank'][:3]:
                name = node_info['node_name'].replace('company_', '').replace('sector_', '').replace('economic_', '')
                print(f"   • {name} ({node_info['node_type']}): {node_info['centrality_value']:.4f}")
        
        print(f"\n✅ Ready for GNN training!")
        print("="*60)

def main():
    """Main execution function"""
    try:
        pipeline = FinGraphPipeline()
        results = pipeline.build_complete_graph()
        
        print(f"\n🎯 Next Steps:")
        print(f"   1. Review graph visualizations in data/processed/")
        print(f"   2. Check analysis report for insights")
        print(f"   3. Graph saved in data/graphs/ ready for model training")
        print(f"   4. Move to Day 4: GNN model development!")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()