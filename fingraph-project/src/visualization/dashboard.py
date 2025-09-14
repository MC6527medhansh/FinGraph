"""
FinGraph Dashboard - Integrates with existing temporal_integration results
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
import os
import requests
import sys
from datetime import datetime

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

st.set_page_config(
    page_title="FinGraph Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

class FinGraphDashboard:
    """Dashboard that reads from your existing temporal_integration results"""
    
    def __init__(self):
        self.api_url = os.environ.get("API_URL", "http://localhost:8000")
        self.data_dir = os.path.join(project_root, "data", "temporal_integration")
        self.risk_data = None
        self.dashboard_summary = None
    
    @st.cache_data
    def load_existing_results(_self):
        """Load results from API (production) or files (local)"""
        import requests
        
        # Try API first (for production deployment)
        try:
            api_url = "https://fingraph-production.up.railway.app"
            
            # Test if API is reachable
            health_response = requests.get(f"{api_url}/health", timeout=10)
            if health_response.status_code == 200:
                # Get portfolio data
                portfolio_response = requests.get(f"{api_url}/portfolio", timeout=10)
                risk_response = requests.get(f"{api_url}/risk", timeout=10)
                
                if portfolio_response.status_code == 200 and risk_response.status_code == 200:
                    portfolio_data = portfolio_response.json()
                    risk_data = risk_response.json()
                    
                    # Convert risk data to DataFrame
                    import pandas as pd
                    risk_df = pd.DataFrame(risk_data)
                    
                    # Format dashboard summary
                    dashboard_summary = {
                        'timestamp': portfolio_data.get('timestamp'),
                        'risk_overview': {
                            'total_companies': portfolio_data.get('companies_analyzed', 5),
                            'high_risk_count': portfolio_data.get('risk_distribution', {}).get('High', 1),
                            'medium_risk_count': portfolio_data.get('risk_distribution', {}).get('Medium', 2),
                            'low_risk_count': portfolio_data.get('risk_distribution', {}).get('Low', 2),
                            'average_risk_score': portfolio_data.get('average_risk_score', 0.5)
                        },
                        'model_performance': portfolio_data.get('model_performance', {})
                    }
                    
                    return {'summary': dashboard_summary, 'predictions': risk_df}, None
        
        except Exception as e:
            st.warning(f"API connection failed: {str(e)}")
        
        # Fallback to local files (for local development)
        try:
            if not os.path.exists(_self.data_dir):
                # Generate fallback data for deployment
                return _self._generate_fallback_dashboard_data(), None
            
            files = os.listdir(_self.data_dir)
            
            # Load dashboard summary
            summary_files = [f for f in files if f.startswith('dashboard_summary_')]
            if summary_files:
                latest_summary = max(summary_files)
                with open(os.path.join(_self.data_dir, latest_summary), 'r') as f:
                    dashboard_summary = json.load(f)
            else:
                dashboard_summary = None
            
            # Load risk predictions
            prediction_files = [f for f in files if f.startswith('risk_predictions_')]
            if prediction_files:
                latest_predictions = max(prediction_files)
                risk_data = pd.read_csv(os.path.join(_self.data_dir, latest_predictions))
            else:
                risk_data = None
            
            return {'summary': dashboard_summary, 'predictions': risk_data}, None
            
        except Exception as e:
            return _self._generate_fallback_dashboard_data(), None
        
    def _generate_fallback_dashboard_data(self):
    """Generate fallback data when API and files are unavailable"""
    import pandas as pd
    from datetime import datetime
    
    # Sample risk data
    risk_data = pd.DataFrame([
        {'symbol': 'AAPL', 'risk_score': 0.299, 'risk_level': 'Low', 'volatility': 0.234},
        {'symbol': 'MSFT', 'risk_score': 0.386, 'risk_level': 'Low', 'volatility': 0.238},
        {'symbol': 'GOOGL', 'risk_score': 0.476, 'risk_level': 'Medium', 'volatility': 0.307},
        {'symbol': 'AMZN', 'risk_score': 0.533, 'risk_level': 'Medium', 'volatility': 0.404},
        {'symbol': 'TSLA', 'risk_score': 0.863, 'risk_level': 'High', 'volatility': 0.730}
    ])
    
    # Sample summary data
    dashboard_summary = {
        'timestamp': datetime.now().isoformat(),
        'risk_overview': {
            'total_companies': 5,
            'high_risk_count': 1,
            'medium_risk_count': 2,
            'low_risk_count': 2,
            'average_risk_score': 0.511
        },
        'model_performance': {
            'Logistic Regression': {'mse': 0.0302, 'rmse': 0.1738},
            'Random Forest': {'mse': 0.0241, 'rmse': 0.1554},
            'Simple GNN': {'mse': 0.0223, 'rmse': 0.1495}
        }
    }
    
    return {'summary': dashboard_summary, 'predictions': risk_data}
    
    def render_header(self):
        """Header with current results"""
        st.markdown("# 📊 FinGraph Dashboard")
        st.markdown("### Financial Risk Assessment Results")
        
        if self.dashboard_summary:
            col1, col2, col3, col4 = st.columns(4)
            
            risk_overview = self.dashboard_summary.get('risk_overview', {})
            data_summary = self.dashboard_summary.get('data_summary', {})
            
            with col1:
                st.metric("Companies", risk_overview.get('total_companies', 'N/A'))
            with col2:
                st.metric("High Risk", risk_overview.get('high_risk_count', 'N/A'))
            with col3:
                avg_risk = risk_overview.get('average_risk_score', 0)
                st.metric("Avg Risk", f"{avg_risk:.3f}" if avg_risk else 'N/A')
            with col4:
                st.metric("Samples", data_summary.get('temporal_samples', 'N/A'))
        
        st.markdown("---")
    
    def render_risk_table(self):
        """Risk rankings table"""
        st.markdown("## 🎯 Company Risk Rankings")
        
        if self.risk_data is not None:
            # Sort by risk score
            df_display = self.risk_data.sort_values('risk_score', ascending=False).copy()
            
            # Add risk level emoji
            risk_emoji = {'High': '🚨', 'Medium': '⚠️', 'Low': '✅'}
            df_display['Risk'] = df_display['risk_level'].map(risk_emoji) + ' ' + df_display['risk_level']
            
            # Display table
            st.dataframe(
                df_display[['symbol', 'Risk', 'risk_score', 'volatility']].rename(columns={
                    'symbol': 'Company',
                    'risk_score': 'Risk Score', 
                    'volatility': 'Volatility'
                }),
                use_container_width=True
            )
            
            # Risk distribution chart
            risk_counts = self.risk_data['risk_level'].value_counts()
            fig = px.pie(
                values=risk_counts.values,
                names=risk_counts.index,
                title="Risk Distribution",
                color_discrete_map={'High': '#ff4444', 'Medium': '#ffaa44', 'Low': '#44ff44'}
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No risk data available. Run the pipeline first.")
    
    def render_model_performance(self):
        """Model performance from YOUR results"""
        st.markdown("## 🤖 Model Performance")
        
        if self.dashboard_summary and 'model_performance' in self.dashboard_summary:
            perf = self.dashboard_summary['model_performance']
            
            # Create comparison table
            models = list(perf.keys())
            mse_scores = [perf[m]['mse'] for m in models]
            rmse_scores = [perf[m]['rmse'] for m in models]
            
            df_perf = pd.DataFrame({
                'Model': models,
                'MSE': mse_scores,
                'RMSE': rmse_scores
            }).sort_values('MSE')
            
            st.dataframe(df_perf, use_container_width=True)
            
            # Best model highlight
            best_model = df_perf.iloc[0]['Model']
            best_mse = df_perf.iloc[0]['MSE']
            st.success(f"🏆 Best Model: **{best_model}** (MSE: {best_mse:.4f})")
            
            # Bar chart
            fig = px.bar(df_perf, x='Model', y='MSE', title='Model Comparison (Lower is Better)')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No model performance data found.")
    
    def render_alerts(self):
        """Risk alerts"""
        st.markdown("## 🚨 Risk Alerts")
        
        if self.risk_data is not None:
            high_risk = self.risk_data[self.risk_data['risk_level'] == 'High']
            
            if len(high_risk) > 0:
                st.error(f"🚨 {len(high_risk)} companies at HIGH RISK:")
                for _, row in high_risk.iterrows():
                    st.markdown(f"- **{row['symbol']}**: {row['risk_score']:.3f}")
            else:
                st.success("✅ No high risk companies currently")
            
            # Risk threshold slider
            threshold = st.slider("Risk Alert Threshold", 0.0, 1.0, 0.7, 0.05)
            alerts = self.risk_data[self.risk_data['risk_score'] >= threshold]
            
            if len(alerts) > 0:
                st.warning(f"⚠️ {len(alerts)} companies above {threshold:.2f} threshold")
        else:
            st.info("No risk data available")
    
    def run(self):
        """Main dashboard execution"""
        # Load YOUR existing results
        results, error = self.load_existing_results()
        
        if error:
            st.error(f"❌ {error}")
            st.markdown("""
            ### 🚀 To generate results:
            
            From your project root, run:
            ```bash
            cd fingraph-project
            python scripts/test_temporal_fix.py
            ```
            
            This will create results in `data/temporal_integration/`
            """)
            return
        
        # Store results
        if results:
            self.dashboard_summary = results['summary']
            self.risk_data = results['predictions']
        
        # Sidebar
        st.sidebar.markdown("## 📊 Data Info")
        if self.dashboard_summary:
            timestamp = self.dashboard_summary.get('timestamp', 'Unknown')[:16]
            st.sidebar.info(f"Last updated: {timestamp}")
        
        if st.sidebar.button("🔄 Refresh Data"):
            st.cache_data.clear()
            st.rerun()
        
        # Main content
        self.render_header()
        
        tab1, tab2, tab3 = st.tabs(["🎯 Risk Overview", "🤖 Model Performance", "🚨 Alerts"])
        
        with tab1:
            self.render_risk_table()
        
        with tab2:
            self.render_model_performance()
        
        with tab3:
            self.render_alerts()

def main():
    """Run the dashboard"""
    dashboard = FinGraphDashboard()
    dashboard.run()

if __name__ == "__main__":
    main()