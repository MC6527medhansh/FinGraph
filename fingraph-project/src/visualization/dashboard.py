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
    """Robust dashboard that connects to live API"""
    
    def __init__(self):
        self.api_base_url = "https://fingraph-production.up.railway.app"
        self.risk_data = None
        self.dashboard_summary = None
    
    def test_api_connection(self):
        """Test API connectivity and return status"""
        try:
            response = requests.get(f"{self.api_base_url}/health", timeout=15)
            if response.status_code == 200:
                return True, response.json()
            else:
                return False, f"API returned status {response.status_code}"
        except requests.exceptions.RequestException as e:
            return False, f"Connection failed: {str(e)}"
    
    def load_live_data(self):
        """Load data directly from live API - no fallbacks"""
        try:
            # Test connection first
            connected, health_data = self.test_api_connection()
            if not connected:
                return None, f"API connection failed: {health_data}"
            
            st.success(f"✅ Connected to API - Last update: {health_data.get('last_update', 'Unknown')}")
            
            # Get risk data
            risk_response = requests.get(f"{self.api_base_url}/risk", timeout=15)
            if risk_response.status_code != 200:
                return None, f"Risk endpoint failed: {risk_response.status_code}"
            
            risk_data = risk_response.json()
            risk_df = pd.DataFrame(risk_data)
            
            # Get portfolio data  
            portfolio_response = requests.get(f"{self.api_base_url}/portfolio", timeout=15)
            if portfolio_response.status_code != 200:
                return None, f"Portfolio endpoint failed: {portfolio_response.status_code}"
            
            portfolio_data = portfolio_response.json()
            
            # Format dashboard summary from API data
            dashboard_summary = {
                'timestamp': portfolio_data.get('timestamp'),
                'risk_overview': {
                    'total_companies': portfolio_data.get('companies_analyzed'),
                    'high_risk_count': portfolio_data.get('risk_distribution', {}).get('High', 0),
                    'medium_risk_count': portfolio_data.get('risk_distribution', {}).get('Medium', 0), 
                    'low_risk_count': portfolio_data.get('risk_distribution', {}).get('Low', 0),
                    'average_risk_score': portfolio_data.get('average_risk_score')
                },
                'model_performance': portfolio_data.get('model_performance', {})
            }
            
            st.info(f"📊 Loaded live data: {len(risk_df)} companies, updated {dashboard_summary['timestamp'][:19]}")
            
            return {'summary': dashboard_summary, 'predictions': risk_df}, None
            
        except requests.exceptions.RequestException as e:
            return None, f"Network error: {str(e)}"
        except Exception as e:
            return None, f"Data processing error: {str(e)}"
    
    def render_header(self):
        """Header with live API status"""
        st.markdown("# 📊 FinGraph Dashboard")
        st.markdown("### Financial Risk Assessment - Live Data")
        
        # Show API connection status
        connected, status_data = self.test_api_connection()
        if connected:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("API Status", "🟢 Live")
            with col2:
                st.metric("Companies", status_data.get('companies_count', 'N/A'))
            with col3:
                st.metric("Data Source", "Real-time API")
            with col4:
                last_update = status_data.get('last_update', '')[:10] if status_data.get('last_update') else 'Unknown'
                st.metric("Last Update", last_update)
        else:
            st.error(f"🔴 API Connection Failed: {status_data}")
        
        st.markdown("---")
    
    def render_risk_table(self):
        """Risk rankings from live API data"""
        st.markdown("## 🎯 Live Company Risk Rankings")
        
        if self.risk_data is not None and not self.risk_data.empty:
            # Sort by risk score (highest first)
            df_display = self.risk_data.sort_values('risk_score', ascending=False).copy()
            
            # Add risk level emoji
            risk_emoji = {'High': '🚨', 'Medium': '⚠️', 'Low': '✅'}
            df_display['Risk Level'] = df_display['risk_level'].map(risk_emoji) + ' ' + df_display['risk_level']
            
            # Display table with live data
            st.dataframe(
                df_display[['symbol', 'Risk Level', 'risk_score', 'volatility']].rename(columns={
                    'symbol': 'Company',
                    'risk_score': 'Risk Score', 
                    'volatility': 'Volatility'
                }),
                use_container_width=True
            )
            
            # Risk distribution chart from live data
            risk_counts = self.risk_data['risk_level'].value_counts()
            fig = px.pie(
                values=risk_counts.values,
                names=risk_counts.index,
                title="Live Risk Distribution",
                color_discrete_map={'High': '#ff4444', 'Medium': '#ffaa44', 'Low': '#44ff44'}
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Show data timestamp
            if self.dashboard_summary and 'timestamp' in self.dashboard_summary:
                st.caption(f"Data timestamp: {self.dashboard_summary['timestamp']}")
        else:
            st.error("❌ No live risk data available")
    
    def render_model_performance(self):
        """Model performance from live API"""
        st.markdown("## 🤖 Live Model Performance")

        if self.dashboard_summary and 'model_performance' in self.dashboard_summary:
            perf_data = self.dashboard_summary['model_performance']

            metrics = {}
            generated_at = None
            declared_best_model = None

            if isinstance(perf_data, dict):
                generated_at = perf_data.get('generated_at')
                declared_best_model = perf_data.get('best_model')
                if isinstance(perf_data.get('metrics'), dict):
                    metrics = perf_data['metrics']
                else:
                    metrics = {k: v for k, v in perf_data.items() if isinstance(v, dict)}

            if metrics:
                records = []
                for model_name, values in metrics.items():
                    if not isinstance(values, dict):
                        continue

                    mse_value = values.get('mse')
                    rmse_value = values.get('rmse')

                    try:
                        mse_value = float(mse_value)
                    except (TypeError, ValueError):
                        continue

                    if np.isnan(mse_value):
                        continue

                    if rmse_value is not None:
                        try:
                            rmse_value = float(rmse_value)
                        except (TypeError, ValueError):
                            rmse_value = None

                    if rmse_value is None:
                        rmse_value = float(np.sqrt(mse_value))

                    records.append({
                        'Model': model_name,
                        'MSE': mse_value,
                        'RMSE': rmse_value
                    })

                if records:
                    df_perf = pd.DataFrame(records).sort_values('MSE')
                    st.dataframe(df_perf, use_container_width=True)

                    # Determine best model from stored metrics
                    best_row = df_perf.iloc[0]
                    highlight_model = declared_best_model if declared_best_model in df_perf['Model'].values else best_row['Model']
                    highlight_mse = df_perf[df_perf['Model'] == highlight_model]['MSE'].iloc[0]

                    st.success(f"🏆 Best Model: **{highlight_model}** (MSE: {highlight_mse:.4f})")

                    # Performance chart using real metrics
                    fig = px.bar(df_perf, x='Model', y='MSE', title='Model Performance (Lower is Better)')
                    st.plotly_chart(fig, use_container_width=True)

                    if generated_at:
                        st.caption(f"Metrics generated at: {generated_at}")
                else:
                    st.warning("No model performance metrics available")
            else:
                st.warning("No model performance data in API response")
        else:
            st.error("❌ No live model performance data available")
    
    def render_alerts(self):
        """Risk alerts from live data"""
        st.markdown("## 🚨 Live Risk Alerts")
        
        if self.risk_data is not None and not self.risk_data.empty:
            # Get high risk companies
            high_risk = self.risk_data[self.risk_data['risk_level'] == 'High']
            
            if len(high_risk) > 0:
                st.error(f"🚨 {len(high_risk)} companies at HIGH RISK:")
                for _, row in high_risk.iterrows():
                    st.markdown(f"- **{row['symbol']}**: Risk Score {row['risk_score']:.3f}")
            else:
                st.success("✅ No high risk companies detected")
            
            # Dynamic risk threshold
            threshold = st.slider("Risk Alert Threshold", 0.0, 1.0, 0.7, 0.05)
            alerts = self.risk_data[self.risk_data['risk_score'] >= threshold]
            
            if len(alerts) > 0:
                st.warning(f"⚠️ {len(alerts)} companies above {threshold:.2f} threshold")
                for _, row in alerts.iterrows():
                    st.text(f"{row['symbol']}: {row['risk_score']:.3f}")
        else:
            st.error("❌ No live risk data for alerts")
    
    def run(self):
        """Main dashboard execution - loads live data only"""
        
        # Force fresh data load on every run (no caching of static data)
        with st.spinner("🔄 Loading live data from API..."):
            results, error = self.load_live_data()
        
        if error:
            st.error(f"❌ Failed to load live data: {error}")
            st.markdown("""
            ### 🔧 Troubleshooting:
            - Check if API is running: https://fingraph-production.up.railway.app/health
            - Verify network connectivity
            - Try refreshing the page
            """)
            return
        
        # Store live data
        if results:
            self.dashboard_summary = results['summary']
            self.risk_data = results['predictions']
        
        # Render dashboard with live data
        self.render_header()
        
        tab1, tab2, tab3 = st.tabs(["🎯 Risk Overview", "🤖 Model Performance", "🚨 Alerts"])
        
        with tab1:
            self.render_risk_table()
        
        with tab2:
            self.render_model_performance()
        
        with tab3:
            self.render_alerts()
        
        # Refresh button for live updates
        if st.button("🔄 Refresh Live Data"):
            st.rerun()

def main():
    """Run the live dashboard - API data only"""
    dashboard = FinGraphDashboard()
    dashboard.run()

if __name__ == "__main__":
    main()