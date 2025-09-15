"""
FinGraph Dashboard - Integrates with existing temporal_integration results
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
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
    """Dashboard capable of using live API data or stored temporal predictions."""

    def __init__(self):
        self.api_base_url = "https://fingraph-production.up.railway.app"
        self.risk_data = None
        self.dashboard_summary = None
        self.local_predictions_path = os.path.join(project_root, 'data', 'temporal_integration', 'predictions.csv')
        self.local_summary_path = os.path.join(project_root, 'data', 'temporal_integration', 'dashboard_summary.json')
        self.data_source = None

    def _categorize_risk(self, risk_score):
        if risk_score >= 0.7:
            return 'High'
        elif risk_score >= 0.4:
            return 'Medium'
        return 'Low'

    def _prepare_predictions_dataframe(self, df):
        prepared = df.copy()
        prepared['symbol'] = prepared['symbol'].astype(str)
        prepared['risk_score'] = pd.to_numeric(prepared['risk_score'], errors='coerce')
        prepared = prepared[prepared['risk_score'].notnull()]

        if 'risk_level' not in prepared.columns:
            prepared['risk_level'] = prepared['risk_score'].apply(self._categorize_risk)

        if 'volatility' not in prepared.columns:
            prepared['volatility'] = np.nan
        else:
            prepared['volatility'] = pd.to_numeric(prepared['volatility'], errors='coerce')

        for column in ['momentum_5d', 'momentum_20d', 'rsi', 'max_drawdown']:
            if column not in prepared.columns:
                prepared[column] = np.nan
            else:
                prepared[column] = pd.to_numeric(prepared[column], errors='coerce')

        if 'last_updated' not in prepared.columns:
            if 'prediction_date' in prepared.columns:
                prepared['last_updated'] = prepared['prediction_date']
            else:
                prepared['last_updated'] = datetime.now().isoformat()

        return prepared

    def _build_summary_from_df(self, df):
        if 'last_updated' in df.columns:
            last_updated_series = pd.to_datetime(df['last_updated'], errors='coerce').dropna()
            timestamp = last_updated_series.max().isoformat() if not last_updated_series.empty else datetime.now().isoformat()
        else:
            timestamp = datetime.now().isoformat()

        summary = {
            'timestamp': timestamp,
            'risk_overview': {
                'total_companies': len(df),
                'high_risk_count': int((df['risk_level'] == 'High').sum()),
                'medium_risk_count': int((df['risk_level'] == 'Medium').sum()),
                'low_risk_count': int((df['risk_level'] == 'Low').sum()),
                'average_risk_score': float(df['risk_score'].mean()) if not df.empty else 0.0
            },
            'model_performance': {}
        }

        return summary

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
        """Load data directly from live API"""
        try:
            connected, health_data = self.test_api_connection()
            if not connected:
                return None, f"API connection failed: {health_data}"

            st.success(f"✅ Connected to API - Last update: {health_data.get('last_update', 'Unknown')}")

            risk_response = requests.get(f"{self.api_base_url}/risk", timeout=15)
            if risk_response.status_code != 200:
                return None, f"Risk endpoint failed: {risk_response.status_code}"

            risk_data = risk_response.json()
            risk_df = pd.DataFrame(risk_data)

            portfolio_response = requests.get(f"{self.api_base_url}/portfolio", timeout=15)
            if portfolio_response.status_code != 200:
                return None, f"Portfolio endpoint failed: {portfolio_response.status_code}"

            portfolio_data = portfolio_response.json()

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

    def load_local_results(self):
        """Load locally stored temporal integration results"""
        if not os.path.exists(self.local_predictions_path):
            return None, "Predictions file not found"

        try:
            predictions_df = pd.read_csv(self.local_predictions_path)
            if predictions_df.empty:
                return None, "Predictions file is empty"

            prepared_df = self._prepare_predictions_dataframe(predictions_df)
            if prepared_df.empty:
                return None, "Predictions data did not contain valid rows"

            summary = None
            if os.path.exists(self.local_summary_path):
                try:
                    with open(self.local_summary_path, 'r', encoding='utf-8') as f:
                        summary_data = json.load(f)
                    if isinstance(summary_data, dict):
                        summary = summary_data
                except (json.JSONDecodeError, OSError):
                    summary = None

            if summary is None:
                summary = self._build_summary_from_df(prepared_df)

            return {'summary': summary, 'predictions': prepared_df}, None
        except Exception as e:
            return None, f"Failed to load stored predictions: {str(e)}"

    def render_header(self):
        """Header with data source status"""
        st.markdown("# 📊 FinGraph Dashboard")
        source_titles = {
            "Live API": "Live Data",
            "Stored Results": "Stored Predictions"
        }
        st.markdown(f"### Financial Risk Assessment - {source_titles.get(self.data_source, 'Data')}")

        if self.data_source == "Live API":
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
                    last_update = status_data.get('last_update', '')[:19] if status_data.get('last_update') else 'Unknown'
                    st.metric("Last Update", last_update)
            else:
                st.error(f"🔴 API Connection Failed: {status_data}")
        elif self.data_source == "Stored Results":
            overview = self.dashboard_summary.get('risk_overview', {}) if self.dashboard_summary else {}
            timestamp = self.dashboard_summary.get('timestamp', 'Unknown') if self.dashboard_summary else 'Unknown'

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Data Source", "Stored Predictions")
            with col2:
                st.metric("Companies", overview.get('total_companies', 'N/A'))
            with col3:
                st.metric("High Risk", overview.get('high_risk_count', 0))
            with col4:
                st.metric("Last Update", timestamp[:19] if timestamp else 'Unknown')

            st.caption(f"Loaded from {self.local_predictions_path}")
        else:
            st.info("Select a data source from the sidebar to load results.")

        st.markdown("---")

    def render_risk_table(self):
        """Render risk rankings for the loaded data"""
        source_label = "Live" if self.data_source == "Live API" else "Stored" if self.data_source == "Stored Results" else "Live"
        st.markdown(f"## 🎯 {source_label} Company Risk Rankings")

        if self.risk_data is not None and not self.risk_data.empty:
            df_display = self.risk_data.sort_values('risk_score', ascending=False).copy()

            risk_emoji = {'High': '🚨', 'Medium': '⚠️', 'Low': '✅'}
            df_display['Risk Level'] = df_display['risk_level'].map(risk_emoji) + ' ' + df_display['risk_level']

            st.dataframe(
                df_display[['symbol', 'Risk Level', 'risk_score', 'volatility']].rename(columns={
                    'symbol': 'Company',
                    'risk_score': 'Risk Score',
                    'volatility': 'Volatility'
                }),
                use_container_width=True
            )

            risk_counts = self.risk_data['risk_level'].value_counts()
            fig = px.pie(
                values=risk_counts.values,
                names=risk_counts.index,
                title=f"{source_label} Risk Distribution",
                color_discrete_map={'High': '#ff4444', 'Medium': '#ffaa44', 'Low': '#44ff44'}
            )
            st.plotly_chart(fig, use_container_width=True)

            if self.dashboard_summary and 'timestamp' in self.dashboard_summary:
                st.caption(f"Data timestamp: {self.dashboard_summary['timestamp']}")
        else:
            st.error(f"❌ No {source_label.lower()} risk data available")

    def render_model_performance(self):
        """Display model performance metrics"""
        source_label = "Live" if self.data_source == "Live API" else "Stored"
        st.markdown(f"## 🤖 {source_label} Model Performance")

        if self.dashboard_summary and 'model_performance' in self.dashboard_summary:
            perf = self.dashboard_summary['model_performance']

            if perf:
                models = []
                mse_scores = []
                rmse_scores = []

                for model_name, values in perf.items():
                    if not isinstance(values, dict):
                        continue

                    mse_value = values.get('mse')
                    rmse_value = values.get('rmse')

                    try:
                        mse_value = float(mse_value)
                    except (TypeError, ValueError):
                        continue

                    if rmse_value is not None:
                        try:
                            rmse_value = float(rmse_value)
                        except (TypeError, ValueError):
                            rmse_value = None

                    if rmse_value is None:
                        rmse_value = float(np.sqrt(mse_value))

                    if np.isnan(mse_value):
                        continue

                    models.append(model_name)
                    mse_scores.append(mse_value)
                    rmse_scores.append(rmse_value)

                if models:
                    df_perf = pd.DataFrame({
                        'Model': models,
                        'MSE': mse_scores,
                        'RMSE': rmse_scores
                    }).sort_values('MSE')

                    st.dataframe(df_perf, use_container_width=True)

                    best_model = df_perf.iloc[0]['Model']
                    best_mse = df_perf.iloc[0]['MSE']
                    st.success(f"🏆 Best Model: **{best_model}** (MSE: {best_mse:.4f})")

                    fig = px.bar(df_perf, x='Model', y='MSE', title='Model Performance (Lower is Better)')
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("No model performance metrics available")
            else:
                st.info("Model performance data is not available for this data source")
        else:
            st.error("❌ No model performance data available")

    def render_alerts(self):
        """Risk alerts for the loaded dataset"""
        source_label = "Live" if self.data_source == "Live API" else "Stored"
        st.markdown(f"## 🚨 {source_label} Risk Alerts")

        if self.risk_data is not None and not self.risk_data.empty:
            high_risk = self.risk_data[self.risk_data['risk_level'] == 'High']

            if len(high_risk) > 0:
                st.error(f"🚨 {len(high_risk)} companies at HIGH RISK:")
                for _, row in high_risk.iterrows():
                    st.markdown(f"- **{row['symbol']}**: Risk Score {row['risk_score']:.3f}")
            else:
                st.success("✅ No high risk companies detected")

            threshold = st.slider("Risk Alert Threshold", 0.0, 1.0, 0.7, 0.05)
            alerts = self.risk_data[self.risk_data['risk_score'] >= threshold]

            if len(alerts) > 0:
                st.warning(f"⚠️ {len(alerts)} companies above {threshold:.2f} threshold")
                for _, row in alerts.iterrows():
                    st.text(f"{row['symbol']}: {row['risk_score']:.3f}")
        else:
            st.error(f"❌ No {source_label.lower()} risk data for alerts")

    def run(self):
        """Main dashboard execution supporting multiple data sources."""
        st.sidebar.header("Data Source")
        mode = st.sidebar.radio(
            "Preferred source",
            ["Auto", "Stored Results", "Live API"],
            index=0
        )
        st.sidebar.caption("Auto loads stored predictions when available and falls back to the live API if needed.")

        stored_error = None
        api_error = None
        results = None
        actual_source = None

        with st.spinner("🔄 Loading risk data..."):
            if mode == "Stored Results":
                results, stored_error = self.load_local_results()
                actual_source = "Stored Results"
            elif mode == "Live API":
                results, api_error = self.load_live_data()
                actual_source = "Live API"
            else:
                results, stored_error = self.load_local_results()
                if results:
                    actual_source = "Stored Results"
                else:
                    results, api_error = self.load_live_data()
                    actual_source = "Live API" if results else None

        if results is None:
            if mode == "Auto" and stored_error and api_error:
                st.error("❌ Failed to load both stored predictions and live API data.")
                st.markdown(f"- Stored predictions error: {stored_error}")
                st.markdown(f"- Live API error: {api_error}")
            else:
                message = stored_error or api_error or "Unknown error"
                st.error(f"❌ Failed to load data: {message}")

            st.markdown("""
            ### 🔧 Troubleshooting:
            - Ensure temporal integration predictions are saved locally
            - Check if API is running: https://fingraph-production.up.railway.app/health
            - Verify network connectivity
            - Try refreshing the page
            """)
            return

        self.dashboard_summary = results['summary']
        self.risk_data = results['predictions']
        self.data_source = actual_source

        if mode == "Auto" and actual_source == "Live API" and stored_error:
            st.warning(f"Stored predictions unavailable ({stored_error}). Loaded live API data instead.")

        self.render_header()

        tab1, tab2, tab3 = st.tabs(["🎯 Risk Overview", "🤖 Model Performance", "🚨 Alerts"])

        with tab1:
            self.render_risk_table()

        with tab2:
            self.render_model_performance()

        with tab3:
            self.render_alerts()

        refresh_label = "🔄 Refresh Live Data" if self.data_source == "Live API" else "🔄 Reload Stored Data"
        if st.button(refresh_label):
            st.rerun()


def main():
    """Run the live dashboard - API data only"""
    dashboard = FinGraphDashboard()
    dashboard.run()


if __name__ == "__main__":
    main()
