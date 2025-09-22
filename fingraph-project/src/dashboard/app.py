import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
import subprocess
import json
from datetime import datetime

st.set_page_config(page_title="FinGraph Trading Dashboard", layout="wide")

def refresh_signals():
    """Generate fresh signals"""
    with st.spinner('Generating fresh signals...'):
        result = subprocess.run(['python', 'scripts/generate_signals.py'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            st.success("✅ Signals refreshed!")
        else:
            st.error(f"Error: {result.stderr}")
    
def load_latest_signals():
    """Load most recent signals"""
    signals_path = Path('data/signals/latest_signals.csv')
    if signals_path.exists():
        return pd.read_csv(signals_path)
    return pd.DataFrame()

# Header
st.title("🚀 FinGraph Trading Dashboard")
st.markdown("---")

# Sidebar controls
with st.sidebar:
    st.header("Controls")
    if st.button("🔄 Refresh Signals", type="primary"):
        refresh_signals()
        st.rerun()
    
    st.markdown("---")
    st.info("Signals auto-generate from latest market data")

# Load signals
signals = load_latest_signals()

if signals.empty:
    st.warning("No signals found. Click 'Refresh Signals' to generate.")
else:
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Signal Date", signals['date'].iloc[0])
    with col2:
        st.metric("Strong Buys", len(signals[signals['recommendation'] == 'STRONG_BUY']))
    with col3:
        avg_return = signals['return_forecast'].mean()
        st.metric("Avg Return Forecast", f"{avg_return:.2%}")
    with col4:
        avg_risk = signals['risk_score'].mean()
        st.metric("Avg Risk Score", f"{avg_risk:.3f}")
    
    st.markdown("---")
    
    # Trading recommendations
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 BUY Signals")
        buys = signals[signals['recommendation'].isin(['BUY', 'STRONG_BUY'])].sort_values('rank')
        if not buys.empty:
            st.dataframe(buys[['symbol', 'recommendation', 'return_forecast', 'risk_score', 'signal_strength']],
                        use_container_width=True)
        else:
            st.info("No buy signals")
    
    with col2:
        st.subheader("📉 SELL Signals")
        sells = signals[signals['recommendation'] == 'SELL'].sort_values('rank')
        if not sells.empty:
            st.dataframe(sells[['symbol', 'recommendation', 'return_forecast', 'risk_score']],
                        use_container_width=True)
        else:
            st.info("No sell signals")
    
    st.markdown("---")
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Risk vs Return")
        fig = px.scatter(signals, x='risk_score', y='return_forecast', 
                        text='symbol', color='recommendation',
                        title='Risk-Return Profile',
                        color_discrete_map={'STRONG_BUY': 'green', 'BUY': 'lightgreen', 
                                          'HOLD': 'gray', 'SELL': 'red'})
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Signal Strength Ranking")
        fig = px.bar(signals.sort_values('signal_strength', ascending=True), 
                    x='signal_strength', y='symbol', orientation='h',
                    color='recommendation',
                    title='Signal Strength by Stock')
        st.plotly_chart(fig, use_container_width=True)
    
    # Full data table
    with st.expander("📊 View All Signals"):
        st.dataframe(signals, use_container_width=True)