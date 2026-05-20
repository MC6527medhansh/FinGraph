"""Production-ready Streamlit Dashboard"""

import os
import sys
import json
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from datetime import datetime

if not os.path.exists('data/signals/latest_signals.csv'):
    st.warning("No signals found. Please run signal generation first.")
    
# Production detection
IS_PRODUCTION = os.environ.get('RENDER', False)

# Page config
st.set_page_config(
    page_title="FinGraph Trading Signals",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stMetric {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
    }
    .warning-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #fff3cd;
        border: 1px solid #ffeeba;
        color: #856404;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=60)
def load_signals():
    """Load latest signals with caching"""
    signals_path = Path('data/signals/latest_signals.csv')
    if signals_path.exists():
        return pd.read_csv(signals_path)
    return pd.DataFrame()


@st.cache_data(ttl=300)
def load_health_status():
    """Load system health status"""
    health_path = Path('data/health/latest_health.json')
    if health_path.exists():
        with open(health_path, 'r') as f:
            return json.load(f)
    return {'overall_health': 'unknown', 'checks': []}


def refresh_signals():
    """Run signal generation with real-time progress"""
    
    # Add project root to path
    project_root = Path(__file__).parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    try:
        # Import after path is set
        from scripts.generate_signals import SignalGenerator
        
        # Create progress display
        status_text = st.empty()
        progress_bar = st.progress(0)
        
        # Progress tracking
        steps = []
        current_step = [0]  # Use list to modify in nested function
        total_steps = 8  # Expected number of progress updates
        
        def update_progress(message: str):
            """Callback for progress updates"""
            steps.append(message)
            current_step[0] += 1
            
            status_text.text(f"Step {current_step[0]}/{total_steps}: {message}")
            progress_bar.progress(min(current_step[0] / total_steps, 1.0))
        
        # Initialize generator with progress callback
        generator = SignalGenerator(progress_callback=update_progress)
        
        # Generate signals
        signals = generator.generate_current_signals()
        
        if signals.empty:
            st.error("Signal generation returned no results")
            return False
        
        # Save signals
        update_progress("Saving signals to disk...")
        generator.save_signals(signals)
        
        # Complete
        progress_bar.progress(1.0)
        status_text.text("✅ Signal generation complete!")
        
        # Clear cache
        st.cache_data.clear()
        
        # Show summary
        with st.expander("Generation Summary", expanded=True):
            st.write("**Steps completed:**")
            for i, step in enumerate(steps, 1):
                st.write(f"{i}. {step}")
            
            st.write("\n**Signal Statistics:**")
            st.write(f"- Total stocks analyzed: {len(signals)}")
            st.write(f"- Strong Buy signals: {(signals['recommendation'] == 'STRONG_BUY').sum()}")
            st.write(f"- Buy signals: {(signals['recommendation'] == 'BUY').sum()}")
            st.write(f"- Top pick: {signals.sort_values('rank').iloc[0]['symbol']}")
        
        st.success("🎉 Signals refreshed successfully! Dashboard will reload...")
        return True
        
    except ImportError as e:
        st.error(f"Import failed: {str(e)}")
        st.info("Make sure you're running from the project root and all dependencies are installed")
        with st.expander("Troubleshooting"):
            st.code(f"Project root: {project_root}\nPython path: {sys.path[:3]}")
        return False
        
    except Exception as e:
        st.error(f"Signal generation failed: {str(e)}")
        with st.expander("Error Details"):
            import traceback
            st.code(traceback.format_exc())
        return False


def main():
    # Header
    st.title("FinGraph Trading Signals Dashboard")
    
    # Sidebar
    with st.sidebar:
        st.header("System Controls")
        
        # Refresh button (only in development)
        if not IS_PRODUCTION:
            if st.button("🔄 Refresh Signals", type="primary", use_container_width=True):
                if refresh_signals():
                    st.rerun()
        else:
            st.info("Signals update automatically at 9pm UTC daily")
        
        # System health
        st.header("System Health")
        health = load_health_status()
        
        health_color = {
            'healthy': '🟢',
            'degraded': '🟡',
            'unhealthy': '🔴',
            'unknown': '⚫'
        }
        
        st.metric(
            "Status",
            health['overall_health'].title(),
            health_color.get(health['overall_health'], '⚫')
        )
        
        # Health checks details
        if health['checks']:
            with st.expander("Health Check Details"):
                for check in health['checks']:
                    status_icon = '✅' if check['status'] == 'pass' else '❌'
                    st.write(f"{status_icon} {check['check']}: {check['message']}")
    
    # Load signals
    signals = load_signals()
    
    if signals.empty:
        st.warning("No signals found. Please refresh or check system status.")
        return
    
    # Key Metrics Row
    st.header("Market Overview")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    signal_date = pd.to_datetime(signals['date'].iloc[0])
    days_old = (datetime.now() - signal_date).days
    
    with col1:
        st.metric(
            "Signal Date",
            signal_date.strftime('%Y-%m-%d'),
            f"{days_old} days old",
            delta_color="inverse" if days_old > 1 else "off"
        )
    
    with col2:
        strong_buys = len(signals[signals['recommendation'] == 'STRONG_BUY'])
        st.metric("Strong Buys", strong_buys)
    
    with col3:
        avg_return = signals['return_forecast'].mean()
        st.metric(
            "Avg Return Forecast",
            f"{avg_return:.2%}",
            delta_color="normal" if avg_return > 0 else "inverse"
        )
    
    with col4:
        avg_risk = signals['risk_score'].mean()
        st.metric("Avg Risk Score", f"{avg_risk:.3f}")
    
    with col5:
        top_pick = signals.sort_values('rank').iloc[0]['symbol']
        st.metric("Top Pick", top_pick)
    
    # Signal Tables
    st.header("Trading Signals")
    
    tab1, tab2, tab3 = st.tabs(["Buy Signals", "Sell Signals", "All Signals"])
    
    with tab1:
        buy_signals = signals[signals['recommendation'].isin(['STRONG_BUY', 'BUY'])]
        if not buy_signals.empty:
            st.dataframe(
                buy_signals[['symbol', 'recommendation', 'return_forecast', 
                           'risk_score', 'signal_strength', 'rank']].sort_values('rank'),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("No buy signals today")
    
    with tab2:
        sell_signals = signals[signals['recommendation'] == 'SELL']
        if not sell_signals.empty:
            st.dataframe(
                sell_signals[['symbol', 'recommendation', 'return_forecast', 
                            'risk_score', 'signal_strength', 'rank']].sort_values('rank'),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("No sell signals today")
    
    with tab3:
        st.dataframe(
            signals.sort_values('rank'),
            use_container_width=True,
            hide_index=True
        )
    
    # Visualizations
    st.header("Signal Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Risk vs Return Scatter
        fig = px.scatter(
            signals,
            x='risk_score',
            y='return_forecast',
            color='recommendation',
            size='signal_strength',
            hover_data=['symbol'],
            title='Risk vs Return Profile',
            color_discrete_map={
                'STRONG_BUY': '#00cc00',
                'BUY': '#66ff66',
                'HOLD': '#ffcc00',
                'SELL': '#ff6666'
            }
        )
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        fig.add_vline(x=signals['risk_score'].median(), line_dash="dash", line_color="gray")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Signal Strength Bar Chart
        fig = px.bar(
            signals.sort_values('signal_strength', ascending=True),
            x='signal_strength',
            y='symbol',
            orientation='h',
            color='recommendation',
            title='Signal Strength by Stock',
            color_discrete_map={
                'STRONG_BUY': '#00cc00',
                'BUY': '#66ff66',
                'HOLD': '#ffcc00',
                'SELL': '#ff6666'
            }
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Distribution Analysis
    st.header("Distribution Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        fig = go.Figure(data=[
            go.Box(y=signals['risk_score'], name='Risk Score', boxmean='sd')
        ])
        fig.update_layout(title='Risk Score Distribution')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = go.Figure(data=[
            go.Box(y=signals['return_forecast'], name='Return Forecast', boxmean='sd')
        ])
        fig.update_layout(title='Return Forecast Distribution')
        fig.add_hline(y=0, line_dash="dash", line_color="red")
        st.plotly_chart(fig, use_container_width=True)
    
    with col3:
        fig = go.Figure(data=[
            go.Box(y=signals['volatility_forecast'], name='Volatility', boxmean='sd')
        ])
        fig.update_layout(title='Volatility Forecast Distribution')
        st.plotly_chart(fig, use_container_width=True)
    
    # Footer
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.caption("Data Source: Yahoo Finance")
    with col2:
        st.caption("Model: Graph Neural Network (GNN)")
    with col3:
        st.caption(f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()