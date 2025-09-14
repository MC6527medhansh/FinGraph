"""
Dashboard Launcher
"""

import sys
import os
import subprocess

# Add src to path
project_root = os.path.dirname(__file__)
sys.path.append(os.path.join(project_root, 'src'))

def launch_dashboard():
    """Launch the FinGraph dashboard"""
    dashboard_path = os.path.join(project_root, 'src', 'visualization', 'dashboard.py')
    
    if not os.path.exists(dashboard_path):
        print(f"❌ Dashboard not found at: {dashboard_path}")
        print("💡 Make sure you saved the dashboard code to src/visualization/dashboard.py")
        return False
    
    print("🚀 Launching FinGraph Dashboard...")
    print(f"📂 Dashboard: {dashboard_path}")
    print("🌐 URL: http://localhost:8501")
    print("⏹️ Press Ctrl+C to stop")
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", dashboard_path
        ])
        return True
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped")
        return True
    except Exception as e:
        print(f"❌ Failed to launch dashboard: {e}")
        return False

if __name__ == "__main__":
    success = launch_dashboard()
    sys.exit(0 if success else 1)