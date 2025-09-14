"""
FinGraph Main Integration Script
"""

import sys
import os
import logging
from datetime import datetime

# Add src to path to use YOUR existing modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import YOUR existing components
from models.temporal_integration import FinGraphTemporalIntegrator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """
    Run complete FinGraph pipeline using YOUR existing components
    """
    print("🚀 FinGraph Integration Pipeline")
    print("=" * 50)
    print("Using your existing temporal integration system...")
    
    try:
        # Use YOUR existing FinGraphTemporalIntegrator
        integrator = FinGraphTemporalIntegrator()
        
        print("\n📋 Pipeline Steps:")
        print("1. Load existing FinGraph data")
        print("2. Run temporal analysis") 
        print("3. Build enhanced graph")
        print("4. Generate risk predictions")
        print("5. Save results for dashboard/API")
        
        # Run YOUR existing integration pipeline
        success = integrator.run_complete_integration()
        
        if success:
            print("\n🎉 Pipeline completed successfully!")
            print(f"📂 Results saved to: data/temporal_integration/")
            
            # Show what was generated
            results_dir = "data/temporal_integration"
            if os.path.exists(results_dir):
                files = os.listdir(results_dir)
                print(f"\n📄 Generated files:")
                for file in sorted(files):
                    print(f"   • {file}")
            
            print(f"\n🚀 Next steps:")
            print(f"   Dashboard: streamlit run src/visualization/dashboard.py")
            print(f"   API: python api/main.py")
            
            return True
            
        else:
            print("\n❌ Pipeline failed. Check the logs above.")
            return False
            
    except ImportError as e:
        print(f"\n❌ Import error: {e}")
        print("💡 Make sure you're running from the fingraph-project/ directory")
        print("💡 Your temporal_integration.py should be in src/models/")
        return False
        
    except Exception as e:
        print(f"\n❌ Pipeline error: {e}")
        logger.exception("Pipeline failed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)