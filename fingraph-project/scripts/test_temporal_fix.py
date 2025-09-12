"""
Test Script for FinGraph Temporal Fix

This script tests the temporal fix integration
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import logging
import warnings
warnings.filterwarnings('ignore')

def setup_logging():
    """Setup logging for the test"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def test_standalone_temporal_predictor():
    """Test the standalone temporal predictor"""
    logger = logging.getLogger(__name__)
    logger.info("🧪 Testing Standalone Temporal Predictor...")
    
    try:
        from src.models.temporal_risk_predictor import TemporalRiskPredictor
        
        # Create predictor and run pipeline
        predictor = TemporalRiskPredictor()
        success = predictor.run_complete_pipeline()
        
        if success:
            logger.info("✅ Standalone temporal predictor working!")
            
            # Show results summary
            if hasattr(predictor, 'results') and predictor.results:
                logger.info("📊 Model Performance:")
                for model, metrics in predictor.results.items():
                    if isinstance(metrics, dict) and 'mse' in metrics:
                        logger.info(f"  {model}: MSE = {metrics['mse']:.4f}")
            
            return True
        else:
            logger.error("❌ Standalone predictor failed")
            return False
            
    except ImportError as e:
        logger.error(f"❌ Import error: {e}")
        logger.info("💡 Make sure temporal_risk_predictor.py is saved in src/models/")
        return False
    except Exception as e:
        logger.error(f"❌ Standalone test failed: {e}")
        return False

def test_integration_with_fingraph():
    """Test integration with existing FinGraph"""
    logger = logging.getLogger(__name__)
    logger.info("🧪 Testing Integration with Existing FinGraph...")
    
    try:
        from src.models.temporal_integration import FinGraphTemporalIntegrator
        
        # Create integrator
        integrator = FinGraphTemporalIntegrator()
        
        # Test just the first few steps to see if integration works
        logger.info("  📂 Testing data loading...")
        if integrator.load_existing_fingraph_data():
            logger.info("  ✅ Data loading successful")
            
            logger.info("  ⚡ Testing temporal analysis...")
            if integrator.run_temporal_analysis():
                logger.info("  ✅ Temporal analysis successful")
                
                logger.info("  🏗️ Testing graph enhancement...")
                if integrator.build_enhanced_graph():
                    logger.info("  ✅ Graph enhancement successful")
                    logger.info("🎉 Integration test successful!")
                    return True
        
        logger.error("❌ Integration test failed at some step")
        return False
        
    except ImportError as e:
        logger.error(f"❌ Import error: {e}")
        logger.info("💡 Make sure you have:")
        logger.info("  1. temporal_integration.py in src/models/")
        logger.info("  2. Existing FinGraph data (run scripts/collect_data.py first)")
        return False
    except Exception as e:
        logger.error(f"❌ Integration test failed: {e}")
        return False

def run_full_integration():
    """Run the complete integration pipeline"""
    logger = logging.getLogger(__name__)
    logger.info("🚀 Running Full Integration Pipeline...")
    
    try:
        from src.models.temporal_integration import FinGraphTemporalIntegrator
        
        integrator = FinGraphTemporalIntegrator()
        success = integrator.run_complete_integration()
        
        if success:
            logger.info("🎉 Full integration successful!")
            logger.info("📁 Check results in data/temporal_integration/")
            return integrator
        else:
            logger.error("❌ Full integration failed")
            return None
            
    except Exception as e:
        logger.error(f"❌ Full integration error: {e}")
        return None

def check_prerequisites():
    """Check if prerequisites are met"""
    logger = logging.getLogger(__name__)
    logger.info("🔍 Checking Prerequisites...")
    
    # Check if required files exist
    required_files = [
        'src/models/temporal_risk_predictor.py',
        'src/models/temporal_integration.py',
        'src/features/graph_data_loader.py',
        'src/features/graph_constructor.py'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        logger.error("❌ Missing required files:")
        for file_path in missing_files:
            logger.error(f"  - {file_path}")
        return False
    
    # Check if data directory exists
    if not os.path.exists('data/raw'):
        logger.warning("⚠️ No data/raw directory found")
        logger.info("💡 Run 'python scripts/collect_data.py' first to collect data")
        return False
    
    # Check if any data files exist
    data_files = os.listdir('data/raw')
    if not data_files:
        logger.warning("⚠️ No data files found in data/raw/")
        logger.info("💡 Run 'python scripts/collect_data.py' first to collect data")
        return False
    
    logger.info("✅ All prerequisites met")
    return True

def demonstrate_temporal_fix():
    """Demonstrate the key temporal fix concepts"""
    logger = logging.getLogger(__name__)
    
    print("\n" + "="*60)
    print("🔬 FINGRAPH TEMPORAL FIX DEMONSTRATION")
    print("="*60)
    
    print("\n❌ PROBLEM IDENTIFIED:")
    print("   Static Features → Dynamic Labels (mathematically impossible)")
    print("   Same P/E ratio cannot predict different risks over time")
    
    print("\n✅ SOLUTION IMPLEMENTED:")
    print("   Point-in-Time Features → Forward Risk Labels")
    print("   Market state at T → Risk from T+1 to T+30")
    print("   Different market states → Different risk predictions")
    
    print("\n📊 KEY COMPONENTS:")
    print("   • Temporal feature extraction (daily market state)")
    print("   • 30-day forward risk calculation (VaR + volatility)")
    print("   • Proper temporal train/val/test splits")
    print("   • Baseline models (Logistic + Random Forest)")
    print("   • Enhanced GNN with temporal awareness")
    
    print("\n🎯 BUSINESS VALUE:")
    print("   • 30-day early warning for portfolio risk")
    print("   • Quantitative risk scoring for companies")
    print("   • Relationship-aware risk propagation")
    print("   • No data leakage in model training")

def main():
    """Main test execution"""
    logger = setup_logging()
    
    print("🧪 FinGraph Temporal Fix Testing Suite")
    print("="*50)
    
    # Demonstrate the fix concepts
    demonstrate_temporal_fix()
    
    print("\n📋 RUNNING TESTS:")
    print("-"*30)
    
    # Step 1: Check prerequisites
    if not check_prerequisites():
        print("\n❌ Prerequisites not met. Please:")
        print("  1. Save the provided files to correct locations")
        print("  2. Run data collection if needed")
        return False
    
    # Step 2: Test standalone predictor
    print("\n🧪 Test 1: Standalone Temporal Predictor")
    standalone_success = test_standalone_temporal_predictor()
    
    # Step 3: Test integration
    print("\n🧪 Test 2: Integration with Existing FinGraph")
    integration_success = test_integration_with_fingraph()
    
    # Step 4: Full pipeline if both tests pass
    if standalone_success and integration_success:
        print("\n🚀 Test 3: Full Integration Pipeline")
        integrator = run_full_integration()
        
        if integrator:
            print("\n🎉 ALL TESTS PASSED!")
            print("="*50)
            print("✅ Temporal issues fixed successfully")
            print("✅ Integration with FinGraph working")
            print("✅ Ready for production deployment")
            
            print("\n📁 Results saved to:")
            print("  • data/temporal_integration/")
            print("  • Enhanced graph with temporal features")
            print("  • Risk predictions for all companies")
            print("  • Dashboard summary data")
            
            print("\n🎯 Next Steps:")
            print("  1. Review prediction results")
            print("  2. Deploy real-time monitoring")
            print("  3. Create Streamlit dashboard")
            print("  4. Week 4: Presentation materials")
            
            return True
    
    # If tests failed
    print("\n❌ SOME TESTS FAILED")
    print("="*30)
    
    if not standalone_success:
        print("❌ Standalone predictor failed")
        print("💡 Check temporal_risk_predictor.py file and dependencies")
    
    if not integration_success:
        print("❌ Integration failed")
        print("💡 Check that FinGraph data exists and integration file is correct")
    
    print("\n🔧 Troubleshooting:")
    print("  1. Ensure all files are saved in correct locations")
    print("  2. Run 'python scripts/collect_data.py' to get data")
    print("  3. Check import paths and dependencies")
    print("  4. Review error messages above")
    
    return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🏆 FinGraph Temporal Fix: COMPLETE SUCCESS!")
    else:
        print("\n🔧 FinGraph Temporal Fix: Needs troubleshooting")
    
    print("\n📚 Documentation:")
    print("  • temporal_risk_predictor.py: Core temporal system")
    print("  • temporal_integration.py: FinGraph integration")
    print("  • test_temporal_fix.py: This testing script")