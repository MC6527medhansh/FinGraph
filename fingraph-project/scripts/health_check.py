#!/usr/bin/env python
"""System health monitoring and validation"""

import sys
import json
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import torch


class HealthChecker:
    """Monitor system health and data freshness"""
    
    def __init__(self):
        self.checks = []
        
    def check_model_exists(self) -> bool:
        """Verify model files exist"""
        model_files = list(Path('data/models').glob('*.pt'))
        status = len(model_files) > 0
        
        self.checks.append({
            'check': 'model_exists',
            'status': 'pass' if status else 'fail',
            'message': f'Found {len(model_files)} model files'
        })
        
        return status
    
    def check_signal_freshness(self) -> bool:
        """Check if signals are recent"""
        signals_file = Path('data/signals/latest_signals.csv')
        
        if not signals_file.exists():
            self.checks.append({
                'check': 'signal_freshness',
                'status': 'fail',
                'message': 'No signals file found'
            })
            return False
        
        # Check modification time
        mod_time = datetime.fromtimestamp(signals_file.stat().st_mtime)
        age_hours = (datetime.now() - mod_time).total_seconds() / 3600
        
        # Allow 30 hours for weekends
        max_age = 30 if datetime.now().weekday() >= 5 else 26
        status = age_hours < max_age
        
        self.checks.append({
            'check': 'signal_freshness',
            'status': 'pass' if status else 'warn',
            'message': f'Signals are {age_hours:.1f} hours old'
        })
        
        return status
    
    def check_predictions_valid(self) -> bool:
        """Validate prediction sanity"""
        signals_file = Path('data/signals/latest_signals.csv')
        
        if not signals_file.exists():
            return False
        
        signals = pd.read_csv(signals_file)
        
        # Check for reasonable values
        checks_passed = True
        messages = []
        
        # Risk should be between 0 and 1
        if signals['risk_score'].min() < 0 or signals['risk_score'].max() > 1:
            checks_passed = False
            messages.append('Risk scores out of range')
        
        # Should have diversity in predictions
        if signals['risk_score'].std() < 0.001:
            checks_passed = False
            messages.append('No diversity in risk predictions')
        
        # Not all predictions should be negative
        if signals['return_forecast'].max() < 0:
            checks_passed = False
            messages.append('All return predictions negative')
        
        self.checks.append({
            'check': 'predictions_valid',
            'status': 'pass' if checks_passed else 'fail',
            'message': '; '.join(messages) if messages else 'Predictions look reasonable'
        })
        
        return checks_passed
    
    def check_data_pipeline(self) -> bool:
        """Verify data pipeline components"""
        required_files = [
            'config/pipeline_config.yaml',
            'src/models/gnn_model.py',
            'src/core/feature_engine.py'
        ]
        
        missing = []
        for file in required_files:
            if not Path(file).exists():
                missing.append(file)
        
        status = len(missing) == 0
        
        self.checks.append({
            'check': 'data_pipeline',
            'status': 'pass' if status else 'fail',
            'message': f'Missing files: {missing}' if missing else 'All required files present'
        })
        
        return status
    
    def run_all_checks(self) -> Dict:
        """Run all health checks"""
        
        overall_health = 'healthy'
        
        # Run checks
        self.check_model_exists()
        self.check_signal_freshness()
        self.check_predictions_valid()
        self.check_data_pipeline()
        
        # Determine overall status
        for check in self.checks:
            if check['status'] == 'fail':
                overall_health = 'unhealthy'
                break
            elif check['status'] == 'warn' and overall_health == 'healthy':
                overall_health = 'degraded'
        
        return {
            'timestamp': datetime.now().isoformat(),
            'overall_health': overall_health,
            'checks': self.checks
        }
    
    def save_health_report(self, report: Dict):
        """Save health report"""
        health_dir = Path('data/health')
        health_dir.mkdir(parents=True, exist_ok=True)
        
        # Save latest
        with open(health_dir / 'latest_health.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        # Save timestamped
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        with open(health_dir / f'health_{timestamp}.json', 'w') as f:
            json.dump(report, f, indent=2)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--validate', action='store_true', 
                       help='Validate deployment readiness')
    args = parser.parse_args()
    
    checker = HealthChecker()
    report = checker.run_all_checks()
    
    print(json.dumps(report, indent=2))
    
    if args.validate and report['overall_health'] == 'unhealthy':
        print("\n❌ System not ready for deployment")
        sys.exit(1)
    else:
        print(f"\n✅ System health: {report['overall_health']}")
        checker.save_health_report(report)