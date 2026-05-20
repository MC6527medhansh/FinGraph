#!/usr/bin/env python3
"""
FinGraph Project State Validator
Checks what actually works vs what's broken
"""

import sys
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple
import json

def run_command(cmd: str) -> Tuple[int, str, str]:
    """Run command and return (returncode, stdout, stderr)"""
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=30
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "Command timed out"
    except Exception as e:
        return -1, "", str(e)

def check_file_exists(path: str) -> bool:
    """Check if file exists"""
    return Path(path).exists()

def check_imports() -> Dict:
    """Check if Python imports work"""
    results = {}
    
    imports = [
        ("UnifiedDataManager", "from src.core.data_manager import UnifiedDataManager"),
        ("UnifiedFeatureEngine", "from src.core.feature_engine import UnifiedFeatureEngine"),
        ("TemporalGraphBuilder", "from src.pipeline.graph_builder import TemporalGraphBuilder"),
        ("FinancialGNN", "from src.models.gnn_model import FinancialGNN"),
        ("ModelTrainer", "from src.pipeline.trainer import ModelTrainer"),
        ("FinGraphBacktester", "from src.backtesting.backtester import FinGraphBacktester"),
    ]
    
    for name, import_cmd in imports:
        code, out, err = run_command(f'python -c "{import_cmd}"')
        results[name] = {
            'status': 'OK' if code == 0 else 'FAIL',
            'error': err if code != 0 else None
        }
    
    return results

def check_data_files() -> Dict:
    """Check for data files"""
    paths = {
        'config': 'config/pipeline_config.yaml',
        'models': 'data/models',
        'processed': 'data/processed',
        'cache': 'data/cache',
        'signals': 'data/signals',
    }
    
    results = {}
    for name, path in paths.items():
        p = Path(path)
        if p.is_dir():
            files = list(p.glob('*'))
            results[name] = {
                'exists': True,
                'is_dir': True,
                'count': len(files),
                'files': [f.name for f in files[:5]]  # First 5
            }
        elif p.is_file():
            results[name] = {
                'exists': True,
                'is_dir': False
            }
        else:
            results[name] = {
                'exists': False
            }
    
    return results

def check_redundancies() -> List[str]:
    """Check for redundant files that should be deleted"""
    redundant = []
    
    files_to_check = [
        'config/logging_config.yaml',
        'src/core/enhanced_features.py',
        'src/core/production_features.py',
        'src/models/baseline_models.py',
        'scripts/monitor_signals.py',
        'scripts/debug_correlations.py',
        'scripts/diagnose_signals.py',
    ]
    
    for f in files_to_check:
        if check_file_exists(f):
            redundant.append(f)
    
    return redundant

def check_critical_issues() -> Dict:
    """Check for the 3 critical issues"""
    issues = {}
    
    # Issue 1: Feature leakage
    if check_file_exists('src/pipeline/graph_builder.py'):
        with open('src/pipeline/graph_builder.py', 'r') as f:
            content = f.read()
            
        has_leakage = True
        if 'forward_return_cs_z' in content and 'forward_volatility_cs_z' in content:
            # Check if they're in LABEL_COLS
            if 'LABEL_COLS' in content:
                label_cols_section = content[content.find('LABEL_COLS'):content.find('LABEL_COLS') + 500]
                if 'forward_return_cs_z' in label_cols_section and 'forward_volatility_cs_z' in label_cols_section:
                    has_leakage = False
        
        issues['feature_leakage'] = {
            'status': 'PRESENT' if has_leakage else 'FIXED',
            'location': 'src/pipeline/graph_builder.py:27-32',
            'critical': True
        }
    
    # Issue 2: Schema mismatch - need to run script to check
    issues['schema_mismatch'] = {
        'status': 'UNKNOWN (need to test)',
        'test': 'python scripts/generate_signals.py',
        'critical': True
    }
    
    # Issue 3: Node-level - need to run script to check
    issues['node_level'] = {
        'status': 'UNKNOWN (need to test)',
        'test': 'python scripts/diagnose_predictions.py',
        'critical': True
    }
    
    return issues

def main():
    print("=" * 70)
    print("FINGRAPH PROJECT STATE VALIDATION")
    print("=" * 70)
    print("")
    
    # Check working directory
    if not check_file_exists('config/pipeline_config.yaml'):
        print("❌ ERROR: Not in FinGraph project directory")
        print("   Run this from the project root where config/ exists")
        sys.exit(1)
    
    print("✅ Project root directory confirmed")
    print("")
    
    # Check imports
    print("📦 Checking Python Imports...")
    print("-" * 70)
    imports = check_imports()
    all_imports_ok = True
    for name, result in imports.items():
        status_icon = "✅" if result['status'] == 'OK' else "❌"
        print(f"  {status_icon} {name:25s} {result['status']}")
        if result['error']:
            print(f"     Error: {result['error'][:60]}")
            all_imports_ok = False
    
    if not all_imports_ok:
        print("")
        print("⚠️  Some imports failed. Check:")
        print("   1. Are you in a virtual environment?")
        print("   2. Did you install requirements: pip install -r requirements.txt")
        print("   3. Is PYTHONPATH set correctly?")
    print("")
    
    # Check data files
    print("📁 Checking Data Files...")
    print("-" * 70)
    data_files = check_data_files()
    for name, info in data_files.items():
        if info['exists']:
            if info.get('is_dir'):
                print(f"  ✅ {name:15s} Directory with {info['count']} files")
                if info['count'] > 0:
                    print(f"     Latest: {info['files']}")
            else:
                print(f"  ✅ {name:15s} File exists")
        else:
            print(f"  ⚠️  {name:15s} Not found")
    print("")
    
    # Check redundancies
    print("🗑️  Checking for Redundant Files...")
    print("-" * 70)
    redundant = check_redundancies()
    if redundant:
        print(f"  Found {len(redundant)} redundant files that should be deleted:")
        for f in redundant:
            print(f"    - {f}")
        print("")
        print("  To remove: bash cleanup_redundancies.sh")
    else:
        print("  ✅ No redundant files found")
    print("")
    
    # Check critical issues
    print("🚨 Checking Critical Issues...")
    print("-" * 70)
    issues = check_critical_issues()
    for name, info in issues.items():
        status = info['status']
        icon = "✅" if status == 'FIXED' else "❌" if status == 'PRESENT' else "❓"
        print(f"  {icon} {name:25s} {status}")
        if 'location' in info:
            print(f"     Location: {info['location']}")
        if 'test' in info:
            print(f"     Test: {info['test']}")
    print("")
    
    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("")
    
    if all_imports_ok:
        print("✅ Core imports working")
    else:
        print("❌ Some imports failing")
    
    if len(redundant) == 0:
        print("✅ No redundant files")
    else:
        print(f"⚠️  {len(redundant)} redundant files need cleanup")
    
    print("")
    print("📋 NEXT STEPS:")
    print("")
    print("1. If imports failing:")
    print("   pip install -r requirements.txt")
    print("")
    print("2. If redundant files exist:")
    print("   bash cleanup_redundancies.sh")
    print("")
    print("3. Test critical issues:")
    print("   python scripts/diagnose_predictions.py")
    print("   python scripts/generate_signals.py")
    print("")
    print("4. Share results with Claude for focused fixes")
    print("")

if __name__ == "__main__":
    main()