#!/usr/bin/env python
"""Model versioning and deployment manager"""

import os
import json
import shutil
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional
import torch
import hashlib


class ModelManager:
    """Handles model versioning, validation, and deployment"""
    
    def __init__(self):
        self.models_dir = Path('data/models')
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.registry_file = self.models_dir / 'model_registry.json'
        
    def register_model(self, model_path: Path, metrics: Dict) -> Dict:
        """Register a new model with its performance metrics"""
        
        # Calculate model hash for versioning
        with open(model_path, 'rb') as f:
            model_hash = hashlib.sha256(f.read()).hexdigest()[:8]
        
        # Create version info
        version_info = {
            'version': f"v{datetime.now().strftime('%Y%m%d')}_{model_hash}",
            'path': str(model_path),
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics,
            'hash': model_hash,
            'deployed': False
        }
        
        # Update registry
        registry = self.load_registry()
        registry[version_info['version']] = version_info
        self.save_registry(registry)
        
        return version_info
    
    def deploy_model(self, version: Optional[str] = None) -> bool:
        """Deploy a model version (latest if version not specified)"""
        
        registry = self.load_registry()
        
        if not registry:
            print("No models in registry")
            return False
        
        if version is None:
            # Get latest version
            version = max(registry.keys())
        
        if version not in registry:
            print(f"Version {version} not found")
            return False
        
        model_info = registry[version]
        model_path = Path(model_info['path'])
        
        # Copy to production location
        prod_path = self.models_dir / 'production_model.pt'
        shutil.copy2(model_path, prod_path)
        
        # Update deployment status
        for v in registry:
            registry[v]['deployed'] = (v == version)
        self.save_registry(registry)
        
        # Commit to git if in production
        if os.environ.get('RENDER'):
            self._commit_to_git(prod_path, version)
        
        print(f"Deployed model version: {version}")
        return True
    
    def validate_model(self, model_path: Path) -> bool:
        """Validate model before deployment"""
        
        try:
            # Load model
            checkpoint = torch.load(model_path, map_location='cpu')
            
            # Check required keys
            required_keys = ['model_state_dict', 'config', 'history']
            for key in required_keys:
                if key not in checkpoint:
                    print(f"Missing required key: {key}")
                    return False
            
            # Check performance thresholds
            history = checkpoint.get('history', {})
            val_losses = history.get('val_loss', [])
            
            if not val_losses:
                print("No validation history found")
                return False
            
            # Model should be improving
            if len(val_losses) > 10:
                recent_avg = sum(val_losses[-5:]) / 5
                earlier_avg = sum(val_losses[-10:-5]) / 5
                if recent_avg > earlier_avg * 1.1:  # 10% degradation
                    print("Model performance degrading")
                    return False
            
            return True
            
        except Exception as e:
            print(f"Model validation failed: {e}")
            return False
    
    def rollback_model(self) -> bool:
        """Rollback to previous model version"""
        
        registry = self.load_registry()
        
        # Find currently deployed version
        current = None
        for v, info in registry.items():
            if info['deployed']:
                current = v
                break
        
        if not current:
            print("No currently deployed model")
            return False
        
        # Find previous version
        versions = sorted(registry.keys())
        current_idx = versions.index(current)
        
        if current_idx == 0:
            print("No previous version to rollback to")
            return False
        
        previous = versions[current_idx - 1]
        
        # Deploy previous version
        return self.deploy_model(previous)
    
    def load_registry(self) -> Dict:
        """Load model registry"""
        if self.registry_file.exists():
            with open(self.registry_file, 'r') as f:
                return json.load(f)
        return {}
    
    def save_registry(self, registry: Dict):
        """Save model registry"""
        with open(self.registry_file, 'w') as f:
            json.dump(registry, f, indent=2)
    
    def _commit_to_git(self, model_path: Path, version: str):
        """Commit model to git"""
        try:
            subprocess.run(['git', 'add', str(model_path)])
            subprocess.run(['git', 'commit', '-m', f'Deploy model {version}'])
            subprocess.run(['git', 'push'])
        except Exception as e:
            print(f"Git commit failed: {e}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--deploy', help='Deploy model version')
    parser.add_argument('--validate', help='Validate model file')
    parser.add_argument('--rollback', action='store_true', help='Rollback to previous version')
    args = parser.parse_args()
    
    manager = ModelManager()
    
    if args.validate:
        valid = manager.validate_model(Path(args.validate))
        print(f"Model valid: {valid}")
    elif args.deploy:
        manager.deploy_model(args.deploy if args.deploy != 'latest' else None)
    elif args.rollback:
        manager.rollback_model()