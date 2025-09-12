"""
Baseline Models for FinGraph
Traditional ML models for risk prediction comparison
"""

import pandas as pd
import numpy as np
import torch
import logging
from typing import Dict, List, Tuple, Optional, Any
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import joblib
import os

logger = logging.getLogger(__name__)

class FinancialBaselineModels:
    """
    Baseline models for financial risk prediction
    
    Models:
    - Logistic Regression: Linear baseline
    - Random Forest: Tree-based ensemble
    - Gradient Boosting: Advanced boosting
    - SVM: Support Vector Machine
    """
    
    def __init__(self):
        """Initialize baseline models"""
        self.models = {
            'logistic_regression': LogisticRegression(random_state=42, max_iter=1000),
            'random_forest': RandomForestClassifier(
                n_estimators=100, 
                random_state=42, 
                max_depth=10,
                min_samples_split=5
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=100,
                random_state=42,
                max_depth=6,
                learning_rate=0.1
            ),
            'svm': SVC(
                random_state=42,
                probability=True,  # For ROC-AUC calculation
                kernel='rbf'
            )
        }
        
        self.scalers = {}
        self.trained_models = {}
        self.feature_names = []
    
    def prepare_features(self, graph, risk_labels: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Prepare features from graph for baseline models - FIXED VERSION
        
        Args:
            graph: PyTorch Geometric graph
            risk_labels: DataFrame with risk labels
            
        Returns:
            Tuple of (features, labels, feature_names)
        """
        logger.info("🔧 Preparing features for baseline models...")
        
        # Extract node features for companies only
        node_features = graph.x.numpy()
        node_types = graph.node_type.numpy()
        
        # Get company nodes (type 0)
        company_mask = node_types == 0
        company_features = node_features[company_mask]
        
        # Get node mapping to extract company symbols
        node_mapping = getattr(graph, 'node_mapping', {})
        reverse_mapping = {v: k for k, v in node_mapping.items()}
        
        company_symbols = []
        company_feature_map = {}
        
        for i, is_company in enumerate(company_mask):
            if is_company:
                node_name = reverse_mapping.get(i, f"node_{i}")
                if node_name.startswith('company_'):
                    symbol = node_name.replace('company_', '')
                    company_symbols.append(symbol)
                    # Map symbol to its feature index
                    company_idx = np.where(company_mask)[0].tolist().index(i)
                    company_feature_map[symbol] = company_idx
        
        logger.info(f"   Found {len(company_symbols)} companies: {company_symbols}")
        logger.info(f"   Node features shape: {company_features.shape}")
        
        # Create feature names
        feature_names = [f'feature_{i}' for i in range(company_features.shape[1])]
        
        # FIXED: Use ALL risk labels, not just latest per company
        matched_features = []
        matched_labels = []
        
        for _, risk_row in risk_labels.iterrows():
            symbol = risk_row['symbol']
            if symbol in company_feature_map:
                # Use the same company features for all time periods
                # (Graph structure is static, risk labels are temporal)
                company_idx = company_feature_map[symbol]
                matched_features.append(company_features[company_idx])
                matched_labels.append(risk_row['risk_binary'])
        
        if not matched_features:
            logger.error("❌ No matching features and labels found")
            return np.array([]), np.array([]), []
        
        features_array = np.array(matched_features)
        labels_array = np.array(matched_labels)
        
        logger.info(f"✅ Prepared {len(matched_features)} samples for baseline models")
        logger.info(f"   Features shape: {features_array.shape}")
        logger.info(f"   Risk distribution: {np.bincount(labels_array)}")
        
        # Check for class imbalance
        if len(np.unique(labels_array)) < 2:
            logger.warning("⚠️ Only one class found in labels - this will cause training issues")
        
        self.feature_names = feature_names
        return features_array, labels_array, feature_names
    
    def train_baseline_models(self, X_train: np.ndarray, y_train: np.ndarray,
                            X_val: np.ndarray = None, y_val: np.ndarray = None) -> Dict:
        """
        Train all baseline models
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            
        Returns:
            Dictionary with training results
        """
        logger.info("🏋️ Training baseline models...")
        
        if len(X_train) == 0:
            logger.error("❌ No training data provided")
            return {}
        
        results = {}
        
        for model_name, model in self.models.items():
            logger.info(f"   Training {model_name}...")
            
            try:
                # Scale features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                self.scalers[model_name] = scaler
                
                # Train model
                model.fit(X_train_scaled, y_train)
                self.trained_models[model_name] = model
                
                # Training metrics
                train_pred = model.predict(X_train_scaled)
                train_prob = model.predict_proba(X_train_scaled)[:, 1] if hasattr(model, 'predict_proba') else train_pred
                
                train_metrics = {
                    'accuracy': accuracy_score(y_train, train_pred),
                    'precision': precision_score(y_train, train_pred, zero_division=0),
                    'recall': recall_score(y_train, train_pred, zero_division=0),
                    'f1': f1_score(y_train, train_pred, zero_division=0),
                    'roc_auc': roc_auc_score(y_train, train_prob) if len(np.unique(y_train)) > 1 else 0.0
                }
                
                model_results = {'train': train_metrics}
                
                # Validation metrics if provided
                if X_val is not None and y_val is not None and len(X_val) > 0:
                    X_val_scaled = scaler.transform(X_val)
                    val_pred = model.predict(X_val_scaled)
                    val_prob = model.predict_proba(X_val_scaled)[:, 1] if hasattr(model, 'predict_proba') else val_pred
                    
                    val_metrics = {
                        'accuracy': accuracy_score(y_val, val_pred),
                        'precision': precision_score(y_val, val_pred, zero_division=0),
                        'recall': recall_score(y_val, val_pred, zero_division=0),
                        'f1': f1_score(y_val, val_pred, zero_division=0),
                        'roc_auc': roc_auc_score(y_val, val_prob) if len(np.unique(y_val)) > 1 else 0.0
                    }
                    model_results['validation'] = val_metrics
                
                results[model_name] = model_results
                logger.info(f"   ✅ {model_name} - Train F1: {train_metrics['f1']:.3f}, ROC-AUC: {train_metrics['roc_auc']:.3f}")
                
            except Exception as e:
                logger.error(f"   ❌ Failed to train {model_name}: {str(e)}")
                results[model_name] = {'error': str(e)}
        
        logger.info(f"✅ Trained {len(self.trained_models)} baseline models")
        return results
    
    def evaluate_models(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """
        Evaluate trained models on test set
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary with test results
        """
        logger.info("📊 Evaluating baseline models on test set...")
        
        if not self.trained_models:
            logger.error("❌ No trained models found. Train models first.")
            return {}
        
        results = {}
        
        for model_name, model in self.trained_models.items():
            try:
                # Scale test features
                scaler = self.scalers[model_name]
                X_test_scaled = scaler.transform(X_test)
                
                # Make predictions
                test_pred = model.predict(X_test_scaled)
                test_prob = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, 'predict_proba') else test_pred
                
                # Calculate metrics
                test_metrics = {
                    'accuracy': accuracy_score(y_test, test_pred),
                    'precision': precision_score(y_test, test_pred, zero_division=0),
                    'recall': recall_score(y_test, test_pred, zero_division=0),
                    'f1': f1_score(y_test, test_pred, zero_division=0),
                    'roc_auc': roc_auc_score(y_test, test_prob) if len(np.unique(y_test)) > 1 else 0.0
                }
                
                results[model_name] = test_metrics
                logger.info(f"   ✅ {model_name} - Test F1: {test_metrics['f1']:.3f}, ROC-AUC: {test_metrics['roc_auc']:.3f}")
                
            except Exception as e:
                logger.error(f"   ❌ Failed to evaluate {model_name}: {str(e)}")
                results[model_name] = {'error': str(e)}
        
        return results
    
    def save_models(self, save_dir: str = 'models/baseline'):
        """Save trained models and scalers"""
        os.makedirs(save_dir, exist_ok=True)
        
        for model_name, model in self.trained_models.items():
            # Save model
            model_path = os.path.join(save_dir, f'{model_name}_model.pkl')
            joblib.dump(model, model_path)
            
            # Save scaler
            scaler_path = os.path.join(save_dir, f'{model_name}_scaler.pkl')
            joblib.dump(self.scalers[model_name], scaler_path)
        
        logger.info(f"💾 Saved {len(self.trained_models)} baseline models to {save_dir}")
    
    def generate_baseline_report(self, train_results: Dict, test_results: Dict) -> str:
        """Generate comprehensive baseline model report"""
        report = []
        report.append("=" * 60)
        report.append("FINANCIAL RISK PREDICTION - BASELINE MODEL RESULTS")
        report.append("=" * 60)
        report.append(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Model comparison table
        report.append("📊 MODEL PERFORMANCE COMPARISON")
        report.append("-" * 40)
        report.append(f"{'Model':<20} {'Train F1':<10} {'Test F1':<10} {'Test ROC-AUC':<12}")
        report.append("-" * 40)
        
        for model_name in self.models.keys():
            if model_name in train_results and model_name in test_results:
                train_f1 = train_results[model_name].get('train', {}).get('f1', 0)
                test_f1 = test_results[model_name].get('f1', 0)
                test_auc = test_results[model_name].get('roc_auc', 0)
                
                report.append(f"{model_name:<20} {train_f1:<10.3f} {test_f1:<10.3f} {test_auc:<12.3f}")
        
        report.append("")
        
        # Best model identification
        if test_results:
            best_model = max(test_results.keys(), 
                           key=lambda x: test_results[x].get('f1', 0) if 'error' not in test_results[x] else 0)
            best_f1 = test_results[best_model].get('f1', 0)
            
            report.append(f"🏆 BEST PERFORMING MODEL: {best_model}")
            report.append(f"   Test F1 Score: {best_f1:.3f}")
            report.append("")
        
        # Detailed results for each model
        report.append("📈 DETAILED RESULTS")
        report.append("-" * 40)
        
        for model_name in self.models.keys():
            if model_name in test_results and 'error' not in test_results[model_name]:
                report.append(f"\n{model_name.upper()}:")
                metrics = test_results[model_name]
                report.append(f"  Accuracy:  {metrics.get('accuracy', 0):.3f}")
                report.append(f"  Precision: {metrics.get('precision', 0):.3f}")
                report.append(f"  Recall:    {metrics.get('recall', 0):.3f}")
                report.append(f"  F1 Score:  {metrics.get('f1', 0):.3f}")
                report.append(f"  ROC-AUC:   {metrics.get('roc_auc', 0):.3f}")
        
        report.append("")
        report.append("=" * 60)
        
        return "\n".join(report)

# Test function
def test_baseline_models():
    """Test baseline models"""
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
    
    from src.features.graph_data_loader import GraphDataLoader
    from src.features.graph_constructor import FinGraphConstructor
    from src.models.risk_labels import RiskLabelGenerator
    
    # Load data and build graph
    loader = GraphDataLoader()
    data = loader.load_latest_data()
    
    constructor = FinGraphConstructor()
    graph = constructor.build_graph(
        data['stock_data'],
        data['company_info'],
        data['economic_data'],
        data['relationship_data']
    )
    
    # Generate risk labels
    risk_gen = RiskLabelGenerator(lookforward_days=20)
    risk_labels = risk_gen.generate_risk_labels(data['stock_data'])
    
    if risk_labels.empty:
        print("❌ No risk labels generated")
        return
    
    # Initialize baseline models
    baseline = FinancialBaselineModels()
    
    # Prepare features
    X, y, feature_names = baseline.prepare_features(graph, risk_labels)
    
    if len(X) == 0:
        print("❌ No features prepared")
        return
    
    # Simple train/test split (since we have limited data)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"✅ Dataset prepared:")
    print(f"   Train: {len(X_train)} samples")
    print(f"   Test: {len(X_test)} samples")
    print(f"   Features: {len(feature_names)}")
    
    # Train models
    train_results = baseline.train_baseline_models(X_train, y_train)
    
    # Evaluate models
    if len(X_test) > 0:
        test_results = baseline.evaluate_models(X_test, y_test)
        
        # Generate report
        report = baseline.generate_baseline_report(train_results, test_results)
        print("\n" + report)
        
        # Save results
        os.makedirs('data/processed', exist_ok=True)
        with open('data/processed/baseline_results.txt', 'w') as f:
            f.write(report)
        
        print(f"\n💾 Saved baseline results to data/processed/baseline_results.txt")
        
        return baseline, train_results, test_results
    else:
        print("⚠️ No test data available")
        return baseline, train_results, {}

if __name__ == "__main__":
    test_baseline_models()