"""
Model Training Module for Air Quality Prediction
Trains multiple ML models and evaluates their performance
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, 
                             confusion_matrix, f1_score, precision_score, recall_score)
from sklearn.model_selection import cross_val_score
import joblib
import os
import json

class AirQualityModelTrainer:
    """
    Trainer for air quality prediction models
    """
    
    def __init__(self):
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        
    def initialize_models(self):
        """Initialize multiple ML models"""
        self.models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ),
            'Decision Tree': DecisionTreeClassifier(
                max_depth=15,
                min_samples_split=10,
                min_samples_leaf=4,
                random_state=42
            ),
            'Logistic Regression': LogisticRegression(
                max_iter=1000,
                multi_class='multinomial',
                solver='lbfgs',
                random_state=42,
                n_jobs=-1
            )
        }
        print(f"Initialized {len(self.models)} models")
        
    def train_model(self, model_name, model, X_train, y_train):
        """Train a single model"""
        print(f"\nTraining {model_name}...")
        model.fit(X_train, y_train)
        print(f"  ✓ {model_name} trained successfully")
        return model
    
    def evaluate_model(self, model_name, model, X_train, y_train, X_test, y_test, label_encoder):
        """Evaluate model performance"""
        print(f"\nEvaluating {model_name}...")
        
        # Predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Calculate metrics
        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        
        test_precision = precision_score(y_test, y_test_pred, average='weighted', zero_division=0)
        test_recall = recall_score(y_test, y_test_pred, average='weighted', zero_division=0)
        test_f1 = f1_score(y_test, y_test_pred, average='weighted', zero_division=0)
        
        # Cross-validation score
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_test_pred)
        
        # Classification report
        target_names = label_encoder.classes_
        report = classification_report(y_test, y_test_pred, 
                                      target_names=target_names,
                                      zero_division=0)
        
        results = {
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'test_precision': test_precision,
            'test_recall': test_recall,
            'test_f1': test_f1,
            'cv_mean': cv_mean,
            'cv_std': cv_std,
            'confusion_matrix': cm,
            'classification_report': report,
            'predictions': y_test_pred
        }
        
        print(f"  Train Accuracy: {train_accuracy:.4f}")
        print(f"  Test Accuracy:  {test_accuracy:.4f}")
        print(f"  Test F1-Score:  {test_f1:.4f}")
        print(f"  CV Score:       {cv_mean:.4f} (+/- {cv_std:.4f})")
        
        return results
    
    def train_all_models(self, X_train, y_train, X_test, y_test, label_encoder):
        """Train and evaluate all models"""
        print("=" * 60)
        print("TRAINING ALL MODELS")
        print("=" * 60)
        
        for model_name, model in self.models.items():
            # Train
            trained_model = self.train_model(model_name, model, X_train, y_train)
            
            # Evaluate
            results = self.evaluate_model(model_name, trained_model, 
                                         X_train, y_train, X_test, y_test,
                                         label_encoder)
            
            self.results[model_name] = results
            
            # Update best model
            if self.best_model is None or results['test_accuracy'] > self.results[self.best_model_name]['test_accuracy']:
                self.best_model = trained_model
                self.best_model_name = model_name
        
        print("\n" + "=" * 60)
        print(f"BEST MODEL: {self.best_model_name}")
        print(f"Test Accuracy: {self.results[self.best_model_name]['test_accuracy']:.4f}")
        print("=" * 60)
    
    def get_feature_importance(self, model_name, feature_names):
        """Get feature importance for tree-based models"""
        model = self.models[model_name]
        
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            feature_importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            return feature_importance_df
        else:
            return None
    
    def save_models(self, output_dir='../models'):
        """Save all trained models"""
        os.makedirs(output_dir, exist_ok=True)
        
        for model_name, model in self.models.items():
            filename = model_name.lower().replace(' ', '_') + '.pkl'
            filepath = os.path.join(output_dir, filename)
            joblib.dump(model, filepath)
            print(f"Saved {model_name} to {filepath}")
        
        # Save results summary
        results_summary = {}
        for model_name, results in self.results.items():
            results_summary[model_name] = {
                'train_accuracy': float(results['train_accuracy']),
                'test_accuracy': float(results['test_accuracy']),
                'test_precision': float(results['test_precision']),
                'test_recall': float(results['test_recall']),
                'test_f1': float(results['test_f1']),
                'cv_mean': float(results['cv_mean']),
                'cv_std': float(results['cv_std'])
            }
        
        with open(os.path.join(output_dir, 'results_summary.json'), 'w') as f:
            json.dump(results_summary, f, indent=4)
        
        print(f"\nResults summary saved to {output_dir}/results_summary.json")
    
    def print_comparison(self):
        """Print comparison of all models"""
        print("\n" + "=" * 60)
        print("MODEL COMPARISON")
        print("=" * 60)
        
        comparison_data = []
        for model_name, results in self.results.items():
            comparison_data.append({
                'Model': model_name,
                'Train Acc': f"{results['train_accuracy']:.4f}",
                'Test Acc': f"{results['test_accuracy']:.4f}",
                'F1-Score': f"{results['test_f1']:.4f}",
                'CV Score': f"{results['cv_mean']:.4f}"
            })
        
        df_comparison = pd.DataFrame(comparison_data)
        print(df_comparison.to_string(index=False))
        print("=" * 60)

def main():
    """Main training pipeline"""
    print("=" * 60)
    print("AIR QUALITY MODEL TRAINING")
    print("=" * 60)
    
    # Load preprocessed data
    print("\n1. Loading preprocessed data...")
    X_train = np.load('../data/X_train.npy')
    X_test = np.load('../data/X_test.npy')
    y_train = np.load('../data/y_train.npy')
    y_test = np.load('../data/y_test.npy')
    
    # Load label encoder
    label_encoder = joblib.load('../models/label_encoder.pkl')
    
    # Load feature names
    with open('../data/feature_names.txt', 'r') as f:
        feature_names = [line.strip() for line in f.readlines()]
    
    print(f"   Training samples: {X_train.shape[0]}")
    print(f"   Testing samples: {X_test.shape[0]}")
    print(f"   Number of features: {X_train.shape[1]}")
    print(f"   Number of classes: {len(label_encoder.classes_)}")
    
    # Initialize trainer
    trainer = AirQualityModelTrainer()
    
    # Initialize models
    print("\n2. Initializing models...")
    trainer.initialize_models()
    
    # Train all models
    print("\n3. Training and evaluating models...")
    trainer.train_all_models(X_train, y_train, X_test, y_test, label_encoder)
    
    # Print comparison
    trainer.print_comparison()
    
    # Feature importance for Random Forest
    print("\n4. Analyzing feature importance (Random Forest)...")
    feature_importance = trainer.get_feature_importance('Random Forest', feature_names)
    if feature_importance is not None:
        print("\nTop 10 Most Important Features:")
        print(feature_importance.head(10).to_string(index=False))
        
        # Save feature importance
        feature_importance.to_csv('../outputs/feature_importance.csv', index=False)
        print("\nFeature importance saved to ../outputs/feature_importance.csv")
    
    # Save models
    print("\n5. Saving models...")
    trainer.save_models()
    
    # Print classification report for best model
    print(f"\n6. Classification Report ({trainer.best_model_name}):")
    print(trainer.results[trainer.best_model_name]['classification_report'])
    
    print("\n" + "=" * 60)
    print("MODEL TRAINING COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()
