"""
Model Classification and Evaluation Module

This module handles model evaluation, predictions, and feature importance analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
from sklearn.metrics import classification_report, roc_auc_score, ConfusionMatrixDisplay


def load_models(model_dir='outputs/models'):
    """
    Load trained models from disk.
    
    Args:
        model_dir: Directory containing saved models
        
    Returns:
        dict: Dictionary of loaded models
    """
    models = {}
    model_files = {
        'Logistic_Regression': 'Logistic_Regression.pkl',
        'Random_Forest': 'Random_Forest.pkl',
        'XGBoost': 'XGBoost.pkl'
    }
    
    for name, filename in model_files.items():
        model_path = os.path.join(model_dir, filename)
        if os.path.exists(model_path):
            models[name] = joblib.load(model_path)
            print(f'Loaded {name} from {model_path}')
    
    return models


def evaluate_models(results, X_val_numeric, y_val, X_test_numeric, y_test):
    """
    Evaluate all models on validation and test sets.
    
    Args:
        results: Dictionary of trained models
        X_val_numeric, y_val: Validation data
        X_test_numeric, y_test: Test data
        
    Returns:
        pd.DataFrame: Performance summary
    """
    performance_summary = []
    
    for name, model in results.items():
        # Validation metrics
        y_val_pred = model.predict(X_val_numeric)
        y_val_proba = (
            model.predict_proba(X_val_numeric)[:, 1]
            if hasattr(model, 'predict_proba')
            else model.predict(X_val_numeric)
        )
        val_report = classification_report(y_val, y_val_pred, output_dict=True)
        val_auc = roc_auc_score(y_val, y_val_proba) if len(set(y_val)) > 1 else None
        
        # Test metrics
        y_test_pred = model.predict(X_test_numeric)
        y_test_proba = (
            model.predict_proba(X_test_numeric)[:, 1]
            if hasattr(model, 'predict_proba')
            else model.predict(X_test_numeric)
        )
        test_report = classification_report(y_test, y_test_pred, output_dict=True)
        test_auc = roc_auc_score(y_test, y_test_proba) if len(set(y_test)) > 1 else None
        
        performance_summary.append({
            'Model': name,
            'Val_Accuracy': val_report['accuracy'],
            'Val_F1_Delayed': val_report['1']['f1-score'],
            'Val_ROC_AUC': val_auc,
            'Test_Accuracy': test_report['accuracy'],
            'Test_F1_Delayed': test_report['1']['f1-score'],
            'Test_ROC_AUC': test_auc
        })
    
    perf_df = pd.DataFrame(performance_summary).sort_values(by='Test_ROC_AUC', ascending=False)
    return perf_df


def plot_confusion_matrix(model, X_test_numeric, y_test, model_name, output_dir='outputs/reports'):
    """
    Plot and save confusion matrix.
    
    Args:
        model: Trained model
        X_test_numeric, y_test: Test data
        model_name: Name of the model
        output_dir: Directory to save plot
    """
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(8, 6))
    ConfusionMatrixDisplay.from_estimator(model, X_test_numeric, y_test, cmap='Blues')
    plt.title(f'Confusion Matrix: {model_name} (Test)')
    plt.grid(False)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, f'confusion_matrix_{model_name}.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f'Saved confusion matrix: {output_path}')
    plt.close()


def plot_feature_importance(final_model, feature_names, output_dir='outputs/reports'):
    """
    Extract and plot feature importance.
    
    Args:
        final_model: Final trained model
        feature_names: Names of features
        output_dir: Directory to save plot
    """
    if hasattr(final_model.named_steps['classifier'], 'feature_importances_'):
        importance = final_model.named_steps['classifier'].feature_importances_
        feat_imp_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance
        }).sort_values('Importance', ascending=False).head(20)
        
        os.makedirs(output_dir, exist_ok=True)
        
        plt.figure(figsize=(8, 10))
        sns.barplot(x='Importance', y='Feature', data=feat_imp_df, palette='viridis')
        plt.title('Top 20 Feature Importances')
        plt.tight_layout()
        
        output_path = os.path.join(output_dir, 'feature_importance.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f'Saved feature importance plot: {output_path}')
        plt.close()
        
        return feat_imp_df
    else:
        print('Model does not expose feature_importances_; skipping plot')
        return None


def predict(model, X):
    """
    Make predictions on new data.
    
    Args:
        model: Trained model
        X: Feature data
        
    Returns:
        np.ndarray: Predictions
    """
    predictions = model.predict(X)
    return predictions


def predict_proba(model, X):
    """
    Get prediction probabilities.
    
    Args:
        model: Trained model
        X: Feature data
        
    Returns:
        np.ndarray: Prediction probabilities
    """
    if hasattr(model, 'predict_proba'):
        return model.predict_proba(X)
    else:
        return None


def load_final_model(model_path='outputs/models/final_model.pkl'):
    """
    Load the final trained model.
    
    Args:
        model_path: Path to the saved model
        
    Returns:
        Model object: Final model
    """
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        print(f'Loaded final model from {model_path}')
        return model
    else:
        print(f'Model not found at {model_path}')
        return None


def main():
    """Main evaluation pipeline."""
    # Load data
    base = 'data/processed/splits/'
    X_val = pd.read_csv(base + 'X_val.csv')
    y_val = pd.read_csv(base + 'y_val.csv').squeeze()
    X_test = pd.read_csv(base + 'X_test.csv')
    y_test = pd.read_csv(base + 'y_test.csv').squeeze()
    
    # Prepare numeric data
    numeric_cols = X_val.select_dtypes(include=[np.number]).columns
    X_val_numeric = X_val[numeric_cols]
    X_test_numeric = X_test[numeric_cols]
    
    # Load models
    results = load_models()
    
    if not results:
        print('No trained models found. Please run train.py first.')
        return
    
    # Evaluate models
    perf_df = evaluate_models(results, X_val_numeric, y_val, X_test_numeric, y_test)
    print('\n=== Model Performance Summary ===')
    print(perf_df.round(4))
    
    # Plot confusion matrix for best model
    best_model_name = perf_df.iloc[0]['Model']
    print(f'\nBest model by Test ROC_AUC: {best_model_name}')
    plot_confusion_matrix(results[best_model_name], X_test_numeric, y_test, best_model_name)
    
    # Load and analyze final model
    final_model = load_final_model()
    if final_model is not None:
        feat_imp_df = plot_feature_importance(final_model, X_test_numeric.columns)
        if feat_imp_df is not None:
            print('\n=== Top 20 Feature Importances ===')
            print(feat_imp_df)
    
    print('\nEvaluation completed successfully!')


if __name__ == '__main__':
    main()
