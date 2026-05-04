"""
Model Training Module

This module handles training of classification models for flight delay prediction.
It loads preprocessed data, trains multiple models, and performs hyperparameter tuning.
"""

import pandas as pd
import numpy as np
import os
import pickle
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline


def load_data(base_path='data/processed/splits/'):
    """
    Load preprocessed training, validation, and test splits.
    
    Args:
        base_path: Path to the splits directory
        
    Returns:
        tuple: (X_train, y_train, X_val, y_val, X_test, y_test)
    """
    X_train = pd.read_csv(os.path.join(base_path, 'X_train.csv'))
    y_train = pd.read_csv(os.path.join(base_path, 'y_train.csv')).squeeze()
    X_val = pd.read_csv(os.path.join(base_path, 'X_val.csv'))
    y_val = pd.read_csv(os.path.join(base_path, 'y_val.csv')).squeeze()
    X_test = pd.read_csv(os.path.join(base_path, 'X_test.csv'))
    y_test = pd.read_csv(os.path.join(base_path, 'y_test.csv')).squeeze()
    
    print('X_train shape:', X_train.shape)
    print('X_val shape  :', X_val.shape)
    print('X_test shape :', X_test.shape)
    
    return X_train, y_train, X_val, y_val, X_test, y_test


def load_class_weights(weights_path='outputs/encoders/class_weights.pkl'):
    """
    Load class weights if available.
    
    Args:
        weights_path: Path to the class weights file
        
    Returns:
        dict or None: Class weights dictionary or None if file not found
    """
    class_weights = None
    try:
        with open(weights_path, 'rb') as f:
            class_weights = pickle.load(f)
        print('Loaded class_weights from', weights_path)
    except Exception:
        print('No class_weights file found; will use balanced/default settings')
    
    return class_weights


def prepare_numeric_data(X_train, X_val, X_test):
    """
    Drop non-numeric columns from datasets.
    
    Args:
        X_train, X_val, X_test: Feature datasets
        
    Returns:
        tuple: (X_train_numeric, X_val_numeric, X_test_numeric)
    """
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns
    X_train_numeric = X_train[numeric_cols]
    X_val_numeric = X_val[numeric_cols]
    X_test_numeric = X_test[numeric_cols]
    
    print(f'X_train shape after dropping non-numeric: {X_train_numeric.shape}')
    print(f'Columns: {list(X_train_numeric.columns)}')
    
    return X_train_numeric, X_val_numeric, X_test_numeric


def train_models(X_train_numeric, y_train, class_weights=None, output_dir='outputs/models'):
    """
    Train multiple classification models.
    
    Args:
        X_train_numeric: Numeric feature training data
        y_train: Training labels
        class_weights: Optional class weight dictionary
        output_dir: Directory to save trained models
        
    Returns:
        dict: Dictionary of trained models
    """
    # Determine scale_pos_weight for XGBoost
    n_on_time = (y_train == 0).sum()
    n_delayed = (y_train == 1).sum()
    scale_pos_weight = max(1, n_on_time / max(1, n_delayed))
    
    models = {
        'Logistic_Regression': LogisticRegression(
            max_iter=1000,
            class_weight=class_weights or 'balanced',
            n_jobs=-1
        ),
        'Random_Forest': RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            class_weight=class_weights or 'balanced',
            n_jobs=-1
        ),
        'XGBoost': XGBClassifier(
            use_label_encoder=False,
            eval_metric='logloss',
            scale_pos_weight=scale_pos_weight,
            n_jobs=-1
        )
    }
    
    results = {}
    os.makedirs(output_dir, exist_ok=True)
    
    for name, model in models.items():
        print(f'Training {name}...')
        model.fit(X_train_numeric, y_train)
        results[name] = model
        model_path = os.path.join(output_dir, f'{name}.pkl')
        joblib.dump(model, model_path)
        print(f'Saved model: {model_path}')
    
    return results, scale_pos_weight


def tune_xgboost(X_train_numeric, y_train, scale_pos_weight, output_dir='outputs/models'):
    """
    Perform hyperparameter tuning for XGBoost.
    
    Args:
        X_train_numeric: Numeric feature training data
        y_train: Training labels
        scale_pos_weight: Class weight for XGBoost
        output_dir: Directory to save tuned model
        
    Returns:
        RandomizedSearchCV: Fitted search object
    """
    param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [3, 6, 10],
        'classifier__learning_rate': [0.01, 0.1],
        'classifier__subsample': [0.8, 1.0],
        'classifier__scale_pos_weight': [scale_pos_weight]
    }
    
    xgb_pipeline = Pipeline(steps=[
        ('classifier', XGBClassifier(use_label_encoder=False, eval_metric='logloss'))
    ])
    
    tuned_xgb = RandomizedSearchCV(
        xgb_pipeline,
        param_distributions=param_grid,
        n_iter=5,
        scoring='roc_auc',
        cv=3,
        verbose=1,
        random_state=42
    )
    
    print('Starting Hyperparameter Tuning on XGBoost...')
    tuned_xgb.fit(X_train_numeric, y_train)
    
    print('Best Parameters:', tuned_xgb.best_params_)
    print('Best ROC-AUC:', tuned_xgb.best_score_)
    
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, 'XGBoost_tuned.pkl')
    joblib.dump(tuned_xgb.best_estimator_, model_path)
    print(f'Saved tuned XGBoost to {model_path}')
    
    return tuned_xgb


def get_final_model(tuned_xgb, results):
    """
    Get final model (tuned XGBoost if available, otherwise baseline XGBoost).
    
    Args:
        tuned_xgb: Tuned RandomizedSearchCV object
        results: Dictionary of trained models
        
    Returns:
        Model object: Final model
    """
    final_model = tuned_xgb.best_estimator_ if tuned_xgb is not None else results.get('XGBoost')
    return final_model


def save_final_model(final_model, output_path='outputs/models/final_model.pkl'):
    """
    Save the final model.
    
    Args:
        final_model: Model to save
        output_path: Path to save the model
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump(final_model, output_path)
    print(f'Saved final model to {output_path}')


def main():
    """Main training pipeline."""
    # Load data
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()
    
    # Load class weights
    class_weights = load_class_weights()
    
    # Prepare numeric data
    X_train_numeric, X_val_numeric, X_test_numeric = prepare_numeric_data(
        X_train, X_val, X_test
    )
    
    # Train models
    results, scale_pos_weight = train_models(X_train_numeric, y_train, class_weights)
    
    # Tune XGBoost
    tuned_xgb = tune_xgboost(X_train_numeric, y_train, scale_pos_weight)
    
    # Get final model
    final_model = get_final_model(tuned_xgb, results)
    
    # Save final model
    save_final_model(final_model)
    
    print('\nTraining completed successfully!')


if __name__ == '__main__':
    main()
