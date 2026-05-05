"""
Model Classification and Evaluation Module

This module handles model evaluation, predictions, and feature importance analysis.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, roc_auc_score, ConfusionMatrixDisplay


def load_models(model_dir: str | Path = "outputs/models"):
    """
    Load trained models from disk.
    
    Args:
        model_dir: Directory containing saved models
        
    Returns:
        dict: Dictionary of loaded models
    """
    models = {}
    model_files = {
        "Logistic_Regression": "Logistic_Regression.pkl",
        "Random_Forest": "Random_Forest.pkl",
        "XGBoost": "XGBoost.pkl",
    }
    base = Path(model_dir)
    for name, filename in model_files.items():
        model_path = base / filename
        if model_path.is_file():
            models[name] = joblib.load(model_path)
            print(f"Loaded {name} from {model_path}")
    
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
        v1 = val_report.get("1", {})

        # Test metrics
        y_test_pred = model.predict(X_test_numeric)
        y_test_proba = (
            model.predict_proba(X_test_numeric)[:, 1]
            if hasattr(model, 'predict_proba')
            else model.predict(X_test_numeric)
        )
        test_report = classification_report(y_test, y_test_pred, output_dict=True)
        test_auc = roc_auc_score(y_test, y_test_proba) if len(set(y_test)) > 1 else None
        t1 = test_report.get("1", {})

        performance_summary.append({
            "Model": name,
            "Val_Accuracy": val_report["accuracy"],
            "Val_Precision_Delayed": v1.get("precision"),
            "Val_Recall_Delayed": v1.get("recall"),
            "Val_F1_Delayed": v1.get("f1-score"),
            "Val_ROC_AUC": val_auc,
            "Test_Accuracy": test_report["accuracy"],
            "Test_Precision_Delayed": t1.get("precision"),
            "Test_Recall_Delayed": t1.get("recall"),
            "Test_F1_Delayed": t1.get("f1-score"),
            "Test_ROC_AUC": test_auc,
        })

    perf_df = pd.DataFrame(performance_summary)
    perf_df = perf_df.sort_values(by="Test_ROC_AUC", ascending=False, na_position="last")
    return perf_df


def plot_confusion_matrix(
    model: Any,
    X_test_numeric: pd.DataFrame,
    y_test: Any,
    model_name: str,
    output_dir: str | Path = "outputs/reports",
) -> None:
    """
    Plot and save confusion matrix.
    
    Args:
        model: Trained model
        X_test_numeric, y_test: Test data
        model_name: Name of the model
        output_dir: Directory to save plot
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 6))
    ConfusionMatrixDisplay.from_estimator(model, X_test_numeric, y_test, cmap="Blues")
    plt.title(f"Confusion Matrix: {model_name} (Test)")
    plt.grid(False)
    plt.tight_layout()

    output_path = out / f"confusion_matrix_{model_name}.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved confusion matrix: {output_path}")
    plt.close()


def _underlying_estimator(final_model: Any) -> Any:
    """Return step with ``feature_importances_`` (Pipeline or bare tree booster)."""
    if hasattr(final_model, "named_steps") and "classifier" in final_model.named_steps:
        return final_model.named_steps["classifier"]
    return final_model


def plot_feature_importance(
    final_model: Any,
    feature_names: pd.Index | list[str],
    output_dir: str | Path = "outputs/reports",
) -> pd.DataFrame | None:
    """
    Extract and plot feature importance.

    Args:
        final_model: Final trained model (sklearn Pipeline or tree-based estimator)
        feature_names: Names of features
        output_dir: Directory to save plot
    """
    est = _underlying_estimator(final_model)
    if not hasattr(est, "feature_importances_"):
        print("Model does not expose feature_importances_; skipping plot")
        return None

    importance = est.feature_importances_
    feat_imp_df = (
        pd.DataFrame({"Feature": list(feature_names), "Importance": importance})
        .sort_values("Importance", ascending=False)
        .head(20)
    )

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 10))
    sns.barplot(x="Importance", y="Feature", data=feat_imp_df, palette="viridis")
    plt.title("Top 20 Feature Importances")
    plt.tight_layout()

    output_path = out / "feature_importance.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved feature importance plot: {output_path}")
    plt.close()

    return feat_imp_df


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


def load_final_model(model_path: str | Path = "outputs/models/final_model.pkl"):
    """
    Load the final trained model.
    
    Args:
        model_path: Path to the saved model
        
    Returns:
        Model object: Final model
    """
    mp = Path(model_path)
    if mp.is_file():
        model = joblib.load(mp)
        print(f"Loaded final model from {mp}")
        return model
    print(f"Model not found at {mp}")
    return None


def run_classification(cfg: dict) -> None:
    """Evaluate saved models; paths from ``cfg`` (``files.splits``, ``models``, ``paths``)."""
    sp = cfg["files"]["splits"]
    X_val = pd.read_csv(sp["x_val"])
    y_val = pd.read_csv(sp["y_val"]).squeeze()
    X_test = pd.read_csv(sp["x_test"])
    y_test = pd.read_csv(sp["y_test"]).squeeze()

    numeric_cols = X_val.select_dtypes(include=[np.number]).columns
    X_val_numeric = X_val[numeric_cols]
    X_test_numeric = X_test[numeric_cols]

    models_cfg = cfg.get("models", {})
    model_dir = models_cfg.get("output_dir", "outputs/models")
    reports_dir = cfg.get("paths", {}).get("reports_dir", "outputs/reports")
    final_path = models_cfg.get("final_model", str(Path(model_dir) / "final_model.pkl"))

    results = load_models(model_dir)
    if not results:
        print("No trained models found. Run the train step first.")
        return

    perf_df = evaluate_models(results, X_val_numeric, y_val, X_test_numeric, y_test)
    print("\n=== Model Performance Summary ===")
    print(perf_df.round(4))

    summary_rel = cfg.get("files", {}).get("reports", {}).get(
        "model_performance_summary", "outputs/reports/model_performance_summary.csv"
    )
    summary_path = Path(summary_rel)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    perf_df.to_csv(summary_path, index=False)
    print(f"\nSaved model performance summary: {summary_path.resolve()}")

    best_model_name = perf_df.iloc[0]["Model"]
    print(f"\nBest model by Test ROC_AUC: {best_model_name}")
    plot_confusion_matrix(results[best_model_name], X_test_numeric, y_test, best_model_name, output_dir=reports_dir)

    final_model = load_final_model(final_path)
    if final_model is not None:
        feat_imp_df = plot_feature_importance(final_model, X_test_numeric.columns, output_dir=reports_dir)
        if feat_imp_df is not None:
            print("\n=== Top 20 Feature Importances ===")
            print(feat_imp_df)

    print("\nEvaluation completed successfully!")


def main() -> None:
    """CLI entry: load default TOML and run evaluation."""
    from src.data.preprocess import load_toml_config

    run_classification(load_toml_config("configs/config.toml"))


if __name__ == "__main__":
    main()
