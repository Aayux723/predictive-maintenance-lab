"""
Evaluation module for predictive maintenance classification models.
Calculates rigorous classification metrics, ROC/PR curves, confusion matrices,
and both permutation and native feature importances.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve
)
from sklearn.inspection import permutation_importance


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Computes all standard academic classification metrics for rare failure detection.
    """
    metrics = {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "F1-Score": float(f1_score(y_true, y_pred, zero_division=0)),
        "Balanced Accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }

    if y_proba is not None and len(np.unique(y_true)) > 1:
        try:
            metrics["ROC-AUC"] = float(roc_auc_score(y_true, y_proba))
        except Exception:
            metrics["ROC-AUC"] = 0.5

        try:
            metrics["Average Precision"] = float(average_precision_score(y_true, y_proba))
        except Exception:
            metrics["Average Precision"] = float(np.mean(y_true))
    else:
        metrics["ROC-AUC"] = 0.5
        metrics["Average Precision"] = float(np.mean(y_true))

    return metrics


def evaluate_pipeline(
    pipeline: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Evaluates a fitted pipeline on untouched test set at a given probability threshold.
    """
    if hasattr(pipeline, "predict_proba"):
        probas = pipeline.predict_proba(X_test)[:, 1]
        preds = (probas >= threshold).astype(int)
    else:
        preds = pipeline.predict(X_test)
        probas = preds.astype(float)

    y_test_arr = np.array(y_test)
    metrics = compute_metrics(y_test_arr, preds, probas)
    cm = confusion_matrix(y_test_arr, preds)

    # Detailed confusion matrix counts
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (cm[0, 0], 0, 0, 0)
    cm_dict = {
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "Matrix": cm.tolist()
    }

    # ROC & PR curve data
    fpr, tpr, roc_thresh = roc_curve(y_test_arr, probas)
    precisions, recalls, pr_thresh = precision_recall_curve(y_test_arr, probas)

    return {
        "metrics": metrics,
        "confusion_matrix": cm_dict,
        "probabilities": probas,
        "predictions": preds,
        "roc_curve": {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "thresholds": roc_thresh.tolist()},
        "pr_curve": {"precision": precisions.tolist(), "recall": recalls.tolist(), "thresholds": pr_thresh.tolist()}
    }


def compute_feature_importances(
    pipeline: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    feature_names: List[str]
) -> Dict[str, pd.DataFrame]:
    """
    Calculates both Native (MDI) and Permutation Importances.
    Permutation importance on held-out test data provides an unbiased metric.
    """
    model = pipeline.named_steps["model"]

    # 1. Native feature importances (if available)
    native_df = None
    if hasattr(model, "feature_importances_"):
        native_imp = model.feature_importances_
        native_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": native_imp
        }).sort_values(by="Importance", ascending=False).reset_index(drop=True)

    # 2. Permutation feature importance on test set
    # Using average_precision scoring because of target imbalance
    perm = permutation_importance(
        pipeline,
        X_test,
        y_test,
        n_repeats=10,
        random_state=42,
        scoring="average_precision"
    )

    # Note: Permutation importance operates on raw input columns
    raw_feature_names = list(X_test.columns)
    perm_df = pd.DataFrame({
        "Feature": raw_feature_names,
        "Importance_Mean": perm.importances_mean,
        "Importance_Std": perm.importances_std
    }).sort_values(by="Importance_Mean", ascending=False).reset_index(drop=True)

    return {
        "native": native_df,
        "permutation": perm_df
    }


def compute_train_vs_test(
    pipeline: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> pd.DataFrame:
    """Computes train vs test performance to evaluate overfitting/underfitting."""
    train_eval = evaluate_pipeline(pipeline, X_train, y_train)
    test_eval = evaluate_pipeline(pipeline, X_test, y_test)

    records = []
    for k in ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC", "Average Precision", "Balanced Accuracy"]:
        records.append({
            "Metric": k,
            "Train": train_eval["metrics"].get(k, 0.0),
            "Test": test_eval["metrics"].get(k, 0.0),
            "Difference (Train - Test)": train_eval["metrics"].get(k, 0.0) - test_eval["metrics"].get(k, 0.0)
        })

    return pd.DataFrame(records)
