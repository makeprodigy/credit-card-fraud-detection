"""
evaluate.py
-----------
Evaluation helpers for fraud detection models.

Because the dataset is highly imbalanced (~0.17% fraud),
we focus on:
  - AUPRC  (Area Under the Precision-Recall Curve) — primary metric
  - F1 Score
  - Recall  (catching fraud is critical — minimise false negatives)
  - Precision
  - ROC-AUC
  - Confusion Matrix
"""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_model(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str = "Model",
    threshold: float = 0.5,
) -> dict:
    """
    Evaluate a trained classifier and return a metrics dictionary.

    Parameters
    ----------
    model       : fitted sklearn estimator (must have predict_proba)
    X_test      : test feature matrix
    y_test      : true labels
    model_name  : display name
    threshold   : classification threshold (default 0.5)

    Returns
    -------
    dict with keys: model, auprc, roc_auc, f1, recall, precision,
                    confusion_matrix, report
    """
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    metrics = {
        "model": model_name,
        "auprc": round(average_precision_score(y_test, y_proba), 4),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
        "f1": round(f1_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "report": classification_report(y_test, y_pred, target_names=["Legit", "Fraud"]),
    }

    print(f"\n{'='*55}")
    print(f"  {model_name}")
    print(f"{'='*55}")
    print(f"  AUPRC     : {metrics['auprc']}")
    print(f"  ROC-AUC   : {metrics['roc_auc']}")
    print(f"  F1 Score  : {metrics['f1']}")
    print(f"  Recall    : {metrics['recall']}")
    print(f"  Precision : {metrics['precision']}")
    print(f"\nConfusion Matrix:\n{metrics['confusion_matrix']}")
    print(f"\nClassification Report:\n{metrics['report']}")

    return metrics


def compare_models(results: list[dict]) -> pd.DataFrame:
    """
    Take a list of metrics dicts (from evaluate_model) and return
    a tidy comparison DataFrame.
    """
    rows = [
        {
            "Model": r["model"],
            "AUPRC": r["auprc"],
            "ROC-AUC": r["roc_auc"],
            "F1": r["f1"],
            "Recall": r["recall"],
            "Precision": r["precision"],
        }
        for r in results
    ]
    df = pd.DataFrame(rows).set_index("Model").sort_values("AUPRC", ascending=False)
    return df
