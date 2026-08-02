"""Shared evaluation metrics, matching the metric set Shimada & Takeuchi
(2025) report (reports/paper-draft.md §4.4): overall accuracy, precision,
recall, F1, Cohen's Kappa, ROC-AUC."""
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    cohen_kappa_score, roc_auc_score,
)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> dict:
    return {
        "overall_accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "kappa": cohen_kappa_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_proba),
    }
