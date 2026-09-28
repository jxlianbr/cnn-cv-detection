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


def mcnemar_test(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> dict:
    """Exact McNemar test for two classifiers scored on the same points.

    Only the discordant points carry information: under the null hypothesis
    that both models have the same error rate, each discordant point is equally
    likely to favour either model, so the smaller count is Binomial(n, 0.5).
    The exact form is used because the discordant counts here are small (tens),
    where the chi-square approximation is unreliable.
    """
    from scipy.stats import binomtest

    correct_a = np.asarray(pred_a) == np.asarray(y_true)
    correct_b = np.asarray(pred_b) == np.asarray(y_true)
    a_only = int(np.sum(correct_a & ~correct_b))
    b_only = int(np.sum(~correct_a & correct_b))
    n = a_only + b_only
    p_value = 1.0 if n == 0 else float(binomtest(min(a_only, b_only), n, 0.5).pvalue)
    return {"a_only_correct": a_only, "b_only_correct": b_only, "p_value": p_value}
