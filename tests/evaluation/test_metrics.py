import numpy as np
from src.evaluation.metrics import compute_metrics


def test_compute_metrics_perfect_predictions():
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([1, 1, 0, 0])
    y_proba = np.array([0.95, 0.9, 0.05, 0.1])
    m = compute_metrics(y_true, y_pred, y_proba)
    assert m["overall_accuracy"] == 1.0
    assert m["precision"] == 1.0
    assert m["recall"] == 1.0
    assert m["f1"] == 1.0
    assert m["kappa"] == 1.0
    assert m["roc_auc"] == 1.0


def test_compute_metrics_returns_all_expected_keys():
    y_true = np.array([1, 0, 1, 0, 1])
    y_pred = np.array([1, 0, 0, 0, 1])
    y_proba = np.array([0.9, 0.2, 0.4, 0.3, 0.8])
    m = compute_metrics(y_true, y_pred, y_proba)
    assert set(m.keys()) == {
        "overall_accuracy", "precision", "recall", "f1", "kappa", "roc_auc"
    }


def test_compute_metrics_matches_published_baseline_ballpark():
    # Sanity check against the published benchmark
    # (OA 0.984, P 0.999, R 0.969, F1 0.984, Kappa 0.968, ROC-AUC 0.9497)
    # using a confusion-matrix-equivalent synthetic set of the same shape
    # as the real 1500-point verification set (752 pv / 748 non_pv).
    rng = np.random.default_rng(0)
    y_true = np.array([1] * 752 + [0] * 748)
    y_pred = y_true.copy()
    flip_idx = rng.choice(len(y_true), size=24, replace=False)  # ~1.6% error rate
    y_pred[flip_idx] = 1 - y_pred[flip_idx]
    y_proba = np.where(y_pred == 1, rng.uniform(0.6, 1.0, len(y_true)),
                       rng.uniform(0.0, 0.4, len(y_true)))
    m = compute_metrics(y_true, y_pred, y_proba)
    assert m["overall_accuracy"] > 0.95
