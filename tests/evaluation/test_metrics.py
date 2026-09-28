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


def test_compute_metrics_reports_nan_auc_for_single_class_split_instead_of_raising():
    # A small block-based split can land entirely in one class. ROC-AUC is
    # undefined there; sklearn currently warns and returns NaN rather than
    # raising, so a run does not abort after training completes. This pins that
    # behaviour -- if a future sklearn goes back to raising, the final scripts
    # would crash at the last step and this test catches it here instead.
    y_true = np.array([0, 0, 0, 0])
    y_pred = np.array([0, 0, 1, 0])
    y_proba = np.array([0.1, 0.2, 0.7, 0.3])

    m = compute_metrics(y_true, y_pred, y_proba)

    assert np.isnan(m["roc_auc"])
    assert m["overall_accuracy"] == 0.75


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


def test_mcnemar_counts_discordant_pairs_in_each_direction():
    from src.evaluation.metrics import mcnemar_test
    y_true = np.array([1, 1, 1, 0, 0, 0])
    pred_a = np.array([1, 1, 0, 0, 1, 0])  # wrong on 2 and 4
    pred_b = np.array([1, 0, 1, 0, 0, 0])  # wrong on 1
    r = mcnemar_test(y_true, pred_a, pred_b)
    assert r["a_only_correct"] == 1   # point 1
    assert r["b_only_correct"] == 2   # points 2 and 4


def test_mcnemar_identical_predictions_give_p_of_one():
    from src.evaluation.metrics import mcnemar_test
    y = np.array([1, 0, 1, 0])
    r = mcnemar_test(y, y, y)
    assert r["a_only_correct"] == r["b_only_correct"] == 0
    assert r["p_value"] == 1.0


def test_mcnemar_uses_the_exact_two_sided_binomial():
    from src.evaluation.metrics import mcnemar_test
    # 3 vs 10 discordant pairs: p = 2 * P(X <= 3), X ~ Bin(13, 0.5) = 2 * 378 / 8192
    y_true = np.ones(13, dtype=int)
    pred_a = np.array([1] * 3 + [0] * 10)
    pred_b = np.array([0] * 3 + [1] * 10)
    r = mcnemar_test(y_true, pred_a, pred_b)
    assert np.isclose(r["p_value"], 2 * 378 / 8192)
