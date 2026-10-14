"""Scoring against the verified visual annotation (scripts/run_failure_case_annotation.py)."""
import numpy as np
import pandas as pd
import pytest

from scripts.run_failure_case_annotation import (exact_mcnemar, load_verified,
                                                 model_rows, wilson_interval)


def test_wilson_interval_known_value():
    lo, hi = wilson_interval(17, 18)
    assert abs(lo - 0.742) < 0.001 and abs(hi - 0.990) < 0.001
    assert all(np.isnan(wilson_interval(0, 0)))


def test_exact_mcnemar_counts_discordant_points():
    a = np.array([True, True, False, False, True])
    b = np.array([True, False, True, True, True])
    a_only, b_only, p = exact_mcnemar(a, b)
    assert (a_only, b_only) == (1, 2) and p == pytest.approx(1.0)


def test_unverified_rows_stop_the_analysis(tmp_path):
    path = tmp_path / "labels.csv"
    pd.DataFrame({"point_id": [1, 2], "verified": ["true", "false"]}).to_csv(path, index=False)
    with pytest.raises(SystemExit):
        load_verified(path)


def test_model_rows_score_no_points_as_non_pv():
    sub = pd.DataFrame({
        "pv_present": ["yes", "yes", "no", "no", "unclear"],
        "xgb_pv": [True, True, True, False, True],
        "cnn_maj_pv": [True, False, True, True, True],
        **{f"cnn{k}_pv": [True, True, True, True, True] for k in range(5)},
    })
    rows = {(r["model"], r["metric"]): r for r in model_rows(sub)}
    assert rows[("xgboost", "label_recall")]["k"] == 4
    assert rows[("xgboost", "recall_verified")]["k"] == 2
    assert rows[("xgboost", "tnr_on_no")]["k"] == 1
    assert rows[("xgboost", "accuracy_verified")]["m"] == 4
    assert rows[("xgboost", "accuracy_verified")]["k"] == 3
    assert rows[("cnn_majority", "accuracy_verified")]["k"] == 1
    assert rows[("cnn_runs", "tnr_on_no")]["value"] == 0.0
