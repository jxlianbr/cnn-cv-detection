import numpy as np
import pandas as pd
from src.config import XGBoostConfig
from src.models.xgboost_baseline import train_baseline, predict_proba, FEATURE_COLUMNS


def _synthetic_training_df(n=200, seed=0):
    rng = np.random.default_rng(seed)
    n_pv, n_nonpv = n // 2, n - n // 2
    pv_pvsi = rng.normal(0.6, 0.05, n_pv)
    nonpv_pvsi = rng.normal(0.1, 0.05, n_nonpv)
    pvsi = np.concatenate([pv_pvsi, nonpv_pvsi])
    other = rng.normal(0.2, 0.05, (n, len(FEATURE_COLUMNS) - 1))
    df = pd.DataFrame(other, columns=[c for c in FEATURE_COLUMNS if c != "PVSI"])
    df["PVSI"] = pvsi
    df["label"] = ["pv"] * n_pv + ["non_pv"] * n_nonpv
    return df


def test_train_baseline_uses_fixed_hyperparameters():
    cfg = XGBoostConfig(max_depth=8, eta=0.01717, n_estimators=1000)
    model = train_baseline(_synthetic_training_df(), cfg)
    assert model.get_params()["max_depth"] == 8
    assert abs(model.get_params()["learning_rate"] - 0.01717) < 1e-9
    assert model.get_params()["n_estimators"] == 1000


def test_predict_proba_separates_synthetic_classes():
    cfg = XGBoostConfig(max_depth=4, eta=0.1, n_estimators=50)  # fast for the test
    train_df = _synthetic_training_df(n=400, seed=1)
    model = train_baseline(train_df, cfg)
    test_df = _synthetic_training_df(n=100, seed=2)
    probs = predict_proba(model, test_df)
    pv_probs = probs[test_df["label"] == "pv"]
    nonpv_probs = probs[test_df["label"] == "non_pv"]
    assert pv_probs.mean() > nonpv_probs.mean() + 0.3
