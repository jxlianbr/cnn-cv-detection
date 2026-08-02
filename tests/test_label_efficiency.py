import numpy as np
import pandas as pd
from src.config import XGBoostConfig
from scripts.run_label_efficiency import run_label_efficiency_curve


def _synthetic_baseline_df(n, seed):
    rng = np.random.default_rng(seed)
    n_pv, n_nonpv = n // 2, n - n // 2
    pvsi = np.concatenate([rng.normal(0.6, 0.05, n_pv), rng.normal(0.1, 0.05, n_nonpv)])
    df = pd.DataFrame(rng.normal(0.2, 0.05, (n, 8)),
                      columns=["B2", "B3", "B4", "B8", "B11", "B12", "NDVI", "MNDWI"])
    df["PVSI"] = pvsi
    df["label"] = ["pv"] * n_pv + ["non_pv"] * n_nonpv
    return df


def _synthetic_patches(n, seed):
    rng = np.random.default_rng(seed)
    n_pv, n_nonpv = n // 2, n - n // 2
    pv = rng.normal(0.8, 0.05, (n_pv, 6, 21, 21)).astype(np.float32)
    nonpv = rng.normal(0.2, 0.05, (n_nonpv, 6, 21, 21)).astype(np.float32)
    patches = np.concatenate([pv, nonpv])
    labels = np.array(["pv"] * n_pv + ["non_pv"] * n_nonpv)
    return patches, labels


def test_run_label_efficiency_curve_covers_all_fractions_and_models():
    train_df = _synthetic_baseline_df(300, seed=0)
    val_df = _synthetic_baseline_df(100, seed=1)
    train_patches, train_labels = _synthetic_patches(300, seed=0)
    val_patches, val_labels = _synthetic_patches(100, seed=1)
    cfg = XGBoostConfig(max_depth=4, eta=0.1, n_estimators=30)

    result = run_label_efficiency_curve(
        train_df, train_patches, train_labels,
        val_df, val_patches, val_labels,
        fractions=[0.25, 1.0], cfg=cfg, cnn_epochs=3,
    )

    assert set(result["fraction"]) == {0.25, 1.0}
    assert set(result["model"]) == {"xgboost_baseline", "cnn"}
    assert {"overall_accuracy", "f1"}.issubset(result.columns)
    assert len(result) == 4  # 2 fractions x 2 models
