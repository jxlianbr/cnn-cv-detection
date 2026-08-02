import numpy as np
import pandas as pd
from src.config import XGBoostConfig
from scripts.run_label_efficiency import (
    run_label_efficiency_curve, subset_indices_for_repeats,
)


def test_subset_indices_have_the_requested_size():
    subsets = subset_indices_for_repeats(n_total=200, fraction=0.25, seed=0, n_repeats=3)
    assert len(subsets) == 3
    assert all(len(s) == 50 for s in subsets)


def test_subset_indices_differ_between_repeats():
    # repeats that all drew the same subset would report zero spread and the
    # averaging would be measuring nothing
    subsets = subset_indices_for_repeats(n_total=200, fraction=0.25, seed=0, n_repeats=3)
    assert set(subsets[0]) != set(subsets[1])
    assert set(subsets[1]) != set(subsets[2])


def test_subset_indices_are_reproducible_for_the_same_seed():
    a = subset_indices_for_repeats(n_total=200, fraction=0.25, seed=7, n_repeats=2)
    b = subset_indices_for_repeats(n_total=200, fraction=0.25, seed=7, n_repeats=2)
    assert [s.tolist() for s in a] == [s.tolist() for s in b]


def test_subset_indices_never_sample_the_same_point_twice():
    subsets = subset_indices_for_repeats(n_total=50, fraction=1.0, seed=0, n_repeats=2)
    assert all(len(set(s)) == len(s) for s in subsets)


def _synthetic_baseline_df(n, seed):
    rng = np.random.default_rng(seed)
    n_pv, n_nonpv = n // 2, n - n // 2
    pvsi = np.concatenate([rng.normal(0.6, 0.05, n_pv), rng.normal(0.1, 0.05, n_nonpv)])
    df = pd.DataFrame(rng.normal(0.2, 0.05, (n, 8)),
                      columns=["B2", "B3", "B4", "B8", "B11", "B12", "NDVI", "MNDWI"])
    df["PVSI"] = pvsi
    df["label"] = ["pv"] * n_pv + ["non_pv"] * n_nonpv
    # spread over many 10km blocks so the early-stopping slice has blocks to
    # choose from, as it does with real sample points
    df["lon"] = 139.0 + (np.arange(n) % 20) * 0.12
    df["lat"] = 36.0 + (np.arange(n) // 20) * 0.12
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


def test_run_label_efficiency_curve_produces_one_row_per_fraction_model_and_repeat():
    train_df = _synthetic_baseline_df(200, seed=0)
    val_df = _synthetic_baseline_df(80, seed=1)
    train_patches, train_labels = _synthetic_patches(200, seed=0)
    val_patches, val_labels = _synthetic_patches(80, seed=1)
    cfg = XGBoostConfig(max_depth=3, eta=0.2, n_estimators=10)

    result = run_label_efficiency_curve(
        train_df, train_patches, train_labels,
        val_df, val_patches, val_labels,
        fractions=[0.5, 1.0], cfg=cfg, cnn_epochs=2, n_repeats=3,
    )

    assert len(result) == 12  # 2 fractions x 2 models x 3 repeats
    assert sorted(result["repeat"].unique()) == [0, 1, 2]
