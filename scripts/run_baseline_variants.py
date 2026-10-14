"""XGBoost row order and the exploratory context variants, reported in full.

Reads the saved CNN predictions (predictions_{val,test}.parquet) and the
processed features and patches; trains only XGBoost models (CPU, a few
minutes). Modifies no existing file and writes two new logs.

(a) Row order. The published baseline was trained once, with the training rows
    in their original order, and XGBoost's result depends slightly on that
    order. Here it is retrained on five permutations of the full training split.
    The permutations are the ones of the label-efficiency experiment at 100%
    (``np.random.default_rng(seed).choice(n, n, replace=False)``, seeds 0-4), so
    gate GA checks that their validation accuracy and F1 equal
    label_efficiency_runs.csv. Per permutation and split: accuracy, F1, exact
    McNemar test against each CNN run, and the 10 km block bootstrap of the
    seed-averaged CNN minus baseline accuracy. A last row per split uses the
    baseline correctness averaged over the five permutations in the bootstrap.

(b) Context variants. Everything the exploratory runs examined that can be
    reconstructed from the saved session logs: the published XGBoost's PV
    probability pooled (mean and median) over the central k x k window of each
    patch, k = 3, 5, 7, 9, 11, 15, 21; the raw per-pixel baseline with a tuned
    threshold; and XGBoost retrained (published hyperparameters, original row
    order) on the nine per-pixel features plus window statistics, either the
    mean and SD of each feature over 3 x 3 (27 features) or over 3 x 3, 7 x 7
    and 21 x 21 (63 features, the exploratory version). Each variant follows the
    protocol of the 3 x 3 control in run_context_control.py: threshold tuned on
    validation (tune_threshold) and applied once to test, plus the untuned 0.5.
    Gate GB checks that the 3 x 3 mean and the raw rows reproduce
    context_control.csv.

Writes results/logs/baseline_rowperm.csv and results/logs/baseline_variants.csv.
"""
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root on sys.path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import f1_score

from scripts.run_context_control import (block_bootstrap, tune_threshold,
                                         window_probabilities)
from src.config import load_config
from src.data.indices import mndwi, ndvi, pvsi
from src.data.split import _block_id
from src.evaluation.metrics import mcnemar_test
from src.models.xgboost_baseline import FEATURE_COLUMNS, predict_proba, train_baseline

LOGS = Path("results/logs")
PROCESSED = Path("data/processed")
ALPHA = 0.05
PERM_SEEDS = range(5)
POOL_WINDOWS = (3, 5, 7, 9, 11, 15, 21)
OUTPUTS = ("baseline_rowperm.csv", "baseline_variants.csv")


# --- helpers ----------------------------------------------------------------

def gate(name: str, ok: bool, detail: str = "") -> bool:
    print(f"Gate {name}: {'PASS' if ok else 'FAILED'} {detail}")
    return ok


def write_new(df: pd.DataFrame, name: str):
    """Only this script's own two logs are written."""
    assert name in OUTPUTS, name
    path = LOGS / name
    df.to_csv(path, index=False)
    print(f"Wrote {path}")


def permutation(n: int, seed: int) -> np.ndarray:
    """Row order of label-efficiency repeat `seed` at fraction 1.0
    (run_label_efficiency.subset_indices_for_repeats with seed 0)."""
    return np.random.default_rng(seed).choice(n, size=n, replace=False)


def load_split(split: str, block_size_km: float) -> dict:
    """Predictions, features and patches of one split, aligned on point_id
    in the order of the saved predictions (as in run_context_control.main)."""
    features = pd.read_parquet(PROCESSED / f"{split}_features.parquet")
    patches = np.load(PROCESSED / f"{split}_patches.npy")
    pred = pd.read_parquet(LOGS / f"predictions_{split}.parquet")
    df = pred.merge(features.assign(row=np.arange(len(features)))
                    .drop(columns=["label", "polygon_area_m2", "lon", "lat"]),
                    on="point_id", how="left")
    assert len(df) == len(pred) and df["row"].notna().all()
    seeds = sorted(c for c in pred.columns if c.startswith("cnn_proba_seed"))
    y = (df["label"] == "pv").values
    cnn = np.column_stack([df[c].values >= 0.5 for c in seeds])
    blocks = np.array([f"{r}_{c}" for r, c in (_block_id(lo, la, block_size_km)
                                               for lo, la in zip(df["lon"], df["lat"]))])
    return {"df": df, "patches": patches[df["row"].values], "y": y, "cnn": cnn,
            "cnn_correct": (cnn == y[:, None]).mean(axis=1), "blocks": blocks}


def compare(data: dict, pred: np.ndarray, correct_b: np.ndarray | None = None) -> dict:
    """A baseline prediction against the CNN runs: accuracy, F1, exact McNemar
    per run (baseline = a) and the block bootstrap of CNN - baseline accuracy."""
    y, cnn = data["y"], data["cnn"]
    pred = np.asarray(pred, bool)
    if correct_b is None:
        correct_b = (pred == y).astype(float)
    tests = [mcnemar_test(y, pred.astype(int), cnn[:, k].astype(int))
             for k in range(cnn.shape[1])]
    p = [r["p_value"] for r in tests]
    diff, lo, hi = block_bootstrap(data["cnn_correct"], correct_b, data["blocks"], seed=0)
    accs = (cnn == y[:, None]).mean(axis=0)
    row = {"accuracy": np.mean(pred == y), "f1": f1_score(y, pred),
           "cnn_accuracy_min": accs.min(), "cnn_accuracy_max": accs.max()}
    for k, r in enumerate(tests):
        row[f"seed{k}_xgb_only"] = r["a_only_correct"]
        row[f"seed{k}_cnn_only"] = r["b_only_correct"]
        row[f"seed{k}_p"] = r["p_value"]
    row.update({"p_min": min(p), "p_max": max(p),
                "n_significant": int(sum(v < ALPHA for v in p)),
                "n_cnn_runs_below": int((accs < np.mean(pred == y)).sum()),
                "diff_pp": 100 * diff, "ci_low_pp": 100 * lo, "ci_high_pp": 100 * hi})
    return row


# --- (a) row order ----------------------------------------------------------

def row_order(train: pd.DataFrame, cfg, data: dict, published) -> pd.DataFrame:
    reference = pd.read_csv(LOGS / "label_efficiency_runs.csv")
    reference = reference[(reference["fraction"] == 1.0)
                          & (reference["model"] == "xgboost_baseline")].set_index("repeat")
    rows, correct, ok = [], {s: [] for s in data}, True
    models = [("original order (published)", None, published)]
    for seed in PERM_SEEDS:
        models.append((f"permutation {seed}", seed,
                       train_baseline(train.iloc[permutation(len(train), seed)], cfg)))
    for name, seed, model in models:
        for split, d in data.items():
            proba = predict_proba(model, d["df"])
            pred = proba >= 0.5
            if seed is not None:
                correct[split].append((pred == d["y"]).astype(float))
            rows.append({"split": split, "baseline": name, "perm_seed": seed,
                         **compare(d, pred)})
            if split == "val" and seed is not None and seed in reference.index:
                ref = reference.loc[seed]
                ok &= (np.isclose(rows[-1]["accuracy"], ref["overall_accuracy"], rtol=0, atol=1e-12)
                       and np.isclose(rows[-1]["f1"], ref["f1"], rtol=0, atol=1e-12))
    gate("GA (permutations = label-efficiency runs at 100%)", ok)
    for split, d in data.items():
        averaged = np.mean(correct[split], axis=0)
        diff, lo, hi = block_bootstrap(d["cnn_correct"], averaged, d["blocks"], seed=0)
        perms = [r for r in rows if r["split"] == split and r["perm_seed"] is not None]
        rows.append({"split": split, "baseline": "permutation-averaged", "perm_seed": np.nan,
                     "accuracy": averaged.mean(),
                     "f1": np.mean([r["f1"] for r in perms]),
                     "cnn_accuracy_min": perms[0]["cnn_accuracy_min"],
                     "cnn_accuracy_max": perms[0]["cnn_accuracy_max"],
                     "p_min": min(r["p_min"] for r in perms),
                     "p_max": max(r["p_max"] for r in perms),
                     "n_significant": min(r["n_significant"] for r in perms),
                     "diff_pp": 100 * diff, "ci_low_pp": 100 * lo, "ci_high_pp": 100 * hi})
    return pd.DataFrame(rows)


# --- (b) context variants ---------------------------------------------------

def pixel_features(patches: np.ndarray) -> np.ndarray:
    """(N, 9, H, W) per-pixel features in FEATURE_COLUMNS order; pixels that
    are zero in all bands (outside the downloaded tile) are NaN. Computed in
    the patches' own dtype (float32), as window_probabilities does, so that the
    pooled 3 x 3 variant is bit-identical to the published control."""
    p = patches
    b2, b3, b4, b8, b11, b12 = (p[:, i] for i in range(6))
    out = np.stack([b2, b3, b4, b8, b11, b12, ndvi(b8, b4), mndwi(b3, b11),
                    pvsi(b11, b12, b8, b2, b4)], axis=1)
    out[np.broadcast_to(np.all(patches == 0, axis=1)[:, None], out.shape)] = np.nan
    return out


def centre(a: np.ndarray, k: int) -> np.ndarray:
    half, c = k // 2, a.shape[-1] // 2
    return a[..., c - half:c + half + 1, c - half:c + half + 1]


def pixel_probabilities(model, patches: np.ndarray) -> np.ndarray:
    """(N, H, W) published-XGBoost PV probability of every patch pixel,
    NaN where the pixel is zero-padded (same features as window_probabilities)."""
    f = pixel_features(patches)
    n, _, h, w = f.shape
    X = np.moveaxis(f, 1, -1).reshape(-1, 9)
    padded = np.isnan(X).any(axis=1)
    proba = np.full(len(X), np.nan, dtype=np.float32)  # as window_probabilities
    # window_probabilities feeds the indices of all-zero pixels too and drops them
    # afterwards; computing only the valid pixels gives the same means.
    proba[~padded] = model.predict_proba(X[~padded])[:, 1]
    return proba.reshape(n, h, w)


def window_stat_features(df: pd.DataFrame, patches: np.ndarray,
                         windows: tuple[int, ...]) -> pd.DataFrame:
    """Per-pixel features (from the feature table) plus the mean and SD of each
    of the nine features over the central k x k window, padded pixels left out."""
    f = pixel_features(patches)
    cols = {c: df[c].values for c in FEATURE_COLUMNS}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for k in windows:
            w = centre(f, k).reshape(len(f), 9, -1)
            mean, sd = np.nanmean(w, axis=2), np.nanstd(w, axis=2)
            for i, c in enumerate(FEATURE_COLUMNS):
                cols[f"{c}_mean{k}"] = mean[:, i]
                cols[f"{c}_sd{k}"] = sd[:, i]
    return pd.DataFrame(cols)


def train_window_stat(train: pd.DataFrame, train_patches: np.ndarray, windows, cfg):
    X = window_stat_features(train, train_patches, windows)
    y = (train["label"] == "pv").astype(int).values
    model = xgb.XGBClassifier(max_depth=cfg.max_depth, learning_rate=cfg.eta,
                              n_estimators=cfg.n_estimators, objective="binary:logistic",
                              eval_metric="logloss", n_jobs=-1)
    model.fit(X.values, y)
    return model, list(X.columns)


def variant_rows(name: str, family: str, window: str, probas: dict, data: dict,
                 in_exploration: bool) -> list[dict]:
    y_val = data["val"]["y"]
    threshold = tune_threshold(y_val, probas["val"])
    # tune_threshold breaks ties in validation accuracy towards 0.5; record how
    # much the test accuracy moves across all thresholds tied on validation.
    grid = np.round(np.arange(0.05, 0.96, 0.01), 2)
    val_acc = np.array([np.mean((probas["val"] >= t) == y_val) for t in grid])
    ties = grid[val_acc == val_acc.max()]
    y_test = data["test"]["y"]
    tie_test = [np.mean((probas["test"] >= t) == y_test) for t in ties]
    rows = []
    for split in ("val", "test"):
        for source, t in (("tuned on val", threshold), ("fixed", 0.5)):
            rows.append({"variant": name, "family": family, "window": window,
                         "in_exploratory_runs": in_exploration, "split": split,
                         "threshold": t, "threshold_source": source,
                         "in_sample": split == "val" and source == "tuned on val",
                         **compare(data[split], probas[split] >= t),
                         "val_tied_threshold_min": ties.min(),
                         "val_tied_threshold_max": ties.max(),
                         "test_accuracy_over_val_ties_min": min(tie_test),
                         "test_accuracy_over_val_ties_max": max(tie_test)})
    return rows


def variants(train: pd.DataFrame, cfg, data: dict, published) -> pd.DataFrame:
    rows = []
    pix = {s: pixel_probabilities(published, d["patches"]) for s, d in data.items()}

    ok = all(np.allclose(np.nanmean(centre(pix[s], 3).reshape(len(pix[s]), -1), axis=1),
                         window_probabilities(published, d["patches"], 3), rtol=0, atol=1e-12)
             for s, d in data.items())
    centre_diff = max(np.abs(pix[s][:, 10, 10] - d["df"]["xgboost_baseline_proba"].values).max()
                      for s, d in data.items())
    ok &= centre_diff < 1e-2  # float32 patch pixels vs float64 feature table
    gate("GB0 (pixel grid = window_probabilities; centre pixel = saved probability)", ok,
         f"max |centre - saved| {centre_diff:.2g}")

    rows += variant_rows("Raw, per pixel", "raw", "1x1",
                         {s: d["df"]["xgboost_baseline_proba"].values for s, d in data.items()},
                         data, True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for stat, fn in (("mean", np.nanmean), ("median", np.nanmedian)):
            for k in POOL_WINDOWS:
                probas = {s: fn(centre(pix[s], k).reshape(len(pix[s]), -1), axis=1)
                          for s in data}
                rows += variant_rows(f"{k}x{k} pooled ({stat})", f"pooled {stat}",
                                     f"{k}x{k}", probas, data, True)

    train_patches = np.load(PROCESSED / "train_patches.npy")
    for windows, name, explored in (((3,), "XGBoost + 3x3 mean/SD (27 features)", False),
                                    ((3, 7, 21), "XGBoost + 3x3/7x7/21x21 mean/SD (63 features)",
                                     True)):
        model, columns = train_window_stat(train, train_patches, windows, cfg)
        probas = {s: model.predict_proba(
                      window_stat_features(d["df"], d["patches"], windows)[columns].values)[:, 1]
                  for s, d in data.items()}
        rows += variant_rows(name, "window statistics",
                             "/".join(f"{k}x{k}" for k in windows), probas, data, explored)
    result = pd.DataFrame(rows)

    reference = pd.read_csv(LOGS / "context_control.csv")
    checks = [("CNN vs 3x3 pooled XGBoost (t=0.5)", "3x3 pooled (mean)", "fixed"),
              ("CNN vs raw XGBoost (t=0.5)", "Raw, per pixel", "fixed")]
    t_ref = reference["threshold"].iloc[0]
    checks.append((f"CNN vs 3x3 pooled XGBoost (t={t_ref})", "3x3 pooled (mean)", "tuned on val"))
    ok = True
    for comparison, variant, source in checks:
        for _, ref in reference[reference["comparison"] == comparison].iterrows():
            mine = result[(result["variant"] == variant) & (result["split"] == ref["split"])
                          & (result["threshold_source"] == source)].iloc[0]
            ok &= all(np.isclose(mine[a], ref[b], rtol=0, atol=1e-9) for a, b in
                      (("accuracy", "baseline_accuracy"), ("p_min", "p_min"),
                       ("p_max", "p_max"), ("diff_pp", "diff_pp"),
                       ("ci_low_pp", "ci_low_pp"), ("ci_high_pp", "ci_high_pp")))
    ok &= np.isclose(result.loc[(result["variant"] == "3x3 pooled (mean)")
                                & (result["threshold_source"] == "tuned on val"),
                                "threshold"].iloc[0], t_ref)
    gate("GB (3x3 mean and raw rows = context_control.csv)", ok)
    return result


def main():
    cfg = load_config()
    train = pd.read_parquet(PROCESSED / "train_features.parquet")
    data = {s: load_split(s, cfg.split.block_size_km) for s in ("val", "test")}
    published = train_baseline(train, cfg.xgboost)
    diffs = [np.abs(predict_proba(published, d["df"]) - d["df"]["xgboost_baseline_proba"]).max()
             for d in data.values()]
    if not gate("G0 (refit = saved XGBoost)", max(diffs) < 1e-9,
                f"max |refit - saved| val {diffs[0]:.3g}, test {diffs[1]:.3g}"):
        raise SystemExit("refit does not reproduce the saved baseline")

    rowperm = row_order(train, cfg.xgboost, data, published)
    variant_table = variants(train, cfg.xgboost, data, published)

    show = ["split", "baseline", "accuracy", "f1", "p_min", "p_max", "n_significant",
            "diff_pp", "ci_low_pp", "ci_high_pp"]
    with pd.option_context("display.width", 250, "display.max_columns", 30,
                           "display.max_rows", 200):
        print(rowperm[show].round(4).to_string(index=False))
        print()
        print(variant_table[["variant", "split", "threshold", "in_sample", "accuracy",
                             "p_min", "p_max", "n_significant", "n_cnn_runs_below",
                             "diff_pp", "ci_low_pp", "ci_high_pp"]]
              .round(4).to_string(index=False))
    write_new(rowperm, "baseline_rowperm.csv")
    write_new(variant_table, "baseline_variants.csv")


if __name__ == "__main__":
    main()
