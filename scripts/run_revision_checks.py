"""Revision checks: four post-hoc analyses added after the main results were known.

All of them read outputs that are already saved (predictions_{val,test}.parquet,
label_efficiency_runs.csv, the processed patches and the polygon release) and
retrain nothing. The XGBoost baseline is refitted once with the published
configuration; gate G0 checks that the refit reproduces the saved
probabilities exactly, so it is the same model, not a new training.

1. Japan-only check. The points inside the bounding boxes of the sampling
   squares that fall outside Japan (all labelled non-PV) are dropped, and raw
   XGBoost is compared with each CNN run again (exact McNemar per run, 10 km
   block bootstrap of the seed-averaged accuracy difference). Gate G1: with
   the filter disabled, the same code reproduces context_control.csv.
2. Per-bin McNemar. Exact McNemar test per capacity bin (PV points only),
   majority vote of the five CNN runs against XGBoost, plus the per-run p range.
   Gate G2: n and hit counts equal make_figures.size_bin_hits (tab_size_recall).
3. Error location at t = 0.5. The pooled baseline's net gain per edge-distance
   stratum at the untuned threshold 0.5. error_location.csv uses the threshold
   tuned on validation (in-sample on val); the t = 0.5 columns here are the
   out-of-sample validation figures. Gate G3: the recomputed raw, tuned-pooled
   and CNN columns equal error_location.csv.
4. Paired label efficiency. Per training size, the five paired CNN - XGBoost
   differences (both models see the same subset within a repeat): mean, SD,
   95% t-interval, paired t-test and exact sign test.
5. Size vs edge share. Share of PV points within 10 m of a polygon edge per
   capacity bin, and recall on both sides of 10 m where the bin allows it.

Writes results/logs/revision_japan_only.csv, revision_size_bin_mcnemar.csv,
revision_error_location_t050.csv, revision_label_efficiency_paired.csv and
revision_size_edge.csv. The logs written by the other scripts are only read.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root on sys.path

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.stats import binomtest, t as t_dist, ttest_1samp

from scripts.make_figures import SIZE_BINS, SIZE_EDGES_KW, size_bin_hits
from scripts.run_context_control import (AREA_CRS, LOCATION_STRATA, WINDOW,
                                         block_bootstrap, edge_distance,
                                         tune_threshold, window_probabilities)
from src.config import load_config
from src.data.split import _block_id
from src.evaluation.failure_cases import DEFAULT_KW_PER_M2, capacity_bin_masks
from src.evaluation.metrics import mcnemar_test
from src.models.xgboost_baseline import predict_proba, train_baseline

LOGS = Path("results/logs")
PROCESSED = Path("data/processed")
ALPHA = 0.05
EDGE_M = 10
MIN_SIDE = 20  # points needed on each side of EDGE_M before recall is split
METRICS = {"f1": "F1", "overall_accuracy": "overall accuracy"}


def outside_japan(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """Conservative boxes around the parts of the sampling squares that lie
    outside Japan (Korea, Russia/Sakhalin), round-1 M-1."""
    return (((lon < 139.3) & (lat > 41.6)) | ((lon < 130.9) & (lat > 34.8))
            | (lat > 45.56))


def blocks_of(df: pd.DataFrame, block_size_km: float) -> np.ndarray:
    return np.array([f"{r}_{c}" for r, c in (_block_id(lo, la, block_size_km)
                                             for lo, la in zip(df["lon"], df["lat"]))])


def seed_columns(df: pd.DataFrame) -> list[str]:
    return sorted(c for c in df.columns if c.startswith("cnn_proba_seed"))


def write_new(df: pd.DataFrame, name: str):
    """Only revision_*.csv files are written; the main logs stay untouched."""
    assert name.startswith("revision_"), name
    path = LOGS / name
    df.to_csv(path, index=False)
    print(f"Wrote {path}")


def gate(name: str, ok: bool, detail: str = "") -> bool:
    print(f"Gate {name}: {'PASS' if ok else 'FAILED'} {detail}")
    return ok


# --- G0 ---------------------------------------------------------------------

def check_refit(model, preds: dict) -> bool:
    diffs = []
    for split, pred in preds.items():
        features = pd.read_parquet(PROCESSED / f"{split}_features.parquet")
        aligned = pred[["point_id"]].merge(features, on="point_id", how="left")
        diffs.append(np.abs(predict_proba(model, aligned)
                            - pred["xgboost_baseline_proba"].values).max())
    return gate("G0 (refit = saved XGBoost)", max(diffs) < 1e-9,
                f"max |refit - saved| val {diffs[0]:.3g}, test {diffs[1]:.3g}")


# --- X-1 --------------------------------------------------------------------

def raw_vs_cnn(pred: pd.DataFrame, keep: np.ndarray, block_size_km: float) -> dict:
    """Raw XGBoost (t = 0.5) against each CNN run on the kept points."""
    df = pred[keep]
    y = (df["label"] == "pv").values
    raw = df["xgboost_baseline_proba"].values >= 0.5
    cnn = np.column_stack([df[c].values >= 0.5 for c in seed_columns(df)])
    accs = (cnn == y[:, None]).mean(axis=0)
    tests = [mcnemar_test(y, raw.astype(int), cnn[:, k].astype(int))
             for k in range(cnn.shape[1])]
    diff, lo, hi = block_bootstrap((cnn == y[:, None]).mean(axis=1),
                                   (raw == y).astype(float),
                                   blocks_of(df, block_size_km), seed=0)
    row = {"n": int(keep.sum()), "baseline_accuracy": np.mean(raw == y),
           "cnn_accuracy_min": accs.min(), "cnn_accuracy_max": accs.max()}
    for k, r in enumerate(tests):
        row.update({f"seed{k}_accuracy": accs[k], f"seed{k}_xgb_only": r["a_only_correct"],
                    f"seed{k}_cnn_only": r["b_only_correct"], f"seed{k}_p": r["p_value"]})
    p = [r["p_value"] for r in tests]
    row.update({"p_min": min(p), "p_max": max(p),
                "n_significant": int(sum(v < ALPHA for v in p)),
                "diff_pp": 100 * diff, "ci_low_pp": 100 * lo, "ci_high_pp": 100 * hi})
    return row


def japan_only(preds: dict, block_size_km: float) -> pd.DataFrame:
    reference = pd.read_csv(LOGS / "context_control.csv")
    reference = reference[reference["comparison"] == "CNN vs raw XGBoost (t=0.5)"]
    rows, ok = [], True
    for split, pred in preds.items():
        y = (pred["label"] == "pv").values
        out = outside_japan(pred["lon"].values, pred["lat"].values)
        for filt, keep in (("none (gate G1)", np.ones(len(pred), bool)),
                           ("Japan only", ~out)):
            rows.append({"split": split, "filter": filt,
                         "n_removed": int((~keep).sum()),
                         "n_removed_pv": int((~keep & y).sum()),
                         **raw_vs_cnn(pred, keep, block_size_km)})
        ref = reference[reference["split"] == split].iloc[0]
        cols = ["baseline_accuracy", "cnn_accuracy_min", "cnn_accuracy_max",
                "p_min", "p_max", "diff_pp", "ci_low_pp", "ci_high_pp"]
        unfiltered = rows[-2]
        ok &= all(np.isclose(unfiltered[c], ref[c], rtol=0, atol=1e-9) for c in cols)
    gate("G1 (unfiltered = context_control.csv)", ok)
    return pd.DataFrame(rows)


# --- X-2 --------------------------------------------------------------------

def size_bin_mcnemar(preds: dict) -> pd.DataFrame:
    rows, ok = [], True
    for split, pred in preds.items():
        y = (pred["label"] == "pv").values
        raw = (pred["xgboost_baseline_proba"].values >= 0.5).astype(int)
        cnn = np.column_stack([pred[c].values >= 0.5 for c in seed_columns(pred)]).astype(int)
        majority_ok = (cnn == y[:, None]).mean(axis=1) >= 0.6  # at least 3 of 5 runs
        majority = np.where(majority_ok, y, ~y).astype(int)
        masks = capacity_bin_masks(pred, SIZE_EDGES_KW, DEFAULT_KW_PER_M2)
        reference = size_bin_hits(split)
        for b in SIZE_BINS:
            m = masks[b] & y
            vote = mcnemar_test(y[m], raw[m], majority[m])
            per_run = [mcnemar_test(y[m], raw[m], cnn[m, k])["p_value"]
                       for k in range(cnn.shape[1])]
            run_hits = [(cnn[m, k] == y[m]).sum() for k in range(cnn.shape[1])]
            rows.append({"split": split, "bin": b, "n": int(m.sum()),
                         "xgb_hits": int((raw[m] == y[m]).sum()),
                         "cnn_majority_hits": int(majority_ok[m].sum()),
                         "cnn_run_hits_min": int(min(run_hits)),
                         "cnn_run_hits_max": int(max(run_hits)),
                         "xgb_only": vote["a_only_correct"],
                         "cnn_majority_only": vote["b_only_correct"],
                         "p_majority": vote["p_value"],
                         "p_run_min": min(per_run), "p_run_max": max(per_run),
                         "n_runs_significant": int(sum(p < ALPHA for p in per_run))})
            ref = reference.loc[b]
            seeds = [c for c in reference.columns if c.startswith("cnn_proba_seed")]
            ok &= (m.sum() == masks[b].sum() == ref["n"]
                   and rows[-1]["xgb_hits"] == ref["xgboost_baseline"]
                   and min(run_hits) == ref[seeds].min()
                   and max(run_hits) == ref[seeds].max())
    gate("G2 (bins = tab_size_recall)", ok)
    return pd.DataFrame(rows)


# --- X-3 --------------------------------------------------------------------

def context_frames(model, cfg) -> dict:
    """As run_context_control.main: pooled probabilities and edge distances."""
    polygons = gpd.read_file(cfg.paths.pv_polygons)[["geometry"]].to_crs(AREA_CRS)
    frames = {}
    for split in ("val", "test"):
        features = pd.read_parquet(PROCESSED / f"{split}_features.parquet")
        patches = np.load(PROCESSED / f"{split}_patches.npy")
        pred = pd.read_parquet(LOGS / f"predictions_{split}.parquet")
        df = pred.merge(features[["point_id"]].assign(row=np.arange(len(features))),
                        on="point_id")
        df["context_proba"] = window_probabilities(model, patches[df["row"].values], WINDOW)
        df["edge_distance_m"] = edge_distance(df, polygons)
        frames[split] = df
        print(f"{split}: context probabilities and edge distances for {len(df)} points")
    return frames


def error_location_t050(frames: dict) -> pd.DataFrame:
    y_val = (frames["val"]["label"] == "pv").values
    threshold = tune_threshold(y_val, frames["val"]["context_proba"].values)
    print(f"Context-pooled threshold tuned on val: {threshold}")
    rows = []
    for split, df in frames.items():
        y = (df["label"] == "pv").values
        cnn = np.column_stack([df[c].values >= 0.5 for c in seed_columns(df)])
        cnn_correct = (cnn == y[:, None]).mean(axis=1)
        raw_correct = ((df["xgboost_baseline_proba"].values >= 0.5) == y).astype(float)
        tuned_correct = ((df["context_proba"].values >= threshold) == y).astype(float)
        t050_correct = ((df["context_proba"].values >= 0.5) == y).astype(float)
        d = df["edge_distance_m"].values
        for name, label, lo_m, hi_m in LOCATION_STRATA:
            m = (df["label"].values == label) & (d >= lo_m) & ((d < hi_m) | np.isinf(hi_m))
            rows.append({
                "split": split, "stratum": name, "n": int(m.sum()),
                "xgboost_accuracy": raw_correct[m].mean() if m.any() else np.nan,
                "pooled_accuracy": tuned_correct[m].mean() if m.any() else np.nan,
                "pooled_accuracy_t050": t050_correct[m].mean() if m.any() else np.nan,
                "cnn_accuracy": cnn_correct[m].mean() if m.any() else np.nan,
                "cnn_net_gain": (cnn_correct[m] - raw_correct[m]).sum(),
                "pooled_net_gain": (tuned_correct[m] - raw_correct[m]).sum(),
                "pooled_net_gain_t050": (t050_correct[m] - raw_correct[m]).sum(),
                "pooled_threshold_tuned": threshold})
    result = pd.DataFrame(rows)

    reference = pd.read_csv(LOGS / "error_location.csv")
    reference = reference[reference["stratum"].isin([s[0] for s in LOCATION_STRATA])]
    merged = reference.merge(result, on=["split", "stratum"], suffixes=("_ref", ""))
    cols = ["n", "xgboost_accuracy", "pooled_accuracy", "cnn_accuracy",
            "cnn_net_gain", "pooled_net_gain"]
    ok = len(merged) == len(result) == len(reference) and all(
        np.allclose(merged[f"{c}_ref"].astype(float), merged[c].astype(float),
                    rtol=0, atol=1e-12, equal_nan=True) for c in cols)
    gate("G3 (recomputed columns = error_location.csv)", ok)
    return result


# --- X-4 --------------------------------------------------------------------

def label_efficiency_paired() -> pd.DataFrame:
    runs = pd.read_csv(LOGS / "label_efficiency_runs.csv")
    wide = runs.pivot_table(index=["fraction", "repeat", "n_train"], columns="model",
                            values=list(METRICS)).reset_index()
    rows = []
    for (fraction, n_train), g in wide.groupby(["fraction", "n_train"]):
        for metric in METRICS:
            d = 100 * (g[(metric, "cnn")] - g[(metric, "xgboost_baseline")]).values
            n = len(d)
            mean, sd = d.mean(), d.std(ddof=1)
            half = t_dist.ppf(0.975, n - 1) * sd / np.sqrt(n)
            positive, nonzero = int((d > 0).sum()), int((d != 0).sum())
            rows.append({"fraction": fraction, "n_train": int(n_train), "metric": metric,
                         "n_repeats": n,
                         **{f"diff_pp_repeat{k}": v for k, v in enumerate(d)},
                         "mean_pp": mean, "sd_pp": sd,
                         "ci_low_pp": mean - half, "ci_high_pp": mean + half,
                         "paired_t_p": ttest_1samp(d, 0.0).pvalue,
                         "n_positive": positive,
                         "sign_test_p": binomtest(positive, nonzero, 0.5).pvalue
                         if nonzero else 1.0})
    return pd.DataFrame(rows)


# --- X-5 --------------------------------------------------------------------

def size_edge(frames: dict) -> pd.DataFrame:
    rows = []
    for split, df in frames.items():
        y = (df["label"] == "pv").values
        cnn = np.column_stack([df[c].values >= 0.5 for c in seed_columns(df)])
        majority_ok = (cnn == y[:, None]).mean(axis=1) >= 0.6
        raw_ok = (df["xgboost_baseline_proba"].values >= 0.5) == y
        near = df["edge_distance_m"].values < EDGE_M
        masks = capacity_bin_masks(df, SIZE_EDGES_KW, DEFAULT_KW_PER_M2)
        for b in SIZE_BINS:
            m = masks[b] & y
            a, z = m & near, m & ~near
            row = {"split": split, "bin": b, "n": int(m.sum()),
                   "n_edge_lt10m": int(a.sum()), "share_edge_lt10m": a.sum() / m.sum()}
            if a.sum() >= MIN_SIDE and z.sum() >= MIN_SIDE:
                row.update({"xgb_recall_lt10m": raw_ok[a].mean(),
                            "cnn_majority_recall_lt10m": majority_ok[a].mean(),
                            "xgb_recall_ge10m": raw_ok[z].mean(),
                            "cnn_majority_recall_ge10m": majority_ok[z].mean()})
            rows.append(row)
    return pd.DataFrame(rows)


def main():
    cfg = load_config()
    train = pd.read_parquet(PROCESSED / "train_features.parquet")
    model = train_baseline(train, cfg.xgboost)
    preds = {s: pd.read_parquet(LOGS / f"predictions_{s}.parquet") for s in ("val", "test")}
    refit_ok = check_refit(model, preds)

    japan = japan_only(preds, cfg.split.block_size_km)
    bins = size_bin_mcnemar(preds)
    paired = label_efficiency_paired()
    frames = context_frames(model, cfg)
    location = error_location_t050(frames) if refit_ok else None
    edges = size_edge(frames)

    with pd.option_context("display.width", 240, "display.max_columns", 40):
        print(japan.round(4).T.to_string())
        print(bins.round(4).to_string(index=False))
        if location is not None:
            print(location.round(3).to_string(index=False))
        print(paired.drop(columns=[c for c in paired if c.startswith("diff_pp_repeat")])
              .round(4).to_string(index=False))
        print(edges.round(3).to_string(index=False))

    write_new(japan, "revision_japan_only.csv")
    write_new(bins, "revision_size_bin_mcnemar.csv")
    if location is not None:
        write_new(location, "revision_error_location_t050.csv")
    write_new(paired, "revision_label_efficiency_paired.csv")
    write_new(edges, "revision_size_edge.csv")


if __name__ == "__main__":
    main()
