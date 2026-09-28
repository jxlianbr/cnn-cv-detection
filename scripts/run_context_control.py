"""Post-hoc control: is the CNN's gain spatial context, or the CNN?

Two analyses on the predictions saved by run_failure_case_evaluation.py:

1. Context-pooled baseline. The published XGBoost is applied to every pixel of
   the 3x3 window around each point (the centre of its 21x21 patch) and its
   PV probabilities are averaged. This is the single pre-declared variant; its
   threshold is tuned on the validation split and then applied to the test
   split once. It is compared with each CNN seed (exact McNemar) and with the
   seed-averaged CNN (10 km block bootstrap of the accuracy difference).

2. Error location. Each point gets a distance to the boundary of the nearby
   released PV polygons (EPSG:6933): inside distance for PV points, outside
   distance for non-PV points. The net gain over raw XGBoost is then split by
   location, which shows where each model's extra correct points come from.

Writes results/logs/context_control.csv and results/logs/error_location.csv.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root on sys.path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.ops import unary_union

from src.config import load_config
from src.data.indices import mndwi, ndvi, pvsi
from src.data.split import _block_id
from src.evaluation.metrics import mcnemar_test
from src.models.xgboost_baseline import FEATURE_COLUMNS, train_baseline

LOGS = Path("results/logs")
PROCESSED = Path("data/processed")
AREA_CRS = "EPSG:6933"
WINDOW = 3
N_BOOTSTRAP = 2000
LOCATION_STRATA = [  # (name, label, lower, upper) on the edge distance in metres
    ("PV < 10 m inside edge", "pv", 0, 10),
    ("PV >= 10 m inside edge", "pv", 10, np.inf),
    ("non-PV <= 20 m outside edge", "non_pv", 0, 20),
    ("non-PV 20-50 m outside edge", "non_pv", 20, 50),
    ("non-PV > 50 m from any PV", "non_pv", 50, np.inf),
]


def window_probabilities(model, patches: np.ndarray, window: int) -> np.ndarray:
    """Mean XGBoost PV probability over the central window of each patch.
    Zero-padded pixels (outside the downloaded tile) are left out."""
    half, c = window // 2, patches.shape[2] // 2
    w = patches[:, :, c - half:c + half + 1, c - half:c + half + 1]  # (N, 6, k, k)
    bands = np.moveaxis(w, 1, -1).reshape(-1, 6)  # B2, B3, B4, B8, B11, B12
    b2, b3, b4, b8, b11, b12 = bands.T
    X = np.column_stack([b2, b3, b4, b8, b11, b12, ndvi(b8, b4), mndwi(b3, b11),
                         pvsi(b11, b12, b8, b2, b4)])
    proba = model.predict_proba(pd.DataFrame(X, columns=FEATURE_COLUMNS).values)[:, 1]
    proba = proba.reshape(len(patches), -1)
    padded = np.all(bands == 0, axis=1).reshape(len(patches), -1)
    proba[padded] = np.nan
    return np.nanmean(proba, axis=1)


def tune_threshold(y: np.ndarray, proba: np.ndarray) -> float:
    grid = np.round(np.arange(0.05, 0.96, 0.01), 2)
    acc = np.array([np.mean((proba >= t) == y) for t in grid])
    best = grid[acc == acc.max()]
    return float(best[np.argmin(np.abs(best - 0.5))])  # ties: closest to 0.5


def block_bootstrap(correct_a: np.ndarray, correct_b: np.ndarray, blocks: np.ndarray,
                    seed: int = 0) -> tuple[float, float, float]:
    """Accuracy difference a - b with a 95% percentile interval, resampling
    whole 10 km blocks (points in one block are not independent)."""
    rng = np.random.default_rng(seed)
    ids, inverse = np.unique(blocks, return_inverse=True)
    diff = np.bincount(inverse, correct_a - correct_b)
    count = np.bincount(inverse)
    draws = rng.integers(0, len(ids), size=(N_BOOTSTRAP, len(ids)))
    boot = diff[draws].sum(axis=1) / count[draws].sum(axis=1)
    point = (correct_a - correct_b).mean()
    return point, *np.percentile(boot, [2.5, 97.5])


def edge_distance(df: pd.DataFrame, polygons: gpd.GeoDataFrame) -> np.ndarray:
    """Metres from each point to the boundary of the PV polygons within 3 km
    (dissolved): inside distance if the point is in PV, else outside distance."""
    points = gpd.GeoSeries(gpd.points_from_xy(df["lon"], df["lat"]), crs=4326).to_crs(AREA_CRS)
    index = polygons.sindex
    out = np.full(len(df), np.inf)
    for i, pt in enumerate(points):
        near = index.query(pt.buffer(3000))
        if len(near) == 0:
            continue
        pv = unary_union(polygons.geometry.values[near])
        out[i] = pt.distance(pv.boundary) if pv.contains(pt) else pt.distance(pv)
    return out


def main():
    cfg = load_config()
    train = pd.read_parquet(PROCESSED / "train_features.parquet")
    model = train_baseline(train, cfg.xgboost)
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
        df["block"] = [f"{r}_{c}" for r, c in (_block_id(lo, la, cfg.split.block_size_km)
                                               for lo, la in zip(df["lon"], df["lat"]))]
        frames[split] = df
        print(f"{split}: context probabilities and edge distances for {len(df)} points")

    y_val = (frames["val"]["label"] == "pv").values
    threshold = tune_threshold(y_val, frames["val"]["context_proba"].values)
    print(f"Context-pooled threshold tuned on val: {threshold}")

    rows, location_rows = [], []
    for split, df in frames.items():
        y = (df["label"] == "pv").values
        seeds = sorted(c for c in df.columns if c.startswith("cnn_proba_seed"))
        raw = (df["xgboost_baseline_proba"].values >= 0.5)
        pooled = {t: df["context_proba"].values >= t for t in (threshold, 0.5)}
        cnn = np.column_stack([df[c].values >= 0.5 for c in seeds])
        cnn_correct = (cnn == y[:, None]).mean(axis=1)

        for label, t in (("tuned", threshold), ("0.5", 0.5)):
            if split == "val" and label == "tuned":
                continue  # the threshold was chosen on val: in-sample
            p = pooled[t]
            tests = [mcnemar_test(y, p.astype(int), cnn[:, k].astype(int)) for k in range(len(seeds))]
            diff, lo, hi = block_bootstrap(cnn_correct, (p == y).astype(float), df["block"].values)
            rows.append({"split": split, "comparison": f"CNN vs 3x3 pooled XGBoost (t={t})",
                         "baseline_accuracy": np.mean(p == y),
                         "cnn_accuracy_min": (cnn == y[:, None]).mean(axis=0).min(),
                         "cnn_accuracy_max": (cnn == y[:, None]).mean(axis=0).max(),
                         "p_min": min(r["p_value"] for r in tests),
                         "p_max": max(r["p_value"] for r in tests),
                         "diff_pp": 100 * diff, "ci_low_pp": 100 * lo, "ci_high_pp": 100 * hi})
        tests = [mcnemar_test(y, raw.astype(int), cnn[:, k].astype(int)) for k in range(len(seeds))]
        diff, lo, hi = block_bootstrap(cnn_correct, (raw == y).astype(float), df["block"].values)
        rows.append({"split": split, "comparison": "CNN vs raw XGBoost (t=0.5)",
                     "baseline_accuracy": np.mean(raw == y),
                     "cnn_accuracy_min": (cnn == y[:, None]).mean(axis=0).min(),
                     "cnn_accuracy_max": (cnn == y[:, None]).mean(axis=0).max(),
                     "p_min": min(r["p_value"] for r in tests),
                     "p_max": max(r["p_value"] for r in tests),
                     "diff_pp": 100 * diff, "ci_low_pp": 100 * lo, "ci_high_pp": 100 * hi})

        d = df["edge_distance_m"].values
        raw_correct = (raw == y).astype(float)
        pooled_correct = (pooled[threshold] == y).astype(float)
        cnn_only = (cnn_correct >= 0.6) & (raw_correct == 0) & y  # majority of seeds
        for name, label, lo_m, hi_m in LOCATION_STRATA:
            # np.inf marks "no PV within 3 km", which belongs to the open-ended stratum
            m = (df["label"].values == label) & (d >= lo_m) & ((d < hi_m) | np.isinf(hi_m))
            location_rows.append({
                "split": split, "stratum": name, "n": int(m.sum()),
                "xgboost_accuracy": raw_correct[m].mean() if m.any() else np.nan,
                "pooled_accuracy": pooled_correct[m].mean() if m.any() else np.nan,
                "cnn_accuracy": cnn_correct[m].mean() if m.any() else np.nan,
                "cnn_net_gain": (cnn_correct[m] - raw_correct[m]).sum(),
                "pooled_net_gain": (pooled_correct[m] - raw_correct[m]).sum(),
            })
        location_rows.append({"split": split, "stratum": "CNN-only PV hits < 10 m from edge",
                              "n": int(cnn_only.sum()),
                              "share": np.mean(d[cnn_only] < 10) if cnn_only.any() else np.nan,
                              "base_share": np.mean(d[y] < 10)})

    extras = []
    both = pd.concat(frames.values(), ignore_index=True)
    for split, df in list(frames.items()) + [("val+test", both)]:
        features = pd.concat(pd.read_parquet(PROCESSED / f"{s}_features.parquet")
                             for s in ("val", "test"))[["point_id", "NDVI", "PVSI", "MNDWI",
                                                        "valid_candidate"]]
        df = df.merge(features, on="point_id", how="left")
        y = (df["label"] == "pv").values
        seeds = sorted(c for c in df.columns if c.startswith("cnn_proba_seed"))
        cnn = np.column_stack([df[c].values >= 0.5 for c in seeds])
        cnn_majority = (cnn == y[:, None]).mean(axis=1) >= 0.6
        raw_ok = (df["xgboost_baseline_proba"].values >= 0.5) == y
        pooled_ok = (df["context_proba"].values >= threshold) == y
        d = df["edge_distance_m"].values
        cnn_only_pv = cnn_majority & ~raw_ok & y
        ring = ~y & (d <= 20)
        ring_test = mcnemar_test(y[ring], (~raw_ok[ring] ^ y[ring]).astype(int),
                                 (~cnn_majority[ring] ^ y[ring]).astype(int))
        far_fa = ~y & (d > 50) & ~raw_ok
        # robustness of CNN vs raw XGBoost on subsets: dry land, and the points
        # the original pre-masking rules would have passed to the classifier
        raw_pred = (df["xgboost_baseline_proba"].values >= 0.5).astype(int)
        subsets = {"dry": (df["MNDWI"] <= 0).values,
                   "premask": df["valid_candidate"].astype(bool).values}
        robust = {}
        for name, m in subsets.items():
            ps = [mcnemar_test(y[m], raw_pred[m], cnn[m, k].astype(int))["p_value"]
                  for k in range(cnn.shape[1])]
            robust.update({f"{name}_n": int(m.sum()),
                           f"{name}_xgb_accuracy": raw_ok[m].mean(),
                           f"{name}_cnn_accuracy_min": (cnn[m] == y[m, None]).mean(axis=0).min(),
                           f"{name}_cnn_accuracy_max": (cnn[m] == y[m, None]).mean(axis=0).max(),
                           f"{name}_p_max": max(ps)})
        extras.append({**robust,
            "split": split,
            "cnn_only_pv_median_ndvi": df.loc[cnn_only_pv, "NDVI"].median(),
            "cnn_only_pv_median_pvsi": df.loc[cnn_only_pv, "PVSI"].median(),
            "all_pv_median_ndvi": df.loc[y, "NDVI"].median(),
            "all_pv_median_pvsi": df.loc[y, "PVSI"].median(),
            "pv_within_20m_inside": int((y & (d < 20)).sum()),
            "nonpv_within_20m_outside": int(ring.sum()),
            "ring_xgb_only_correct": ring_test["a_only_correct"],
            "ring_cnn_only_correct": ring_test["b_only_correct"],
            "ring_p_value": ring_test["p_value"],
            "far_false_alarms_xgb": int(far_fa.sum()),
            "far_fixed_by_cnn": int((far_fa & cnn_majority).sum()),
            "far_fixed_by_both": int((far_fa & cnn_majority & pooled_ok).sum()),
        })

    context = pd.DataFrame(rows).assign(threshold=threshold)
    pd.DataFrame(extras).to_csv(LOGS / "context_extras.csv", index=False)
    print(pd.DataFrame(extras).round(3).T.to_string())
    location = pd.DataFrame(location_rows)
    context.to_csv(LOGS / "context_control.csv", index=False)
    location.to_csv(LOGS / "error_location.csv", index=False)
    with pd.option_context("display.width", 220, "display.max_columns", 20):
        print(context.round(4).to_string(index=False))
        print()
        print(location.round(3).to_string(index=False))
    print(f"\nWrote {LOGS}/context_control.csv and error_location.csv")


if __name__ == "__main__":
    main()
