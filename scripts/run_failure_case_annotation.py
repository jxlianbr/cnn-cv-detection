"""Failure-case analysis against the author-verified visual annotation.

Reads the 240 PV-labelled evaluation points of the annotation worksheet
(data/external/annotation_worksheet_v2.csv, 40 per capacity bin, validation and
test splits) as verified by the author on GSI aerial photos
(reports/review/annotation/verified_labels.csv), joins the saved predictions of
the runs behind the paper (results/logs/predictions_{val,test}.parquet) and
scores both models per stratum:

  label_recall          share of points classified PV (agreement with the
                        released label, which is PV for every point here)
  recall_verified       share classified PV among points with PV visible at
                        the point (pv_present = yes)
  tnr_on_no             share classified non-PV among points with no PV
                        visible at the point (pv_present = no)
  accuracy_verified     share classified correctly among the yes and no points
                        (unclear points excluded)

for XGBoost (single run), the CNN majority (at least 3 of the 5 runs at 0.5)
and the five CNN runs (mean, min, max). Proportions carry Wilson score 95%
intervals; XGBoost and the CNN majority (and each CNN run) are compared on the
same points with an exact McNemar test. The annotation itself is summarised
per stratum as model "reference"; the last rows reweight these shares from the
equal-per-bin sample to the bin sizes of the evaluation splits (for weighted
rows, k is the number of PV evaluation points and m the sample size).

Only rows with verified = true are used; the script stops if any row is not
verified, so unverified AI pre-labels can never enter the results. It also
writes the per-field change rate of the pre-labels.

Validation and test CNN runs with the same seed are separate trainings; "run k"
pools seed k of both splits. Several points can lie in the same polygon, so the
points of a stratum are not independent; n_polygons gives the number of
distinct polygons.

    python scripts/run_failure_case_annotation.py
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root on sys.path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, norm

from src.evaluation.failure_cases import AREA_CRS, DEFAULT_KW_PER_M2

LOGS = Path("results/logs")
ANNOTATION = Path("reports/review/annotation")
SIZE_BINS = ["0-100 kW", "100-250 kW", "250-500 kW", "500-1000 kW", "1000-2500 kW",
             ">2500 kW"]
FIELDS = ["pv_present", "outline_match", "polygon_fragment", "failure_case", "confidence"]
N_RUNS = 5


def wilson_interval(k: int, n: int, level: float = 0.95) -> tuple[float, float]:
    """Wilson score interval; (nan, nan) for an empty stratum."""
    if n == 0:
        return float("nan"), float("nan")
    z = norm.ppf(0.5 + level / 2)
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return float(centre - half), float(centre + half)


def exact_mcnemar(correct_a: np.ndarray, correct_b: np.ndarray) -> tuple[int, int, float]:
    """(a only correct, b only correct, two-sided exact p)."""
    a_only = int(np.sum(correct_a & ~correct_b))
    b_only = int(np.sum(~correct_a & correct_b))
    n = a_only + b_only
    p = 1.0 if n == 0 else float(binomtest(min(a_only, b_only), n, 0.5).pvalue)
    return a_only, b_only, p


def load_verified(path: Path) -> pd.DataFrame:
    labels = pd.read_csv(path, encoding="utf-8-sig", keep_default_na=False)
    verified = labels["verified"].astype(str).str.lower() == "true"
    if not verified.all():
        raise SystemExit(f"{(~verified).sum()} rows of {path} are not verified; "
                         f"unverified pre-labels are not data. Stopping.")
    return labels


def prelabel_changes(prelabels: pd.DataFrame, verified: pd.DataFrame) -> pd.DataFrame:
    """Per-field count of pre-labels the author changed, with the transitions."""
    merged = prelabels.merge(verified, on="point_id", suffixes=("_pre", "_ver"))
    rows = [{"field": "any", "n": len(merged),
             "n_changed": int((verified["changed_by_author"].astype(str).str.lower()
                               == "true").sum()), "transitions": ""}]
    for field in FIELDS:
        before = merged[f"{field}_pre"].astype(str)
        after = merged[f"{field}_ver"].astype(str)
        changed = before != after
        pairs = (before[changed] + "->" + after[changed]).value_counts()
        rows.append({"field": field, "n": len(merged), "n_changed": int(changed.sum()),
                     "transitions": "; ".join(f"{k} {v}" for k, v in pairs.items())})
    out = pd.DataFrame(rows)
    out["share_changed"] = out["n_changed"] / out["n"]
    return out


def attach_predictions(labels: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for split in ("val", "test"):
        pred = pd.read_parquet(LOGS / f"predictions_{split}.parquet")
        pred["split"] = split
        frames.append(pred)
    pred = pd.concat(frames, ignore_index=True)
    run_cols = [f"cnn_proba_seed{k}" for k in range(N_RUNS)]
    df = labels.merge(pred[["point_id", "split", "label", "xgboost_baseline_proba"]
                           + run_cols], on=["point_id", "split"], how="left",
                      validate="one_to_one")
    if df["label"].isna().any() or (df["label"] != "pv").any():
        raise SystemExit("Some annotated points are missing from the predictions or "
                         "not labelled PV there.")
    df["xgb_pv"] = df["xgboost_baseline_proba"] >= 0.5
    for k in range(N_RUNS):
        df[f"cnn{k}_pv"] = df[f"cnn_proba_seed{k}"] >= 0.5
    df["cnn_votes"] = df[[f"cnn{k}_pv" for k in range(N_RUNS)]].sum(axis=1)
    df["cnn_maj_pv"] = df["cnn_votes"] >= 3
    # consistency with the chip metadata the annotator saw
    if not ((df["xgb_pv"] == (df["xgb_pred"] == "PV")).all()
            and (df["cnn_votes"] == df["cnn_votes_pv"].astype(int)).all()):
        raise SystemExit("Predictions disagree with the chip metadata.")
    return df


def attach_polygon_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Id of the smallest released polygon containing each point, as in the
    analysis (src.evaluation.failure_cases.attach_pv_polygon_area)."""
    import geopandas as gpd
    points = gpd.GeoDataFrame(df[["point_id", "split"]].copy(),
                              geometry=gpd.points_from_xy(df["lon"], df["lat"]),
                              crs="EPSG:4326")
    minx, miny, maxx, maxy = points.total_bounds
    polys = gpd.read_file("data/external/pv_polygons.gpkg",
                          bbox=(minx - 0.01, miny - 0.01, maxx + 0.01, maxy + 0.01))
    polys = polys[["geometry"]].reset_index(names="polygon_id")
    polys["area_m2"] = polys.to_crs(AREA_CRS).geometry.area.values
    joined = gpd.sjoin(points, polys, predicate="within", how="left")
    joined = joined.sort_values("area_m2").groupby(level=0).first()
    out = df.copy()
    out["polygon_id"] = joined["polygon_id"].reindex(df.index).values
    area_check = joined["area_m2"].reindex(df.index).values
    if np.nanmax(np.abs(area_check - df["polygon_area_m2"].astype(float))) > 1.0:
        raise SystemExit("Polygon areas do not match the worksheet.")
    return out


def define_strata(df: pd.DataFrame) -> list[tuple[str, str, np.ndarray]]:
    edge = df["edge_distance_m"].astype(float)
    no = df["pv_present"] == "no"
    strata = [("all", "all", np.ones(len(df), bool)),
              ("split", "val", (df["split"] == "val").to_numpy()),
              ("split", "test", (df["split"] == "test").to_numpy()),
              ("size", "sub_50kw", (df["sub_50kw"] == "yes").to_numpy())]
    strata += [("size", b, (df["size_bin"] == b).to_numpy()) for b in SIZE_BINS]
    strata += [("size", "below_1MW", df["size_bin"].isin(SIZE_BINS[:4]).to_numpy()),
               ("size", "above_1MW", df["size_bin"].isin(SIZE_BINS[4:]).to_numpy())]
    strata += [("failure_case", v, (df["failure_case"] == v).to_numpy())
               for v in ("none", "spectrally_ambiguous", "non_crystalline_silicon",
                         "both", "unclear")]
    strata += [("pv_present", v, (df["pv_present"] == v).to_numpy())
               for v in ("yes", "no", "unclear")]
    # outline n/a: no array visible anywhere in the polygon, so the photo may
    # predate construction; otherwise the outline was judged and the point lies
    # on a visibly non-PV surface (road, roof, water, grass beside an array)
    strata += [("pv_present_no", "no_no_array_visible",
                (no & (df["outline_match"] == "n/a")).to_numpy()),
               ("pv_present_no", "no_outline_judged",
                (no & (df["outline_match"] != "n/a")).to_numpy()),
               ("date_robust", "yes_or_no_outline_judged",
                ((df["pv_present"] == "yes")
                 | (no & (df["outline_match"] != "n/a"))).to_numpy())]
    strata += [("outline_match", v, (df["outline_match"] == v).to_numpy())
               for v in ("good", "partial", "poor", "n/a")]
    strata += [("polygon_fragment", v, (df["polygon_fragment"] == v).to_numpy())
               for v in ("yes", "no", "unclear")]
    strata += [("edge", "edge_lt10m", (edge < 10).to_numpy()),
               ("edge", "edge_ge10m", (edge >= 10).to_numpy())]
    strata += [("size_x_failure", "sub_50kw_spectrally_ambiguous",
                ((df["sub_50kw"] == "yes")
                 & (df["failure_case"] == "spectrally_ambiguous")).to_numpy()),
               ("size_x_failure", "below_1MW_spectrally_ambiguous",
                (df["size_bin"].isin(SIZE_BINS[:4])
                 & (df["failure_case"] == "spectrally_ambiguous")).to_numpy())]
    return strata


def _proportion(k, n):
    lo, hi = wilson_interval(int(k), int(n))
    return {"k": int(k), "m": int(n), "value": k / n if n else float("nan"),
            "ci_low": lo, "ci_high": hi}


def reference_rows(sub: pd.DataFrame) -> list[dict]:
    n = len(sub)
    decided = sub["pv_present"].isin(["yes", "no"])
    judged = sub["outline_match"] != "n/a"
    rows = [("share_pv_yes", (sub["pv_present"] == "yes").sum(), n),
            ("share_pv_no", (sub["pv_present"] == "no").sum(), n),
            ("share_pv_unclear", (sub["pv_present"] == "unclear").sum(), n),
            ("share_pv_yes_of_decided", (sub["pv_present"] == "yes").sum(), decided.sum()),
            ("share_outline_good_of_judged", (sub["outline_match"] == "good").sum(),
             judged.sum()),
            ("share_outline_partial_or_poor_of_judged",
             sub["outline_match"].isin(["partial", "poor"]).sum(), judged.sum()),
            ("share_fragment_yes", (sub["polygon_fragment"] == "yes").sum(), n),
            ("share_pv_yes_fragment", ((sub["pv_present"] == "yes")
                                       & (sub["polygon_fragment"] == "yes")).sum(), n),
            ("share_pv_yes_whole_installation",
             ((sub["pv_present"] == "yes") & (sub["polygon_fragment"] == "no")).sum(), n),
            ("share_pv_no_outline_judged", ((sub["pv_present"] == "no")
                                            & (sub["outline_match"] != "n/a")).sum(), n),
            ("share_pv_no_no_array_visible", ((sub["pv_present"] == "no")
                                              & (sub["outline_match"] == "n/a")).sum(), n),
            ("share_spectrally_ambiguous",
             (sub["failure_case"] == "spectrally_ambiguous").sum(), n),
            ("share_non_crystalline_silicon",
             sub["failure_case"].isin(["non_crystalline_silicon", "both"]).sum(), n)]
    return [{"model": "reference", "metric": m, **_proportion(k, d)} for m, k, d in rows]


def model_rows(sub: pd.DataFrame) -> list[dict]:
    yes = (sub["pv_present"] == "yes").to_numpy()
    no = (sub["pv_present"] == "no").to_numpy()
    metrics = {"label_recall": np.ones(len(sub), bool), "recall_verified": yes,
               "tnr_on_no": no, "accuracy_verified": yes | no}

    def correct(pv: np.ndarray, metric: str) -> np.ndarray:
        # PV is correct for the released label and for a verified yes,
        # non-PV for a verified no
        return pv if metric in ("label_recall", "recall_verified") else np.where(yes, pv, ~pv)

    xgb = sub["xgb_pv"].to_numpy()
    maj = sub["cnn_maj_pv"].to_numpy()
    runs = [sub[f"cnn{k}_pv"].to_numpy() for k in range(N_RUNS)]
    rows = []
    for metric, mask in metrics.items():
        c_xgb = correct(xgb, metric)[mask]
        c_maj = correct(maj, metric)[mask]
        c_runs = [correct(r, metric)[mask] for r in runs]
        m = int(mask.sum())
        rows.append({"model": "xgboost", "metric": metric, **_proportion(c_xgb.sum(), m)})
        xo, co, p = exact_mcnemar(c_xgb, c_maj)
        rows.append({"model": "cnn_majority", "metric": metric,
                     **_proportion(c_maj.sum(), m), "mcnemar_xgb_only": xo,
                     "mcnemar_cnn_only": co, "mcnemar_p": p})
        tests = [exact_mcnemar(c_xgb, c) for c in c_runs]
        values = [c.mean() if m else float("nan") for c in c_runs]
        rows.append({"model": "cnn_runs", "metric": metric, "m": m,
                     "value": float(np.mean(values)) if m else float("nan"),
                     "run_min": float(np.min(values)) if m else float("nan"),
                     "run_max": float(np.max(values)) if m else float("nan"),
                     "mcnemar_p_min": min(t[2] for t in tests),
                     "mcnemar_p_max": max(t[2] for t in tests),
                     "runs_cnn_ahead": sum(t[1] > t[0] for t in tests),
                     "runs_xgb_ahead": sum(t[0] > t[1] for t in tests)})
    return rows


def eval_split_bin_counts(kw_per_m2: float = DEFAULT_KW_PER_M2) -> pd.Series:
    """PV points per capacity bin in the validation and test splits together,
    the population the worksheet was drawn from (40 points per bin)."""
    pred = pd.concat(pd.read_parquet(LOGS / f"predictions_{s}.parquet")
                     for s in ("val", "test"))
    kw = pred.loc[pred["label"] == "pv", "polygon_area_m2"] * kw_per_m2
    bins = pd.cut(kw, [0, 100, 250, 500, 1000, 2500, np.inf], right=False,
                  labels=SIZE_BINS)
    return bins.value_counts().reindex(SIZE_BINS)


def weighted_reference_rows(df: pd.DataFrame, counts: pd.Series) -> list[dict]:
    """Reference shares reweighted from the equal-per-bin sample to the bin
    sizes of the evaluation splits. The interval is a normal approximation
    for a stratified sample (no finite-population correction; points sharing
    a polygon are treated as independent, so it is too narrow)."""
    weights = counts / counts.sum()
    rows = []
    per_bin = {b: pd.DataFrame(reference_rows(df[df["size_bin"] == b]))
               .set_index("metric") for b in SIZE_BINS}
    for metric in per_bin[SIZE_BINS[0]].index:
        if metric.endswith("_of_decided") or metric.endswith("_of_judged"):
            continue
        p = np.array([per_bin[b].loc[metric, "value"] for b in SIZE_BINS])
        n = np.array([per_bin[b].loc[metric, "m"] for b in SIZE_BINS], dtype=float)
        w = weights.to_numpy(dtype=float)
        value = float(np.sum(w * p))
        se = float(np.sqrt(np.sum(w**2 * p * (1 - p) / n)))
        rows.append({"model": "reference", "metric": metric, "value": value,
                     "ci_low": max(0.0, value - 1.96 * se),
                     "ci_high": min(1.0, value + 1.96 * se),
                     "m": int(n.sum()), "k": int(counts.sum())})
    return rows


def evaluate(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for group, name, mask in define_strata(df):
        sub = df[mask]
        base = {"stratum_group": group, "stratum": name, "n": len(sub),
                "n_polygons": int(sub["polygon_id"].nunique()) if len(sub) else 0,
                "n_yes": int((sub["pv_present"] == "yes").sum()),
                "n_no": int((sub["pv_present"] == "no").sum()),
                "n_unclear": int((sub["pv_present"] == "unclear").sum())}
        if len(sub) == 0:
            out.append({**base, "model": "none", "metric": "empty_stratum"})
            continue
        for row in reference_rows(sub) + model_rows(sub):
            out.append({**base, **row})
    counts = eval_split_bin_counts()
    base = {"stratum_group": "weighted_to_eval_splits", "stratum": "all_bin_weighted",
            "n": len(df), "n_polygons": int(df["polygon_id"].nunique()),
            "n_yes": int((df["pv_present"] == "yes").sum()),
            "n_no": int((df["pv_present"] == "no").sum()),
            "n_unclear": int((df["pv_present"] == "unclear").sum())}
    out += [{**base, **row} for row in weighted_reference_rows(df, counts)]
    cols = ["stratum_group", "stratum", "n", "n_polygons", "n_yes", "n_no", "n_unclear",
            "model", "metric", "k", "m", "value", "ci_low", "ci_high", "run_min",
            "run_max", "mcnemar_xgb_only", "mcnemar_cnn_only", "mcnemar_p",
            "mcnemar_p_min", "mcnemar_p_max", "runs_cnn_ahead", "runs_xgb_ahead"]
    return pd.DataFrame(out).reindex(columns=cols)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", default=str(ANNOTATION / "verified_labels.csv"))
    parser.add_argument("--prelabels", default=str(ANNOTATION / "prelabels.csv"))
    parser.add_argument("--out", default=str(LOGS / "annotation_failure_cases.csv"))
    parser.add_argument("--changes-out",
                        default=str(LOGS / "annotation_prelabel_changes.csv"))
    args = parser.parse_args()
    outputs = [Path(args.out), Path(args.changes_out)]
    existing = [str(p) for p in outputs if p.exists()]
    if existing:
        raise SystemExit(f"Refusing to overwrite {', '.join(existing)}")

    labels = load_verified(Path(args.labels))
    prelabels = pd.read_csv(args.prelabels, keep_default_na=False)
    changes = prelabel_changes(prelabels, labels)
    df = attach_polygon_ids(attach_predictions(labels))
    result = evaluate(df)

    LOGS.mkdir(parents=True, exist_ok=True)
    result.to_csv(outputs[0], index=False)
    changes.to_csv(outputs[1], index=False)
    with pd.option_context("display.width", 220, "display.max_columns", 30,
                           "display.max_colwidth", 90):
        print(changes.to_string(index=False))
        print(result[result["model"].isin(["reference"])
                     & (result["metric"] == "share_pv_yes")]
              [["stratum", "n", "n_polygons", "n_yes", "n_no", "n_unclear"]]
              .to_string(index=False))
    print(f"Wrote {outputs[0]} and {outputs[1]}")


if __name__ == "__main__":
    main()
