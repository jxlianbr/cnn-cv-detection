"""Operational definitions for the three failure-case subsets named in
Shimada & Takeuchi (2025) and reports/paper-draft.md §3.5. See this
task's design note in plan.md for why two of the three require manual
annotation rather than an automated proxy."""
import numpy as np
import pandas as pd
import geopandas as gpd

from src.evaluation.metrics import compute_metrics

DEFAULT_KW_PER_M2 = 0.086  # 0.86 MW/ha, the linear area-to-capacity fit Shimada &
                           # Takeuchi (2025) derived from 50 Japanese solar farms and
                           # used on this same polygon release (50 kW ~ 600 m^2 there)

AREA_CRS = "EPSG:6933"  # equal-area; Japan spans several UTM zones, so a single
                        # UTM projection would bias areas at the edges


def attach_pv_polygon_area(points_gdf: gpd.GeoDataFrame, pv_gdf: gpd.GeoDataFrame,
                           area_crs: str = AREA_CRS) -> gpd.GeoDataFrame:
    """Add `polygon_area_m2`: the area of the PV polygon containing each point.

    NaN for points that sit in no polygon (the non-PV class). This is what
    turns a sample point into something `flag_sub_50kw` can size, since the
    released dataset carries footprints but no capacity figures.
    """
    polygons = pv_gdf[["geometry"]].copy()
    polygons["polygon_area_m2"] = polygons.to_crs(area_crs).geometry.area.values

    joined = gpd.sjoin(points_gdf, polygons, predicate="within", how="left")
    # A point inside overlapping footprints matches more than once. Keep the
    # smallest, which is the individual installation rather than any enclosing
    # block; groupby collapses the duplicates without depending on row order.
    smallest = joined.groupby(level=0)["polygon_area_m2"].min()

    out = points_gdf.copy()
    out["polygon_area_m2"] = smallest.reindex(points_gdf.index).values
    return out


def capacity_bin_masks(eval_df: pd.DataFrame, edges_kw: list[float],
                       kw_per_m2: float = DEFAULT_KW_PER_M2) -> dict[str, np.ndarray]:
    """Boolean masks for consecutive capacity intervals [edge_i, edge_i+1).

    Lets detection rate be plotted against installation size, which is the
    empirical form of the question "is small-PV detection a method problem or a
    resolution problem?". Points with no footprint belong to no bin.
    """
    area = eval_df["polygon_area_m2"].to_numpy(dtype=float)
    sized = ~np.isnan(area)
    capacity_kw = np.full(len(eval_df), np.nan)
    capacity_kw[sized] = estimate_capacity_kw(area[sized], kw_per_m2)

    masks = {}
    for low, high in zip(edges_kw[:-1], edges_kw[1:]):
        name = f"{low:g}-{high:g}kW" if np.isfinite(high) else f">{low:g}kW"
        masks[name] = sized & (capacity_kw >= low) & (capacity_kw < high)
    return masks


def evaluate_on_subsets(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray,
                        subsets: dict[str, np.ndarray]) -> pd.DataFrame:
    """Metrics for each named boolean subset. Empty subsets are skipped rather
    than reported as zeros, so a stratum with no members is visibly absent."""
    rows = []
    for name, mask in subsets.items():
        mask = np.asarray(mask, dtype=bool)
        if not mask.any():
            continue
        rows.append({"subset": name, "n": int(mask.sum()),
                     "n_pv": int((y_true[mask] == 1).sum()),
                     **compute_metrics(y_true[mask], y_pred[mask], y_proba[mask])})
    return pd.DataFrame(rows)


def evaluate_saved_predictions(predictions: pd.DataFrame,
                               subsets: dict[str, np.ndarray],
                               threshold: float = 0.5) -> pd.DataFrame:
    """Subset metrics for every model run stored in a predictions frame.

    `predictions` is the per-point file run_failure_case_evaluation.py writes:
    a `label` column, `xgboost_baseline_proba`, and one `cnn_proba_seed<k>`
    column per CNN run. Recomputing strata from it changes how points are
    grouped without retraining, so the runs behind the paper stay the same.
    Rows: the baseline (repeat 0), then the CNN seeds in order.
    """
    y_true = (predictions["label"] == "pv").astype(int).to_numpy()
    runs = [("xgboost_baseline", 0, "xgboost_baseline_proba")]
    seed_cols = sorted((c for c in predictions.columns if c.startswith("cnn_proba_seed")),
                       key=lambda c: int(c.removeprefix("cnn_proba_seed")))
    runs += [("cnn", int(c.removeprefix("cnn_proba_seed")), c) for c in seed_cols]

    frames = []
    for model, repeat, column in runs:
        proba = predictions[column].to_numpy(dtype=float)
        report = evaluate_on_subsets(y_true, (proba >= threshold).astype(int), proba,
                                     subsets)
        report.insert(0, "repeat", repeat)
        report.insert(0, "model", model)
        frames.append(report)
    return pd.concat(frames, ignore_index=True)


def estimate_capacity_kw(polygon_area_m2: np.ndarray, kw_per_m2: float = DEFAULT_KW_PER_M2
                         ) -> np.ndarray:
    return polygon_area_m2 * kw_per_m2


def flag_sub_50kw(polygons_gdf: gpd.GeoDataFrame, kw_per_m2: float = DEFAULT_KW_PER_M2
                  ) -> pd.Series:
    areas_m2 = polygons_gdf.geometry.area.values
    capacity_kw = estimate_capacity_kw(areas_m2, kw_per_m2)
    return pd.Series(capacity_kw < 50.0, index=polygons_gdf.index)


ANNOTATION_COLUMNS = ["non_crystalline_silicon", "spectrally_ambiguous"]


def stratified_annotation_sample(points_df: pd.DataFrame, n_per_bin: int,
                                 edges_kw: list[float],
                                 kw_per_m2: float = DEFAULT_KW_PER_M2,
                                 seed: int = 0) -> pd.DataFrame:
    """Draw an equal-per-capacity-bin sample for manual annotation.

    Uniform random sampling is useless here: small installations are ~0.1% of
    the pool, so a hand-annotatable sample of a few hundred points would
    contain almost none of exactly the stratum the research question is about.
    Taking a fixed quota per bin oversamples the rare small end on purpose.

    Returns the points with blank ANNOTATION_COLUMNS for a human to fill in;
    the completed file is what `load_manual_failure_flags` reads back.
    """
    masks = capacity_bin_masks(points_df, edges_kw, kw_per_m2)
    rng = np.random.default_rng(seed)

    area = points_df["polygon_area_m2"].to_numpy(dtype=float)
    chosen = []
    for mask in masks.values():
        positions = np.flatnonzero(mask)
        if len(positions) > n_per_bin:
            positions = rng.choice(positions, size=n_per_bin, replace=False)
        chosen.extend(sorted(positions))

    sample = points_df.iloc[chosen].copy()
    sample["estimated_kw"] = estimate_capacity_kw(area[chosen], kw_per_m2)
    for column in ANNOTATION_COLUMNS:
        sample[column] = pd.NA
    return sample.reset_index(drop=True)


def load_manual_failure_flags(csv_path) -> pd.DataFrame:
    """Loads a hand-annotated CSV (point_id, non_crystalline_silicon,
    spectrally_ambiguous) produced by visually inspecting a stratified
    subsample against Google Earth imagery. This function does not create
    or infer the annotations -- see the design note above."""
    df = pd.read_csv(csv_path)
    expected_cols = {"point_id", "non_crystalline_silicon", "spectrally_ambiguous"}
    if set(df.columns) != expected_cols:
        raise ValueError(f"Expected columns {expected_cols}, got {set(df.columns)}")
    return df
