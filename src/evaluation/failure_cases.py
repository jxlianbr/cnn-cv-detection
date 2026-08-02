"""Operational definitions for the three failure-case subsets named in
Shimada & Takeuchi (2025) and reports/paper-draft.md §3.5. See this
task's design note in plan.md for why two of the three require manual
annotation rather than an automated proxy."""
import numpy as np
import pandas as pd
import geopandas as gpd

from src.evaluation.metrics import compute_metrics

DEFAULT_KW_PER_M2 = 0.15  # mid-range published capacity density for
                          # crystalline-silicon ground-mount PV; approximation only

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


def estimate_capacity_kw(polygon_area_m2: np.ndarray, kw_per_m2: float = DEFAULT_KW_PER_M2
                         ) -> np.ndarray:
    return polygon_area_m2 * kw_per_m2


def flag_sub_50kw(polygons_gdf: gpd.GeoDataFrame, kw_per_m2: float = DEFAULT_KW_PER_M2
                  ) -> pd.Series:
    areas_m2 = polygons_gdf.geometry.area.values
    capacity_kw = estimate_capacity_kw(areas_m2, kw_per_m2)
    return pd.Series(capacity_kw < 50.0, index=polygons_gdf.index)


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
