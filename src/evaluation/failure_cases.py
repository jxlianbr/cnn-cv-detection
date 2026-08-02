"""Operational definitions for the three failure-case subsets named in
Shimada & Takeuchi (2025) and reports/paper-draft.md §3.5. See this
task's design note in plan.md for why two of the three require manual
annotation rather than an automated proxy."""
import numpy as np
import pandas as pd
import geopandas as gpd

DEFAULT_KW_PER_M2 = 0.15  # mid-range published capacity density for
                          # crystalline-silicon ground-mount PV; approximation only


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
