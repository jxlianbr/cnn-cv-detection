"""Assemble the 9-input-feature matrix (6 reflectance bands + NDVI/MNDWI/
PVSI) and pre-masking validity flag, per papers/spectral-baseline-specifications.md."""
import geopandas as gpd
import pandas as pd
from src.data.indices import ndvi, mndwi, pvsi, bsi, premask


def build_feature_matrix(band_values: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
    df = band_values.copy()
    df["NDVI"] = ndvi(df["B8"].values, df["B4"].values)
    df["MNDWI"] = mndwi(df["B3"].values, df["B11"].values)
    df["PVSI"] = pvsi(df["B11"].values, df["B12"].values, df["B8"].values,
                      df["B2"].values, df["B4"].values)
    bsi_val = bsi(df["B2"].values, df["B4"].values)
    df["valid_candidate"] = premask(df["NDVI"].values, df["MNDWI"].values, bsi_val)
    df["label"] = labels.values
    return df[["B2", "B3", "B4", "B8", "B11", "B12", "NDVI", "MNDWI", "PVSI",
               "valid_candidate", "label"]]


def sample_band_values_at_points(composite, points_gdf: gpd.GeoDataFrame,
                                 bands: list[str], scale: int = 10) -> pd.DataFrame:
    """GEE-networked: samples `composite` at each point in points_gdf.
    Not unit-tested (see Task 5's testing-philosophy note) -- verify with
    the manual smoke test in Step 4."""
    import ee
    fc = ee.FeatureCollection([
        ee.Feature(ee.Geometry.Point([geom.x, geom.y]))
        for geom in points_gdf.geometry
    ])
    sampled = composite.sampleRegions(collection=fc, scale=scale, geometries=False)
    records = sampled.getInfo()["features"]
    rows = [r["properties"] for r in records]
    return pd.DataFrame(rows)[bands]
