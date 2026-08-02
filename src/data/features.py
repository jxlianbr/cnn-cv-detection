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


def rejoin_sampled_bands(records: list[dict], points_gdf: gpd.GeoDataFrame,
                         bands: list[str], id_column: str = "point_id"
                         ) -> tuple[pd.DataFrame, gpd.GeoDataFrame]:
    """Realign GEE `sampleRegions` output against the points that produced it.

    `sampleRegions` returns features in no guaranteed order and silently omits
    points that fall on masked (cloud / no-data) pixels. Assigning labels
    positionally against the original points_gdf therefore shifts every label
    after the first omission. Joining on an explicit id is the only safe way.

    Returns (band_values, kept_points), positionally aligned row-for-row.
    """
    sampled = pd.DataFrame(records).set_index(id_column)
    kept = points_gdf[points_gdf[id_column].isin(sampled.index)].reset_index(drop=True)
    band_values = sampled.loc[kept[id_column].values, bands].reset_index(drop=True)
    return band_values, kept


def fetch_sampled_records(composite, points_gdf: gpd.GeoDataFrame, bands: list[str],
                          scale: int = 10, id_column: str = "point_id",
                          batch_size: int = 1000) -> list[dict]:
    """GEE-networked: samples `composite` at each point, tagging every feature
    with its `id_column` value so the result can be rejoined safely.

    Batched because `getInfo()` fails on large FeatureCollections. Not
    unit-tested (see Task 5's testing-philosophy note) -- verify with the
    manual smoke test.
    """
    import ee
    records = []
    ids = points_gdf[id_column].tolist()
    geoms = list(points_gdf.geometry)
    for start in range(0, len(ids), batch_size):
        chunk = list(zip(ids[start:start + batch_size], geoms[start:start + batch_size]))
        fc = ee.FeatureCollection([
            ee.Feature(ee.Geometry.Point([geom.x, geom.y]), {id_column: int(pid)})
            for pid, geom in chunk
        ])
        sampled = composite.sampleRegions(collection=fc, scale=scale, geometries=False)
        records.extend(r["properties"] for r in sampled.getInfo()["features"])
    return records


def sample_band_values_at_points(composite, points_gdf: gpd.GeoDataFrame,
                                 bands: list[str], scale: int = 10,
                                 id_column: str = "point_id", batch_size: int = 1000
                                 ) -> tuple[pd.DataFrame, gpd.GeoDataFrame]:
    """Sample `composite` at each point and realign the result against the
    points that survived masking. Returns (band_values, kept_points), aligned
    row-for-row -- see `rejoin_sampled_bands` for why the rejoin is required."""
    records = fetch_sampled_records(composite, points_gdf, bands, scale,
                                    id_column, batch_size)
    return rejoin_sampled_bands(records, points_gdf, bands, id_column)
