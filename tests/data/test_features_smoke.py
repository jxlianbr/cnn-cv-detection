"""Run manually: python tests/data/test_features_smoke.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root on sys.path

import ee
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

from src.config import load_config
from src.data.features import sample_band_values_at_points, build_feature_matrix
from src.data.sentinel2 import get_autumn_2022_composite


def main():
    cfg = load_config()
    ee.Initialize(project=cfg.gee_project_id)
    aoi = ee.Geometry.Rectangle([139.6, 35.5, 139.8, 35.7])
    composite = get_autumn_2022_composite(aoi, cfg.sentinel2)
    points = gpd.GeoDataFrame(
        {"point_id": [0, 1],
         "label": ["pv", "non_pv"],
         "geometry": [Point(139.70, 35.60), Point(139.75, 35.65)]}, crs="EPSG:4326"
    )
    band_values, kept = sample_band_values_at_points(composite, points, cfg.sentinel2.bands)
    print(band_values)
    print(f"{len(kept)}/{len(points)} points survived masking")
    features = build_feature_matrix(band_values, kept["label"])
    print(features)
    assert not features.isna().any().any(), "feature matrix contains NaNs"
    assert len(features) == len(kept), "feature rows must match surviving points"


if __name__ == "__main__":
    main()
