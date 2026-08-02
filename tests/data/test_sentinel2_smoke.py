"""Run manually: python tests/data/test_sentinel2_smoke.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root on sys.path

import ee
from src.config import load_config
from src.data.sentinel2 import get_autumn_2022_composite


def main():
    cfg = load_config()
    ee.Initialize(project=cfg.gee_project_id)
    small_aoi = ee.Geometry.Rectangle([139.6, 35.5, 139.8, 35.7])  # Tokyo area

    composite = get_autumn_2022_composite(small_aoi, cfg.sentinel2)
    band_names = composite.bandNames().getInfo()
    print("Bands:", band_names)
    assert band_names == cfg.sentinel2.bands, band_names

    stats = composite.reduceRegion(
        reducer=ee.Reducer.mean(), geometry=small_aoi, scale=10, maxPixels=1e9
    ).getInfo()
    print("Mean reflectance per band:", stats)
    for band, value in stats.items():
        assert value is not None and 0.0 <= value <= 1.0, f"{band}={value} out of range"


if __name__ == "__main__":
    main()
