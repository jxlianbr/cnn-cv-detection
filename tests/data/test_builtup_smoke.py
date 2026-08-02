"""Not collected by pytest automatically (requires live GEE auth + network).
Run manually: python tests/data/test_builtup_smoke.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root on sys.path

import ee
from src.config import load_config
from src.data.builtup import get_populated_buffer, sample_urban_core_points


def main():
    cfg = load_config()
    ee.Initialize(project=cfg.gee_project_id)
    japan_aoi = ee.Geometry.Rectangle([129.0, 31.0, 146.0, 46.0])  # rough Japan bbox

    buffer_fc = get_populated_buffer(japan_aoi, buffer_km=cfg.sampling.buffer_km)
    print("Populated buffer feature count:", buffer_fc.size().getInfo())

    urban_points = sample_urban_core_points(japan_aoi, n_points=10)
    print("Urban core sample points:", urban_points.size().getInfo())
    first = urban_points.first().getInfo()
    print("First point geometry:", first["geometry"])


if __name__ == "__main__":
    main()
