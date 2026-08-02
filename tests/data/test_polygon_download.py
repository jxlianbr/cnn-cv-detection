import geopandas as gpd
import pytest
from pathlib import Path

POLYGON_PATH = Path("data/external/pv_polygons.gpkg")

# These two verify the *downloaded dataset*, not this repository's code, so they
# need a 66 MB file that is gitignored and fetched by
# scripts/download_pv_polygons.py. Skip rather than fail where it is absent
# (CI, a fresh clone): a red build for a missing optional download would be
# noise, and re-downloading it on every CI run would make the suite depend on
# Zenodo being up.
pytestmark = pytest.mark.skipif(
    not POLYGON_PATH.exists(),
    reason=f"{POLYGON_PATH} not present -- run scripts/download_pv_polygons.py",
)


def test_polygon_file_exists_and_loads():
    gdf = gpd.read_file(POLYGON_PATH)
    assert len(gdf) > 0
    assert gdf.crs is not None


def test_polygon_total_area_matches_published_figure():
    gdf = gpd.read_file(POLYGON_PATH)
    gdf_m = gdf.to_crs(epsg=6690)  # JGD2011 / Japan Plane Rectangular equal-area-ish for a national extent check
    total_km2 = gdf_m.geometry.area.sum() / 1e6
    # Shimada & Takeuchi (2025) report 571 km^2; allow 5% tolerance for
    # reprojection/precision differences.
    assert 542 <= total_km2 <= 600, f"got {total_km2:.1f} km^2, expected ~571"
