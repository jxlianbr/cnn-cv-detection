import geopandas as gpd
from pathlib import Path

POLYGON_PATH = Path("data/external/pv_polygons.gpkg")


def test_polygon_file_exists_and_loads():
    assert POLYGON_PATH.exists(), "run scripts/download_pv_polygons.py first"
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
