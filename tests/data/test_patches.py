import numpy as np
import geopandas as gpd
from shapely.geometry import Point
from rasterio.transform import from_origin
from src.data.patches import extract_patch, extract_all_patches


def _synthetic_raster(n_bands=6, size=100):
    raster = np.arange(n_bands * size * size, dtype=float).reshape(n_bands, size, size)
    transform = from_origin(139.0, 36.0, 0.0001, 0.0001)  # west, north, xsize, ysize
    return raster, transform


def test_extract_patch_returns_correct_shape():
    raster, transform = _synthetic_raster()
    point = Point(139.005, 35.995)  # well inside the raster
    patch = extract_patch(raster, transform, point, size_px=21)
    assert patch.shape == (6, 21, 21)


def test_extract_patch_returns_none_near_edge():
    raster, transform = _synthetic_raster()
    point = Point(139.0001, 35.9999)  # near the top-left corner
    patch = extract_patch(raster, transform, point, size_px=21)
    assert patch is None


def test_extract_patch_is_centered_on_point():
    raster, transform = _synthetic_raster(n_bands=1, size=101)
    raster[0, 50, 50] = 9999.0  # known center pixel
    from rasterio.transform import xy
    cx, cy = xy(transform, 50, 50)
    patch = extract_patch(raster, transform, Point(cx, cy), size_px=21)
    assert patch[0, 10, 10] == 9999.0  # center of a 21x21 patch is index 10


def test_extract_all_patches_drops_edge_points_and_keeps_labels_aligned():
    raster, transform = _synthetic_raster()
    points = gpd.GeoDataFrame(
        {"label": ["pv", "non_pv"],
         "geometry": [Point(139.005, 35.995), Point(139.0001, 35.9999)]},
        crs="EPSG:4326",
    )
    patches, labels = extract_all_patches(raster, transform, points, size_px=21)
    assert patches.shape == (1, 6, 21, 21)
    assert labels.tolist() == ["pv"]
