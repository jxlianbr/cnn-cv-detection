"""Fixed-size image-patch extraction centered on sample points, for CNN
input (reports/paper-draft.md §3.3). Patch size default (21px at 10m
Sentinel-2 resolution = ~210m context window) is a config-driven design
choice, not hardcoded here -- see configs/config.yaml `patch.size_px`."""
import numpy as np
import geopandas as gpd
from rasterio.transform import rowcol


def extract_patch(raster: np.ndarray, transform, point, size_px: int) -> np.ndarray | None:
    n_bands, height, width = raster.shape
    row, col = rowcol(transform, point.x, point.y)
    half = size_px // 2
    row_start, row_end = row - half, row + half + 1
    col_start, col_end = col - half, col + half + 1
    if row_start < 0 or col_start < 0 or row_end > height or col_end > width:
        return None
    return raster[:, row_start:row_end, col_start:col_end]


def extract_all_patches(raster: np.ndarray, transform, points_gdf: gpd.GeoDataFrame,
                        size_px: int) -> tuple[np.ndarray, np.ndarray]:
    patches, labels = [], []
    for _, row in points_gdf.iterrows():
        patch = extract_patch(raster, transform, row.geometry, size_px)
        if patch is not None:
            patches.append(patch)
            labels.append(row["label"])
    return np.stack(patches), np.array(labels)
