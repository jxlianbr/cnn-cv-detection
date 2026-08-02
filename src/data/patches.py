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


def extract_patches_with_ids(raster: np.ndarray, transform, points_gdf: gpd.GeoDataFrame,
                             size_px: int, id_column: str = "point_id"
                             ) -> tuple[np.ndarray, np.ndarray]:
    """Like `extract_all_patches`, but reports *which* points survived.

    Points whose window would run off the raster edge are dropped, so the
    output is shorter than the input. Callers that also hold per-point feature
    rows need the surviving ids to keep the two in step -- see
    `src.data.alignment.align_features_and_patches`.
    """
    n_bands = raster.shape[0]
    patches, ids = [], []
    for _, row in points_gdf.iterrows():
        patch = extract_patch(raster, transform, row.geometry, size_px)
        if patch is not None:
            patches.append(patch)
            ids.append(row[id_column])
    if not patches:
        # np.stack raises on an empty list; a square where every point fell
        # near the edge is normal, not an error.
        return (np.empty((0, n_bands, size_px, size_px), dtype=raster.dtype),
                np.array([], dtype=points_gdf[id_column].dtype))
    return np.stack(patches), np.array(ids)


def extract_all_patches(raster: np.ndarray, transform, points_gdf: gpd.GeoDataFrame,
                        size_px: int) -> tuple[np.ndarray, np.ndarray]:
    patches, labels = [], []
    for _, row in points_gdf.iterrows():
        patch = extract_patch(raster, transform, row.geometry, size_px)
        if patch is not None:
            patches.append(patch)
            labels.append(row["label"])
    return np.stack(patches), np.array(labels)
