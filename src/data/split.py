"""Block-based spatial train/test split: partitions the AOI into
block_size_km x block_size_km grid cells and assigns whole cells to
train or test, so no two spatially adjacent points can leak across
the split (reports/paper-draft.md §3.3, §7)."""
import geopandas as gpd
import numpy as np

_KM_PER_DEGREE_LAT = 111.0


def _block_id(lon: float, lat: float, block_size_km: float) -> tuple[int, int]:
    km_per_degree_lon = _KM_PER_DEGREE_LAT * np.cos(np.radians(lat))
    col = int(lon * km_per_degree_lon // block_size_km)
    row = int(lat * _KM_PER_DEGREE_LAT // block_size_km)
    return row, col


def spatial_train_test_split(gdf: gpd.GeoDataFrame, test_fraction: float,
                             block_size_km: float, seed: int
                             ) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    blocks = gdf.geometry.apply(lambda p: _block_id(p.x, p.y, block_size_km))
    unique_blocks = sorted(set(blocks))
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(len(unique_blocks))
    n_test_blocks = max(1, int(len(unique_blocks) * test_fraction))
    test_block_set = {unique_blocks[i] for i in shuffled[:n_test_blocks]}

    is_test = blocks.isin(test_block_set)
    return gdf[~is_test], gdf[is_test]


def spatial_train_val_test_split(gdf: gpd.GeoDataFrame, test_fraction: float,
                                 val_fraction: float, block_size_km: float, seed: int
                                 ) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Three-way block split. Both fractions are shares of the *whole* dataset,
    so (0.2, 0.2) gives roughly 60/20/20.

    The test split is carved off first and never touched by the training
    scripts, leaving it clean for the final Results-chapter numbers; val is
    then carved out of what remains, for model selection. Applying the same
    whole-block assignment at each stage keeps all three mutually
    block-disjoint, which is what prevents patch leakage between them.
    """
    train_pool, test = spatial_train_test_split(gdf, test_fraction, block_size_km, seed)
    val_share_of_pool = val_fraction / (1.0 - test_fraction)
    train, val = spatial_train_test_split(train_pool, val_share_of_pool, block_size_km, seed + 1)
    return train, val, test
