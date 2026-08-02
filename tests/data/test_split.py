import geopandas as gpd
import numpy as np
from shapely.geometry import Point
from src.data.split import (
    spatial_train_test_split, spatial_train_val_test_split, block_holdout_mask, _block_id,
)


def _grid_points_gdf(n_per_side=20, spacing_deg=0.01):
    pts = [Point(139.0 + i * spacing_deg, 35.0 + j * spacing_deg)
           for i in range(n_per_side) for j in range(n_per_side)]
    return gpd.GeoDataFrame({"geometry": pts}, crs="EPSG:4326")


def test_split_produces_requested_fraction_approximately():
    gdf = _grid_points_gdf()
    train, test = spatial_train_test_split(gdf, test_fraction=0.2, block_size_km=5, seed=0)
    total = len(train) + len(test)
    assert total == len(gdf)
    assert 0.1 < len(test) / total < 0.35  # blocky split won't hit exactly 20%


def test_split_is_deterministic_given_seed():
    gdf = _grid_points_gdf()
    train1, test1 = spatial_train_test_split(gdf, test_fraction=0.2, block_size_km=5, seed=7)
    train2, test2 = spatial_train_test_split(gdf, test_fraction=0.2, block_size_km=5, seed=7)
    assert set(train1.index) == set(train2.index)
    assert set(test1.index) == set(test2.index)


def test_split_no_point_appears_in_both_sets():
    gdf = _grid_points_gdf()
    train, test = spatial_train_test_split(gdf, test_fraction=0.3, block_size_km=5, seed=3)
    assert set(train.index).isdisjoint(set(test.index))


def test_split_points_in_same_small_neighborhood_stay_together():
    # Two points ~50m apart must fall in the same block at block_size_km=5
    gdf = gpd.GeoDataFrame(
        {"geometry": [Point(139.000, 35.000), Point(139.0005, 35.0005)]},
        crs="EPSG:4326",
    )
    # pad with a spread-out grid so the split isn't trivially all-train
    gdf = gpd.GeoDataFrame(
        list(gdf.geometry) + list(_grid_points_gdf().geometry),
        columns=["geometry"], crs="EPSG:4326",
    )
    train, test = spatial_train_test_split(gdf, test_fraction=0.2, block_size_km=5, seed=0)
    both_in_train = 0 in train.index and 1 in train.index
    both_in_test = 0 in test.index and 1 in test.index
    assert both_in_train or both_in_test


def test_block_holdout_mask_holds_out_roughly_the_requested_share():
    gdf = _grid_points_gdf(n_per_side=30)
    lon = gdf.geometry.x.values
    lat = gdf.geometry.y.values

    mask = block_holdout_mask(lon, lat, holdout_fraction=0.2, block_size_km=5, seed=0)

    assert 0.1 < mask.mean() < 0.35  # blocky split won't hit exactly 20%


def test_block_holdout_mask_keeps_a_whole_block_on_one_side():
    # an early-stopping slice drawn at random would put near-neighbours of the
    # fitting points into the stopping set, which is the leak being closed
    gdf = _grid_points_gdf(n_per_side=30)
    lon = gdf.geometry.x.values
    lat = gdf.geometry.y.values

    mask = block_holdout_mask(lon, lat, holdout_fraction=0.3, block_size_km=5, seed=1)

    by_block = {}
    for l0, l1, m in zip(lon, lat, mask):
        by_block.setdefault(_block_id(l0, l1, 5), set()).add(bool(m))
    assert all(len(v) == 1 for v in by_block.values())


def test_block_holdout_mask_is_deterministic_given_seed():
    gdf = _grid_points_gdf(n_per_side=20)
    lon, lat = gdf.geometry.x.values, gdf.geometry.y.values
    a = block_holdout_mask(lon, lat, 0.2, 5, seed=3)
    b = block_holdout_mask(lon, lat, 0.2, 5, seed=3)
    assert (a == b).all()


def test_block_holdout_mask_always_leaves_points_on_both_sides():
    # a tiny training subset must still yield a usable stopping set
    gdf = _grid_points_gdf(n_per_side=4)
    lon, lat = gdf.geometry.x.values, gdf.geometry.y.values

    mask = block_holdout_mask(lon, lat, holdout_fraction=0.2, block_size_km=50, seed=0)

    assert 0 < mask.sum() < len(mask)


def _blocks_of(gdf, block_size_km):
    return {_block_id(p.x, p.y, block_size_km) for p in gdf.geometry}


def test_three_way_split_shares_no_block_between_any_two_splits():
    gdf = _grid_points_gdf(n_per_side=30)
    train, val, test = spatial_train_val_test_split(
        gdf, test_fraction=0.2, val_fraction=0.2, block_size_km=5, seed=0)

    train_b, val_b, test_b = (_blocks_of(g, 5) for g in (train, val, test))
    assert train_b.isdisjoint(val_b)
    assert train_b.isdisjoint(test_b)
    assert val_b.isdisjoint(test_b)


def test_three_way_split_partitions_every_point_exactly_once():
    gdf = _grid_points_gdf(n_per_side=30)
    train, val, test = spatial_train_val_test_split(
        gdf, test_fraction=0.2, val_fraction=0.2, block_size_km=5, seed=0)

    assert len(train) + len(val) + len(test) == len(gdf)
    assert set(train.index) | set(val.index) | set(test.index) == set(gdf.index)


def test_three_way_split_gives_each_split_a_nonempty_share():
    gdf = _grid_points_gdf(n_per_side=30)
    train, val, test = spatial_train_val_test_split(
        gdf, test_fraction=0.2, val_fraction=0.2, block_size_km=5, seed=0)

    assert len(test) > 0 and len(val) > 0
    assert len(train) > len(val)  # train must remain the majority share
