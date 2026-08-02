import geopandas as gpd
import numpy as np
from shapely.geometry import Point
from src.data.split import spatial_train_test_split


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
