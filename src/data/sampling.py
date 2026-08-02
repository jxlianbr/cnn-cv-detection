"""Reconstructed version of Shimada & Takeuchi (2025)'s point-sampling
protocol (papers/zenodo-dataset-verification.md §2), built from the
polygon-only public release rather than the original's unreleased point
samples."""
import geopandas as gpd
import numpy as np
from shapely.geometry import Point, box
from src.config import SamplingConfig


def _random_squares_within(bounds_gdf: gpd.GeoDataFrame, n: int, side_deg: float,
                           rng: np.random.Generator) -> list:
    minx, miny, maxx, maxy = bounds_gdf.total_bounds
    squares = []
    while len(squares) < n:
        x = rng.uniform(minx, maxx - side_deg)
        y = rng.uniform(miny, maxy - side_deg)
        squares.append(box(x, y, x + side_deg, y + side_deg))
    return squares


def _squares_centered_on_features(features_gdf: gpd.GeoDataFrame, n: int, side_deg: float,
                                  rng: np.random.Generator) -> list:
    idx = rng.integers(0, len(features_gdf), size=n)
    squares = []
    for i in idx:
        cx, cy = features_gdf.geometry.iloc[i].centroid.coords[0]
        squares.append(box(cx - side_deg / 2, cy - side_deg / 2, cx + side_deg / 2, cy + side_deg / 2))
    return squares


def generate_sampling_squares(pv_gdf: gpd.GeoDataFrame, populated_buffer_gdf: gpd.GeoDataFrame,
                              cfg: SamplingConfig, seed: int, side_deg: float = 0.009
                              ) -> gpd.GeoDataFrame:
    """~1000 1km x 1km squares (side_deg ~= 0.009 deg at Japan's latitude),
    split pv/buffer/urban strata exactly as in the original protocol."""
    rng = np.random.default_rng(seed)
    pv_squares = _squares_centered_on_features(pv_gdf, cfg.pv_squares, side_deg, rng)
    buffer_squares = _random_squares_within(populated_buffer_gdf, cfg.buffer_squares, side_deg, rng)
    urban_squares = _random_squares_within(populated_buffer_gdf, cfg.urban_squares, side_deg, rng)

    geoms = pv_squares + buffer_squares + urban_squares
    strata = (["pv"] * len(pv_squares) + ["buffer"] * len(buffer_squares)
              + ["urban"] * len(urban_squares))
    return gpd.GeoDataFrame({"stratum": strata, "geometry": geoms}, crs=pv_gdf.crs)


def _random_points_in_polygon(poly, n: int, rng: np.random.Generator) -> list[Point]:
    minx, miny, maxx, maxy = poly.bounds
    points = []
    attempts = 0
    max_attempts = n * 2000
    while len(points) < n and attempts < max_attempts:
        p = Point(rng.uniform(minx, maxx), rng.uniform(miny, maxy))
        if poly.contains(p):
            points.append(p)
        attempts += 1
    return points


def sample_points_in_squares(squares_gdf: gpd.GeoDataFrame, pv_gdf: gpd.GeoDataFrame,
                             n_pv: int, n_nonpv: int, seed: int) -> gpd.GeoDataFrame:
    rng = np.random.default_rng(seed)
    pv_union = pv_gdf.geometry.union_all()
    squares_union = squares_gdf.geometry.union_all()

    pv_region = squares_union.intersection(pv_union)
    pv_points = _random_points_in_polygon(pv_region, n_pv, rng) if not pv_region.is_empty else []

    nonpv_region = squares_union.difference(pv_union)
    nonpv_points = (_random_points_in_polygon(nonpv_region, n_nonpv, rng)
                    if not nonpv_region.is_empty else [])

    geoms = pv_points + nonpv_points
    labels = ["pv"] * len(pv_points) + ["non_pv"] * len(nonpv_points)
    return gpd.GeoDataFrame({"label": labels, "geometry": geoms}, crs=squares_gdf.crs)
