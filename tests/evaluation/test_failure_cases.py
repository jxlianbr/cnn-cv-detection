import numpy as np
import pandas as pd
import pytest
import geopandas as gpd
from shapely.geometry import box
from shapely.geometry import Point
from src.evaluation.failure_cases import (
    estimate_capacity_kw, flag_sub_50kw, load_manual_failure_flags,
    attach_pv_polygon_area, evaluate_on_subsets,
)


def _pv_polygons():
    return gpd.GeoDataFrame(
        {"geometry": [box(139.100, 36.100, 139.101, 36.101),
                      box(139.200, 36.200, 139.203, 36.203)]},
        crs="EPSG:4326",
    )


def test_attach_pv_polygon_area_reports_the_containing_polygons_area():
    pv = _pv_polygons()
    points = gpd.GeoDataFrame(
        {"point_id": [0, 1], "label": ["pv", "pv"],
         "geometry": [Point(139.1005, 36.1005), Point(139.2015, 36.2015)]},
        crs="EPSG:4326",
    )

    out = attach_pv_polygon_area(points, pv)

    expected = pv.to_crs("EPSG:6933").geometry.area.values
    np.testing.assert_allclose(out["polygon_area_m2"].values, expected, rtol=1e-6)


def test_attach_pv_polygon_area_leaves_non_pv_points_as_nan():
    points = gpd.GeoDataFrame(
        {"point_id": [0, 1], "label": ["pv", "non_pv"],
         "geometry": [Point(139.1005, 36.1005), Point(139.900, 36.900)]},
        crs="EPSG:4326",
    )

    out = attach_pv_polygon_area(points, _pv_polygons())

    assert out["polygon_area_m2"].notna().iloc[0]
    assert pd.isna(out["polygon_area_m2"].iloc[1])


def test_attach_pv_polygon_area_handles_points_inside_overlapping_polygons():
    # A point can sit inside both an installation and a larger enclosing
    # footprint, so the spatial join emits several rows for it. Each point must
    # still yield exactly one row, carrying the smallest containing footprint.
    outer = box(139.100, 36.100, 139.120, 36.120)
    inner = box(139.101, 36.101, 139.102, 36.102)
    lone = box(139.300, 36.300, 139.310, 36.310)
    pv = gpd.GeoDataFrame({"geometry": [outer, inner, lone]}, crs="EPSG:4326")
    points = gpd.GeoDataFrame(
        {"point_id": [0, 1, 2],
         "label": ["pv", "pv", "non_pv"],
         "geometry": [Point(139.1015, 36.1015),   # inside outer AND inner
                      Point(139.3050, 36.3050),   # inside lone only
                      Point(139.900, 36.900)]},   # inside nothing
        crs="EPSG:4326",
    )

    out = attach_pv_polygon_area(points, pv)

    assert len(out) == 3
    areas = pv.to_crs("EPSG:6933").geometry.area.values
    assert out["polygon_area_m2"].iloc[0] == pytest.approx(areas[1])  # the inner one
    assert out["polygon_area_m2"].iloc[1] == pytest.approx(areas[2])
    assert pd.isna(out["polygon_area_m2"].iloc[2])


def test_attach_pv_polygon_area_preserves_row_order_and_count():
    points = gpd.GeoDataFrame(
        {"point_id": [5, 6, 7], "label": ["non_pv", "pv", "non_pv"],
         "geometry": [Point(139.9, 36.9), Point(139.1005, 36.1005),
                      Point(139.95, 36.95)]},
        crs="EPSG:4326",
    )

    out = attach_pv_polygon_area(points, _pv_polygons())

    assert out["point_id"].tolist() == [5, 6, 7]


def test_evaluate_on_subsets_reports_one_row_per_named_subset():
    y_true = np.array([1, 1, 0, 0, 1, 0])
    y_pred = np.array([1, 0, 0, 0, 1, 1])
    y_proba = np.array([0.9, 0.4, 0.1, 0.2, 0.8, 0.6])
    subsets = {"all": np.ones(6, dtype=bool),
               "small": np.array([True, True, True, False, False, False])}

    out = evaluate_on_subsets(y_true, y_pred, y_proba, subsets)

    assert out["subset"].tolist() == ["all", "small"]
    assert out["n"].tolist() == [6, 3]
    assert "f1" in out.columns


def test_evaluate_on_subsets_skips_a_subset_with_no_members():
    y_true = np.array([1, 0])
    y_pred = np.array([1, 0])
    y_proba = np.array([0.9, 0.1])
    subsets = {"all": np.ones(2, dtype=bool), "empty": np.zeros(2, dtype=bool)}

    out = evaluate_on_subsets(y_true, y_pred, y_proba, subsets)

    assert out["subset"].tolist() == ["all"]


def test_estimate_capacity_kw_scales_with_area():
    areas = np.array([100.0, 1000.0])
    kw = estimate_capacity_kw(areas, kw_per_m2=0.15)
    np.testing.assert_allclose(kw, [15.0, 150.0])


def test_flag_sub_50kw_identifies_small_installations():
    # areas in m^2, in EPSG:3857-like projected units for the test polygons
    small = box(0, 0, 18, 18)   # 324 m^2 -> ~48.6 kW at 0.15 kW/m^2 -> sub-50kW
    large = box(0, 0, 100, 100)  # 10,000 m^2 -> 1500 kW -> not sub-50kW
    gdf = gpd.GeoDataFrame({"geometry": [small, large]}, crs="EPSG:3857")
    flags = flag_sub_50kw(gdf, kw_per_m2=0.15)
    assert flags.tolist() == [True, False]


def test_load_manual_failure_flags_reads_csv(tmp_path):
    csv_path = tmp_path / "manual_flags.csv"
    csv_path.write_text(
        "point_id,non_crystalline_silicon,spectrally_ambiguous\n"
        "1,True,False\n"
        "2,False,True\n"
    )
    df = load_manual_failure_flags(csv_path)
    assert list(df.columns) == ["point_id", "non_crystalline_silicon", "spectrally_ambiguous"]
    assert df["non_crystalline_silicon"].tolist() == [True, False]
    assert df["spectrally_ambiguous"].tolist() == [False, True]
