import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box
from src.evaluation.failure_cases import (
    estimate_capacity_kw, flag_sub_50kw, load_manual_failure_flags,
)


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
