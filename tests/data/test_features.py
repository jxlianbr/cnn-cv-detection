import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
from src.data.features import build_feature_matrix, rejoin_sampled_bands


def _three_points():
    return gpd.GeoDataFrame(
        {"point_id": [0, 1, 2],
         "label": ["pv", "non_pv", "pv"],
         "geometry": [Point(0, 0), Point(1, 1), Point(2, 2)]},
        crs="EPSG:4326",
    )


def test_rejoin_sampled_bands_reorders_records_to_match_points():
    # GEE's sampleRegions gives no ordering guarantee
    records = [{"point_id": 2, "B2": 0.3}, {"point_id": 0, "B2": 0.1},
               {"point_id": 1, "B2": 0.2}]

    bands_df, kept = rejoin_sampled_bands(records, _three_points(), bands=["B2"])

    assert bands_df["B2"].tolist() == [0.1, 0.2, 0.3]
    assert kept["point_id"].tolist() == [0, 1, 2]


def test_rejoin_sampled_bands_drops_points_missing_from_records_without_shifting_labels():
    # point 1 landed on a cloud-masked pixel, so sampleRegions returns nothing
    # for it. Assigning labels positionally against the original points would
    # label the 0.3 sample "non_pv" -- it is point 2, which is "pv".
    records = [{"point_id": 0, "B2": 0.1}, {"point_id": 2, "B2": 0.3}]

    bands_df, kept = rejoin_sampled_bands(records, _three_points(), bands=["B2"])

    assert bands_df["B2"].tolist() == [0.1, 0.3]
    assert kept["point_id"].tolist() == [0, 2]
    assert kept["label"].tolist() == ["pv", "pv"]


def test_build_feature_matrix_computes_indices_and_validity():
    band_values = pd.DataFrame({
        "B2": [0.10, 0.30], "B3": [0.15, 0.20], "B4": [0.12, 0.25],
        "B8": [0.40, 0.22], "B11": [0.20, 0.10], "B12": [0.15, 0.08],
    })
    labels = pd.Series(["pv", "non_pv"])

    df = build_feature_matrix(band_values, labels)

    assert list(df.columns) == [
        "B2", "B3", "B4", "B8", "B11", "B12", "NDVI", "MNDWI", "PVSI",
        "valid_candidate", "label",
    ]
    assert len(df) == 2
    assert df["label"].tolist() == ["pv", "non_pv"]
    assert df["valid_candidate"].dtype == bool


def test_build_feature_matrix_flags_dense_vegetation_as_invalid():
    band_values = pd.DataFrame({
        "B2": [0.05], "B3": [0.08], "B4": [0.04],
        "B8": [0.50], "B11": [0.10], "B12": [0.08],
    })
    labels = pd.Series(["non_pv"])
    df = build_feature_matrix(band_values, labels)
    assert df["NDVI"].iloc[0] > 0.5
    assert df["valid_candidate"].iloc[0] == False  # noqa: E712
