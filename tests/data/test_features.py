import pandas as pd
from src.data.features import build_feature_matrix


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
