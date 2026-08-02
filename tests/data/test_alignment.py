import numpy as np
import pandas as pd
from src.data.alignment import align_features_and_patches


def _features(point_ids):
    return pd.DataFrame({"point_id": point_ids,
                         "B2": [0.1 * i for i in point_ids],
                         "label": ["pv"] * len(point_ids)})


def _patches_stamped_with_ids(point_ids):
    """Each patch is filled with its own point_id, so a misalignment is visible."""
    return np.stack([np.full((6, 21, 21), float(i)) for i in point_ids])


def test_align_features_and_patches_orders_patches_to_match_feature_rows():
    # patches come back grouped by sampling square, not in point order
    features = _features([10, 11, 12])
    patch_ids = np.array([12, 10, 11])
    patches = _patches_stamped_with_ids(patch_ids)

    df, aligned = align_features_and_patches(features, patches, patch_ids)

    assert df["point_id"].tolist() == [10, 11, 12]
    assert [aligned[i, 0, 0, 0] for i in range(3)] == [10.0, 11.0, 12.0]


def test_align_features_and_patches_keeps_only_the_intersection():
    # point 11 was cloud-masked out of the band sampling; point 13's patch ran
    # off the raster edge. Neither can be used, and neither may shift the rest.
    features = _features([10, 12, 13])
    patch_ids = np.array([10, 11, 12])
    patches = _patches_stamped_with_ids(patch_ids)

    df, aligned = align_features_and_patches(features, patches, patch_ids)

    assert df["point_id"].tolist() == [10, 12]
    assert aligned.shape == (2, 6, 21, 21)
    assert [aligned[i, 0, 0, 0] for i in range(2)] == [10.0, 12.0]


def test_align_features_and_patches_resets_the_frame_index():
    # downstream code indexes train_df.iloc[idx] and patches[idx] with the same
    # integer array, which only agrees if the frame index is 0..n-1
    features = _features([10, 11, 12])
    patch_ids = np.array([11, 12])
    patches = _patches_stamped_with_ids(patch_ids)

    df, aligned = align_features_and_patches(features, patches, patch_ids)

    assert df.index.tolist() == [0, 1]
