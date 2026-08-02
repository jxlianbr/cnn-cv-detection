import numpy as np
import pandas as pd
from scripts.run_failure_case_evaluation import build_subset_masks

# 0.15 kW/m^2 -> 50 kW at 333.3 m^2
SMALL_M2 = 200.0    # ~30 kW
LARGE_M2 = 4000.0   # ~600 kW


def _eval_df():
    return pd.DataFrame({
        "point_id": [10, 11, 12, 13],
        "label": ["pv", "pv", "non_pv", "non_pv"],
        "polygon_area_m2": [SMALL_M2, LARGE_M2, np.nan, np.nan],
    })


def test_build_subset_masks_always_includes_an_all_stratum():
    masks = build_subset_masks(_eval_df(), kw_per_m2=0.15, manual_flags=None)
    assert masks["all"].all()
    assert len(masks["all"]) == 4


def test_build_subset_masks_splits_pv_points_by_the_50kw_threshold():
    masks = build_subset_masks(_eval_df(), kw_per_m2=0.15, manual_flags=None)
    assert masks["sub_50kw"].tolist() == [True, False, False, False]
    assert masks["over_50kw"].tolist() == [False, True, False, False]


def test_build_subset_masks_excludes_unsized_points_from_both_capacity_strata():
    # non-PV points have no footprint, so they belong to neither capacity
    # stratum -- NaN must not silently compare as "under 50kW"
    masks = build_subset_masks(_eval_df(), kw_per_m2=0.15, manual_flags=None)
    unsized = _eval_df()["polygon_area_m2"].isna().to_numpy()
    assert not masks["sub_50kw"][unsized].any()
    assert not masks["over_50kw"][unsized].any()


def test_build_subset_masks_omits_manual_strata_when_no_annotations_given():
    masks = build_subset_masks(_eval_df(), kw_per_m2=0.15, manual_flags=None)
    assert "non_crystalline_silicon" not in masks
    assert "spectrally_ambiguous" not in masks


def test_build_subset_masks_aligns_manual_flags_by_point_id_not_row_order():
    # the annotation CSV covers a subset of points, in its own order
    manual = pd.DataFrame({
        "point_id": [13, 11],
        "non_crystalline_silicon": [True, False],
        "spectrally_ambiguous": [False, True],
    })

    masks = build_subset_masks(_eval_df(), kw_per_m2=0.15, manual_flags=manual)

    assert masks["non_crystalline_silicon"].tolist() == [False, False, False, True]
    assert masks["spectrally_ambiguous"].tolist() == [False, True, False, False]


def test_build_subset_masks_respects_a_different_capacity_density():
    # at 0.30 kW/m^2 the 200 m^2 footprint is 60 kW and no longer sub-50kW
    masks = build_subset_masks(_eval_df(), kw_per_m2=0.30, manual_flags=None)
    assert masks["sub_50kw"].tolist() == [False, False, False, False]
