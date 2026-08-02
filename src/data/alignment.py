"""Keeps the per-point feature rows and the per-point image patches in step.

Two independent stages drop points: GEE's `sampleRegions` omits points on
cloud-masked pixels, and patch extraction omits points whose window runs off
the raster edge. The two sets of survivors differ. Downstream code
(`scripts/run_label_efficiency.py`) indexes `train_df.iloc[idx]` and
`train_patches[idx]` with the *same* integer array, so the two must be reduced
to their intersection and given a common row order before being written out.
"""
import numpy as np
import pandas as pd


def align_features_and_patches(features_df: pd.DataFrame, patches: np.ndarray,
                               patch_ids: np.ndarray, id_column: str = "point_id"
                               ) -> tuple[pd.DataFrame, np.ndarray]:
    """Reduce both inputs to the points present in each, ordered identically.

    Returns (features, patches) where row i of the frame describes patches[i].
    """
    position_of_id = {pid: i for i, pid in enumerate(patch_ids)}
    keep = features_df[id_column].isin(position_of_id)
    aligned_df = features_df[keep].reset_index(drop=True)
    order = [position_of_id[pid] for pid in aligned_df[id_column]]
    return aligned_df, patches[order]
