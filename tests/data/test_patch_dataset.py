import numpy as np
import torch
from src.data.patch_dataset import PVPatchDataset


def test_dataset_length_matches_input():
    patches = np.random.rand(10, 6, 21, 21).astype(np.float32)
    labels = np.array(["pv"] * 5 + ["non_pv"] * 5)
    ds = PVPatchDataset(patches, labels)
    assert len(ds) == 10


def test_dataset_returns_correct_types_and_shapes():
    patches = np.random.rand(4, 6, 21, 21).astype(np.float32)
    labels = np.array(["pv", "non_pv", "pv", "non_pv"])
    ds = PVPatchDataset(patches, labels)
    x, y = ds[0]
    assert isinstance(x, torch.Tensor)
    assert x.dtype == torch.float32
    assert x.shape == (6, 21, 21)
    assert isinstance(y, torch.Tensor)
    assert y.dtype == torch.long


def test_dataset_label_encoding_is_consistent():
    patches = np.random.rand(2, 6, 21, 21).astype(np.float32)
    labels = np.array(["pv", "non_pv"])
    ds = PVPatchDataset(patches, labels)
    _, y_pv = ds[0]
    _, y_nonpv = ds[1]
    assert y_pv.item() == 1
    assert y_nonpv.item() == 0


def test_dataset_works_with_dataloader():
    from torch.utils.data import DataLoader
    patches = np.random.rand(16, 6, 21, 21).astype(np.float32)
    labels = np.array(["pv", "non_pv"] * 8)
    ds = PVPatchDataset(patches, labels)
    loader = DataLoader(ds, batch_size=4, shuffle=True)
    batch_x, batch_y = next(iter(loader))
    assert batch_x.shape == (4, 6, 21, 21)
    assert batch_y.shape == (4,)
