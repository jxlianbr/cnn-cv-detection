import numpy as np
import torch
from src.data.patch_dataset import PVPatchDataset, compute_channel_stats


def _asymmetric_patches(n=8):
    """Patches with no symmetry, so any dihedral transform is detectable."""
    base = np.arange(6 * 21 * 21, dtype=np.float32).reshape(6, 21, 21)
    return np.stack([base + i for i in range(n)])


def test_compute_channel_stats_returns_per_channel_mean_and_std():
    patches = np.random.rand(20, 6, 21, 21).astype(np.float32)

    mean, std = compute_channel_stats(patches)

    assert mean.shape == (6,)
    assert std.shape == (6,)
    np.testing.assert_allclose(mean, patches.mean(axis=(0, 2, 3)), rtol=1e-5)


def test_compute_channel_stats_never_returns_zero_std():
    # a constant channel would otherwise divide by zero
    patches = np.ones((4, 6, 21, 21), dtype=np.float32)

    _, std = compute_channel_stats(patches)

    assert (std > 0).all()


def test_dataset_standardizes_channels_when_given_stats():
    patches = np.random.rand(64, 6, 21, 21).astype(np.float32) * 3.0 + 5.0
    labels = np.array(["pv", "non_pv"] * 32)
    mean, std = compute_channel_stats(patches)

    ds = PVPatchDataset(patches, labels, mean=mean, std=std)
    stacked = torch.stack([ds[i][0] for i in range(len(ds))])

    assert abs(stacked.mean().item()) < 0.05
    assert abs(stacked.std().item() - 1.0) < 0.1


def test_dataset_leaves_values_untouched_without_stats():
    patches = np.random.rand(4, 6, 21, 21).astype(np.float32)
    ds = PVPatchDataset(patches, np.array(["pv"] * 4))
    np.testing.assert_allclose(ds[0][0].numpy(), patches[0])


def test_augmentation_varies_the_same_patch_across_reads():
    ds = PVPatchDataset(_asymmetric_patches(), np.array(["pv"] * 8),
                        augment=True, seed=0)

    reads = [ds[0][0].numpy().copy() for _ in range(20)]

    assert any(not np.array_equal(reads[0], r) for r in reads[1:])


def test_augmentation_only_produces_dihedral_views_of_the_original():
    ds = PVPatchDataset(_asymmetric_patches(), np.array(["pv"] * 8),
                        augment=True, seed=0)
    original = _asymmetric_patches()[0]
    allowed = [np.rot90(f, k, axes=(1, 2))
               for f in (original, original[:, :, ::-1])
               for k in range(4)]

    for _ in range(30):
        got = ds[0][0].numpy()
        assert any(np.array_equal(got, a) for a in allowed)


def test_augmentation_preserves_the_label():
    ds = PVPatchDataset(_asymmetric_patches(4), np.array(["pv", "non_pv"] * 2),
                        augment=True, seed=0)
    assert all(ds[1][1].item() == 0 for _ in range(10))


def test_dataset_is_deterministic_without_augmentation():
    ds = PVPatchDataset(_asymmetric_patches(), np.array(["pv"] * 8))
    np.testing.assert_array_equal(ds[3][0].numpy(), ds[3][0].numpy())


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
