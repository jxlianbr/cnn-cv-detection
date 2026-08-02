"""PyTorch Dataset wrapping extracted patches for CNN training."""
import numpy as np
import torch
from torch.utils.data import Dataset

LABEL_TO_INT = {"non_pv": 0, "pv": 1}
_MIN_STD = 1e-6


def compute_channel_stats(patches: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-band mean and standard deviation over a patch stack.

    Derive these from the training split only and pass the same values to the
    validation and test datasets; computing them per split would leak split
    statistics into the inputs.
    """
    mean = patches.mean(axis=(0, 2, 3)).astype(np.float32)
    std = patches.std(axis=(0, 2, 3)).astype(np.float32)
    return mean, np.maximum(std, _MIN_STD)


class PVPatchDataset(Dataset):
    """Patches as CNN input, optionally standardized and augmented.

    Augmentation is the dihedral group of the square (4 rotations x optional
    mirror). PV arrays carry no canonical orientation -- a rotated array is
    still an array -- so these transforms produce genuinely plausible samples
    rather than distortions. Enable it for training only; validation and test
    reads must stay deterministic.
    """

    def __init__(self, patches: np.ndarray, labels: np.ndarray,
                 mean: np.ndarray | None = None, std: np.ndarray | None = None,
                 augment: bool = False, seed: int = 0):
        self.patches = patches.astype(np.float32)
        self.labels = np.array([LABEL_TO_INT[label] for label in labels], dtype=np.int64)
        self.augment = augment
        self._rng = np.random.default_rng(seed)
        if mean is None or std is None:
            self.mean = self.std = None
        else:
            self.mean = np.asarray(mean, dtype=np.float32).reshape(-1, 1, 1)
            self.std = np.asarray(std, dtype=np.float32).reshape(-1, 1, 1)

    def __len__(self) -> int:
        return len(self.patches)

    def __getitem__(self, idx: int):
        patch = self.patches[idx]
        if self.augment:
            if self._rng.random() < 0.5:
                patch = patch[:, :, ::-1]
            patch = np.rot90(patch, self._rng.integers(4), axes=(1, 2))
            patch = np.ascontiguousarray(patch)
        if self.mean is not None:
            patch = (patch - self.mean) / self.std
        x = torch.from_numpy(np.ascontiguousarray(patch, dtype=np.float32))
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y
