"""PyTorch Dataset wrapping extracted patches for CNN training."""
import numpy as np
import torch
from torch.utils.data import Dataset

LABEL_TO_INT = {"non_pv": 0, "pv": 1}


class PVPatchDataset(Dataset):
    def __init__(self, patches: np.ndarray, labels: np.ndarray):
        self.patches = patches.astype(np.float32)
        self.labels = np.array([LABEL_TO_INT[label] for label in labels], dtype=np.int64)

    def __len__(self) -> int:
        return len(self.patches)

    def __getitem__(self, idx: int):
        x = torch.from_numpy(self.patches[idx])
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y
