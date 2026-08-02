"""Entry point: train the CNN on the reconstructed sample, evaluate,
print metrics. Assumes Tasks 1-14's outputs already exist on disk."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root on sys.path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.config import load_config
from src.data.patch_dataset import PVPatchDataset
from src.models.cnn import PVPatchCNN
from src.models.train_cnn import train_cnn, evaluate_cnn


def main():
    cfg = load_config()
    patches = np.load("data/processed/train_patches.npy")
    labels = np.load("data/processed/train_labels.npy", allow_pickle=True)
    val_patches = np.load("data/processed/val_patches.npy")
    val_labels = np.load("data/processed/val_labels.npy", allow_pickle=True)

    train_loader = DataLoader(PVPatchDataset(patches, labels), batch_size=32, shuffle=True)
    val_loader = DataLoader(PVPatchDataset(val_patches, val_labels), batch_size=32, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PVPatchCNN(in_channels=len(cfg.sentinel2.bands), input_size=cfg.patch.size_px)
    # epochs is a ceiling; early stopping picks the budget and leaves the
    # best-val-loss weights in the model rather than the last epoch's
    history = train_cnn(model, train_loader, val_loader, epochs=200, lr=1e-3,
                        device=device, patience=15)
    metrics = evaluate_cnn(model, val_loader, device=device)

    print(f"Stopped after {len(history['val_loss'])} epochs")
    print("Final train loss:", history["train_loss"][-1])
    print("Best val loss:", min(history["val_loss"]))
    print("Validation metrics:", metrics)
    Path("results/logs").mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), "results/logs/pv_patch_cnn.pt")


if __name__ == "__main__":
    main()
