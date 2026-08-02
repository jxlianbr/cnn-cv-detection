"""Label-efficiency comparison: both models trained on shrinking fractions
of the training set (reports/paper-draft.md §4.5, secondary RQ)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root on sys.path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.config import XGBoostConfig
from src.models.xgboost_baseline import train_baseline, predict_proba, FEATURE_COLUMNS
from src.data.patch_dataset import PVPatchDataset
from src.models.cnn import PVPatchCNN
from src.models.train_cnn import train_cnn, evaluate_cnn
from src.evaluation.metrics import compute_metrics


def run_label_efficiency_curve(train_df, train_patches, train_labels,
                               val_df, val_patches, val_labels,
                               fractions: list[float], cfg: XGBoostConfig,
                               cnn_epochs: int = 20, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_total = len(train_df)
    rows = []

    for fraction in fractions:
        n_subset = max(2, int(n_total * fraction))
        idx = rng.choice(n_total, size=n_subset, replace=False)

        # --- XGBoost baseline ---
        subset_df = train_df.iloc[idx]
        model = train_baseline(subset_df, cfg)
        val_proba = predict_proba(model, val_df)
        val_pred = (val_proba >= 0.5).astype(int)
        val_true = (val_df["label"] == "pv").astype(int).values
        baseline_metrics = compute_metrics(val_true, val_pred, val_proba)
        rows.append({"fraction": fraction, "model": "xgboost_baseline", **baseline_metrics})

        # --- CNN ---
        subset_patches = train_patches[idx]
        subset_labels = train_labels[idx]
        train_loader = DataLoader(
            PVPatchDataset(subset_patches, subset_labels), batch_size=16, shuffle=True
        )
        val_loader = DataLoader(
            PVPatchDataset(val_patches, val_labels), batch_size=16, shuffle=False
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        cnn_model = PVPatchCNN(in_channels=train_patches.shape[1], input_size=train_patches.shape[2])
        train_cnn(cnn_model, train_loader, val_loader, epochs=cnn_epochs, lr=1e-3, device=device)
        cnn_metrics = evaluate_cnn(cnn_model, val_loader, device=device)
        rows.append({"fraction": fraction, "model": "cnn", **cnn_metrics})

    return pd.DataFrame(rows)


def main():
    from src.config import load_config
    cfg = load_config()
    train_df = pd.read_parquet("data/processed/train_features.parquet")
    val_df = pd.read_parquet("data/processed/val_features.parquet")
    train_patches = np.load("data/processed/train_patches.npy")
    train_labels = np.load("data/processed/train_labels.npy", allow_pickle=True)
    val_patches = np.load("data/processed/val_patches.npy")
    val_labels = np.load("data/processed/val_labels.npy", allow_pickle=True)

    result = run_label_efficiency_curve(
        train_df, train_patches, train_labels,
        val_df, val_patches, val_labels,
        fractions=[0.1, 0.25, 0.5, 0.75, 1.0], cfg=cfg.xgboost,
    )
    Path("results/logs").mkdir(parents=True, exist_ok=True)
    result.to_csv("results/logs/label_efficiency_curve.csv", index=False)
    print(result)


if __name__ == "__main__":
    main()
