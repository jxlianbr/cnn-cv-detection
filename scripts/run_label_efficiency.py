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
from src.data.patch_dataset import PVPatchDataset, compute_channel_stats
from src.data.split import block_holdout_mask
from src.models.cnn import PVPatchCNN
from src.models.train_cnn import train_cnn, evaluate_cnn
from src.evaluation.metrics import compute_metrics


METRIC_COLUMNS = ["overall_accuracy", "precision", "recall", "f1", "kappa", "roc_auc"]
EARLY_STOP_FRACTION = 0.2
EARLY_STOP_BLOCK_KM = 10


def subset_indices_for_repeats(n_total: int, fraction: float, seed: int,
                               n_repeats: int) -> list[np.ndarray]:
    """One independent training subset per repeat.

    A single draw per fraction cannot be distinguished from noise: on a small
    validation set the spread between draws is the same size as the differences
    being measured, which is why a one-seed curve comes out non-monotonic.
    Each repeat gets its own seed so the results can be averaged.
    """
    n_subset = max(2, int(n_total * fraction))
    return [np.random.default_rng(seed + repeat).choice(n_total, size=n_subset,
                                                        replace=False)
            for repeat in range(n_repeats)]


def run_label_efficiency_curve(train_df, train_patches, train_labels,
                               val_df, val_patches, val_labels,
                               fractions: list[float], cfg: XGBoostConfig,
                               cnn_epochs: int = 100, seed: int = 0,
                               n_repeats: int = 1, patience: int = 10
                               ) -> pd.DataFrame:
    """One row per (fraction, model, repeat). Aggregate with `summarize_curve`."""
    n_total = len(train_df)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    val_true = (val_df["label"] == "pv").astype(int).values
    rows = []

    for fraction in fractions:
        for repeat, idx in enumerate(
                subset_indices_for_repeats(n_total, fraction, seed, n_repeats)):
            tag = {"fraction": fraction, "repeat": repeat, "n_train": len(idx)}

            # --- XGBoost baseline ---
            model = train_baseline(train_df.iloc[idx], cfg)
            val_proba = predict_proba(model, val_df)
            val_pred = (val_proba >= 0.5).astype(int)
            rows.append({**tag, "model": "xgboost_baseline",
                         **compute_metrics(val_true, val_pred, val_proba)})

            # --- CNN ---
            # Early stopping must not watch `val`: choosing the stopping epoch
            # by val loss and then scoring on val would hand the CNN an
            # advantage the baseline never gets. The stopping slice is carved
            # out of this repeat's own training subset, blockwise.
            subset = train_df.iloc[idx]
            stop_mask = block_holdout_mask(
                subset["lon"].values, subset["lat"].values,
                holdout_fraction=EARLY_STOP_FRACTION,
                block_size_km=EARLY_STOP_BLOCK_KM, seed=seed + repeat)
            fit_idx, stop_idx = idx[~stop_mask], idx[stop_mask]

            # stats from this repeat's fitting patches only
            mean, std = compute_channel_stats(train_patches[fit_idx])
            train_loader = DataLoader(
                PVPatchDataset(train_patches[fit_idx], train_labels[fit_idx],
                               mean=mean, std=std, augment=True, seed=seed + repeat),
                batch_size=16, shuffle=True,
            )
            stop_loader = DataLoader(
                PVPatchDataset(train_patches[stop_idx], train_labels[stop_idx],
                               mean=mean, std=std),
                batch_size=16, shuffle=False,
            )
            val_loader = DataLoader(
                PVPatchDataset(val_patches, val_labels, mean=mean, std=std),
                batch_size=16, shuffle=False,
            )
            cnn_model = PVPatchCNN(in_channels=train_patches.shape[1],
                                   input_size=train_patches.shape[2])
            train_cnn(cnn_model, train_loader, stop_loader, epochs=cnn_epochs,
                      lr=1e-3, device=device, patience=patience)
            rows.append({**tag, "model": "cnn",
                         **evaluate_cnn(cnn_model, val_loader, device=device)})
            print(f"  fraction={fraction} repeat={repeat} n={len(idx)} done", flush=True)

    return pd.DataFrame(rows)


def summarize_curve(result: pd.DataFrame) -> pd.DataFrame:
    """Mean and standard deviation across repeats."""
    grouped = result.groupby(["fraction", "model"])[METRIC_COLUMNS]
    summary = grouped.agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    return summary.reset_index()


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=5,
                        help="independent training subsets per fraction")
    parser.add_argument("--max-epochs", type=int, default=100,
                        help="ceiling; early stopping decides the actual budget")
    parser.add_argument("--patience", type=int, default=10)
    args = parser.parse_args()

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
        cnn_epochs=args.max_epochs, n_repeats=args.repeats, patience=args.patience,
    )
    summary = summarize_curve(result)
    Path("results/logs").mkdir(parents=True, exist_ok=True)
    result.to_csv("results/logs/label_efficiency_runs.csv", index=False)
    summary.to_csv("results/logs/label_efficiency_curve.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(summary[["fraction", "model", "overall_accuracy_mean",
                       "overall_accuracy_std", "f1_mean", "f1_std"]].round(3))


if __name__ == "__main__":
    main()
