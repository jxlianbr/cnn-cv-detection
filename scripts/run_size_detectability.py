"""Detection rate as a function of installation size, for both models.

Answers whether small-PV detection is a *method* problem or a *resolution*
problem. If both a pixel-wise spectral classifier and a spatial CNN degrade
together as installations shrink, the limit is the 10 m sensor, not the choice
of method -- and no amount of spatial context recovers it.

For scale: at 0.086 kW/m^2 a 50 kW installation is ~580 m^2, roughly 6 Sentinel-2
pixels. Recall is the metric of interest since every bin is PV-only.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root on sys.path

import numpy as np
import pandas as pd
import torch

from src.config import load_config
from src.evaluation.failure_cases import (DEFAULT_KW_PER_M2, capacity_bin_masks,
                                          evaluate_on_subsets)
from src.models.xgboost_baseline import predict_proba, train_baseline
from scripts.run_failure_case_evaluation import _cnn_probabilities, _train_cnn_on_split

PROCESSED = Path("data/processed")
DEFAULT_EDGES = [0, 100, 250, 500, 1000, 2500, np.inf]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--kw-per-m2", type=float, default=DEFAULT_KW_PER_M2)
    args = parser.parse_args()

    cfg = load_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_df = pd.read_parquet(PROCESSED / "train_features.parquet")
    train_patches = np.load(PROCESSED / "train_patches.npy")
    train_labels = np.load(PROCESSED / "train_labels.npy", allow_pickle=True)
    eval_df = pd.read_parquet(PROCESSED / f"{args.split}_features.parquet")
    eval_patches = np.load(PROCESSED / f"{args.split}_patches.npy")
    eval_labels = np.load(PROCESSED / f"{args.split}_labels.npy", allow_pickle=True)

    y_true = (eval_df["label"] == "pv").astype(int).values
    print(f"Evaluating on {args.split} ({len(eval_df)} points, {y_true.sum()} PV)")

    print("Training spectral baseline...")
    baseline_proba = predict_proba(train_baseline(train_df, cfg.xgboost), eval_df)
    print("Training CNN...")
    cnn, mean, std = _train_cnn_on_split(train_df, train_patches, train_labels,
                                         cfg, device)
    cnn_proba = _cnn_probabilities(cnn, eval_patches, eval_labels, device, mean, std)

    masks = capacity_bin_masks(eval_df, DEFAULT_EDGES, args.kw_per_m2)
    frames = []
    for name, proba in (("xgboost_baseline", baseline_proba), ("cnn", cnn_proba)):
        report = evaluate_on_subsets(y_true, (proba >= 0.5).astype(int), proba, masks)
        report.insert(0, "model", name)
        frames.append(report)
    result = pd.concat(frames, ignore_index=True)

    Path("results/logs").mkdir(parents=True, exist_ok=True)
    out_csv = Path("results/logs") / f"size_detectability_{args.split}.csv"
    result.to_csv(out_csv, index=False)

    pivot = result.pivot(index="subset", columns="model", values="recall").round(3)
    counts = result[result["model"] == "cnn"].set_index("subset")["n"]
    pivot.insert(0, "n", counts)
    print()
    print(pivot.to_string())
    print(f"\nWrote {out_csv}")


if __name__ == "__main__":
    main()
