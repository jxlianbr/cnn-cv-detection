"""Head-to-head comparison of the spectral baseline and the CNN, in aggregate
and on the failure-case strata from Shimada & Takeuchi (2025)
(reports/paper-draft.md 3.5) -- the primary research question.

Three strata are named in the source study. Only one can be derived
automatically:

  sub_50kw               area-based proxy, from the released footprints
  non_crystalline_silicon  manual annotation required
  spectrally_ambiguous     manual annotation required

The two manual strata are read from a hand-annotated CSV if one exists at
--manual-flags; without it they are simply absent from the report rather than
guessed at. See the design note in src/evaluation/failure_cases.py.

Evaluates on the validation split by default. Pass --split test only when
producing final numbers: that split is meant to be looked at once.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root on sys.path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.config import load_config
from src.data.patch_dataset import PVPatchDataset, compute_channel_stats
from src.data.split import block_holdout_mask
from src.evaluation.failure_cases import (
    DEFAULT_KW_PER_M2, estimate_capacity_kw, evaluate_on_subsets,
    load_manual_failure_flags,
)
from src.models.cnn import PVPatchCNN
from src.models.train_cnn import train_cnn
from src.models.xgboost_baseline import predict_proba, train_baseline

PROCESSED = Path("data/processed")


def build_subset_masks(eval_df: pd.DataFrame, kw_per_m2: float,
                       manual_flags: pd.DataFrame | None) -> dict[str, np.ndarray]:
    """Named boolean strata over the evaluation frame.

    Points with no footprint (the non-PV class) belong to neither capacity
    stratum: an unknown size is not a small size. Manual strata appear only
    when annotations were supplied.
    """
    n = len(eval_df)
    masks = {"all": np.ones(n, dtype=bool)}

    area = eval_df["polygon_area_m2"].to_numpy(dtype=float)
    sized = ~np.isnan(area)
    capacity_kw = np.full(n, np.nan)
    capacity_kw[sized] = estimate_capacity_kw(area[sized], kw_per_m2)
    masks["sub_50kw"] = sized & (capacity_kw < 50.0)
    masks["over_50kw"] = sized & (capacity_kw >= 50.0)

    if manual_flags is not None:
        flags = eval_df[["point_id"]].merge(manual_flags, on="point_id", how="left")
        for column in ("non_crystalline_silicon", "spectrally_ambiguous"):
            masks[column] = flags[column].fillna(False).to_numpy(dtype=bool)
    return masks


def _train_cnn_on_split(train_df, train_patches, train_labels, cfg, device):
    stop_mask = block_holdout_mask(train_df["lon"].values, train_df["lat"].values,
                                   holdout_fraction=0.2,
                                   block_size_km=cfg.split.block_size_km,
                                   seed=cfg.split.seed)
    mean, std = compute_channel_stats(train_patches[~stop_mask])
    fit_loader = DataLoader(
        PVPatchDataset(train_patches[~stop_mask], train_labels[~stop_mask],
                       mean=mean, std=std, augment=True, seed=cfg.split.seed),
        batch_size=32, shuffle=True)
    stop_loader = DataLoader(
        PVPatchDataset(train_patches[stop_mask], train_labels[stop_mask],
                       mean=mean, std=std),
        batch_size=32, shuffle=False)
    model = PVPatchCNN(in_channels=train_patches.shape[1],
                       input_size=train_patches.shape[2])
    history = train_cnn(model, fit_loader, stop_loader, epochs=200, lr=1e-3,
                        device=device, patience=15)
    print(f"  CNN stopped after {len(history['val_loss'])} epochs")
    return model, mean, std


def _cnn_probabilities(model, patches, labels, device, mean, std) -> np.ndarray:
    loader = DataLoader(PVPatchDataset(patches, labels, mean=mean, std=std),
                        batch_size=32, shuffle=False)
    model.to(device).eval()
    out = []
    with torch.no_grad():
        for x, _ in loader:
            logits = model(x.to(device))
            out.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
    return np.concatenate(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--manual-flags", default="data/external/manual_failure_flags.csv")
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
    print(f"Evaluating on the {args.split} split ({len(eval_df)} points)")

    if "polygon_area_m2" not in eval_df.columns:
        raise RuntimeError(
            "No polygon_area_m2 column -- rebuild the dataset with "
            "scripts/build_dataset.py so footprint areas are attached.")

    manual_path = Path(args.manual_flags)
    manual_flags = None
    if manual_path.exists():
        manual_flags = load_manual_failure_flags(manual_path)
        print(f"Loaded {len(manual_flags)} manual annotations from {manual_path}")
    else:
        print(f"No manual annotations at {manual_path}; reporting the "
              f"area-based stratum only")

    y_true = (eval_df["label"] == "pv").astype(int).values

    print("Training spectral baseline...")
    baseline = train_baseline(train_df, cfg.xgboost)
    baseline_proba = predict_proba(baseline, eval_df)

    print("Training CNN...")
    cnn, mean, std = _train_cnn_on_split(train_df, train_patches, train_labels,
                                         cfg, device)
    cnn_proba = _cnn_probabilities(cnn, eval_patches, eval_labels, device, mean, std)

    masks = build_subset_masks(eval_df, args.kw_per_m2, manual_flags)
    frames = []
    for name, proba in (("xgboost_baseline", baseline_proba), ("cnn", cnn_proba)):
        report = evaluate_on_subsets(y_true, (proba >= 0.5).astype(int), proba, masks)
        report.insert(0, "model", name)
        frames.append(report)
    result = pd.concat(frames, ignore_index=True)

    Path("results/logs").mkdir(parents=True, exist_ok=True)
    out_csv = Path("results/logs") / f"failure_case_comparison_{args.split}.csv"
    result.to_csv(out_csv, index=False)

    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print()
        print(result[["subset", "model", "n", "n_pv", "overall_accuracy",
                      "recall", "f1"]].round(3).to_string(index=False))
    print(f"\nWrote {out_csv}")


if __name__ == "__main__":
    main()
