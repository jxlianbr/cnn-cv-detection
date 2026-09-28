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
from src.evaluation.metrics import mcnemar_test
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


def _train_cnn_on_split(train_df, train_patches, train_labels, cfg, device,
                        seed: int | None = None):
    """`seed` fixes weight init, batch order and augmentation, so repeats differ
    only in training stochasticity. The early-stopping slice stays the same
    across repeats. None keeps the old unseeded behaviour."""
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    stop_mask = block_holdout_mask(train_df["lon"].values, train_df["lat"].values,
                                   holdout_fraction=0.2,
                                   block_size_km=cfg.split.block_size_km,
                                   seed=cfg.split.seed)
    mean, std = compute_channel_stats(train_patches[~stop_mask])
    generator = torch.Generator().manual_seed(seed) if seed is not None else None
    fit_loader = DataLoader(
        PVPatchDataset(train_patches[~stop_mask], train_labels[~stop_mask],
                       mean=mean, std=std, augment=True,
                       seed=cfg.split.seed if seed is None else seed),
        batch_size=32, shuffle=True, generator=generator)
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
    parser.add_argument("--repeats", type=int, default=5,
                        help="CNN training runs with seeds 0..N-1; the baseline "
                             "is deterministic and trained once")
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

    masks = build_subset_masks(eval_df, args.kw_per_m2, manual_flags)
    baseline_pred = (baseline_proba >= 0.5).astype(int)
    report = evaluate_on_subsets(y_true, baseline_pred, baseline_proba, masks)
    report.insert(0, "repeat", 0)
    report.insert(0, "model", "xgboost_baseline")
    frames = [report]

    # one row per evaluation point, so later analyses (paired tests, size bins)
    # use exactly these runs instead of retraining
    predictions = eval_df[["point_id", "label", "polygon_area_m2", "lon", "lat"]].copy()
    predictions["xgboost_baseline_proba"] = baseline_proba
    tests = []
    for repeat in range(args.repeats):
        print(f"Training CNN (seed {repeat})...")
        cnn, mean, std = _train_cnn_on_split(train_df, train_patches, train_labels,
                                             cfg, device, seed=repeat)
        cnn_proba = _cnn_probabilities(cnn, eval_patches, eval_labels, device,
                                       mean, std)
        cnn_pred = (cnn_proba >= 0.5).astype(int)
        predictions[f"cnn_proba_seed{repeat}"] = cnn_proba
        report = evaluate_on_subsets(y_true, cnn_pred, cnn_proba, masks)
        report.insert(0, "repeat", repeat)
        report.insert(0, "model", "cnn")
        frames.append(report)
        test = mcnemar_test(y_true, baseline_pred, cnn_pred)
        tests.append({"repeat": repeat,
                      "xgboost_only_correct": test["a_only_correct"],
                      "cnn_only_correct": test["b_only_correct"],
                      "p_value": test["p_value"]})
    result = pd.concat(frames, ignore_index=True)
    tests = pd.DataFrame(tests)

    logs = Path("results/logs")
    logs.mkdir(parents=True, exist_ok=True)
    out_csv = logs / f"failure_case_comparison_{args.split}.csv"
    result.to_csv(out_csv, index=False)
    predictions.to_parquet(logs / f"predictions_{args.split}.parquet", index=False)
    tests.to_csv(logs / f"mcnemar_{args.split}.csv", index=False)

    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print()
        print(result[result["subset"] == "all"][
            ["model", "repeat", "n", "overall_accuracy", "precision", "recall",
             "f1"]].round(4).to_string(index=False))
        print("\nExact McNemar test, baseline vs CNN (same points)")
        print(tests.to_string(index=False))
    print(f"\nWrote {out_csv}, predictions_{args.split}.parquet and "
          f"mcnemar_{args.split}.csv to {logs}/")


if __name__ == "__main__":
    main()
