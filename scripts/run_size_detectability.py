"""Detection rate as a function of installation size, for both models.

Answers whether small-PV detection is a *method* problem or a *resolution*
problem. If both a pixel-wise spectral classifier and a spatial CNN degrade
together as installations shrink, the limit is the 10 m sensor, not the choice
of method -- and no amount of spatial context recovers it.

For scale: at 0.086 kW/m^2 a 50 kW installation is ~580 m^2, roughly 6 Sentinel-2
pixels. Recall is the metric of interest since every bin is PV-only.

By default this retrains both models once (unseeded). --from-predictions instead
bins the per-point predictions saved by run_failure_case_evaluation.py -- the
baseline run and every CNN seed behind the paper -- so it retrains nothing and
gives the same counts as the size-recall table from make_figures.py.
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
                                          evaluate_on_subsets,
                                          evaluate_saved_predictions)
from src.models.xgboost_baseline import predict_proba, train_baseline
from scripts.run_failure_case_evaluation import _cnn_probabilities, _train_cnn_on_split

PROCESSED = Path("data/processed")
LOGS = Path("results/logs")
DEFAULT_EDGES = [0, 100, 250, 500, 1000, 2500, np.inf]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--kw-per-m2", type=float, default=DEFAULT_KW_PER_M2)
    parser.add_argument("--from-predictions", action="store_true",
                        help="retrain nothing: bin results/logs/predictions_<split>"
                             ".parquet, one row per model run")
    parser.add_argument("--suffix", default="",
                        help='inserted before the split in the output name, e.g. "_v2"')
    parser.add_argument("--overwrite", action="store_true",
                        help="allow replacing an existing output file")
    args = parser.parse_args()

    out_csv = LOGS / f"size_detectability{args.suffix}_{args.split}.csv"
    if out_csv.exists() and not args.overwrite:
        raise SystemExit(f"Refusing to overwrite {out_csv}; pass --suffix to write a "
                         f"new file or --overwrite to replace it.")

    if args.from_predictions:
        source = LOGS / f"predictions_{args.split}.parquet"
        predictions = pd.read_parquet(source)
        print(f"Binning {source} ({len(predictions)} points, {args.kw_per_m2} kW/m^2)")
        masks = capacity_bin_masks(predictions, DEFAULT_EDGES, args.kw_per_m2)
        _write_and_print(evaluate_saved_predictions(predictions, masks), out_csv)
        return

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
    _write_and_print(pd.concat(frames, ignore_index=True), out_csv)


def _write_and_print(result: pd.DataFrame, out_csv: Path):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_csv, index=False)

    # first run of each model; with saved predictions that is CNN seed 0
    first = result[result["repeat"] == 0] if "repeat" in result else result
    pivot = first.pivot(index="subset", columns="model", values="recall").round(3)
    pivot.insert(0, "n", first[first["model"] == "cnn"].set_index("subset")["n"])
    print()
    print(pivot.reindex(list(dict.fromkeys(first["subset"]))).to_string())
    print(f"\nWrote {out_csv}")


if __name__ == "__main__":
    main()
