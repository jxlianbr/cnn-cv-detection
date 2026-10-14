"""Label-efficiency rerun with the main-run CNN configuration, scored on the
test split as well as on validation (review round 3, task T2).

Differences from scripts/run_label_efficiency.py (which is left unchanged and
whose logs are not touched):

* The CNN is trained through the same function as the main comparison
  (run_failure_case_evaluation._train_cnn_on_split): batch size 32, at most
  200 epochs, patience 15, Adam lr 1e-3. The settings are asserted against the
  main-run code path and checked again at run time on every call.
* Weight initialisation, batch order and augmentation are seeded per repeat
  (seed = repeat index, 0-4, as in the main comparison).
* The early-stopping slice is chosen as in the main runs: 10 km blocks
  (config split.block_size_km), 20 % of the blocks, the fixed seed
  config split.seed, applied to the training subset the CNN receives.
* Each subset is drawn with an explicit seed (default_rng(repeat), i.e. the
  same subsets as the original experiment) and kept in the original row order.
  XGBoost is trained on exactly the same subset (all of its points; the CNN
  fits on the non-stopping part, as in the main runs).
* At 100 % the subset is the full training split in its original order, so the
  five repeats are not five permutations of the same points but one run per
  seed (0-4), exactly the main comparison; XGBoost there is the main baseline.
* Both models are scored on the test and on the validation split.

Writes results/logs/label_efficiency_v2_runs.csv, label_efficiency_v2_curve.csv
and label_efficiency_v2_paired.csv. Refuses to overwrite existing files.
"""
import inspect
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root on sys.path

import numpy as np
import pandas as pd
import torch
from scipy.stats import binomtest, t as t_dist, ttest_1samp

import scripts.run_cnn_training as main_training
import scripts.run_failure_case_evaluation as main_eval
from scripts.run_label_efficiency import (subset_indices_for_repeats,
                                          summarize_curve)
from src.config import load_config
from src.evaluation.metrics import compute_metrics
from src.models.xgboost_baseline import predict_proba, train_baseline

LOGS = Path("results/logs")
PROCESSED = Path("data/processed")
FRACTIONS = [0.1, 0.25, 0.5, 0.75, 1.0]  # as in run_label_efficiency.main
N_REPEATS = 5
SUBSET_SEED = 0  # subset for repeat r drawn with default_rng(SUBSET_SEED + r)
SPLITS = ("test", "val")
PAIRED_METRICS = ("f1", "overall_accuracy")

# The main-run CNN configuration (scripts/run_cnn_training.py and
# scripts/run_failure_case_evaluation.py). Asserted below, not assumed.
MAIN_CNN = {"batch_size": 32, "epochs": 200, "patience": 15, "lr": 1e-3,
            "holdout_fraction": 0.2}


def assert_main_configuration(cfg) -> None:
    """Static check: the main-run code paths use the settings in MAIN_CNN."""
    for fn in (main_training.main, main_eval._train_cnn_on_split):
        src = inspect.getsource(fn)
        for token in (f"batch_size={MAIN_CNN['batch_size']}",
                      f"epochs={MAIN_CNN['epochs']}",
                      f"patience={MAIN_CNN['patience']}", "lr=1e-3",
                      f"holdout_fraction={MAIN_CNN['holdout_fraction']}",
                      "block_size_km=cfg.split.block_size_km", "seed=cfg.split.seed"):
            assert token in src, f"{fn.__qualname__} no longer contains {token!r}"
    assert cfg.split.block_size_km == 10 and cfg.split.seed == 42, cfg.split


class TrainingRecorder:
    """Run-time check of the CNN configuration on every training call.

    Wraps the train_cnn and block_holdout_mask that _train_cnn_on_split looks
    up in its module, asserts the arguments it passes, and records the number
    of epochs actually run and the sizes of the fitting and stopping sets.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.last = {}
        self._train_cnn = main_eval.train_cnn
        self._holdout = main_eval.block_holdout_mask

    def install(self):
        main_eval.train_cnn = self.train_cnn
        main_eval.block_holdout_mask = self.block_holdout_mask

    def block_holdout_mask(self, lon, lat, holdout_fraction, block_size_km, seed):
        assert holdout_fraction == MAIN_CNN["holdout_fraction"]
        assert block_size_km == self.cfg.split.block_size_km
        assert seed == self.cfg.split.seed
        mask = self._holdout(lon, lat, holdout_fraction=holdout_fraction,
                             block_size_km=block_size_km, seed=seed)
        self.last["n_stop"] = int(mask.sum())
        self.last["n_fit"] = int((~mask).sum())
        return mask

    def train_cnn(self, model, train_loader, val_loader, epochs, lr, device, patience):
        assert train_loader.batch_size == MAIN_CNN["batch_size"]
        assert val_loader.batch_size == MAIN_CNN["batch_size"]
        assert epochs == MAIN_CNN["epochs"] and patience == MAIN_CNN["patience"]
        assert lr == MAIN_CNN["lr"]
        assert train_loader.generator is not None, "batch order is not seeded"
        history = self._train_cnn(model, train_loader, val_loader, epochs=epochs,
                                  lr=lr, device=device, patience=patience)
        self.last["epochs_run"] = len(history["val_loss"])
        return history


def subsets_for_fraction(n_total: int, fraction: float) -> list[np.ndarray]:
    """Seeded subsets, kept in the original row order. At 100 % every repeat
    gets the full split in its original order (one run per seed)."""
    if fraction >= 1.0:
        return [np.arange(n_total) for _ in range(N_REPEATS)]
    return [np.sort(idx) for idx in
            subset_indices_for_repeats(n_total, fraction, SUBSET_SEED, N_REPEATS)]


def paired_differences(runs: pd.DataFrame) -> pd.DataFrame:
    """CNN - XGBoost per repeat, in pp; same method as run_revision_checks X-4."""
    rows = []
    for (split, fraction, n_train), g in runs.groupby(["split", "fraction", "n_train"]):
        wide = g.pivot_table(index="repeat", columns="model", values=list(PAIRED_METRICS))
        for metric in PAIRED_METRICS:
            d = 100 * (wide[(metric, "cnn")] - wide[(metric, "xgboost_baseline")]).values
            n = len(d)
            mean, sd = d.mean(), d.std(ddof=1)
            half = t_dist.ppf(0.975, n - 1) * sd / np.sqrt(n)
            positive, nonzero = int((d > 0).sum()), int((d != 0).sum())
            rows.append({"split": split, "fraction": fraction, "n_train": int(n_train),
                         "metric": metric, "n_repeats": n,
                         **{f"diff_pp_repeat{k}": v for k, v in enumerate(d)},
                         "mean_pp": mean, "sd_pp": sd,
                         "ci_low_pp": mean - half, "ci_high_pp": mean + half,
                         "paired_t_p": ttest_1samp(d, 0.0).pvalue,
                         "n_positive": positive,
                         "sign_test_p": binomtest(positive, nonzero, 0.5).pvalue
                         if nonzero else 1.0})
    return pd.DataFrame(rows)


def write_new(df: pd.DataFrame, name: str):
    assert name.startswith("label_efficiency_v2_"), name
    path = LOGS / name
    if path.exists():
        raise FileExistsError(f"{path} exists; refusing to overwrite a log")
    df.to_csv(path, index=False)
    print(f"Wrote {path}")


def main():
    start = time.time()
    cfg = load_config()
    assert_main_configuration(cfg)
    for name in ("runs", "curve", "paired"):
        if (LOGS / f"label_efficiency_v2_{name}.csv").exists():
            raise FileExistsError(f"label_efficiency_v2_{name}.csv already exists")
    recorder = TrainingRecorder(cfg)
    recorder.install()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    train_df = pd.read_parquet(PROCESSED / "train_features.parquet")
    train_patches = np.load(PROCESSED / "train_patches.npy")
    train_labels = np.load(PROCESSED / "train_labels.npy", allow_pickle=True)
    evals = {}
    for split in SPLITS:
        df = pd.read_parquet(PROCESSED / f"{split}_features.parquet")
        evals[split] = {"df": df,
                        "patches": np.load(PROCESSED / f"{split}_patches.npy"),
                        "labels": np.load(PROCESSED / f"{split}_labels.npy",
                                          allow_pickle=True),
                        "y": (df["label"] == "pv").astype(int).values}
    n_total = len(train_df)

    rows = []
    full_proba = {split: {} for split in SPLITS}  # 100 % runs, for the sanity check
    for fraction in FRACTIONS:
        for repeat, idx in enumerate(subsets_for_fraction(n_total, fraction)):
            t0 = time.time()
            sub_df = train_df.iloc[idx].reset_index(drop=True)
            tag = {"fraction": fraction, "repeat": repeat, "n_train": len(idx),
                   "subset_seed": SUBSET_SEED + repeat if fraction < 1.0 else None,
                   "train_seed": repeat}

            baseline = train_baseline(sub_df, cfg.xgboost)
            cnn, mean, std = main_eval._train_cnn_on_split(
                sub_df, train_patches[idx], train_labels[idx], cfg, device, seed=repeat)
            info = dict(recorder.last)
            assert info["n_fit"] + info["n_stop"] == len(idx)

            for split, e in evals.items():
                xgb_proba = predict_proba(baseline, e["df"])
                rows.append({**tag, "model": "xgboost_baseline", "split": split,
                             "n_fit": len(idx), "n_stop": 0, "epochs_run": None,
                             **compute_metrics(e["y"], (xgb_proba >= 0.5).astype(int),
                                               xgb_proba)})
                cnn_proba = main_eval._cnn_probabilities(cnn, e["patches"], e["labels"],
                                                         device, mean, std)
                rows.append({**tag, "model": "cnn", "split": split, **info,
                             **compute_metrics(e["y"], (cnn_proba >= 0.5).astype(int),
                                               cnn_proba)})
                if fraction >= 1.0:
                    full_proba[split][f"cnn_seed{repeat}"] = cnn_proba
                    full_proba[split]["xgb"] = xgb_proba
            print(f"fraction={fraction} repeat={repeat} n={len(idx)} "
                  f"fit={info['n_fit']} stop={info['n_stop']} "
                  f"epochs={info['epochs_run']} ({time.time() - t0:.0f} s)", flush=True)

    runs = pd.DataFrame(rows)
    curve = pd.concat([summarize_curve(g).assign(split=split)
                       for split, g in runs.groupby("split")], ignore_index=True)
    n_train = runs.groupby("fraction")["n_train"].first()
    curve.insert(1, "n_train", curve["fraction"].map(n_train))
    curve = curve[["split"] + [c for c in curve.columns if c != "split"]]
    paired = paired_differences(runs)

    LOGS.mkdir(parents=True, exist_ok=True)
    write_new(runs, "label_efficiency_v2_runs.csv")
    write_new(curve, "label_efficiency_v2_curve.csv")
    write_new(paired, "label_efficiency_v2_paired.csv")

    # Sanity: the 100 % runs are the main comparison's runs (same function,
    # seeds and data). Compare with the saved predictions of the main runs.
    print("\nSanity check against predictions_{split}.parquet (100 % runs)")
    for split in SPLITS:
        saved = pd.read_parquet(LOGS / f"predictions_{split}.parquet")
        assert (saved["point_id"].values == evals[split]["df"]["point_id"].values).all()
        xgb_diff = np.abs(saved["xgboost_baseline_proba"].values
                          - full_proba[split]["xgb"]).max()
        print(f"  {split}: XGBoost max |proba diff| = {xgb_diff:.2e}")
        for seed in range(N_REPEATS):
            ours = full_proba[split][f"cnn_seed{seed}"]
            ref = saved[f"cnn_proba_seed{seed}"].values
            agree = ((ours >= 0.5) == (ref >= 0.5)).mean()
            print(f"  {split}: CNN seed {seed} decision agreement {agree:.4f}, "
                  f"max |proba diff| {np.abs(ours - ref).max():.3f}")

    with pd.option_context("display.width", 220, "display.max_columns", 40):
        print(curve[["split", "fraction", "n_train", "model", "overall_accuracy_mean",
                     "overall_accuracy_std", "f1_mean", "f1_std"]].round(4)
              .to_string(index=False))
        print(paired[["split", "n_train", "metric", "mean_pp", "sd_pp", "ci_low_pp",
                      "ci_high_pp", "paired_t_p", "n_positive", "sign_test_p"]]
              .round(4).to_string(index=False))
    print(f"\nTotal runtime: {(time.time() - start) / 60:.1f} min")


if __name__ == "__main__":
    main()
