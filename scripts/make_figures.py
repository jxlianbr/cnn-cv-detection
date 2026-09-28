"""Figures and summary numbers for the Results chapter (reports/paper-draft.md 5).

Reads only what the experiment scripts already wrote to results/logs/ plus the
polygon release and the processed sample, so it retrains nothing:

    fig_size_distribution.pdf   capacity distribution, released polygons vs sample
    fig_label_efficiency.pdf    F1 / precision / recall against training-set size
    fig_size_recall.pdf         recall per capacity bin, val and test, Wilson 95% CI

and the chapter's LaTeX tables (booktabs) into a sibling tables/ directory.
Also prints the Wilson intervals quoted in the text, so every number in the
chapter can be traced back to a command.
"""
import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root on sys.path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

from src.evaluation.failure_cases import (AREA_CRS, DEFAULT_KW_PER_M2,
                                          capacity_bin_masks)

LOGS = Path("results/logs")
PROCESSED = Path("data/processed")

# two-series palette, validated for CVD separation (dataviz validator, light mode);
# marker and line style carry identity too, so the figures survive greyscale print
MODELS = {
    "xgboost_baseline": {"label": "XGBoost + spectral indices", "color": "#2a78d6",
                         "marker": "o", "linestyle": "-"},
    "cnn": {"label": "CNN (6-band patches)", "color": "#eb6834",
            "marker": "s", "linestyle": "--"},
}
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
TEXT_WIDTH_IN = 5.9  # ~15 cm, A4 with 3 cm margins

SIZE_BINS = ["0-100kW", "100-250kW", "250-500kW", "500-1000kW",
             "1000-2500kW", ">2500kW"]
SIZE_BIN_LABELS = ["<0.1", "0.1–\n0.25", "0.25–\n0.5", "0.5–1", "1–2.5", ">2.5"]


def _style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 8.5,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK_2,
        "axes.titlesize": 9,
        "axes.titlecolor": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK_2,
        "ytick.labelcolor": INK_2,
        "legend.frameon": False,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,  # embed TrueType so the PDF stays text-selectable
    })


def wilson_interval(successes: np.ndarray, n: np.ndarray, level: float = 0.95):
    """Wilson score interval for a binomial proportion; stable at p near 1."""
    successes = np.asarray(successes, dtype=float)
    n = np.asarray(n, dtype=float)
    z = norm.ppf(0.5 + level / 2)
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return centre - half, centre + half


def fig_size_distribution(out: Path):
    polygons = gpd.read_file("data/external/pv_polygons.gpkg")
    population_kw = polygons.to_crs(AREA_CRS).geometry.area.values * DEFAULT_KW_PER_M2
    sample = pd.concat(pd.read_parquet(PROCESSED / f"{s}_features.parquet")
                       for s in ("train", "val", "test"))
    sample_kw = sample["polygon_area_m2"].dropna().values * DEFAULT_KW_PER_M2

    bins = np.logspace(1, 5.5, 46)
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH_IN, 2.4))
    for values, label, color, ls in (
            (population_kw, f"Released polygons (n = {len(population_kw):,})",
             MODELS["xgboost_baseline"]["color"], "-"),
            (sample_kw, f"Sampled PV points (n = {len(sample_kw):,})",
             MODELS["cnn"]["color"], "--")):
        weights = np.full(len(values), 100.0 / len(values))
        ax.hist(values, bins=bins, weights=weights, histtype="step", linewidth=1.6,
                color=color, linestyle=ls, label=label)
    ax.axvline(50, color=INK_2, linewidth=0.9, linestyle=":")
    ax.text(50 * 0.88, ax.get_ylim()[1] * 0.97, "50 kW", color=INK_2, va="top",
            ha="right")
    ax.set_xscale("log")
    ax.set_xlim(10, 10**5.5)
    ax.set_xlabel(f"Estimated capacity (kW, footprint × {DEFAULT_KW_PER_M2} kW/m²)")
    ax.set_ylabel("Share of items (%)")
    ax.legend(loc="upper right")
    fig.savefig(out / "fig_size_distribution.pdf")
    plt.close(fig)

    print("Size distribution")
    for name, kw in (("population", population_kw), ("sample", sample_kw)):
        print(f"  {name}: n={len(kw)}, median={np.median(kw):.0f} kW, "
              f"<50 kW={np.sum(kw < 50)} ({100 * np.mean(kw < 50):.2f}%)")


def fig_label_efficiency(out: Path):
    runs = pd.read_csv(LOGS / "label_efficiency_runs.csv")
    curve = pd.read_csv(LOGS / "label_efficiency_curve.csv")
    n_by_fraction = runs.groupby("fraction")["n_train"].first()
    curve["n_train"] = curve["fraction"].map(n_by_fraction)

    metrics = (("f1", "F1"), ("precision", "Precision"), ("recall", "Recall"))
    fig, axes = plt.subplots(1, 3, figsize=(TEXT_WIDTH_IN, 2.5), sharey=True)
    for ax, (metric, title) in zip(axes, metrics):
        for model, spec in MODELS.items():
            c = curve[curve["model"] == model].sort_values("n_train")
            r = runs[runs["model"] == model]
            mean, std = c[f"{metric}_mean"].values, c[f"{metric}_std"].values
            ax.fill_between(c["n_train"], mean - std, mean + std,
                            color=spec["color"], alpha=0.15, linewidth=0)
            ax.scatter(r["n_train"], r[metric], s=6, color=spec["color"],
                       alpha=0.35, linewidths=0)
            ax.plot(c["n_train"], mean, color=spec["color"], linewidth=1.6,
                    linestyle=spec["linestyle"], marker=spec["marker"],
                    markersize=4, label=spec["label"])
        ax.set_title(title, loc="left")
        ax.set_xlabel("Training points")
        # label every other size; the unlabelled ticks still mark the fractions
        ax.set_xticks(n_by_fraction.values)
        ax.set_xticklabels([f"{n:,}" if i % 2 == 0 else ""
                            for i, n in enumerate(n_by_fraction.values)])
    axes[0].set_ylabel("Validation score")
    axes[0].set_ylim(0.935, 1.0)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2,
               bbox_to_anchor=(0.5, 1.07))
    fig.savefig(out / "fig_label_efficiency.pdf")
    plt.close(fig)


SIZE_EDGES_KW = [0, 100, 250, 500, 1000, 2500, np.inf]


def size_bin_hits(split: str) -> pd.DataFrame:
    """Per capacity bin: n, baseline hits, and CNN hits for every seed, all from
    the predictions saved by run_failure_case_evaluation.py."""
    pred = pd.read_parquet(LOGS / f"predictions_{split}.parquet")
    masks = capacity_bin_masks(pred, SIZE_EDGES_KW, DEFAULT_KW_PER_M2)
    seeds = sorted(c for c in pred.columns if c.startswith("cnn_proba_seed"))
    rows = []
    for b in SIZE_BINS:
        m = masks[b]
        row = {"bin": b, "n": int(m.sum()),
               "xgboost_baseline": int((pred.loc[m, "xgboost_baseline_proba"] >= 0.5).sum())}
        for col in seeds:
            row[col] = int((pred.loc[m, col] >= 0.5).sum())
        rows.append(row)
    return pd.DataFrame(rows).set_index("bin")


def fig_size_recall(out: Path):
    """Baseline: one deterministic run. CNN: mean over seeds (marker) and the
    individual seeds (small dots). Error bars are Wilson intervals over the
    points in the bin, i.e. sampling uncertainty, not training spread."""
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH_IN, 2.7), sharey=True)
    offsets = {"xgboost_baseline": -0.13, "cnn": 0.13}
    print("Per-bin recall, Wilson 95% CI (CNN: mean over seeds)")
    for ax, split in zip(axes, ("val", "test")):
        hits = size_bin_hits(split)
        seeds = [c for c in hits.columns if c.startswith("cnn_proba_seed")]
        n = hits["n"].values
        x = np.arange(len(SIZE_BINS))
        series = {"xgboost_baseline": hits["xgboost_baseline"].values,
                  "cnn": hits[seeds].mean(axis=1).values}
        for model, spec in MODELS.items():
            r = series[model] / n
            lo, hi = wilson_interval(series[model], n)
            ax.errorbar(x + offsets[model], r, yerr=[r - lo, hi - r],
                        fmt=spec["marker"], color=spec["color"], markersize=4.5,
                        elinewidth=1.1, capsize=2, label=spec["label"], zorder=3)
            for b, rr, a, z in zip(SIZE_BINS, r, lo, hi):
                print(f"  {split:4s} {model:16s} {b:12s} {rr:.3f} [{a:.3f}, {z:.3f}]")
        for col in seeds:
            ax.scatter(x + offsets["cnn"] + 0.1, hits[col].values / n, s=5,
                       color=MODELS["cnn"]["color"], alpha=0.5, linewidths=0, zorder=2)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{lab}\nn={k}" for lab, k in zip(SIZE_BIN_LABELS, n)],
                           fontsize=6.5)
        ax.set_title("Validation split" if split == "val" else "Test split",
                     loc="left")
        ax.set_xlabel("Estimated capacity (MW)")
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Recall (share of PV points detected)")
    axes[0].set_ylim(0.45, 1.01)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2,
               bbox_to_anchor=(0.5, 1.06))
    fig.savefig(out / "fig_size_recall.pdf")
    plt.close(fig)


def _aggregate(split: str) -> pd.DataFrame:
    df = pd.read_csv(LOGS / f"failure_case_comparison_{split}.csv")
    return df[df["subset"] == "all"]


def print_aggregate_intervals():
    print("Aggregate overall accuracy, Wilson 95% CI")
    for split in ("val", "test"):
        for _, row in _aggregate(split).iterrows():
            lo, hi = wilson_interval(round(row.overall_accuracy * row.n), row.n)
            print(f"  {split:4s} {row.model:16s} seed {row['repeat']} "
                  f"OA={row.overall_accuracy:.4f} [{float(lo):.4f}, {float(hi):.4f}]")
    for split in ("val", "test"):
        print(f"McNemar ({split})")
        print(pd.read_csv(LOGS / f"mcnemar_{split}.csv").to_string(index=False))


METRIC_LABELS = [("overall_accuracy", "Overall accuracy"), ("precision", "Precision"),
                 ("recall", "Recall"), ("f1", "F1"), ("kappa", "Cohen's $\\kappa$"),
                 ("roc_auc", "ROC-AUC")]
# Shimada & Takeuchi (2025), post-processed map on their 1,500-point
# verification sample (papers/spectral-baseline-specifications.md 6)
PUBLISHED = {"overall_accuracy": 0.984, "precision": 0.999, "recall": 0.969,
             "f1": 0.984, "kappa": 0.968, "roc_auc": 0.9497}


def _write(path: Path, lines: list[str]):
    # a leading hyphen before a number prints as a hyphen, not a minus sign
    text = re.sub(r"(?<=[\s\[])-(?=\d)", "$-$", "\n".join(lines))
    path.write_text(text + "\n", encoding="utf-8")


def tab_replication(out: Path):
    test = pd.read_csv(LOGS / "failure_case_comparison_test.csv")
    row = test[(test["model"] == "xgboost_baseline") & (test["subset"] == "all")].iloc[0]
    lines = [r"\begin{tabular}{lrr}", r"\toprule",
             r"Metric & Published & Replicated \\",
             r"\midrule"]
    for key, label in METRIC_LABELS:
        lines.append(f"{label} & {PUBLISHED[key]:.3f} & {row[key]:.3f} \\\\")
    lines += [r"\midrule",
              f"Evaluation points & 1,500 & {int(row.n):,} \\\\",
              r"\bottomrule", r"\end{tabular}"]
    _write(out / "tab_replication.tex", lines)


def tab_aggregate(out: Path):
    """Baseline: deterministic, one run. CNN: mean and sd over the seeded runs.
    Bold marks the higher value (CNN by its mean)."""
    frames = {s: _aggregate(s) for s in ("val", "test")}
    lines = [r"\begin{tabular}{lcccc}", r"\toprule",
             r"& \multicolumn{2}{c}{Validation} & \multicolumn{2}{c}{Test} \\",
             r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
             r"Metric & XGBoost & CNN & XGBoost & CNN \\", r"\midrule"]
    for key, label in METRIC_LABELS:
        cells = []
        for split in ("val", "test"):
            df = frames[split]
            base = df[df["model"] == "xgboost_baseline"][key].iloc[0]
            cnn = df[df["model"] == "cnn"][key]
            b_txt, c_mean = f"{base:.3f}", f"{cnn.mean():.3f}"
            if round(cnn.mean(), 3) > round(base, 3):
                c_mean = f"\\textbf{{{c_mean}}}"
            elif round(base, 3) > round(cnn.mean(), 3):
                b_txt = f"\\textbf{{{b_txt}}}"
            cells += [b_txt, f"{c_mean} $\\pm$ {cnn.std():.3f}"]
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")
    n = [int(frames[s]["n"].iloc[0]) for s in ("val", "test")]
    n_pv = [int(frames[s]["n_pv"].iloc[0]) for s in ("val", "test")]
    lines += [r"\midrule",
              f"Points (PV) & \\multicolumn{{2}}{{c}}{{{n[0]:,} ({n_pv[0]:,})}} & "
              f"\\multicolumn{{2}}{{c}}{{{n[1]:,} ({n_pv[1]:,})}} \\\\",
              r"\bottomrule", r"\end{tabular}"]
    _write(out / "tab_aggregate.tex", lines)


def tab_context_control(out: Path):
    """From run_context_control.py: CNN against raw and 3x3-pooled XGBoost."""
    df = pd.read_csv(LOGS / "context_control.csv")
    t = df["threshold"].iloc[0]
    order = [("test", "CNN vs raw XGBoost (t=0.5)", "Raw, per pixel", "0.50"),
             ("test", f"CNN vs 3x3 pooled XGBoost (t={t})", "3$\\times$3 pooled", f"{t:.2f}$^{{a}}$"),
             ("test", "CNN vs 3x3 pooled XGBoost (t=0.5)", "3$\\times$3 pooled", "0.50"),
             ("val", "CNN vs raw XGBoost (t=0.5)", "Raw, per pixel", "0.50"),
             ("val", "CNN vs 3x3 pooled XGBoost (t=0.5)", "3$\\times$3 pooled", "0.50")]
    lines = [r"\begin{tabular}{llcccc}", r"\toprule",
             r"Split & XGBoost variant & Threshold & Accuracy & CNN $-$ XGBoost (pp) & McNemar $p$ \\",
             r"\midrule"]
    for split, comparison, name, thr in order:
        r = df[(df["split"] == split) & (df["comparison"] == comparison)].iloc[0]
        p = (f"{r.p_min:.3f}--{r.p_max:.3f}" if r.p_min >= 0.001
             else f"$<$0.001--{r.p_max:.3f}")
        lines.append(f"{'Test' if split == 'test' else 'Validation'} & {name} & {thr} & "
                     f"{r.baseline_accuracy:.3f} & {r.diff_pp:+.2f} [{r.ci_low_pp:+.2f}, "
                     f"{r.ci_high_pp:+.2f}] & {p} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    _write(out / "tab_context_control.tex", lines)


def tab_error_location(out: Path):
    df = pd.read_csv(LOGS / "error_location.csv")
    strata = df[df["stratum"].str.contains("edge|any PV") & df["n"].notna()
                & ~df["stratum"].str.startswith("CNN-only")]
    # the pooled threshold was tuned on val, so only the CNN is reported there
    lines = [r"\begin{tabular}{lrcccrrr}", r"\toprule",
             r"& \multicolumn{6}{c}{Test} & Validation \\",
             r"\cmidrule(lr){2-7}\cmidrule(lr){8-8}",
             r"& & \multicolumn{3}{c}{Accuracy} & \multicolumn{2}{c}{Net gain} & Net gain \\",
             r"\cmidrule(lr){3-5}\cmidrule(lr){6-7}\cmidrule(lr){8-8}",
             r"Location & $n$ & Raw & Pooled & CNN & Pooled & CNN & CNN \\",
             r"\midrule"]
    names = strata["stratum"].unique()
    for name in names:
        t = strata[(strata["split"] == "test") & (strata["stratum"] == name)].iloc[0]
        v = strata[(strata["split"] == "val") & (strata["stratum"] == name)].iloc[0]
        label = (name.replace(">=", "$\\geq$").replace("<=", "$\\leq$")
                 .replace("< ", "$<$ ").replace("> ", "$>$ ").replace("20-50", "20--50"))
        lines.append(f"{label} & {int(t.n):,} & {t.xgboost_accuracy:.3f} & {t.pooled_accuracy:.3f} & "
                     f"{t.cnn_accuracy:.3f} & {t.pooled_net_gain:+.1f} & {t.cnn_net_gain:+.1f} & "
                     f"{v.cnn_net_gain:+.1f} \\\\")
    tot = {s: strata[strata["split"] == s][["pooled_net_gain", "cnn_net_gain"]].sum()
           for s in ("test", "val")}
    lines += [r"\midrule",
              f"Total & & & & & {tot['test'].pooled_net_gain:+.1f} & {tot['test'].cnn_net_gain:+.1f} & "
              f"{tot['val'].cnn_net_gain:+.1f} \\\\",
              r"\bottomrule", r"\end{tabular}"]
    _write(out / "tab_error_location.tex", lines)


def tab_mcnemar(out: Path):
    tests = {s: pd.read_csv(LOGS / f"mcnemar_{s}.csv").set_index("repeat")
             for s in ("val", "test")}
    lines = [r"\begin{tabular}{crrrrrr}", r"\toprule",
             r"& \multicolumn{3}{c}{Validation} & \multicolumn{3}{c}{Test} \\",
             r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
             r"CNN seed & XGB only & CNN only & $p$ & XGB only & CNN only & $p$ \\",
             r"\midrule"]
    for seed in tests["test"].index:
        cells = []
        for split in ("val", "test"):
            t = tests[split].loc[seed]
            p = "$<$0.001" if t.p_value < 0.001 else f"{t.p_value:.3f}"
            cells += [str(int(t.xgboost_only_correct)), str(int(t.cnn_only_correct)), p]
        lines.append(f"{seed} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    _write(out / "tab_mcnemar.tex", lines)


def tab_size_recall(out: Path):
    """Baseline recall with its hit count; CNN mean recall over seeds with the
    range of hit counts across seeds."""
    hits = {s: size_bin_hits(s) for s in ("val", "test")}
    labels = [lab.replace("\n", "") for lab in SIZE_BIN_LABELS]
    labels = [lab.replace("<", "$<$").replace(">", "$>$") for lab in labels]
    lines = [r"\begin{tabular}{lrrrrrr}", r"\toprule",
             r"& \multicolumn{3}{c}{Validation} & \multicolumn{3}{c}{Test} \\",
             r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
             r"Capacity (MW) & $n$ & XGBoost & CNN & $n$ & XGBoost & CNN \\",
             r"\midrule"]
    for b, label in zip(SIZE_BINS, labels):
        cells = []
        for split in ("val", "test"):
            h = hits[split].loc[b]
            cnn = h[[c for c in h.index if c.startswith("cnn_proba_seed")]].astype(int)
            n, k = int(h["n"]), int(h["xgboost_baseline"])
            cells += [str(n), f"{k / n:.3f} ({k})",
                      f"{cnn.mean() / n:.3f} ({cnn.min()}--{cnn.max()})"]
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    _write(out / "tab_size_recall.tex", lines)


def tab_label_efficiency(out: Path):
    runs = pd.read_csv(LOGS / "label_efficiency_runs.csv")
    curve = pd.read_csv(LOGS / "label_efficiency_curve.csv")
    n_by_fraction = runs.groupby("fraction")["n_train"].first()
    lines = [r"\begin{tabular}{rrcccc}", r"\toprule",
             r"& & \multicolumn{2}{c}{F1} & \multicolumn{2}{c}{Recall} \\",
             r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
             r"Share & Points & XGBoost & CNN & XGBoost & CNN \\", r"\midrule"]
    for fraction, n in n_by_fraction.items():
        c = curve[curve["fraction"] == fraction].set_index("model")
        cells = [f"{c.loc[m, f'{k}_mean']:.3f} $\\pm$ {c.loc[m, f'{k}_std']:.3f}"
                 for k in ("f1", "recall") for m in ("xgboost_baseline", "cnn")]
        lines.append(f"{fraction:.0%}".replace("%", r"\%") + f" & {n:,} & "
                     + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    _write(out / "tab_label_efficiency.tex", lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="reports/latex/figures")
    parser.add_argument("--skip-population", action="store_true",
                        help="skip the slow polygon-area figure")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    _style()
    if not args.skip_population:
        fig_size_distribution(out)
    fig_label_efficiency(out)
    fig_size_recall(out)
    tables = out.parent / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    tab_replication(tables)
    tab_aggregate(tables)
    tab_size_recall(tables)
    tab_mcnemar(tables)
    tab_context_control(tables)
    tab_error_location(tables)
    tab_label_efficiency(tables)
    print_aggregate_intervals()
    print(f"Wrote figures to {out}/ and tables to {tables}/")


if __name__ == "__main__":
    main()
