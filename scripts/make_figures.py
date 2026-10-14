"""Figures and summary numbers for the Results chapter (reports/paper-draft.md 5).

Reads only what the experiment scripts already wrote to results/logs/ plus the
polygon release and the processed sample, so it retrains nothing:

    fig_size_distribution.pdf   capacity distribution, released polygons vs sample
    fig_label_efficiency.pdf    test and validation F1 against training-set size
    fig_size_recall.pdf         recall per capacity bin, val and test, Wilson 95% CI

and the chapter's LaTeX tables (booktabs) into a sibling tables/ directory.
With --review-figures DIR it also draws, into DIR,

    fig_study_area.pdf          sampling squares per split on Natural Earth 1:10m
    fig_example_patches.pdf     seven test patches chosen to illustrate Chapter 5

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
    ax.axvline(50, color=INK_2, linewidth=0.9, linestyle=":", label="50 kW")
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
    """F1 on the test split (left) and the validation split (right) against the
    number of training points, from run_label_efficiency_v2.py: mean of five
    repeats (line), +- 1 sd (band) and the individual repeats (dots)."""
    runs = pd.read_csv(LOGS / "label_efficiency_v2_runs.csv")
    curve = pd.read_csv(LOGS / "label_efficiency_v2_curve.csv")
    n_by_fraction = runs.groupby("fraction")["n_train"].first()

    fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH_IN, 2.1), sharey=True)
    for ax, (split, title) in zip(axes, (("test", "Test split"), ("val", "Validation split"))):
        for model, spec in MODELS.items():
            c = curve[(curve["model"] == model) & (curve["split"] == split)].sort_values("n_train")
            r = runs[(runs["model"] == model) & (runs["split"] == split)]
            mean, std = c["f1_mean"].values, c["f1_std"].values
            ax.fill_between(c["n_train"], mean - std, mean + std,
                            color=spec["color"], alpha=0.15, linewidth=0)
            ax.scatter(r["n_train"], r["f1"], s=6, color=spec["color"],
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
    axes[0].set_ylabel("F1")
    lo, hi = runs["f1"].min(), runs["f1"].max()
    pad = 0.05 * (hi - lo)
    axes[0].set_ylim(lo - pad, hi + pad)  # every repeat stays visible
    # legend in the empty lower right of the validation panel keeps the figure low
    axes[1].legend(loc="lower right", fontsize=7.5, handlelength=2.6)
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
    """Baseline: a single run with the original row order. CNN: mean over seeds (marker) and the
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
    axes[0].set_ylim(0.70, 1.01)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2,
               bbox_to_anchor=(0.5, 1.06))
    fig.savefig(out / "fig_size_recall.pdf")
    plt.close(fig)


# --- study-area map and example patches (review round; not part of the default run) ---

NATURAL_EARTH = Path("data/external/naturalearth")
RAW_SQUARES = Path("data/raw/japan")  # one GeoTIFF per sampling square (build_dataset.py)
AOI_BBOX = (129.0, 31.0, 146.0, 46.0)  # configs/config.yaml aoi.bbox
SPLITS = (("train", "Training"), ("val", "Validation"), ("test", "Test"))
# stratum identity: shape + fill + hue (slots 7/3/4 of the dataviz palette; the
# three validate all-pairs for CVD, and shape alone carries it in greyscale)
STRATA = {
    "pv": {"label": "PV-centred square", "marker": "o", "color": "#4a3aa7", "size": 7},
    "buffer": {"label": "Buffer square", "marker": "^", "color": "#1baf7a", "size": 9},
    "urban": {"label": "“Urban” square", "marker": "s", "color": "#eda100",
              "size": 7},
}
LAND = "#efeeea"
LAND_OTHER = "#f7f7f5"
OUTLINE = "#898781"


def outside_japan_boxes():
    """The boxes of scripts/run_revision_checks.outside_japan (the Japan-only
    check) as one geometry: lon < 139.3 & lat > 41.6, lon < 130.9 & lat > 34.8,
    and lat > 45.56, clipped to the study area."""
    from shapely.geometry import box
    from shapely.ops import unary_union
    w, s, e, n = AOI_BBOX
    return unary_union([box(w, 41.6, 139.3, n), box(w, 34.8, 130.9, n),
                        box(w, 45.56, e, n)])


def _point_splits() -> gpd.GeoDataFrame:
    points = gpd.read_file(PROCESSED / "sample_points.gpkg")
    split = pd.concat(pd.read_parquet(PROCESSED / f"{s}_features.parquet",
                                      columns=["point_id"]).assign(split=s)
                      for s, _ in SPLITS)
    return points.merge(split, on="point_id", how="inner")


def _scale_bar(ax, lon: float, lat: float, km: float):
    deg = km / (111.32 * np.cos(np.radians(lat)))
    ax.plot([lon, lon + deg], [lat, lat], color=INK, linewidth=1.6,
            solid_capstyle="butt", zorder=6)
    ax.text(lon + deg / 2, lat + 0.35, f"{km:.0f} km", ha="center", va="bottom",
            fontsize=6.5, color=INK, zorder=6)


def fig_study_area(out: Path):
    """Sampling squares on Natural Earth 1:10m admin-0/admin-1, one panel per
    split. A square is drawn in every split it contributed points to (squares
    straddling a 10 km block boundary appear in more than one panel); the 45
    squares without a usable point are not drawn."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    w, s, e, n = AOI_BBOX
    countries = gpd.read_file(NATURAL_EARTH / "ne_10m_admin_0_countries"
                              / "ne_10m_admin_0_countries.shp", bbox=(w - 1, s - 1, e + 1, n + 1))
    provinces = gpd.read_file(NATURAL_EARTH / "ne_10m_admin_1_states_provinces"
                              / "ne_10m_admin_1_states_provinces.shp", bbox=(w, s, e, n))
    # clip to the frame and drop detail below ~0.5 km (invisible at this scale)
    # to keep the vector PDF small
    def _prepare(gdf):
        gdf = gdf.clip((w - 0.5, s - 0.5, e + 0.5, n + 0.5))
        return gdf.set_geometry(gdf.geometry.simplify(0.005))
    japan = _prepare(countries[countries["ADM0_A3"] == "JPN"])
    others = _prepare(countries[countries["ADM0_A3"] != "JPN"])
    prefectures = _prepare(provinces[provinces["adm0_a3"] == "JPN"])

    squares = gpd.read_file(PROCESSED / "sampling_squares.gpkg")
    centres = squares.to_crs(AREA_CRS).geometry.centroid.to_crs(4326)
    squares["lon"], squares["lat"] = centres.x.values, centres.y.values
    points = _point_splits()
    # one square per point, the first match, as build_dataset.py assigned them for
    # patch extraction (overlapping squares would otherwise count a point twice)
    joined = gpd.sjoin(points, squares[["geometry"]], predicate="within",
                       how="inner").drop_duplicates(subset="point_id")
    outside = points.geometry.within(outside_japan_boxes())

    fig, axes = plt.subplots(1, 3, figsize=(TEXT_WIDTH_IN, 2.75), sharey=True,
                             gridspec_kw={"wspace": 0.06})
    plt.rcParams["hatch.linewidth"] = 0.4  # restored below; only this figure hatches
    print("Study-area map")
    for ax, (split, title) in zip(axes, SPLITS):
        others.plot(ax=ax, color=LAND_OTHER, edgecolor=MUTED, linewidth=0.3, zorder=1)
        prefectures.plot(ax=ax, color=LAND, edgecolor="#d3d2cb", linewidth=0.2, zorder=1)
        japan.boundary.plot(ax=ax, color=OUTLINE, linewidth=0.35, zorder=2)
        gpd.GeoSeries([outside_japan_boxes()]).plot(
            ax=ax, facecolor="none", edgecolor="#a9a8a1", hatch="////", linewidth=0.0, zorder=2)
        gpd.GeoSeries([outside_japan_boxes()]).boundary.plot(
            ax=ax, color=MUTED, linewidth=0.5, linestyle="--", zorder=2)
        in_split = set(joined.loc[joined["split"] == split, "index_right"])
        mine = squares[squares.index.isin(in_split)]
        rest = squares[~squares.index.isin(in_split)
                       & squares.index.isin(set(joined["index_right"]))]
        ax.scatter(rest["lon"], rest["lat"], s=2, color="#c3c2b7", linewidths=0, zorder=3)
        for stratum in ("pv", "buffer", "urban"):  # fewer, larger symbols on top
            spec, sq = STRATA[stratum], mine[mine["stratum"] == stratum]
            ax.scatter(sq["lon"], sq["lat"], s=spec["size"], marker=spec["marker"],
                       facecolor=spec["color"], edgecolor=INK, linewidths=0.25,
                       alpha=0.9, zorder=4)
        n_pts = int((points["split"] == split).sum())
        n_out = int((outside & (points["split"] == split)).sum())
        ax.set_title(f"{title}\n{n_pts:,} points \u00b7 {len(mine)} squares", loc="left",
                     fontsize=7.5, linespacing=1.3)
        ax.text(w + 0.3, n - 0.3, f"{n_out} non-PV points\noutside Japan",
                ha="left", va="top", fontsize=6.3, color=INK_2, zorder=6,
                bbox={"boxstyle": "square,pad=0.2", "facecolor": "white",
                      "edgecolor": "none", "alpha": 0.85})
        ax.set_xlim(w, e)
        ax.set_ylim(s, n)
        ax.set_aspect(1 / np.cos(np.radians((s + n) / 2)))
        ax.set_xticks([130, 135, 140, 145])
        ax.set_xticklabels([f"{x}°E" for x in (130, 135, 140, 145)], fontsize=6.5)
        ax.set_yticks([35, 40, 45])
        ax.set_yticklabels([f"{y}°N" for y in (35, 40, 45)], fontsize=6.5)
        ax.tick_params(length=0, pad=2)
        for side in ("top", "right", "bottom", "left"):
            ax.spines[side].set_visible(True)
            ax.spines[side].set_color(AXIS)
            ax.spines[side].set_linewidth(0.6)
        print(f"  {split:5s} points={n_pts} squares={len(mine)} outside Japan={n_out} "
              f"(strata {mine['stratum'].value_counts().to_dict()})")
    _scale_bar(axes[0], 142.0, 31.6, 300)
    straddle = joined.groupby("index_right")["split"].nunique()
    print(f"  squares with points={len(straddle)}, in >1 split={int((straddle > 1).sum())}, "
          f"in 3 splits={int((straddle > 2).sum())}, "
          f"outside-Japan points={int(outside.sum())}")
    handles = [Line2D([], [], linestyle="", marker=STRATA[k]["marker"],
                      markerfacecolor=STRATA[k]["color"], markeredgecolor=INK,
                      markeredgewidth=0.4, markersize=5, label=STRATA[k]["label"])
               for k in ("pv", "buffer", "urban")]
    handles += [Line2D([], [], linestyle="", marker="o", markerfacecolor="#c3c2b7",
                       markeredgecolor="none", markersize=3, label="Square in other split"),
                Patch(facecolor="white", edgecolor=MUTED, hatch="////", linestyle="--",
                      linewidth=0.5, label="Outside Japan (Japan-only check)")]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=6.5,
               bbox_to_anchor=(0.5, 0.90), handletextpad=0.3, columnspacing=1.0)
    fig.savefig(out / "fig_study_area.pdf")
    plt.close(fig)
    plt.rcParams["hatch.linewidth"] = matplotlib.rcParamsDefault["hatch.linewidth"]


# Test-split points chosen to illustrate the findings of Chapter 5; each is
# checked against the saved predictions when the figure is drawn.
EXAMPLE_PATCHES = [
    (19, "Large array", "large"),
    (2127, "Array edge", "edge_cnn_only"),
    (553, "Small installation", "small_cnn_only"),
    (1277, "Small installation", "small_xgb_only"),
    (7091, "Just outside an array", "cnn_extends"),
    (6778, "Isolated false alarm", "isolated_false_alarm"),
    (6204, "Lookalike far from PV", "lookalike_both_pv"),
]
EDGE_M, FAR_M, SMALL_KW = 10, 50, 100


def _example_case_holds(case: str, r: pd.Series) -> bool:
    """The selection rule of each panel, on the label, the per-pixel XGBoost
    decision (t = 0.5), the CNN majority vote (>= 3 of 5 runs) and the distance
    to the nearest polygon edge."""
    pv, x, c, d = r["label"] == "pv", r["xgb_pv"], r["cnn_pv"], r["edge_m"]
    return {
        "large": pv and x and r["cnn_votes"] == 5 and r["kw"] > 2500 and d >= EDGE_M,
        "edge_cnn_only": pv and not x and c and d < EDGE_M,
        "small_cnn_only": pv and not x and c and r["kw"] < SMALL_KW,
        "small_xgb_only": pv and x and not c and r["kw"] < SMALL_KW,
        "cnn_extends": not pv and not x and c and d <= 20,
        "isolated_false_alarm": not pv and x and not c and d > FAR_M,
        "lookalike_both_pv": not pv and x and c and d > FAR_M,
    }[case]


def _locate_patch(point, squares: gpd.GeoDataFrame, patch: np.ndarray):
    """Affine transform and top-left pixel of a saved patch, found by cutting it
    again from the square's GeoTIFF; the cut must equal the saved patch."""
    import rasterio
    from rasterio.transform import rowcol
    from rasterio.windows import Window
    half = patch.shape[-1] // 2
    for idx in squares.index[squares.contains(point)]:
        with rasterio.open(RAW_SQUARES / f"square_{idx:05d}.tif") as src:
            row, col = rowcol(src.transform, point.x, point.y)
            cut = src.read(window=Window(col - half, row - half, patch.shape[-1],
                                         patch.shape[-1]))
            if cut.shape == patch.shape and np.allclose(cut, patch):
                return src.transform, row - half, col - half
    raise RuntimeError(f"no GeoTIFF reproduces the saved patch at {point}")


def fig_example_patches(out: Path):
    """Sentinel-2 RGB (B4, B3, B2) of 21 x 21-pixel test patches, with the
    released PV polygons and the centre (sample) pixel. Pixels are shown
    square, as the CNN receives them; on the ground they are 8.98e-5 degrees,
    i.e. 10 m north-south and about 7 to 8.6 m east-west."""
    import matplotlib.patheffects as pe
    from shapely.ops import unary_union
    pred = pd.read_parquet(LOGS / "predictions_test.parquet").set_index("point_id")
    features = pd.read_parquet(PROCESSED / "test_features.parquet", columns=["point_id"])
    row_of = {pid: i for i, pid in enumerate(features["point_id"])}
    patches = np.load(PROCESSED / "test_patches.npy", mmap_mode="r")
    points = gpd.read_file(PROCESSED / "sample_points.gpkg").set_index("point_id")
    squares = gpd.read_file(PROCESSED / "sampling_squares.gpkg")
    seeds = sorted(c for c in pred.columns if c.startswith("cnn_proba_seed"))

    ids = [pid for pid, _, _ in EXAMPLE_PATCHES]
    rgb = np.stack([patches[row_of[pid]][[2, 1, 0]] for pid in ids])  # B4, B3, B2
    lo, hi = np.percentile(rgb, [1, 99], axis=(0, 2, 3))  # one stretch for all panels

    fig, axes = plt.subplots(2, 4, figsize=(TEXT_WIDTH_IN, 4.1),
                             gridspec_kw={"hspace": 0.78, "wspace": 0.17})
    halo = [pe.withStroke(linewidth=2.2, foreground=INK)]
    print("Example patches (test split)")
    for k, (ax, (pid, title, case)) in enumerate(zip(axes.flat, EXAMPLE_PATCHES)):
        point = points.loc[pid].geometry
        transform, r0, c0 = _locate_patch(point, squares, patches[row_of[pid]])
        size = patches.shape[-1]
        west, north = transform * (c0, r0)
        east, south = transform * (c0 + size, r0 + size)
        polys = gpd.read_file("data/external/pv_polygons.gpkg",
                              bbox=(west - 0.03, south - 0.03, east + 0.03, north + 0.03))
        p = pred.loc[pid]
        votes = int((p[seeds] >= 0.5).sum())
        pt_m = gpd.GeoSeries([point], crs=4326).to_crs(AREA_CRS).iloc[0]
        pv_m = unary_union(polys.to_crs(AREA_CRS).geometry.values)
        edge_m = pt_m.distance(pv_m.boundary) if pv_m.contains(pt_m) else pt_m.distance(pv_m)
        kw = p["polygon_area_m2"] * DEFAULT_KW_PER_M2
        r = {"label": p["label"], "xgb_pv": p["xgboost_baseline_proba"] >= 0.5,
             "cnn_pv": votes >= 3, "cnn_votes": votes, "edge_m": edge_m, "kw": kw}
        assert _example_case_holds(case, r), (pid, case, r)

        img = np.clip((rgb[k] - lo[:, None, None]) / (hi - lo)[:, None, None], 0, 1)
        ax.imshow(img.transpose(1, 2, 0) ** (1 / 1.4), interpolation="nearest",
                  extent=(-0.5, size - 0.5, size - 0.5, -0.5))
        inverse = ~transform
        for geom in polys.geometry:
            for part in getattr(geom, "geoms", [geom]):
                xy = np.asarray(part.exterior.coords)
                cols, rows = inverse * (xy[:, 0], xy[:, 1])
                ax.plot(cols - c0 - 0.5, rows - r0 - 0.5, color="#ffd60a",
                        linewidth=1.0, path_effects=halo)
        c = size // 2
        ax.add_patch(plt.Rectangle((c - 0.5, c - 0.5), 1, 1, fill=False,
                                   edgecolor="white", linewidth=0.9, zorder=5,
                                   path_effects=halo))
        for x0, y0, dx, dy in ((c, -0.5, 0, 2), (c, size - 0.5, 0, -2),
                               (-0.5, c, 2, 0), (size - 0.5, c, -2, 0)):
            ax.plot([x0, x0 + dx], [y0, y0 + dy], color="white", linewidth=1.0,
                    path_effects=halo, solid_capstyle="butt", zorder=5)
        ax.set_xlim(-0.5, size - 0.5)
        ax.set_ylim(size - 0.5, -0.5)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        for side in ax.spines.values():
            side.set_visible(True)
            side.set_color(AXIS)
        ax.set_title(f"({'abcdefgh'[k]}) {title}", loc="left", fontsize=7.5, pad=3)
        name = {"pv": "PV", "non_pv": "non-PV"}
        dist = "<1" if edge_m < 1 else f"{edge_m:.0f}"
        where = (f"{dist} m inside polygon edge" if p["label"] == "pv"
                 else f"{dist} m from nearest polygon")
        size_txt = f" \u00b7 {kw / 1000:.2f} MW" if p["label"] == "pv" else ""
        lines = [f"Label: {name[p['label']]}",
                 f"XGBoost: {'PV' if r['xgb_pv'] else 'non-PV'} "
                 f"(p = {p['xgboost_baseline_proba']:.2f})",
                 f"CNN: {'PV' if r['cnn_pv'] else 'non-PV'} ({votes}/{len(seeds)} runs PV)",
                 f"id {pid}{size_txt}\n{where}"]
        ax.text(0, -0.04, "\n".join(lines[:3]), transform=ax.transAxes, ha="left",
                va="top", fontsize=6.3, color=INK, linespacing=1.25)
        ax.text(0, -0.37, lines[3], transform=ax.transAxes, ha="left", va="top",
                fontsize=6, color=INK_2, linespacing=1.2)
        print(f"  ({'abcdefgh'[k]}) {pid:6d} {case:22s} label={p['label']:6s} "
              f"xgb={p['xgboost_baseline_proba']:.3f} cnn_votes={votes} "
              f"edge={edge_m:.1f} m kw={kw:.0f} lon={point.x:.5f} lat={point.y:.5f}")

    key = axes.flat[-1]
    key.axis("off")
    key.set_xlim(0, 1)
    key.set_ylim(0, 1)
    key.plot([0.02, 0.2], [0.82, 0.82], color="#ffd60a", linewidth=1.0, path_effects=halo)
    key.text(0.26, 0.82, "PV polygon outline\n(released map)", va="center", fontsize=6.5,
             color=INK_2)
    key.add_patch(plt.Rectangle((0.07, 0.52), 0.08, 0.08, fill=False, edgecolor="white",
                                linewidth=0.9, path_effects=halo))
    key.text(0.26, 0.56, "Centre pixel\n(sample point)", va="center", fontsize=6.5,
             color=INK_2)
    key.text(0.02, 0.32, "RGB = B4, B3, B2; one\nlinear 1–99% stretch\nfor all panels",
             va="top", fontsize=6.3, color=MUTED)
    fig.savefig(out / "fig_example_patches.pdf")
    plt.close(fig)


def _aggregate(split: str) -> pd.DataFrame:
    df = pd.read_csv(LOGS / f"failure_case_comparison_v2_{split}.csv")
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


def _sd(x: float) -> str:
    """Standard deviation to three decimals; one that rounds to zero is shown
    as below 0.001 rather than as an exact 0.000."""
    return "$<$0.001" if round(x, 3) == 0 else f"{x:.3f}"


def _write(path: Path, lines: list[str]):
    # a leading hyphen before a number prints as a hyphen, not a minus sign
    text = re.sub(r"(?<=[\s\[])-(?=\d)", "$-$", "\n".join(lines))
    path.write_text(text + "\n", encoding="utf-8")


def tab_replication(out: Path):
    test = pd.read_csv(LOGS / "failure_case_comparison_v2_test.csv")
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
    """Baseline: a single run with the original row order. CNN: mean and sd over the seeded runs.
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
            cells += [b_txt, f"{c_mean} $\\pm$ {_sd(cnn.std())}"]
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")
    n = [int(frames[s]["n"].iloc[0]) for s in ("val", "test")]
    n_pv = [int(frames[s]["n_pv"].iloc[0]) for s in ("val", "test")]
    lines += [r"\midrule",
              f"Points (PV) & \\multicolumn{{2}}{{c}}{{{n[0]:,} ({n_pv[0]:,})}} & "
              f"\\multicolumn{{2}}{{c}}{{{n[1]:,} ({n_pv[1]:,})}} \\\\",
              r"\bottomrule", r"\end{tabular}"]
    _write(out / "tab_aggregate.tex", lines)


def _p_range(p_min: float, p_max: float) -> str:
    """McNemar p range over the CNN runs, in the style of the retired
    tab_context_control: values below 0.001 are shown as "$<$0.001"."""
    if p_max < 0.001:
        return "$<$0.001"
    if p_min < 0.001:
        return f"$<$0.001--{p_max:.3f}"
    return f"{p_min:.3f}--{p_max:.3f}"


def _diff_cell(r) -> str:
    return f"{r.diff_pp:+.2f} [{r.ci_low_pp:+.2f}, {r.ci_high_pp:+.2f}]"


# rows of the appendix variants table: (variant in baseline_variants.csv, label);
# the medians of 5x5 and more and the windows of 9x9 and more are only in the log
VARIANT_ROWS = [
    ("3x3 pooled (mean)", "3$\\times$3 pooled, mean"),
    ("5x5 pooled (mean)", "5$\\times$5 pooled, mean"),
    ("7x7 pooled (mean)", "7$\\times$7 pooled, mean"),
    ("3x3 pooled (median)", "3$\\times$3 pooled, median"),
    ("XGBoost + 3x3 mean/SD (27 features)", "Window statistics, 3$\\times$3$^{b}$"),
    ("XGBoost + 3x3/7x7/21x21 mean/SD (63 features)", "Window statistics, 3/7/21$^{c}$"),
]


def tab_baseline_variants(out: Path):
    """From run_baseline_variants.py: the CNN against the per-pixel baseline and
    its context variants, out of sample only (test at the validation-tuned
    threshold and at 0.5, validation at 0.5). The raw and 3x3 rows reproduce
    the retired context-control table (context_control.csv) exactly."""
    df = pd.read_csv(LOGS / "baseline_variants.csv")

    def row(split, variant, source):
        r = df[(df["split"] == split) & (df["variant"] == variant)
               & (df["threshold_source"] == source)]
        assert len(r) == 1, (split, variant, source)
        return r.iloc[0]

    control = pd.read_csv(LOGS / "context_control.csv")
    for split, comparison, variant, source in (
            ("test", "CNN vs raw XGBoost (t=0.5)", "Raw, per pixel", "fixed"),
            ("test", "CNN vs 3x3 pooled XGBoost (t=0.46)", "3x3 pooled (mean)", "tuned on val"),
            ("test", "CNN vs 3x3 pooled XGBoost (t=0.5)", "3x3 pooled (mean)", "fixed"),
            ("val", "CNN vs raw XGBoost (t=0.5)", "Raw, per pixel", "fixed"),
            ("val", "CNN vs 3x3 pooled XGBoost (t=0.5)", "3x3 pooled (mean)", "fixed")):
        c = control[(control["split"] == split) & (control["comparison"] == comparison)].iloc[0]
        v = row(split, variant, source)
        for a, b in (("baseline_accuracy", "accuracy"), ("diff_pp", "diff_pp"),
                     ("ci_low_pp", "ci_low_pp"), ("ci_high_pp", "ci_high_pp")):
            assert abs(c[a] - v[b]) < 1e-9, (split, comparison, a)

    lines = [r"\begin{tabular}{lcccc}", r"\toprule",
             r"XGBoost variant & Threshold & Accuracy & CNN $-$ XGBoost (pp) & McNemar $p$ \\",
             r"\midrule"]
    for split, name in (("test", "Test split"), ("val", "Validation split")):
        if split == "val":
            lines.append(r"\addlinespace")
        lines.append(r"\multicolumn{5}{l}{\textit{%s}} \\" % name)
        rows = [("Raw, per pixel", "Raw, per pixel", "fixed")]
        for variant, label in VARIANT_ROWS:
            if split == "test":  # the tuned threshold is out of sample only on test
                rows.append((variant, label, "tuned on val"))
            rows.append((variant, label, "fixed"))
        for variant, label, source in rows:
            r = row(split, variant, source)
            thr = f"{r.threshold:.2f}" + ("$^{a}$" if source == "tuned on val" else "")
            lines.append(f"{label} & {thr} & {r.accuracy:.3f} & {_diff_cell(r)} & "
                         f"{_p_range(r.p_min, r.p_max)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    _write(out / "tab_baseline_variants.tex", lines)


def tab_baseline_rowperm(out: Path):
    """From run_baseline_variants.py: the published baseline retrained on five
    random permutations of the training rows, against the same CNN runs."""
    df = pd.read_csv(LOGS / "baseline_rowperm.csv")
    lines = [r"\begin{tabular}{lccccc}", r"\toprule",
             r"Baseline run & Accuracy & F1 & CNN $-$ XGBoost (pp) & McNemar $p$ & $p<0.05$ \\",
             r"\midrule"]
    for split, name in (("test", "Test split"), ("val", "Validation split")):
        if split == "val":
            lines.append(r"\addlinespace")
        lines.append(r"\multicolumn{6}{l}{\textit{%s}} \\" % name)
        d = df[df["split"] == split]
        perms = d[d["baseline"].str.startswith("permutation ")].sort_values("perm_seed")
        rows = ([("Original order", d[d["baseline"] == "original order (published)"].iloc[0], "")]
                + [(f"Permutation {int(r.perm_seed)}", r, "") for _, r in perms.iterrows()]
                + [("Permutation mean$^{a}$",
                    d[d["baseline"] == "permutation-averaged"].iloc[0], "$^{b}$")])
        for label, r, note in rows:
            lines.append(f"{label} & {r.accuracy:.3f} & {r.f1:.3f} & {_diff_cell(r)} & "
                         f"{_p_range(r.p_min, r.p_max)} & {int(r.n_significant)} of 5{note} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    _write(out / "tab_baseline_rowperm.tex", lines)


def tab_error_location(out: Path):
    """From run_revision_checks.py: the error_location.csv strata (identical
    values) plus the pooled baseline's net gain at the untuned threshold 0.5."""
    df = pd.read_csv(LOGS / "revision_error_location_t050.csv")
    strata = df[df["stratum"].str.contains("edge|any PV") & df["n"].notna()
                & ~df["stratum"].str.startswith("CNN-only")]
    # test: pooled at the val-tuned threshold; val: pooled at the untuned 0.5 (out of sample)
    lines = [r"\begin{tabular}{lrcccrrrr}", r"\toprule",
             r"& \multicolumn{6}{c}{Test} & \multicolumn{2}{c}{Validation} \\",
             r"\cmidrule(lr){2-7}\cmidrule(lr){8-9}",
             r"& & \multicolumn{3}{c}{Accuracy} & \multicolumn{2}{c}{Net gain} & \multicolumn{2}{c}{Net gain} \\",
             r"\cmidrule(lr){3-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}",
             r"Location & $n$ & Raw & Pooled & CNN & Pooled & CNN & Pooled & CNN \\",
             r"\midrule"]
    names = strata["stratum"].unique()
    for name in names:
        t = strata[(strata["split"] == "test") & (strata["stratum"] == name)].iloc[0]
        v = strata[(strata["split"] == "val") & (strata["stratum"] == name)].iloc[0]
        label = (name.replace(">=", "$\\geq$").replace("<=", "$\\leq$")
                 .replace("< ", "$<$ ").replace("> ", "$>$ ").replace("20-50", "20--50"))
        lines.append(f"{label} & {int(t.n):,} & {t.xgboost_accuracy:.3f} & {t.pooled_accuracy:.3f} & "
                     f"{t.cnn_accuracy:.3f} & {t.pooled_net_gain:+.1f} & {t.cnn_net_gain:+.1f} & "
                     f"{v.pooled_net_gain_t050:+.1f} & {v.cnn_net_gain:+.1f} \\\\")
    tot = {s: strata[strata["split"] == s][["pooled_net_gain", "pooled_net_gain_t050",
                                            "cnn_net_gain"]].sum()
           for s in ("test", "val")}
    lines += [r"\midrule",
              f"Total & & & & & {tot['test'].pooled_net_gain:+.1f} & {tot['test'].cnn_net_gain:+.1f} & "
              f"{tot['val'].pooled_net_gain_t050:+.1f} & {tot['val'].cnn_net_gain:+.1f} \\\\",
              r"\bottomrule", r"\end{tabular}"]
    _write(out / "tab_error_location.tex", lines)


def tab_mcnemar(out: Path):
    tests = {s: pd.read_csv(LOGS / f"mcnemar_{s}.csv").set_index("repeat")
             for s in ("val", "test")}
    lines = [r"\begin{tabular}{crrrrrr}", r"\toprule",
             r"& \multicolumn{3}{c}{Validation} & \multicolumn{3}{c}{Test} \\",
             r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
             r"Seed & XGBoost only & CNN only & $p$ & XGBoost only & CNN only & $p$ \\",
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
    labels = [lab.replace("<", "$<$").replace(">", "$>$").replace("–", "--")
              for lab in labels]
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
    """From run_label_efficiency_v2.py: F1 and recall (mean and sd over five
    repeats) on the test and the validation split, and the paired CNN - XGBoost
    F1 difference with its 95% t-interval. A standard deviation of exactly zero
    (the baseline on the full training split) is not printed."""
    curve = pd.read_csv(LOGS / "label_efficiency_v2_curve.csv")
    paired = pd.read_csv(LOGS / "label_efficiency_v2_paired.csv")

    def cell(mean: float, std: float) -> str:
        return f"{mean:.3f}" if std == 0 else f"{mean:.3f} $\\pm$ {_sd(std)}"

    lines = [r"\begin{tabular}{rccccc}", r"\toprule",
             r"& \multicolumn{2}{c}{F1} & \multicolumn{2}{c}{Recall} & CNN $-$ XGBoost \\",
             r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
             r"Points & XGBoost & CNN & XGBoost & CNN & F1 (pp) [95\%] \\", r"\midrule"]
    for split, name in (("test", "Test split"), ("val", "Validation split")):
        if split == "val":
            lines.append(r"\addlinespace")
        lines.append(r"\multicolumn{6}{l}{\textit{%s}} \\" % name)
        for n in sorted(curve["n_train"].unique()):
            c = curve[(curve["split"] == split) & (curve["n_train"] == n)].set_index("model")
            d = paired[(paired["split"] == split) & (paired["n_train"] == n)
                       & (paired["metric"] == "f1")].iloc[0]
            cells = [cell(c.loc[m, f"{k}_mean"], c.loc[m, f"{k}_std"])
                     for k in ("f1", "recall") for m in ("xgboost_baseline", "cnn")]
            lines.append(f"{n:,} & " + " & ".join(cells) + f" & {d.mean_pp:+.2f} "
                         f"[{d.ci_low_pp:+.2f}, {d.ci_high_pp:+.2f}] \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    _write(out / "tab_label_efficiency.tex", lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="reports/latex/figures")
    parser.add_argument("--skip-population", action="store_true",
                        help="skip the slow polygon-area figure")
    parser.add_argument("--review-figures", metavar="DIR", default=None,
                        help="also write fig_study_area.pdf and fig_example_patches.pdf "
                             "to DIR (needs data/external/naturalearth and data/raw/japan)")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    _style()
    if args.review_figures:
        review = Path(args.review_figures)
        review.mkdir(parents=True, exist_ok=True)
        fig_study_area(review)
        fig_example_patches(review)
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
    tab_baseline_variants(tables)
    tab_baseline_rowperm(tables)
    tab_error_location(tables)
    tab_label_efficiency(tables)
    print_aggregate_intervals()
    print(f"Wrote figures to {out}/ and tables to {tables}/")


if __name__ == "__main__":
    main()
