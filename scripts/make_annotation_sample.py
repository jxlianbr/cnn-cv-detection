"""Draw a manual-annotation worksheet, deliberately oversampling small
installations.

Two of the three failure-case strata -- non-crystalline-silicon panels and
spectrally ambiguous arrangements -- cannot be derived from the released data
and need a human looking at high-resolution imagery. Uniform sampling would be
wasted effort: small installations are a fraction of a percent of the pool, so
a few hundred random points would contain essentially none of the stratum the
research question is about. This takes a fixed quota per capacity bin instead.

Fill in the two annotation columns (True/False) against Google Earth or other
VHR imagery, save as CSV, and point
scripts/run_failure_case_evaluation.py --manual-flags at it.

    python scripts/make_annotation_sample.py --n-per-bin 40
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root on sys.path

import numpy as np
import pandas as pd

from src.evaluation.failure_cases import (
    ANNOTATION_COLUMNS, DEFAULT_KW_PER_M2, stratified_annotation_sample,
)

PROCESSED = Path("data/processed")
DEFAULT_EDGES = [0, 100, 250, 500, 1000, 2500, np.inf]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", nargs="+", default=["val", "test"],
                        help="which splits to draw from")
    parser.add_argument("--n-per-bin", type=int, default=40)
    parser.add_argument("--kw-per-m2", type=float, default=DEFAULT_KW_PER_M2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="data/external/annotation_worksheet.csv")
    args = parser.parse_args()

    frames = []
    for split in args.splits:
        frame = pd.read_parquet(PROCESSED / f"{split}_features.parquet")
        frame["split"] = split
        frames.append(frame)
    pool = pd.concat(frames, ignore_index=True)
    pv_pool = pool[pool["label"] == "pv"].reset_index(drop=True)
    print(f"Pool: {len(pv_pool)} PV points across {args.splits}")

    sample = stratified_annotation_sample(
        pv_pool, n_per_bin=args.n_per_bin, edges_kw=DEFAULT_EDGES,
        kw_per_m2=args.kw_per_m2, seed=args.seed)

    columns = (["point_id", "split", "lon", "lat", "polygon_area_m2", "estimated_kw"]
               + ANNOTATION_COLUMNS)
    sample = sample[columns].sort_values("estimated_kw").reset_index(drop=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(out_path, index=False)

    bins = pd.cut(sample["estimated_kw"], bins=DEFAULT_EDGES)
    print(sample.groupby(bins, observed=True).size().to_string())
    print(f"\n{len(sample)} points written to {out_path}")
    print("Fill non_crystalline_silicon and spectrally_ambiguous with True/False.")
    print("Google Earth link for the first row:")
    first = sample.iloc[0]
    print(f"  https://www.google.com/maps/@{first['lat']:.6f},{first['lon']:.6f},200m/data=!3m1!1e3")


if __name__ == "__main__":
    main()
