"""Build the on-disk training sample that run_cnn_training.py and
run_label_efficiency.py consume.

Chains Tasks 1, 4, 5, 6, 7, 11 and 12 against real Earth Engine imagery and
writes nine files to `paths.processed_dir`:

    {train,val,test}_features.parquet   9 spectral features + label per point
    {train,val,test}_patches.npy        (N, 6, size_px, size_px) float32
    {train,val,test}_labels.npy         (N,) str

Row i of a split's feature frame describes row i of its patch array; keeping
that true across two independent point-dropping stages is what
`src.data.alignment` exists for.

Run the pilot region first:
    python scripts/build_dataset.py --max-squares 40 --n-pv 400 --n-nonpv 400
Then the full configured run:
    python scripts/build_dataset.py
"""
import argparse
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root on sys.path

import geopandas as gpd
import numpy as np
import pandas as pd

from src.config import load_config
from src.data.alignment import align_features_and_patches
from src.data.builtup import get_populated_buffer
from src.data.features import build_feature_matrix, sample_band_values_at_points
from src.data.patches import extract_patches_with_ids
from src.data.raster import download_square_geotiff, read_geotiff
from src.data.sampling import generate_sampling_squares, sample_points_in_squares
from src.data.sentinel2 import get_autumn_2022_composite
from src.data.split import spatial_train_val_test_split

_DEG_PER_M = 1.0 / 111_000.0  # crude, but only used to pad a download window


def _scaled_sampling_config(cfg_sampling, max_squares: int | None):
    """Shrink the stratum counts proportionally for a pilot run."""
    if max_squares is None or max_squares >= cfg_sampling.n_sampling_squares:
        return cfg_sampling
    ratio = max_squares / cfg_sampling.n_sampling_squares
    return replace(
        cfg_sampling,
        n_sampling_squares=max_squares,
        pv_squares=max(1, round(cfg_sampling.pv_squares * ratio)),
        buffer_squares=max(1, round(cfg_sampling.buffer_squares * ratio)),
        urban_squares=max(1, round(cfg_sampling.urban_squares * ratio)),
    )


def _buffer_extent_gdf(buffer_fc, fallback_bbox: list[float]) -> gpd.GeoDataFrame:
    """`generate_sampling_squares` only reads `.total_bounds` off this frame, so
    the buffer's server-side extent is all that needs to come back -- which
    avoids a getInfo() over hundreds of thousands of vectorised GHSL polygons.

    Note this means buffer/urban squares are drawn from the buffer's *bounding
    box*, not its actual footprint. That is the behaviour of the sampling code
    as written and tested; see the follow-up note in the run summary.
    """
    from shapely.geometry import box
    try:
        coords = buffer_fc.geometry().bounds().getInfo()["coordinates"][0]
        xs = [c[0] for c in coords]
        ys = [c[1] for c in coords]
        extent = box(min(xs), min(ys), max(xs), max(ys))
    except Exception as exc:  # noqa: BLE001 - degrade to the AOI, but say so
        print(f"  ! could not resolve buffer extent ({exc}); falling back to AOI bbox")
        extent = box(*fallback_bbox)
    return gpd.GeoDataFrame({"geometry": [extent]}, crs="EPSG:4326")


def _extract_patches_per_square(composite, points_gdf, squares_gdf, bands, size_px,
                                raw_dir: Path, scale: int = 10):
    """Download one GeoTIFF per sampling square and cut patches from it.

    The download window is padded by half a patch so points near a square's
    edge still have a full neighbourhood and are not silently dropped.
    """
    margin_deg = (size_px // 2 + 1) * scale * _DEG_PER_M
    located = gpd.sjoin(points_gdf, squares_gdf[["geometry"]], predicate="within",
                        how="inner").drop_duplicates(subset="point_id")

    all_patches, all_ids = [], []
    n_squares = located["index_right"].nunique()
    for n, (square_idx, group) in enumerate(located.groupby("index_right"), start=1):
        west, south, east, north = squares_gdf.geometry.loc[square_idx].bounds
        tif = raw_dir / f"square_{square_idx:05d}.tif"
        try:
            download_square_geotiff(
                composite,
                (west - margin_deg, south - margin_deg,
                 east + margin_deg, north + margin_deg),
                bands, tif, scale=scale,
            )
            raster, transform = read_geotiff(tif)
        except Exception as exc:  # noqa: BLE001 - one bad square must not end the run
            print(f"  ! square {square_idx} failed ({exc}); skipping its "
                  f"{len(group)} points")
            continue
        patches, ids = extract_patches_with_ids(raster, transform, group, size_px)
        all_patches.append(patches)
        all_ids.append(ids)
        print(f"  [{n}/{n_squares}] square {square_idx}: "
              f"{len(ids)}/{len(group)} patches", flush=True)

    if not all_patches:
        raise RuntimeError("No patches could be extracted from any sampling square.")
    return np.concatenate(all_patches), np.concatenate(all_ids)


def _write_split(name: str, features_gdf, patches, out_dir: Path):
    frame = pd.DataFrame(features_gdf.drop(columns="geometry"))
    rows = features_gdf.index.values
    frame.to_parquet(out_dir / f"{name}_features.parquet", index=False)
    np.save(out_dir / f"{name}_patches.npy", patches[rows].astype(np.float32))
    np.save(out_dir / f"{name}_labels.npy", frame["label"].to_numpy())
    counts = frame["label"].value_counts().to_dict()
    print(f"  {name}: {len(frame)} points {counts}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--max-squares", type=int, default=None,
                        help="cap sampling squares (each costs one GEE download)")
    parser.add_argument("--n-pv", type=int, default=2000)
    parser.add_argument("--n-nonpv", type=int, default=2000)
    args = parser.parse_args()

    import ee

    cfg = load_config(args.config)
    ee.Initialize(project=cfg.gee_project_id)
    bbox = cfg.aoi.bbox
    aoi = ee.Geometry.Rectangle(bbox)
    out_dir = Path(cfg.paths.processed_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = Path("data/raw") / cfg.aoi.name
    print(f"AOI '{cfg.aoi.name}' = {bbox}")

    print("Loading PV polygons...")
    pv_gdf = gpd.read_file(cfg.paths.pv_polygons).cx[bbox[0]:bbox[2], bbox[1]:bbox[3]]
    print(f"  {len(pv_gdf)} polygons inside the AOI")
    if pv_gdf.empty:
        raise RuntimeError("No PV polygons in the AOI; check aoi.bbox.")

    print("Resolving GHSL populated buffer extent...")
    buffer_fc = get_populated_buffer(aoi, buffer_km=cfg.sampling.buffer_km)
    buffer_gdf = _buffer_extent_gdf(buffer_fc, bbox)

    sampling_cfg = _scaled_sampling_config(cfg.sampling, args.max_squares)
    print(f"Generating {sampling_cfg.n_sampling_squares} sampling squares "
          f"(pv={sampling_cfg.pv_squares}, buffer={sampling_cfg.buffer_squares}, "
          f"urban={sampling_cfg.urban_squares})...")
    squares = generate_sampling_squares(pv_gdf, buffer_gdf, sampling_cfg,
                                        seed=cfg.split.seed)

    print(f"Sampling {args.n_pv} PV / {args.n_nonpv} non-PV points...")
    points = sample_points_in_squares(squares, pv_gdf, args.n_pv, args.n_nonpv,
                                      seed=cfg.split.seed)
    points = points.reset_index(drop=True)
    points["point_id"] = np.arange(len(points))
    print(f"  {len(points)} points ({points['label'].value_counts().to_dict()})")

    print("Building Sentinel-2 autumn-2022 composite...")
    composite = get_autumn_2022_composite(aoi, cfg.sentinel2)

    print("Sampling band values at points (GEE)...")
    band_values, kept = sample_band_values_at_points(
        composite, points, cfg.sentinel2.bands)
    print(f"  {len(kept)}/{len(points)} points survived cloud masking")
    features = build_feature_matrix(band_values, kept["label"])
    features["point_id"] = kept["point_id"].values
    features_gdf = gpd.GeoDataFrame(features, geometry=kept.geometry.values,
                                    crs=points.crs)

    print("Extracting patches (one GeoTIFF per square)...")
    patches, patch_ids = _extract_patches_per_square(
        composite, kept, squares, cfg.sentinel2.bands, cfg.patch.size_px, raw_dir)
    print(f"  {len(patch_ids)} patches extracted")

    aligned_gdf, aligned_patches = align_features_and_patches(
        features_gdf, patches, patch_ids)
    aligned_gdf = gpd.GeoDataFrame(aligned_gdf, geometry="geometry", crs=points.crs)
    print(f"Aligned to {len(aligned_gdf)} points with both features and a patch")
    if len(aligned_gdf) < 10:
        raise RuntimeError("Too few aligned samples to split; widen the AOI or "
                           "raise --n-pv/--n-nonpv.")

    train, val, test = spatial_train_val_test_split(
        aligned_gdf, test_fraction=cfg.split.test_fraction,
        val_fraction=cfg.split.val_fraction,
        block_size_km=cfg.split.block_size_km, seed=cfg.split.seed)

    print(f"Writing to {out_dir}/ ...")
    for name, split in (("train", train), ("val", val), ("test", test)):
        _write_split(name, split, aligned_patches, out_dir)
    aligned_gdf.to_file(out_dir / "sample_points.gpkg", driver="GPKG")
    squares.to_file(out_dir / "sampling_squares.gpkg", driver="GPKG")
    print("Done.")


if __name__ == "__main__":
    main()
