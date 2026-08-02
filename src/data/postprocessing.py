"""Post-processing pipeline replicating Shimada & Takeuchi (2025):
white top-hat denoise -> probability threshold -> Felzenszwalb segmentation
-> DID (built-up-area) false-positive filtering.
(papers/spectral-baseline-specifications.md §5)"""
import numpy as np
from shapely.geometry import Polygon
from skimage.morphology import white_tophat, disk
from skimage.segmentation import felzenszwalb
from skimage.measure import find_contours, label as cc_label


def denoise_top_hat(prob_raster: np.ndarray, radius_px: int = 1) -> np.ndarray:
    footprint = disk(radius_px)
    noise = white_tophat(prob_raster, footprint=footprint)
    return np.clip(prob_raster - noise, 0.0, 1.0)


def threshold_probability(prob_raster: np.ndarray, threshold: float = 0.6) -> np.ndarray:
    return prob_raster >= threshold


def segment_polygons(binary_mask: np.ndarray, prob_raster: np.ndarray,
                     scale: float = 50, sigma: float = 0.0,
                     min_size: int = 1) -> list[Polygon]:
    """Delineate PV polygons inside `binary_mask`, using Felzenszwalb
    segmentation of `prob_raster` to split adjacent-but-distinct
    installations that the threshold alone would merge into one blob.

    `sigma` defaults to 0: the probability raster reaching this function has
    already been smoothed by `denoise_top_hat`, and a second Gaussian blur
    only turns each installation's sharp edge into a gradient ramp, which
    Felzenszwalb then splits into concentric rings -- over-segmenting a
    single homogeneous installation into many spurious polygons.
    """
    segments = felzenszwalb(prob_raster, scale=scale, sigma=sigma, min_size=min_size)
    # Shift to a strictly positive range so 0 can mean "outside the mask",
    # then re-label connected components: one Felzenszwalb segment may be cut
    # into spatially disjoint pieces by the mask, and each piece is its own
    # installation.
    masked = np.where(binary_mask, segments + 1, 0)
    regions = cc_label(masked, background=0, connectivity=1)

    polygons = []
    for region_id in np.unique(regions):
        if region_id == 0:
            continue
        # Pad by one pixel so regions touching the raster edge still yield a
        # closed contour; subtract the pad offset from the resulting coords.
        blob_mask = np.pad((regions == region_id).astype(float), 1)
        contours = find_contours(blob_mask, level=0.5)
        if not contours:
            continue
        candidates = []
        for contour in contours:
            coords = [(x - 1, y - 1) for y, x in contour]
            if len(coords) >= 3:
                poly = Polygon(coords)
                if poly.is_valid and poly.area > 0:
                    candidates.append(poly)
        if candidates:
            # The outer boundary is the largest contour; inner contours are
            # holes, not separate installations.
            polygons.append(max(candidates, key=lambda p: p.area))
    return polygons


def filter_did_false_positives(polygons: list[Polygon], built_up_mask_raster: np.ndarray,
                               min_overlap_fraction: float = 0.5,
                               area_threshold_px: float = 20.0) -> list[Polygon]:
    """Drop small polygons that mostly overlap a Densely Inhabited District
    proxy (Task 4's GHSL built-up mask), matching the original study's DID
    filter for eliminating small urban false positives."""
    kept = []
    for poly in polygons:
        if poly.area >= area_threshold_px:
            kept.append(poly)
            continue
        minx, miny, maxx, maxy = poly.bounds
        window = built_up_mask_raster[int(miny):int(maxy) + 1, int(minx):int(maxx) + 1]
        if window.size == 0:
            kept.append(poly)
            continue
        overlap_fraction = window.mean()
        if overlap_fraction < min_overlap_fraction:
            kept.append(poly)
    return kept
