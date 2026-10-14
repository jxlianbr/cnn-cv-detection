"""Registration check: is the Sentinel-2 grid offset from the PV label polygons?

The examiner asked for the offset between the imagery and the released label
polygons to be measured rather than assumed. Two complementary checks:

1. Lattice check (exact, geometric). The tiles were exported from Earth Engine
   in EPSG:4326 with a pixel spacing of 8.983e-5 degrees, and the released
   polygons are vectorised raster outlines whose vertices also lie on a
   regular EPSG:4326 lattice. If both lattices share spacing and phase, every
   polygon edge coincides with a pixel edge of the tiles: there is no
   grid-level (sub-pixel phase) offset between the two.

2. Image-based offset (empirical). For each large, compact polygon with
   enough straight edge inside a downloaded tile, the polygon mask (together
   with every other released polygon nearby) is translated over a +/-30 m
   search window on a sub-pixel grid (1/10 pixel: about 0.8 m east-west and
   1.0 m north-south). For each shift, the fractional coverage of every image
   pixel by the translated polygons is computed and smoothed with a Gaussian
   point-spread function (PSF); the shift is scored by how well this template
   explains the image (mean squared Pearson correlation over the four native
   10 m bands B2, B3, B4, B8; all six bands, PVSI and NDVI are kept as
   sensitivity signals). The PSF width matters: the image is softer than a
   binary mask (sensor PSF, sub-pixel position of the true edge, nearest-
   neighbour resampling from the native grid), and with a sharp template any
   small shift that makes the edge pixels fractional improves the fit, so a
   sharp template is biased *away* from zero offset in a random direction
   (shown in the validation as the `sharp_template` rows). One PSF width is
   therefore chosen for all polygons, the one in PSF_SIGMAS_M that maximises
   the summed peak r^2 over all polygons (a pooled fit; per-polygon fits of the
   width were found to over-fit background texture); offsets at every other
   width are kept as a sensitivity check. The best shift is refined with a
   parabolic fit through the peak, and converted from degrees to metres at
   the polygon's latitude.

   Sign convention: (dx, dy) is the displacement (east, north; metres) that has
   to be applied to the polygons to best match the image, i.e. the position of
   the image content relative to the labels.

The estimator is validated (a) on synthetic images built from the real
polygon masks with a known shift (zero, whole-pixel and continuous), blur and
noise, and (b) on the real imagery by translating the polygons by a known
amount and checking that the recovered shift changes by exactly that amount.

Writes (new files only; refuses to overwrite):
  results/logs/registration_offsets.csv     per-polygon offsets
  results/logs/registration_validation.csv  synthetic / known-shift recovery
  results/logs/registration_edge_sides.csv  baseline/CNN misses by edge side
  reports/review/fig_registration_proposed.pdf
and prints the summary numbers used in reports/review/results_T4.md.
"""
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root on sys.path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from rasterio.transform import Affine
from scipy.ndimage import gaussian_filter
from shapely.affinity import translate
from shapely.geometry import box

from src.data.indices import ndvi, pvsi

RAW_DIR = Path("data/raw/japan")
POLYGONS = Path("data/external/pv_polygons.gpkg")
SQUARES = Path("data/processed/sampling_squares.gpkg")
LOGS = Path("results/logs")
OUT_CSV = LOGS / "registration_offsets.csv"
OUT_VALIDATION = LOGS / "registration_validation.csv"
OUT_SIDES = LOGS / "registration_edge_sides.csv"
OUT_FIG = Path("reports/review/fig_registration_proposed.pdf")

BANDS = ["B2", "B3", "B4", "B8", "B11", "B12"]  # configs/default.yaml order
BAND10_IDX = [0, 1, 2, 3]                       # native 10 m bands
AREA_CRS = "EPSG:6933"
MIN_AREA_M2 = 10_000          # 1 ha; results are also given for >= 2 ha
MIN_SOLIDITY = 0.5            # area / convex-hull area: compact outlines only
MIN_EDGE_M = 100.0            # polygon edge inside the tile, per orientation
SEARCH_M = 30.0               # +/- search radius in metres, per axis
SUPERSAMPLE = 10              # sub-pixel steps per pixel
PSF_SIGMAS_M = (0.0, 3.0, 5.0, 7.0, 10.0, 14.0)  # candidate template PSF widths
WINDOW_MARGIN_PX = 6          # image pixels kept around the polygon bbox
MIN_R2_RELIABLE = 0.2
SEED = 20261008


# ---------------------------------------------------------------- geometry

def metres_per_degree(lat_deg: float) -> tuple[float, float]:
    """(east-west, north-south) metres per degree on WGS84 at `lat_deg`."""
    phi = np.radians(lat_deg)
    m_lat = 111132.954 - 559.822 * np.cos(2 * phi) + 1.175 * np.cos(4 * phi)
    m_lon = 111412.84 * np.cos(phi) - 93.5 * np.cos(3 * phi) + 0.118 * np.cos(5 * phi)
    return float(m_lon), float(m_lat)


def lattice_phase(values, spacing: float) -> np.ndarray:
    """Distance (in pixels) of each coordinate from the nearest multiple of `spacing`."""
    frac = np.mod(np.asarray(values, dtype=np.float64) / spacing, 1.0)
    return np.minimum(frac, 1.0 - frac)


def edge_lengths_m(geom, clip, lat: float) -> tuple[float, float]:
    """(north-south running, east-west running) boundary length in metres of
    `geom` inside `clip`. The outlines are axis-aligned staircases."""
    m_lon, m_lat = metres_per_degree(lat)
    lines = geom.boundary.intersection(clip)
    parts = getattr(lines, "geoms", [lines])
    vertical = horizontal = 0.0
    for part in parts:
        if part.is_empty or part.geom_type != "LineString":
            continue
        c = np.asarray(part.coords)
        d = np.diff(c, axis=0)
        vertical += np.abs(d[np.abs(d[:, 0]) < 1e-12, 1]).sum() * m_lat
        horizontal += np.abs(d[np.abs(d[:, 1]) < 1e-12, 0]).sum() * m_lon
    return float(vertical), float(horizontal)


# ---------------------------------------------------------------- estimator

def coverage_for_shifts(fine_mask: np.ndarray, k: int, pad_px: int, n_rows: int,
                        n_cols: int, sy: int, sx_values: np.ndarray) -> np.ndarray:
    """Pixel means of `fine_mask` translated by (sx, sy) fine steps.

    `fine_mask` covers the image window plus `pad_px` pixels on every side at
    `k` fine cells per pixel. Positive sx moves the mask east, positive sy
    moves it south (down the rows). Returns (len(sx_values), n_rows * n_cols).
    """
    csum = np.zeros((fine_mask.shape[0] + 1, fine_mask.shape[1] + 1))
    csum[1:, 1:] = fine_mask.cumsum(0).cumsum(1)
    rows = pad_px * k - sy + k * np.arange(n_rows)
    cols = pad_px * k - sx_values[:, None] + k * np.arange(n_cols)[None, :]
    r0, r1 = rows[:, None], rows[:, None] + k
    c0, c1 = cols[:, None, :], cols[:, None, :] + k
    sums = csum[r1, c1] - csum[r0, c1] - csum[r1, c0] + csum[r0, c0]
    return (sums / (k * k)).reshape(len(sx_values), -1)


def score_shifts(fine_mask: np.ndarray, signals: np.ndarray, valid: np.ndarray,
                 k: int, pad_px: int, sy_values: np.ndarray,
                 sx_values: np.ndarray) -> np.ndarray:
    """Squared correlation between the template and each signal for every shift.

    signals: (n_signals, n_rows, n_cols). Returns (n_signals, len(sy), len(sx)).
    """
    n_sig, n_rows, n_cols = signals.shape
    v = valid.ravel()
    z = signals.reshape(n_sig, -1)[:, v]
    z = (z - z.mean(1, keepdims=True)) / (z.std(1, keepdims=True) + 1e-12)
    n = v.sum()
    out = np.zeros((n_sig, len(sy_values), len(sx_values)))
    for a, sy in enumerate(sy_values):
        cov = coverage_for_shifts(fine_mask, k, pad_px, n_rows, n_cols, int(sy), sx_values)[:, v]
        sd = cov.std(1)
        r = (cov @ z.T) / (n * np.where(sd > 0, sd, np.inf))[:, None]
        out[:, a, :] = (r ** 2).T
    return out


def _parabolic(fm1: float, f0: float, fp1: float) -> float:
    denom = fm1 - 2 * f0 + fp1
    if denom >= 0:  # flat or not a maximum
        return 0.0
    return float(np.clip(0.5 * (fm1 - fp1) / denom, -0.5, 0.5))


def peak(surface: np.ndarray) -> tuple[float, float, bool]:
    """Sub-step (row, col) position of the maximum and whether it hit the border."""
    i, j = np.unravel_index(np.argmax(surface), surface.shape)
    ni, nj = surface.shape
    border = i in (0, ni - 1) or j in (0, nj - 1)
    di = _parabolic(surface[i - 1, j], surface[i, j], surface[i + 1, j]) if 0 < i < ni - 1 else 0.0
    dj = _parabolic(surface[i, j - 1], surface[i, j], surface[i, j + 1]) if 0 < j < nj - 1 else 0.0
    return i + di, j + dj, bool(border)


def fine_raster(geoms, west: float, north: float, res: float, k: int,
                n_rows: int, n_cols: int, pad_px: int) -> np.ndarray:
    """Rasterise `geoms` at k cells per pixel over the window plus padding."""
    fine_res = res / k
    tr = Affine(fine_res, 0, west - pad_px * res, 0, -fine_res, north + pad_px * res)
    shape = ((n_rows + 2 * pad_px) * k, (n_cols + 2 * pad_px) * k)
    if len(geoms) == 0:
        return np.zeros(shape, dtype=np.float64)
    return rasterize(((g, 1) for g in geoms), out_shape=shape, transform=tr,
                     fill=0, dtype="uint8").astype(np.float64)


def estimate_shift(geoms, signals: dict, valid: np.ndarray, west: float,
                   north: float, res: float, lat: float,
                   sigmas_m=PSF_SIGMAS_M, search_m: float = SEARCH_M,
                   k: int = SUPERSAMPLE) -> dict:
    """Best (dx, dy) in metres for each signal set.

    `signals` maps a name to an (n_signals, rows, cols) array over the window
    whose north-west corner is (west, north). For every PSF width in
    `sigmas_m` the r^2 surface is computed; the width and shift with the
    highest r^2 are kept.
    """
    m_lon, m_lat = metres_per_degree(lat)
    step_x, step_y = res / k * m_lon, res / k * m_lat
    nx, ny = int(np.ceil(search_m / step_x)), int(np.ceil(search_m / step_y))
    max_sigma_cells = max(sigmas_m) / min(step_x, step_y)
    pad_px = int(np.ceil((max(nx, ny) + 3 * max_sigma_cells) / k)) + 2
    sx_values, sy_values = np.arange(-nx, nx + 1), np.arange(-ny, ny + 1)
    n_rows, n_cols = valid.shape
    sharp = fine_raster(geoms, west, north, res, k, n_rows, n_cols, pad_px)
    names = list(signals)
    stacked = np.concatenate([signals[n] for n in names], axis=0)
    best = {}
    for sigma in sigmas_m:
        fm = sharp if sigma == 0 else gaussian_filter(sharp, (sigma / step_y, sigma / step_x))
        surf = score_shifts(fm, stacked, valid, k, pad_px, sy_values, sx_values)
        start = 0
        for name in names:
            n_sig = signals[name].shape[0]
            s = surf[start:start + n_sig].mean(0)
            start += n_sig
            if name in best and s.max() <= best[name]["r2_peak"]:
                continue
            pi, pj, border = peak(s)
            best[name] = {
                "dx_m": (pj - nx) * step_x,       # mask moved east
                "dy_m": -(pi - ny) * step_y,      # rows grow southwards
                "r2_peak": float(s.max()),
                "r2_zero": float(s[ny, nx]),
                "psf_sigma_m": float(sigma),
                "border": border,
            }
    return best


# ---------------------------------------------------------------- data

def image_signals(img: np.ndarray) -> dict:
    """img: (6, rows, cols) reflectance in configs/default.yaml band order."""
    b2, b3, b4, b8, b11, b12 = img
    return {
        "bands10": img[BAND10_IDX],
        "all6": img,
        "pvsi": pvsi(b11, b12, b8, b2, b4)[None],
        "ndvi": ndvi(b8, b4)[None],
    }


def load_tiles(squares: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    rows = []
    for path in sorted(RAW_DIR.glob("square_*.tif")):
        idx = int(path.stem.split("_")[1])
        with rasterio.open(path) as src:
            b = src.bounds
            rows.append({"square": idx, "path": str(path), "res_x": src.transform.a,
                         "res_y": -src.transform.e, "west": b.left, "east": b.right,
                         "south": b.bottom, "north": b.top})
    df = pd.DataFrame(rows)
    df["stratum"] = squares.loc[df["square"], "stratum"].to_numpy()
    geometry = [box(r.west, r.south, r.east, r.north) for r in df.itertuples()]
    return gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")


def select_polygons(polys: gpd.GeoDataFrame, tiles: gpd.GeoDataFrame) -> pd.DataFrame:
    """Large, compact polygons with >= MIN_EDGE_M of north-south and of
    east-west edge inside a tile (one pixel in from its border). A polygon may
    extend beyond the tile: only the window inside the tile is scored, and
    the vector outline beyond it is still used for the template."""
    big = polys[polys["area"] >= MIN_AREA_M2].copy()
    metric = big.geometry.to_crs(AREA_CRS)
    big["solidity"] = metric.area / metric.convex_hull.area
    big = big[big["solidity"] >= MIN_SOLIDITY]
    joined = gpd.sjoin(big, tiles[["square", "res_x", "geometry"]], predicate="intersects")
    tile_geom = tiles.set_index("square").geometry
    keep = []
    for pid, r in joined.iterrows():
        inner = tile_geom.loc[r.square].buffer(-r.res_x, join_style=2)
        part = r.geometry.intersection(inner)
        if part.is_empty:
            continue
        lat = part.centroid.y
        m_lon, m_lat = metres_per_degree(lat)
        inside_m2 = part.area * m_lon * m_lat
        v, h = edge_lengths_m(r.geometry, inner, lat)
        if min(v, h) >= MIN_EDGE_M:
            keep.append((pid, int(r.square), lat, part.centroid.x, r["area"],
                         inside_m2, r.solidity, v, h))
    sel = pd.DataFrame(keep, columns=["polygon_id", "square", "lat", "lon", "area_m2",
                                      "inside_tile_m2", "solidity", "edge_ns_m", "edge_ew_m"])
    # a polygon cut by two overlapping tiles is measured once, where most of it lies
    sel = sel.sort_values("inside_tile_m2", ascending=False).drop_duplicates("polygon_id")
    return sel.sort_values(["square", "polygon_id"]).reset_index(drop=True)


def window_for(geom, tile_tr: Affine, shape: tuple[int, int]):
    w, s, e, n = geom.bounds
    inv = ~tile_tr
    c0, r0 = inv * (w, n)
    c1, r1 = inv * (e, s)
    r0 = max(int(np.floor(r0)) - WINDOW_MARGIN_PX, 0)
    c0 = max(int(np.floor(c0)) - WINDOW_MARGIN_PX, 0)
    r1 = min(int(np.ceil(r1)) + WINDOW_MARGIN_PX, shape[0])
    c1 = min(int(np.ceil(c1)) + WINDOW_MARGIN_PX, shape[1])
    return r0, r1, c0, c1


def nearby_geoms(polys: gpd.GeoDataFrame, west: float, north: float, n_rows: int,
                 n_cols: int, res: float) -> list:
    pad = 2 * SEARCH_M / 80_000 + 20 * res  # generous: > search radius + PSF tail
    query = box(west - pad, north - n_rows * res - pad, west + n_cols * res + pad, north + pad)
    return list(polys.geometry.iloc[polys.sindex.query(query)])


def measure(sel: pd.DataFrame, polys: gpd.GeoDataFrame, tiles: gpd.GeoDataFrame,
            known_shift_m: tuple[float, float] | None = None,
            sigmas_m=PSF_SIGMAS_M, signal_names=None) -> pd.DataFrame:
    """Run the estimator on every selected polygon. With `known_shift_m`, all
    polygons are first translated by that (east, north) amount in metres."""
    tile_by_square = tiles.set_index("square")
    rows, cache = [], {}
    for r in sel.itertuples():
        if r.square not in cache:
            cache.clear()
            with rasterio.open(tile_by_square.loc[r.square].path) as src:
                cache[r.square] = (src.read().astype(np.float64), src.transform)
        img, tr = cache[r.square]
        r0, r1, c0, c1 = window_for(polys.geometry.loc[r.polygon_id], tr, img.shape[1:])
        sub = img[:, r0:r1, c0:c1]
        valid = np.all(np.isfinite(sub) & (sub > 0), axis=0)
        if valid.mean() < 0.9:
            continue
        west, north = tr * (c0, r0)
        geoms = nearby_geoms(polys, west, north, r1 - r0, c1 - c0, tr.a)
        if known_shift_m is not None:
            m_lon, m_lat = metres_per_degree(r.lat)
            geoms = [translate(g, known_shift_m[0] / m_lon, known_shift_m[1] / m_lat) for g in geoms]
        sig = image_signals(sub)
        if signal_names is not None:
            sig = {n: sig[n] for n in signal_names}
        sig = {n: np.where(valid, s, 0.0) for n, s in sig.items()}
        est = estimate_shift(geoms, sig, valid, west, north, tr.a, r.lat, sigmas_m=sigmas_m)
        row = {"polygon_id": r.polygon_id, "square": r.square, "lat": r.lat, "lon": r.lon,
               "area_ha": r.area_m2 / 1e4, "inside_tile_ha": r.inside_tile_m2 / 1e4,
               "solidity": r.solidity, "edge_ns_m": r.edge_ns_m, "edge_ew_m": r.edge_ew_m,
               "n_pixels": int(valid.sum()), "n_polygons_in_window": len(geoms)}
        for name, e in est.items():
            for key, val in e.items():
                row[f"{name}_{key}"] = val
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- validation

def synthetic_image(geoms, true_dx: float, true_dy: float, west: float, north: float,
                    res: float, lat: float, n_rows: int, n_cols: int, psf_m: float,
                    cnr: float, rng, n_bands: int = 4, k: int = 20) -> np.ndarray:
    """Area coverage of the polygons translated by (true_dx, true_dy) metres,
    blurred by a Gaussian PSF of `psf_m` metres, plus Gaussian noise at
    contrast-to-noise ratio `cnr`, in `n_bands` independent bands."""
    m_lon, m_lat = metres_per_degree(lat)
    moved = [translate(g, true_dx / m_lon, true_dy / m_lat) for g in geoms]
    pad = 4
    fm = fine_raster(moved, west, north, res, k, n_rows, n_cols, pad)
    if psf_m > 0:
        fm = gaussian_filter(fm, (psf_m / (res / k * m_lat), psf_m / (res / k * m_lon)))
    fm = fm[pad * k:(pad + n_rows) * k, pad * k:(pad + n_cols) * k]
    cov = fm.reshape(n_rows, k, n_cols, k).mean((1, 3))
    return np.stack([cov + rng.normal(0, 1 / cnr, size=cov.shape) for _ in range(n_bands)])


def synthetic_variants(template_sigma: float) -> list:
    """(name, true PSF sigma m, contrast-to-noise, template PSF widths)."""
    t = (template_sigma,)
    return [
        ("psf0_cnr2", 0.0, 2.0, t),
        ("psf5_cnr2", 5.0, 2.0, t),
        ("psf8_cnr2", 8.0, 2.0, t),
        ("psf5_cnr1", 5.0, 1.0, t),
        ("psf5_cnr2_sharp_template", 5.0, 2.0, (0.0,)),
    ]


def synthetic_validation(sel: pd.DataFrame, polys: gpd.GeoDataFrame,
                         tiles: gpd.GeoDataFrame, template_sigma: float,
                         n: int = 120) -> pd.DataFrame:
    """Recover known shifts from synthetic images built on the real masks.

    For each polygon the true shift is drawn from three kinds: zero, a whole
    number of pixels (up to +/-2 per axis; template stays lattice-aligned),
    and continuous uniform in +/-15 m per axis.
    """
    rng = np.random.default_rng(SEED)
    sub = sel.sample(n=min(n, len(sel)), random_state=SEED)
    tile_by_square = tiles.set_index("square")
    rows = []
    for i, r in enumerate(sub.itertuples()):
        with rasterio.open(tile_by_square.loc[r.square].path) as src:
            tr, shape = src.transform, (src.height, src.width)
        r0, r1, c0, c1 = window_for(polys.geometry.loc[r.polygon_id], tr, shape)
        n_rows, n_cols = r1 - r0, c1 - c0
        west, north = tr * (c0, r0)
        res = tr.a
        geoms = nearby_geoms(polys, west, north, n_rows, n_cols, res)
        m_lon, m_lat = metres_per_degree(r.lat)
        kind = ["zero", "whole_pixel", "continuous"][i % 3]
        if kind == "zero":
            true_dx = true_dy = 0.0
        elif kind == "whole_pixel":
            px, py = rng.integers(-2, 3, size=2)
            true_dx, true_dy = px * res * m_lon, py * res * m_lat
        else:
            true_dx, true_dy = rng.uniform(-15, 15, size=2)
        valid = np.ones((n_rows, n_cols), dtype=bool)
        for variant, psf, cnr, sigmas in synthetic_variants(template_sigma):
            img = synthetic_image(geoms, true_dx, true_dy, west, north, res, r.lat,
                                  n_rows, n_cols, psf, cnr, rng)
            est = estimate_shift(geoms, {"syn": img}, valid, west, north, res, r.lat,
                                 sigmas_m=sigmas)["syn"]
            rows.append({"kind": "synthetic", "variant": variant, "shift_kind": kind,
                         "polygon_id": r.polygon_id, "area_ha": r.area_m2 / 1e4,
                         "true_dx_m": true_dx, "true_dy_m": true_dy,
                         "est_dx_m": est["dx_m"], "est_dy_m": est["dy_m"],
                         "err_dx_m": est["dx_m"] - true_dx, "err_dy_m": est["dy_m"] - true_dy,
                         "psf_sigma_m": est["psf_sigma_m"], "border": est["border"]})
    return pd.DataFrame(rows)


def known_shift_validation(base: pd.DataFrame, sel: pd.DataFrame, polys, tiles,
                           template_sigma: float,
                           shifts=((7.3, -4.1), (-12.0, 9.5))) -> pd.DataFrame:
    """Translate the polygons by a known amount on the real imagery; the
    recovered shift must move by minus that amount."""
    rows = []
    sel_ok = sel[sel.polygon_id.isin(base.polygon_id)]
    for dx0, dy0 in shifts:
        moved = measure(sel_ok, polys, tiles, known_shift_m=(dx0, dy0),
                        sigmas_m=(template_sigma,), signal_names=["bands10"])
        m = base.merge(moved, on="polygon_id", suffixes=("", "_moved"))
        rows.append(pd.DataFrame({
            "kind": "known_shift_real", "variant": f"dx{dx0:+.1f}_dy{dy0:+.1f}",
            "shift_kind": "continuous", "polygon_id": m.polygon_id, "area_ha": m.area_ha,
            "true_dx_m": -dx0, "true_dy_m": -dy0,
            "est_dx_m": m.bands10_dx_m_moved - m.bands10_dx_m,
            "est_dy_m": m.bands10_dy_m_moved - m.bands10_dy_m,
            "err_dx_m": (m.bands10_dx_m_moved - m.bands10_dx_m) + dx0,
            "err_dy_m": (m.bands10_dy_m_moved - m.bands10_dy_m) + dy0,
            "psf_sigma_m": m.bands10_psf_sigma_m_moved,
            "border": m.bands10_border | m.bands10_border_moved,
        }))
    return pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------- edge sides

def edge_side_analysis(polys: gpd.GeoDataFrame) -> pd.DataFrame:
    """Do the models' misses at array edges depend on which side of the
    facility the edge is on?

    A systematic offset (dx, dy) between image and labels moves PV content
    away from the labelled outline on one side and beyond it on the other, so
    PV points just inside the edge on the side the image is displaced away
    from would look less like PV and be missed more often. For every PV
    evaluation point within 10 m of the dissolved polygon boundary
    (EPSG:6933, as in run_context_control.py), the direction to the nearest
    boundary point is classified as N, S, E or W and the miss rates of the
    baseline and of the CNN (majority of five runs) are compared by side.
    """
    from scipy.stats import fisher_exact
    from shapely.ops import nearest_points, unary_union
    metric = polys[["geometry"]].to_crs(AREA_CRS)
    index = metric.sindex
    rows = []
    for split in ("val", "test"):
        pred = pd.read_parquet(LOGS / f"predictions_{split}.parquet")
        pv = pred[pred.label == "pv"].reset_index(drop=True)
        pts = gpd.GeoSeries(gpd.points_from_xy(pv.lon, pv.lat), crs=4326).to_crs(AREA_CRS)
        seeds = sorted(c for c in pv.columns if c.startswith("cnn_proba_seed"))
        for i, pt in enumerate(pts):
            near = index.query(pt.buffer(3000))
            if len(near) == 0:
                continue
            union = unary_union(metric.geometry.values[near])
            if not union.contains(pt):
                continue
            q = nearest_points(pt, union.boundary)[1]
            vx, vy = q.x - pt.x, q.y - pt.y
            d = float(np.hypot(vx, vy))
            if d >= 10:
                continue
            side = ("E" if vx > 0 else "W") if abs(vx) >= abs(vy) else ("N" if vy > 0 else "S")
            cnn_hits = sum(pv.loc[i, c] >= 0.5 for c in seeds)
            rows.append({"split": split, "point_id": pv.loc[i, "point_id"], "side": side,
                         "edge_distance_m": d,
                         "xgb_miss": bool(pv.loc[i, "xgboost_baseline_proba"] < 0.5),
                         "cnn_majority_miss": bool(cnn_hits < 3)})
    df = pd.DataFrame(rows)
    print("\nEdge sides (PV points < 10 m inside an edge, val + test):")
    for side, g in df.groupby("side"):
        print(f"  {side}: n={len(g):4d}  baseline miss rate {g.xgb_miss.mean():.3f}  "
              f"CNN-majority miss rate {g.cnn_majority_miss.mean():.3f}")
    for a, b in (("N", "S"), ("E", "W")):
        ga, gb = df[df.side == a], df[df.side == b]
        for col in ("xgb_miss", "cnn_majority_miss"):
            table = [[int(ga[col].sum()), int((~ga[col]).sum())],
                     [int(gb[col].sum()), int((~gb[col]).sum())]]
            p = fisher_exact(table)[1]
            print(f"  {a} vs {b} {col}: {ga[col].mean():.3f} vs {gb[col].mean():.3f}, "
                  f"Fisher exact p = {p:.3f}")
    return df


# ---------------------------------------------------------------- summary

def iqr(x) -> tuple[float, float]:
    return float(np.percentile(x, 25)), float(np.percentile(x, 75))


def summarise(df: pd.DataFrame, prefix: str = "bands10") -> dict:
    dx, dy = df[f"{prefix}_dx_m"].to_numpy(), df[f"{prefix}_dy_m"].to_numpy()
    mag = np.hypot(dx, dy)
    return {"n": len(df), "median_dx": float(np.median(dx)), "iqr_dx": iqr(dx),
            "median_dy": float(np.median(dy)), "iqr_dy": iqr(dy),
            "mean_dx": float(dx.mean()), "mean_dy": float(dy.mean()),
            "median_abs": float(np.median(mag)), "iqr_abs": iqr(mag),
            "share_gt5": float((mag > 5).mean()), "share_gt10": float((mag > 10).mean())}


def bootstrap_median_ci(df: pd.DataFrame, col: str, n_boot: int = 2000) -> tuple[float, float]:
    """95% interval of the median, resampling sampling squares (clusters)."""
    rng = np.random.default_rng(SEED)
    groups = [g[col].to_numpy() for _, g in df.groupby("square")]
    meds = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(groups), len(groups))
        meds.append(np.median(np.concatenate([groups[i] for i in pick])))
    return float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))


def fmt(s: dict) -> str:
    return (f"n={s['n']:4d}  dx med {s['median_dx']:+.2f} IQR [{s['iqr_dx'][0]:+.2f}, "
            f"{s['iqr_dx'][1]:+.2f}] mean {s['mean_dx']:+.2f}  dy med {s['median_dy']:+.2f} "
            f"IQR [{s['iqr_dy'][0]:+.2f}, {s['iqr_dy'][1]:+.2f}] mean {s['mean_dy']:+.2f}  "
            f"|s| med {s['median_abs']:.2f} IQR [{s['iqr_abs'][0]:.2f}, {s['iqr_abs'][1]:.2f}]  "
            f">5 m {s['share_gt5']:.2f}  >10 m {s['share_gt10']:.2f}")


def make_figure(df: pd.DataFrame, val: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 7.2))
    ax = axes.ravel()
    big = df.area_ha >= 2
    for k, (prefix, name) in enumerate([("bands10", "B2-B8 (primary)"), ("pvsi", "PVSI")]):
        dx, dy = df[f"{prefix}_dx_m"], df[f"{prefix}_dy_m"]
        a = ax[k]
        a.scatter(dx[~big], dy[~big], s=7, alpha=0.5, color="#8fb3d1", lw=0, label="1-2 ha")
        a.scatter(dx[big], dy[big], s=7, alpha=0.7, color="#1f5a8a", lw=0, label=r"$\geq$2 ha")
        for rad in (5, 10):
            a.add_patch(plt.Circle((0, 0), rad, fill=False, ls="--", lw=0.8, color="0.4"))
        a.axhline(0, color="0.8", lw=0.5, zorder=0)
        a.axvline(0, color="0.8", lw=0.5, zorder=0)
        a.plot(np.median(dx), np.median(dy), "x", color="#c0392b", ms=8, mew=2, label="median")
        a.set(xlim=(-30, 30), ylim=(-30, 30), aspect="equal", xlabel="dx (m, east)",
              ylabel="dy (m, north)", title=f"({'ab'[k]}) Offset per polygon, {name}")
        a.legend(frameon=False, fontsize=7, loc="lower left")
    for prefix, color, name in [("bands10", "#1f5a8a", "B2-B8"), ("pvsi", "#e08a1e", "PVSI")]:
        ax[2].scatter(df.area_ha, np.hypot(df[f"{prefix}_dx_m"], df[f"{prefix}_dy_m"]),
                      s=7, alpha=0.6, color=color, lw=0, label=name)
    ax[2].set(xscale="log", xlabel="Polygon area (ha)", ylabel="|offset| (m)",
              title="(c) Offset vs. polygon size")
    for h in (5, 10):
        ax[2].axhline(h, color="0.4", ls="--", lw=0.8)
    ax[2].legend(frameon=False, fontsize=7)
    v = val[(val.kind == "synthetic") & (val.variant == "psf5_cnr2")]
    ax[3].scatter(v.true_dx_m, v.est_dx_m, s=9, label="dx", color="#1f5a8a", lw=0)
    ax[3].scatter(v.true_dy_m, v.est_dy_m, s=9, label="dy", color="#e08a1e", lw=0)
    lim = max(20.0, float(np.abs(v[["true_dx_m", "true_dy_m"]]).max().max()) + 2)
    ax[3].plot([-lim, lim], [-lim, lim], color="0.4", lw=0.8)
    ax[3].set(xlim=(-lim, lim), ylim=(-lim, lim), aspect="equal", xlabel="True shift (m)",
              ylabel="Recovered shift (m)", title="(d) Synthetic recovery")
    ax[3].legend(frameon=False, loc="upper left", fontsize=7)
    fig.tight_layout()
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG)
    plt.close(fig)


def main() -> None:
    for path in (OUT_CSV, OUT_VALIDATION, OUT_SIDES):
        if path.exists():
            sys.exit(f"{path} exists; refusing to overwrite an existing log.")
    warnings.filterwarnings("ignore", category=UserWarning)
    squares = gpd.read_file(SQUARES)
    polys = gpd.read_file(POLYGONS)
    tiles = load_tiles(squares)
    res = float(tiles.res_x.iloc[0])
    assert np.allclose(tiles.res_x, res) and np.allclose(tiles.res_y, res)

    # 1. lattice check
    tile_phase = np.concatenate([lattice_phase(tiles.west, res), lattice_phase(tiles.north, res)])
    coords = [np.asarray(ring.coords) for g in polys.geometry
              for ring in [g.exterior, *g.interiors]]
    xy = np.concatenate(coords)
    print(f"Lattice: {len(tiles)} tiles, spacing {res:.15e} deg")
    print(f"  tile origins: max distance from the 0-anchored lattice = {tile_phase.max():.2e} px")
    print(f"  polygon vertices ({len(xy)}): max distance = "
          f"{max(lattice_phase(xy[:, 0], res).max(), lattice_phase(xy[:, 1], res).max()):.2e} px")

    # 2. image-based offsets
    sel = select_polygons(polys, tiles)
    print(f"Selected {len(sel)} polygons (>= {MIN_AREA_M2/1e4:.0f} ha, solidity >= "
          f"{MIN_SOLIDITY}, >= {MIN_EDGE_M:.0f} m of edge per orientation inside a tile) "
          f"in {sel.square.nunique()} squares; {(sel.area_m2 >= 20_000).sum()} are >= 2 ha")
    per_sigma = {}
    for sigma in PSF_SIGMAS_M:
        per_sigma[sigma] = measure(sel, polys, tiles, sigmas_m=(sigma,), signal_names=["bands10"])
    pooled = {s: d.bands10_r2_peak.sum() for s, d in per_sigma.items()}
    sigma_star = max(pooled, key=pooled.get)
    print("  pooled peak r2 by template PSF sigma (m): "
          + ", ".join(f"{s:g}: {v:.2f}" for s, v in pooled.items()) + f" -> {sigma_star:g} m")
    df = measure(sel, polys, tiles, sigmas_m=(sigma_star,))
    for sigma, d in per_sigma.items():
        d = d.set_index("polygon_id")
        for c in ["dx_m", "dy_m", "r2_peak", "border"]:
            df[f"bands10_{c}_psf{sigma:g}"] = df.polygon_id.map(d[f"bands10_{c}"])
    df["reliable"] = ~df.bands10_border & (df.bands10_r2_peak >= MIN_R2_RELIABLE)
    LOGS.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"  measured {len(df)} (>= 90% valid pixels); reliable {df.reliable.sum()}")

    val = pd.concat([synthetic_validation(sel, polys, tiles, sigma_star),
                     known_shift_validation(df[df.reliable], sel, polys, tiles, sigma_star)],
                    ignore_index=True)
    val.to_csv(OUT_VALIDATION, index=False)

    print("\nValidation (error = recovered - true, metres):")
    for (kind, variant, sk), g in val.groupby(["kind", "variant", "shift_kind"]):
        err = np.hypot(g.err_dx_m, g.err_dy_m)
        print(f"  {kind:16s} {variant:26s} {sk:11s} n={len(g):4d}  med err dx "
              f"{g.err_dx_m.median():+.2f} dy {g.err_dy_m.median():+.2f}  MAE dx "
              f"{g.err_dx_m.abs().mean():.2f} dy {g.err_dy_m.abs().mean():.2f}  "
              f"med |err| {err.median():.2f}  |err|>2 m {(err > 2).mean():.2f}  "
              f">5 m {(err > 5).mean():.2f}")

    print("\nOffsets (dx east, dy north; metres):")
    rel = df[df.reliable]
    for label, sub in [("all", df), ("reliable", rel), ("rel>=2ha", rel[rel.area_ha >= 2])]:
        for prefix in ["bands10", "all6", "pvsi", "ndvi"]:
            print(f"  {label:8s} {prefix:7s} {fmt(summarise(sub, prefix))}")
    for label, sub in [("reliable", rel), ("rel>=2ha", rel[rel.area_ha >= 2])]:
        for col in ["bands10_dx_m", "bands10_dy_m", "pvsi_dx_m", "pvsi_dy_m"]:
            lo, hi = bootstrap_median_ci(sub, col)
            print(f"  {label} median {col}: 95% square-bootstrap CI [{lo:+.2f}, {hi:+.2f}]")
    print(f"  r2 (bands10, reliable): median at zero shift {rel.bands10_r2_zero.median():.3f}, "
          f"at peak {rel.bands10_r2_peak.median():.3f}")
    print("  sensitivity to the template PSF width (reliable polygons, bands10):")
    for sigma in PSF_SIGMAS_M:
        sub = rel.rename(columns={f"bands10_{c}_psf{sigma:g}": f"psf_{c}" for c in ["dx_m", "dy_m"]})
        print(f"    sigma {sigma:4.1f} m  {fmt(summarise(sub, 'psf'))}")
    agree = np.hypot(rel.bands10_dx_m - rel.pvsi_dx_m, rel.bands10_dy_m - rel.pvsi_dy_m)
    print(f"  bands10 vs pvsi per-polygon difference: median {agree.median():.2f} m")

    print("\nBy latitude (reliable, bands10):")
    for b, g in rel.groupby(pd.cut(rel.lat, [0, 33, 34.5, 36, 37.5, 50]), observed=True):
        print(f"  {str(b):14s} {fmt(summarise(g))}")
    print("\nBy longitude (reliable, bands10):")
    for b, g in rel.groupby(pd.cut(rel.lon, [0, 132, 135, 138, 140, 150]), observed=True):
        print(f"  {str(b):14s} {fmt(summarise(g))}")
    print("\nBy size (reliable, bands10):")
    for b, g in rel.groupby(pd.cut(rel.area_ha, [1, 2, 3, 5, 10, 1000]), observed=True):
        print(f"  {str(b):14s} {fmt(summarise(g))}")
    rho = pd.Series(np.hypot(rel.bands10_dx_m, rel.bands10_dy_m)).corr(
        pd.Series(np.log(rel.area_ha.to_numpy())), method="spearman")
    print(f"  Spearman rho(|offset|, log area) = {rho:+.3f}")
    print(f"  Spearman rho(dx, lat) = {rel.bands10_dx_m.corr(rel.lat, method='spearman'):+.3f}, "
          f"rho(dy, lat) = {rel.bands10_dy_m.corr(rel.lat, method='spearman'):+.3f}")

    sides = edge_side_analysis(polys)
    sides.to_csv(OUT_SIDES, index=False)
    make_figure(rel, val)
    print(f"\nWrote {OUT_CSV}, {OUT_VALIDATION}, {OUT_SIDES}, {OUT_FIG}")


if __name__ == "__main__":
    main()
