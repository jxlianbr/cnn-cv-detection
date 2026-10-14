"""Build review chips for the annotation worksheet from GSI aerial photos.

For every worksheet point this fetches the GSI seamless aerial photo tiles
(zoom 18) around the point, caches them under data/external/gsi_tiles/, and
writes one PNG with three panels: the aerial chip with the point and the PV
polygon outlines, the Sentinel-2 RGB patch the CNN saw, and a text panel with
the point's size and both models' predictions.

The chips support a human visual check; nothing here produces labels.

    python scripts/make_annotation_chips.py --fetch-only   # tiles only
    python scripts/make_annotation_chips.py                # tiles + chips

Imagery: "Source: GSI seamless photo (国土地理院)",
https://maps.gsi.go.jp/development/ichiran.html
"""
import argparse
import json
import math
import sys
import time
from io import BytesIO
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root on sys.path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = ["DejaVu Sans", "Yu Gothic", "MS Gothic"]
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from matplotlib.patches import Rectangle
from PIL import Image
from shapely.geometry import Point, box

from src.evaluation.failure_cases import AREA_CRS, DEFAULT_KW_PER_M2

TILE_URL = "https://cyberjapandata.gsi.go.jp/xyz/seamlessphoto/{z}/{x}/{y}.jpg"
ATTRIBUTION = "Source: GSI seamless photo (国土地理院)"
TILE_CACHE = Path("data/external/gsi_tiles")
EARTH_CIRC = 2 * math.pi * 6378137.0
MIN_INTERVAL_S = 0.5  # at most 2 requests per second
SIZE_EDGES_KW = [0, 100, 250, 500, 1000, 2500, np.inf]


# --- Web-Mercator tile arithmetic -------------------------------------------

def lonlat_to_global_px(lon: float, lat: float, z: int) -> tuple[float, float]:
    n = 256 * 2 ** z
    x = (lon + 180.0) / 360.0 * n
    lat_r = math.radians(lat)
    y = (1 - math.log(math.tan(lat_r) + 1 / math.cos(lat_r)) / math.pi) / 2 * n
    return x, y


def metres_per_px(lat: float, z: int) -> float:
    return EARTH_CIRC * math.cos(math.radians(lat)) / (256 * 2 ** z)


class TileFetcher:
    def __init__(self, cache: Path, zoom: int):
        self.cache, self.zoom = cache, zoom
        self.meta_path = cache / f"tile_meta_z{zoom}.json"
        self.meta = json.loads(self.meta_path.read_text()) if self.meta_path.exists() else {}
        self.session = requests.Session()
        self.session.headers["User-Agent"] = "pv-annotation-chips (research; throttled 2 req/s)"
        self._last = 0.0
        self.n_requests = 0

    def path(self, x: int, y: int) -> Path:
        return self.cache / str(self.zoom) / str(x) / f"{y}.jpg"

    def get(self, x: int, y: int) -> Image.Image | None:
        key = f"{self.zoom}/{x}/{y}"
        p = self.path(x, y)
        if p.exists():
            return Image.open(p).convert("RGB")
        if self.meta.get(key, {}).get("status") == 404:
            return None
        wait = MIN_INTERVAL_S - (time.monotonic() - self._last)
        if wait > 0:
            time.sleep(wait)
        self._last = time.monotonic()
        self.n_requests += 1
        for attempt in range(3):
            try:
                r = self.session.get(TILE_URL.format(z=self.zoom, x=x, y=y), timeout=30)
                break
            except requests.RequestException:
                if attempt == 2:
                    raise
                time.sleep(2.0)
        self.meta[key] = {"status": r.status_code,
                          "last_modified": r.headers.get("Last-Modified")}
        if r.status_code == 404:
            return None
        r.raise_for_status()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(r.content)
        return Image.open(BytesIO(r.content)).convert("RGB")

    def save_meta(self):
        self.cache.mkdir(parents=True, exist_ok=True)
        self.meta_path.write_text(json.dumps(self.meta, indent=1, sort_keys=True))


def aerial_window(fetcher: TileFetcher, lon: float, lat: float, half_m: float):
    """Mosaic and crop a north-up window of +-half_m around the point.

    Returns the image (or None if no tile existed), metres per pixel, the
    window's top-left global pixel, and the tile keys used."""
    z = fetcher.zoom
    mpp = metres_per_px(lat, z)
    cx, cy = lonlat_to_global_px(lon, lat, z)
    half_px = half_m / mpp
    x0, y0, x1, y1 = cx - half_px, cy - half_px, cx + half_px, cy + half_px
    tx0, ty0, tx1, ty1 = int(x0 // 256), int(y0 // 256), int(x1 // 256), int(y1 // 256)
    mosaic = Image.new("RGB", ((tx1 - tx0 + 1) * 256, (ty1 - ty0 + 1) * 256), (128, 128, 128))
    keys, n_found = [], 0
    for tx in range(tx0, tx1 + 1):
        for ty in range(ty0, ty1 + 1):
            keys.append(f"{z}/{tx}/{ty}")
            tile = fetcher.get(tx, ty)
            if tile is not None:
                mosaic.paste(tile, ((tx - tx0) * 256, (ty - ty0) * 256))
                n_found += 1
    left, top = x0 - tx0 * 256, y0 - ty0 * 256
    crop = mosaic.crop((int(round(left)), int(round(top)),
                        int(round(left + 2 * half_px)), int(round(top + 2 * half_px))))
    origin = (tx0 * 256 + int(round(left)), ty0 * 256 + int(round(top)))
    return (crop if n_found else None), mpp, origin, keys, n_found


# --- chip rendering -----------------------------------------------------------

def s2_rgb(patch: np.ndarray) -> np.ndarray:
    rgb = np.stack([patch[2], patch[1], patch[0]], axis=-1).astype(float)  # B4 B3 B2
    lo, hi = np.percentile(rgb, 2), np.percentile(rgb, 98)
    return np.clip((rgb - lo) / max(hi - lo, 1e-6), 0, 1)


def size_bin(kw: float) -> str:
    for low, high in zip(SIZE_EDGES_KW[:-1], SIZE_EDGES_KW[1:]):
        if low <= kw < high:
            return f"{low:g}-{high:g} kW" if np.isfinite(high) else f">{low:g} kW"
    return "n/a"


def render_chip(row, img, mpp, polys_window, containing_idx, patch, info, out_path, z,
                zoom_half_m: float = 30.0):
    """Four panels: aerial overview, aerial zoom on the point, S2 patch, text."""
    fig, axes = plt.subplots(1, 4, figsize=(19.5, 5.6),
                             gridspec_kw={"width_ratios": [1.0, 1.0, 0.8, 0.75]})
    half_m = info["half_m"]
    cx, cy = lonlat_to_global_px(row.lon, row.lat, z)

    def to_local(lon, lat):
        gx, gy = lonlat_to_global_px(lon, lat, z)
        return (gx - cx) * mpp, -(gy - cy) * mpp

    rings = []
    for idx, geom in polys_window.items():
        for part in getattr(geom, "geoms", [geom]):
            for ring in [part.exterior, *part.interiors]:
                xs, ys = zip(*[to_local(lon, lat) for lon, lat in ring.coords])
                rings.append((xs, ys, idx == containing_idx))

    for ax, h, lw_scale in ((axes[0], half_m, 1.0), (axes[1], zoom_half_m, 1.4)):
        if img is not None:
            ax.imshow(img, extent=(-half_m, half_m, -half_m, half_m))
        else:
            ax.set_facecolor("0.5")
            ax.text(0, 0, "no GSI tile", ha="center", va="center", color="w")
        for xs, ys, own in rings:
            ax.plot(xs, ys, color="yellow" if own else "cyan",
                    lw=(1.6 if own else 1.0) * lw_scale, alpha=0.9)
        # the point's approximate 10 m pixel, with tick marks outside it
        ax.add_patch(Rectangle((-5, -5), 10, 10, fill=False, ec="red", lw=1.4 * lw_scale))
        t0, t1 = (7, 14) if h > 50 else (6, 9)
        for sx, sy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ax.plot([sx * t0, sx * t1], [sy * t0, sy * t1], color="red", lw=1.4 * lw_scale)
        bar = 50 if h > 50 else 10
        ax.plot([-h * 0.92, -h * 0.92 + bar], [-h * 0.9, -h * 0.9], color="white", lw=3)
        ax.text(-h * 0.92 + bar / 2, -h * 0.84, f"{bar} m", color="white", ha="center",
                fontsize=8, bbox=dict(fc="black", alpha=0.4, lw=0, pad=1))
        ax.set_xlim(-h, h)
        ax.set_ylim(-h, h)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.text(h * 0.98, -h * 0.98, ATTRIBUTION, color="white", fontsize=7,
                ha="right", va="bottom", bbox=dict(fc="black", alpha=0.5, lw=0))
    axes[0].add_patch(Rectangle((-105, -105), 210, 210, fill=False, ec="white", lw=0.8, ls="--"))
    axes[0].add_patch(Rectangle((-zoom_half_m, -zoom_half_m), 2 * zoom_half_m, 2 * zoom_half_m,
                                fill=False, ec="orange", lw=0.8))
    axes[0].set_title(f"GSI aerial z{z}, {2 * half_m:.0f} m window, {mpp:.2f} m/px\n"
                      "yellow = point's polygon, cyan = other polygons, dashed = approx. S2 patch",
                      fontsize=8)
    axes[1].set_title(f"zoom: {2 * zoom_half_m:.0f} m window (orange box at left)\n"
                      "red square = approx. 10 m pixel at the point", fontsize=8)

    ax = axes[2]
    ax.imshow(s2_rgb(patch), interpolation="nearest")
    ax.add_patch(Rectangle((9.5, 9.5), 1, 1, fill=False, ec="red", lw=1.5))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Sentinel-2 RGB patch (21 x 21 px, autumn 2022)\n"
                 "per-chip 2-98% stretch; red = centre pixel", fontsize=8)

    ax = axes[3]
    ax.axis("off")
    ax.text(0.0, 1.0, info["text"], va="top", ha="left", family="monospace", fontsize=9,
            transform=ax.transAxes)
    fig.suptitle(f"point {row.point_id}  ({row.split})", fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--worksheet", default="data/external/annotation_worksheet_v2.csv")
    parser.add_argument("--zoom", type=int, default=18)
    parser.add_argument("--half-m", type=float, default=125.0)
    parser.add_argument("--out-dir", default="reports/review/annotation/chips")
    parser.add_argument("--fetch-only", action="store_true")
    parser.add_argument("--point-ids", nargs="*", type=int)
    args = parser.parse_args()

    ws = pd.read_csv(args.worksheet)
    if args.point_ids:
        ws = ws[ws["point_id"].isin(args.point_ids)]
    fetcher = TileFetcher(TILE_CACHE, args.zoom)

    if args.fetch_only:
        try:
            for i, row in enumerate(ws.itertuples(), 1):
                aerial_window(fetcher, row.lon, row.lat, args.half_m)
                if i % 20 == 0:
                    fetcher.save_meta()
                    print(f"{i}/{len(ws)} points, {fetcher.n_requests} requests", flush=True)
        finally:
            fetcher.save_meta()
        return

    frames, patches, preds = {}, {}, {}
    for split in ws["split"].unique():
        feats = pd.read_parquet(f"data/processed/{split}_features.parquet")
        frames[split] = {pid: i for i, pid in enumerate(feats["point_id"].values)}
        patches[split] = np.load(f"data/processed/{split}_patches.npy", mmap_mode="r")
        preds[split] = pd.read_parquet(f"results/logs/predictions_{split}.parquet")

    pad = 0.004
    bounds = (ws.lon.min() - pad, ws.lat.min() - pad, ws.lon.max() + pad, ws.lat.max() + pad)
    polys = gpd.read_file("data/external/pv_polygons.gpkg", bbox=bounds)
    polys["area_m2"] = polys.to_crs(AREA_CRS).geometry.area.values
    sindex = polys.sindex
    seed_cols = [c for c in preds[ws["split"].iloc[0]].columns if c.startswith("cnn_proba_seed")]

    meta_rows = []
    try:
        for row in ws.itertuples():
            img, mpp, origin, keys, n_found = aerial_window(fetcher, row.lon, row.lat, args.half_m)
            dlat = args.half_m / 111_320.0
            dlon = args.half_m / (111_320.0 * math.cos(math.radians(row.lat)))
            win = box(row.lon - dlon, row.lat - dlat, row.lon + dlon, row.lat + dlat)
            hits = polys.iloc[sindex.query(win, predicate="intersects")]
            pt = Point(row.lon, row.lat)
            containing = hits[hits.contains(pt)]
            containing_idx = containing["area_m2"].idxmin() if len(containing) else None

            edge_m = np.nan
            if containing_idx is not None:
                g = gpd.GeoSeries([polys.geometry[containing_idx].boundary, pt], crs=polys.crs)
                g = g.to_crs(g.estimate_utm_crs())  # local UTM: distances in metres
                edge_m = g.iloc[0].distance(g.iloc[1])

            i = frames[row.split][row.point_id]
            p = preds[row.split].iloc[i]
            assert p["point_id"] == row.point_id
            xgb = float(p["xgboost_baseline_proba"])
            cnn = np.array([p[c] for c in seed_cols], dtype=float)
            votes = int((cnn >= 0.5).sum())
            kw = row.polygon_area_m2 * DEFAULT_KW_PER_M2
            lm = sorted({fetcher.meta.get(k, {}).get("last_modified") or "cached"
                         for k in keys})
            text = "\n".join([
                f"point_id        {row.point_id}",
                f"split           {row.split}",
                f"lon, lat        {row.lon:.5f}, {row.lat:.5f}",
                "reference label PV (released map)",
                f"polygon area    {row.polygon_area_m2:,.0f} m2",
                f"capacity @0.086 {kw:,.0f} kW",
                f"size bin        {size_bin(kw)}",
                f"sub-50 kW       {'YES' if kw < 50 else 'no'}",
                f"dist. to edge   {edge_m:.1f} m (approx.)",
                "",
                f"XGBoost proba   {xgb:.3f} -> {'PV' if xgb >= 0.5 else 'non-PV'}",
                f"CNN seeds       {' '.join(f'{v:.2f}' for v in cnn)}",
                f"CNN majority    {'PV' if votes >= 3 else 'non-PV'} ({votes}/5 runs >= 0.5)",
                "",
                f"polygons in window {len(hits)}",
                f"GSI tiles {n_found}/{len(keys)} found",
                "",
                "Aerial photo date: not provided",
                "by GSI tile metadata.",
            ])
            render_chip(row, img, mpp, dict(zip(hits.index, hits.geometry)),
                        containing_idx, patches[row.split][i], {"half_m": args.half_m, "text": text},
                        Path(args.out_dir) / f"{row.point_id}.png", args.zoom)
            meta_rows.append({"point_id": row.point_id, "split": row.split,
                              "estimated_kw": kw, "xgb_proba": xgb, "xgb_pred_pv": xgb >= 0.5,
                              "cnn_votes_pv": votes, "cnn_majority_pv": votes >= 3,
                              "edge_distance_m": edge_m, "n_polygons_window": len(hits),
                              "tiles_found": n_found, "tiles_needed": len(keys),
                              "m_per_px": mpp, "tile_last_modified": ";".join(lm)})
    finally:
        fetcher.save_meta()
    out = Path(args.out_dir).parent / "chip_index.csv"
    pd.DataFrame(meta_rows).to_csv(out, index=False)
    print(f"{len(meta_rows)} chips written to {args.out_dir}; index {out}; "
          f"{fetcher.n_requests} tile requests")


if __name__ == "__main__":
    main()
