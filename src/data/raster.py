"""Pulling pixels out of Earth Engine and onto disk.

`get_autumn_2022_composite` returns an `ee.Image` -- a server-side handle, not
data. CNN patch extraction needs a real numpy array, so the composite has to be
materialised. It cannot be materialised in one piece: the full Japan bbox at
10 m is roughly 150000 x 166000 px x 6 bands, about 600 GB. Downloading one
small GeoTIFF per sampling square keeps each request inside GEE's direct-download
size limit and brings the total for a 1000-square run to a few hundred MB.
"""
from pathlib import Path

import numpy as np
import rasterio
import requests

DOWNLOAD_TIMEOUT_S = 300


def download_square_geotiff(image, bounds: tuple[float, float, float, float],
                            bands: list[str], out_path: Path, scale: int = 10,
                            crs: str = "EPSG:4326") -> Path:
    """GEE-networked: fetch `image` over `bounds` as a multi-band GeoTIFF.

    Skips the request if `out_path` already holds a non-empty file, so an
    interrupted run resumes instead of re-downloading. Not unit-tested (see
    Task 5's testing-philosophy note) -- verify with the manual smoke test.
    """
    import ee

    out_path = Path(out_path)
    if out_path.exists() and out_path.stat().st_size > 0:
        return out_path

    west, south, east, north = bounds
    url = image.getDownloadURL({
        "bands": bands,
        "region": ee.Geometry.Rectangle([west, south, east, north]),
        "scale": scale,
        "crs": crs,
        "format": "GEO_TIFF",
    })
    response = requests.get(url, timeout=DOWNLOAD_TIMEOUT_S)
    response.raise_for_status()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(response.content)
    return out_path


def read_geotiff(path: Path) -> tuple[np.ndarray, object]:
    """Read a multi-band GeoTIFF into the (bands, rows, cols) array plus affine
    transform that `src.data.patches` expects."""
    with rasterio.open(path) as src:
        return src.read().astype(np.float32), src.transform
