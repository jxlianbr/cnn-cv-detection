"""Download the Shimada & Takeuchi (2025) PV polygon dataset from Zenodo
(DOI 10.5281/zenodo.10674681) and save it as a single GeoPackage.

Note on file selection: the record publishes the layer as a bare ESRI
Shapefile *set* (.shp plus its .shx/.dbf/.prj/.cpg sidecars), not as a single
self-contained file. All sidecars sharing the .shp basename must be
downloaded together or the layer is unreadable, so this script groups by
basename rather than picking one file.
"""
import geopandas as gpd
import requests
from pathlib import Path

ZENODO_RECORD_ID = "10674681"
OUT_DIR = Path("data/external")
RAW_DIR = OUT_DIR / "raw"
OUT_PATH = OUT_DIR / "pv_polygons.gpkg"

SHAPEFILE_SIDECAR_EXTS = (".shp", ".shx", ".dbf", ".prj", ".cpg")
SELF_CONTAINED_EXTS = (".gpkg", ".geojson", ".zip")


def fetch_file_list() -> list[dict]:
    resp = requests.get(f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}", timeout=30)
    resp.raise_for_status()
    return resp.json()["files"]


def download_file(file_entry: dict, dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / file_entry["key"]
    if dest.exists() and dest.stat().st_size == file_entry.get("size", -1):
        print(f"  cached: {dest.name}")
        return dest
    url = file_entry["links"]["self"]
    print(f"  downloading: {dest.name} ({file_entry.get('size', 0) / 1e6:.1f} MB)")
    with requests.get(url, stream=True, timeout=600) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    return dest


def select_vector_files(files: list[dict]) -> tuple[list[dict], str]:
    """Return (files_to_download, key_of_layer_file)."""
    self_contained = [f for f in files if f["key"].lower().endswith(SELF_CONTAINED_EXTS)]
    if self_contained:
        return [self_contained[0]], self_contained[0]["key"]

    shp_entries = [f for f in files if f["key"].lower().endswith(".shp")]
    if not shp_entries:
        raise RuntimeError(
            f"No vector file found in Zenodo record {ZENODO_RECORD_ID}. "
            f"Files present: {[f['key'] for f in files]}. "
            "Inspect the record manually and adjust the extension lists above."
        )
    stem = Path(shp_entries[0]["key"]).stem
    sidecars = [
        f for f in files
        if Path(f["key"]).stem == stem
        and f["key"].lower().endswith(SHAPEFILE_SIDECAR_EXTS)
    ]
    return sidecars, shp_entries[0]["key"]


def main() -> None:
    files = fetch_file_list()
    to_download, layer_key = select_vector_files(files)
    print(f"Zenodo record {ZENODO_RECORD_ID}: fetching {len(to_download)} file(s)")
    for entry in to_download:
        download_file(entry, RAW_DIR)

    raw_path = RAW_DIR / layer_key
    gdf = gpd.read_file(raw_path)
    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=4326)
    else:
        gdf = gdf.to_crs(epsg=4326)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    gdf.to_file(OUT_PATH, driver="GPKG")
    print(f"Saved {len(gdf)} polygons to {OUT_PATH}")


if __name__ == "__main__":
    main()
