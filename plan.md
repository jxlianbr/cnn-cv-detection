# Spatial-vs-Spectral PV Detection Pipeline — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the full data-and-model pipeline needed to answer the thesis's two research questions: replicate Shimada & Takeuchi's (2025) spectral/XGBoost PV-detection baseline for Japan, build a CNN patch classifier on the same data, and evaluate both — in aggregate and on three named failure-case subsets — to determine whether spatial context improves detection where the spectral baseline is known to struggle.

**Architecture:** A shared data foundation (PV polygons → reconstructed point sample → Sentinel-2 features) feeds two independent classifiers — an XGBoost baseline (pixel-wise spectral features) and a CNN (patch-wise spatial features) — which are scored through one shared evaluation module so aggregate and stratified metrics are directly comparable. GEE-networked components (Sentinel-2 fetch, built-up mask) are kept as thin, isolated wrappers verified by manual smoke test rather than mocked unit tests, since mocking Earth Engine's query-building API produces false confidence, not real coverage; everything else (index math, sampling geometry, post-processing, metrics, CNN, patch extraction) is pure/local and gets full TDD.

**Tech Stack:** Python (see venv note in Task 0), `earthengine-api` (Sentinel-2/GHSL access), `geopandas`/`shapely` (polygon/point geometry), `numpy`/`rasterio`/`scikit-image` (raster math, morphology, Felzenszwalb segmentation), `xgboost` + `scikit-learn` (baseline classifier + metrics), `torch`/`torchvision` (CNN), `pytest` (tests), `pyyaml` (config).

## Global Constraints

- Sentinel-2 imagery: Level-2A surface reflectance, autumn 2022 window — must match Shimada & Takeuchi (2025) exactly (see `reports/paper-draft.md` §3.1).
- Bands: B2 (Blue, 490nm), B3 (Green, 560nm), B4 (Red, 665nm), B8 (NIR, 842nm), B11 (SWIR1, 1610nm), B12 (SWIR2, 2190nm) — from `papers/spectral-baseline-specifications.md`.
- Index formulas and XGBoost hyperparameters (`max_depth=8`, `eta=0.01717`, `n_estimators=1000`) are fixed by the original study — do not re-tune; replication must use these exact values (`papers/spectral-baseline-specifications.md`).
- PV label source: Shimada & Takeuchi (2025) Zenodo polygon release (DOI `10.5281/zenodo.10674681`) — **polygons only, no point-level samples included** (`papers/zenodo-dataset-verification.md`). All point samples in this pipeline are reconstructed, not reused.
- Reconstructed sampling must mirror the original's stratified structure (populated-area buffer sampling, deliberate urban non-PV sampling, an independent held-out verification set never touched during training) — see `papers/zenodo-dataset-verification.md` §2–3.
- Every classifier (baseline and CNN) must be evaluated with the same metric set: overall accuracy, precision, recall, F1, Cohen's Kappa, ROC-AUC — for direct comparability with the published baseline (`reports/paper-draft.md` §4.4).
- CNN input channels = the same 6 reflectance bands as the baseline (not an RGB subset) — motivated by Ishii et al.'s (2016) finding that spectral-channel completeness materially affects CNN accuracy on Japanese imagery (`reports/literature-review.md` #13).
- Target hardware: local RTX 2070 Super (8GB VRAM) — CNN batch size and patch size must fit this budget; Kaggle's free GPU tier is the documented fallback, not this plan's target.

---

## File Structure

```
D:\SCIENCE\CNN\
├── configs/
│   └── config.yaml              # all tunable paths, dates, hyperparams, thresholds
├── data/
│   ├── external/                # downloaded Zenodo polygons, GHSL refs (gitignored)
│   ├── raw/                     # raw GEE exports (gitignored)
│   └── processed/               # feature matrices, patch arrays (gitignored)
├── scripts/
│   ├── download_pv_polygons.py  # Task 1
│   ├── run_baseline.py          # Task 8-9 entry point
│   ├── run_cnn_training.py      # Task 15 entry point
│   └── run_label_efficiency.py  # Task 17 entry point
├── src/
│   ├── config.py                 # Task 2
│   ├── data/
│   │   ├── indices.py             # Task 3
│   │   ├── builtup.py             # Task 4
│   │   ├── sentinel2.py           # Task 5
│   │   ├── sampling.py            # Task 6
│   │   ├── features.py            # Task 7
│   │   ├── postprocessing.py      # Task 9
│   │   ├── split.py               # Task 11
│   │   ├── patches.py             # Task 12
│   │   └── patch_dataset.py       # Task 13
│   ├── models/
│   │   ├── xgboost_baseline.py    # Task 8
│   │   ├── cnn.py                 # Task 14
│   │   └── train_cnn.py           # Task 15
│   └── evaluation/
│       ├── metrics.py             # Task 10
│       └── failure_cases.py       # Task 16
└── tests/
    ├── data/  (test_indices.py, test_sampling.py, test_postprocessing.py,
    │           test_split.py, test_patches.py, test_patch_dataset.py)
    ├── models/ (test_xgboost_baseline.py, test_cnn.py)
    └── evaluation/ (test_metrics.py, test_failure_cases.py)
```

---

## Task 0: Repository and environment setup

**Files:**
- Create: `.gitignore`
- Modify: `requirements.txt`
- Test: none (setup task; verification step replaces the test cycle)

- [ ] **Step 1: Initialize git**

```powershell
git init
git add README.md requirements.txt reports configs src data papers
git status
```

- [ ] **Step 2: Write `.gitignore`**

```
# Environments
.venv/
__pycache__/
*.pyc

# Data (large, regenerable, not versioned)
data/raw/*
data/processed/*
!data/raw/.gitkeep
!data/processed/.gitkeep
data/external/*
!data/external/.gitkeep

# Reference PDFs (large binaries; keep the .md notes)
papers/*.pdf

# Results
results/figures/*
results/logs/*
!results/figures/.gitkeep
!results/logs/.gitkeep

# Editors
.ipynb_checkpoints/
```

- [ ] **Step 3: Create and activate a virtual environment, verify Python version compatibility**

The system Python is 3.14.5 — very recent, and PyTorch/XGBoost wheel availability for it is not guaranteed. Verify before committing to it:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

If this fails with "no matching distribution," Python 3.14 is not yet supported by PyTorch's published wheels. Fall back to a pinned interpreter:

```powershell
winget install Python.Python.3.12
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

- [ ] **Step 4: Update `requirements.txt`**

```
numpy
pandas
scikit-learn
scikit-image
matplotlib
rasterio
geopandas
shapely
xgboost
torch
torchvision
earthengine-api
pyyaml
pytest
tqdm
requests
```

```powershell
pip install -r requirements.txt
```

- [ ] **Step 5: Verify the install**

```powershell
python -c "import torch, xgboost, ee, geopandas, rasterio, skimage; print('torch cuda:', torch.cuda.is_available()); print('all imports OK')"
```

Expected: `torch cuda: True` (confirms the RTX 2070 Super is visible to PyTorch) and `all imports OK`.

- [ ] **Step 6: Authenticate Earth Engine (manual, one-time, cannot be scripted here)**

```powershell
python -c "import ee; ee.Authenticate()"
```

This opens a browser flow tied to a Google Cloud project. You need a registered (free, non-commercial) Earth Engine Cloud project ID — note it down, it goes into `configs/config.yaml` in Task 2.

- [ ] **Step 7: Commit**

```powershell
git add .gitignore requirements.txt
git commit -m "chore: initialize repository, environment, and dependencies"
```

---

## Task 1: Acquire and verify the PV polygon dataset

**Files:**
- Create: `scripts/download_pv_polygons.py`
- Test: `tests/data/test_polygon_download.py`

**Interfaces:**
- Produces: `data/external/pv_polygons.gpkg` (GeoPackage, EPSG:4326), used by Task 6 (sampling) and Task 9 (post-processing DID-proxy filter).

- [ ] **Step 1: Write the failing test**

```python
# tests/data/test_polygon_download.py
import geopandas as gpd
from pathlib import Path

POLYGON_PATH = Path("data/external/pv_polygons.gpkg")

def test_polygon_file_exists_and_loads():
    assert POLYGON_PATH.exists(), "run scripts/download_pv_polygons.py first"
    gdf = gpd.read_file(POLYGON_PATH)
    assert len(gdf) > 0
    assert gdf.crs is not None

def test_polygon_total_area_matches_published_figure():
    gdf = gpd.read_file(POLYGON_PATH)
    gdf_m = gdf.to_crs(epsg=6690)  # JGD2011 / Japan Plane Rectangular equal-area-ish for a national extent check
    total_km2 = gdf_m.geometry.area.sum() / 1e6
    # Shimada & Takeuchi (2025) report 571 km^2; allow 5% tolerance for
    # reprojection/precision differences.
    assert 542 <= total_km2 <= 600, f"got {total_km2:.1f} km^2, expected ~571"
```

- [ ] **Step 2: Run test to verify it fails**

```powershell
pytest tests/data/test_polygon_download.py -v
```

Expected: FAIL — `data/external/pv_polygons.gpkg` does not exist.

- [ ] **Step 3: Write the download script**

```python
# scripts/download_pv_polygons.py
"""Download the Shimada & Takeuchi (2025) PV polygon dataset from Zenodo
(DOI 10.5281/zenodo.10674681) and save it as a single GeoPackage."""
import geopandas as gpd
import requests
from pathlib import Path

ZENODO_RECORD_ID = "10674681"
OUT_DIR = Path("data/external")
OUT_PATH = OUT_DIR / "pv_polygons.gpkg"


def fetch_file_list() -> list[dict]:
    resp = requests.get(f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}", timeout=30)
    resp.raise_for_status()
    return resp.json()["files"]


def download_file(file_entry: dict, dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / file_entry["key"]
    url = file_entry["links"]["self"]
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    return dest


def main() -> None:
    files = fetch_file_list()
    vector_exts = (".gpkg", ".shp", ".geojson", ".zip")
    vector_files = [f for f in files if f["key"].lower().endswith(vector_exts)]
    if not vector_files:
        raise RuntimeError(
            f"No vector file found in Zenodo record {ZENODO_RECORD_ID}. "
            f"Files present: {[f['key'] for f in files]}. "
            "Inspect the record manually and adjust vector_exts / extraction logic."
        )
    raw_path = download_file(vector_files[0], OUT_DIR / "raw")
    gdf = gpd.read_file(raw_path)
    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=4326)
    else:
        gdf = gdf.to_crs(epsg=4326)
    gdf.to_file(OUT_PATH, driver="GPKG")
    print(f"Saved {len(gdf)} polygons to {OUT_PATH}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the script, then run the test**

```powershell
python scripts/download_pv_polygons.py
pytest tests/data/test_polygon_download.py -v
```

Expected: PASS. If the area assertion fails, open the Zenodo record page manually in a browser, confirm which file is the actual polygon layer (the record may bundle multiple files), and adjust `vector_exts`/file selection in the script accordingly — do not loosen the area tolerance to force a pass.

- [ ] **Step 5: Commit**

```powershell
git add scripts/download_pv_polygons.py tests/data/test_polygon_download.py
git commit -m "feat: download and verify Shimada & Takeuchi PV polygon dataset"
```

---

## Task 2: Config module

**Files:**
- Create: `src/config.py`
- Create: `configs/config.yaml`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `load_config(path: str | Path = "configs/config.yaml") -> Config` dataclass, consumed by every later task that needs paths, dates, or hyperparameters.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_config.py
from src.config import load_config

def test_load_config_returns_expected_fields(tmp_path):
    cfg_yaml = tmp_path / "config.yaml"
    cfg_yaml.write_text(
        """
gee_project_id: "my-test-project"
sentinel2:
  start_date: "2022-09-01"
  end_date: "2022-11-30"
  bands: [B2, B3, B4, B8, B11, B12]
xgboost:
  max_depth: 8
  eta: 0.01717
  n_estimators: 1000
sampling:
  n_sampling_squares: 1000
  pv_squares: 600
  buffer_squares: 300
  urban_squares: 100
  buffer_km: 7
patch:
  size_px: 21
paths:
  pv_polygons: "data/external/pv_polygons.gpkg"
"""
    )
    cfg = load_config(cfg_yaml)
    assert cfg.gee_project_id == "my-test-project"
    assert cfg.sentinel2.bands == ["B2", "B3", "B4", "B8", "B11", "B12"]
    assert cfg.xgboost.max_depth == 8
    assert cfg.xgboost.eta == 0.01717
    assert cfg.sampling.pv_squares == 600
    assert cfg.patch.size_px == 21
```

- [ ] **Step 2: Run test to verify it fails**

```powershell
pytest tests/test_config.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'src.config'`.

- [ ] **Step 3: Implement the config module**

```python
# src/config.py
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import yaml


@dataclass
class Sentinel2Config:
    start_date: str
    end_date: str
    bands: list[str]


@dataclass
class XGBoostConfig:
    max_depth: int
    eta: float
    n_estimators: int


@dataclass
class SamplingConfig:
    n_sampling_squares: int
    pv_squares: int
    buffer_squares: int
    urban_squares: int
    buffer_km: float


@dataclass
class PatchConfig:
    size_px: int


@dataclass
class PathsConfig:
    pv_polygons: str


@dataclass
class Config:
    gee_project_id: str
    sentinel2: Sentinel2Config
    xgboost: XGBoostConfig
    sampling: SamplingConfig
    patch: PatchConfig
    paths: PathsConfig


def load_config(path: str | Path = "configs/config.yaml") -> Config:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return Config(
        gee_project_id=raw["gee_project_id"],
        sentinel2=Sentinel2Config(**raw["sentinel2"]),
        xgboost=XGBoostConfig(**raw["xgboost"]),
        sampling=SamplingConfig(**raw["sampling"]),
        patch=PatchConfig(**raw["patch"]),
        paths=PathsConfig(**raw["paths"]),
    )
```

- [ ] **Step 4: Run test to verify it passes**

```powershell
pytest tests/test_config.py -v
```

Expected: PASS.

- [ ] **Step 5: Write the real `configs/config.yaml`**

```yaml
gee_project_id: "REPLACE-WITH-YOUR-GEE-CLOUD-PROJECT-ID"
sentinel2:
  start_date: "2022-09-01"
  end_date: "2022-11-30"
  bands: [B2, B3, B4, B8, B11, B12]
xgboost:
  max_depth: 8
  eta: 0.01717
  n_estimators: 1000
sampling:
  n_sampling_squares: 1000
  pv_squares: 600
  buffer_squares: 300
  urban_squares: 100
  buffer_km: 7
patch:
  size_px: 21
paths:
  pv_polygons: "data/external/pv_polygons.gpkg"
```

- [ ] **Step 6: Commit**

```powershell
git add src/config.py configs/config.yaml tests/test_config.py
git commit -m "feat: add YAML-backed config module"
```

---

## Task 3: Spectral indices and pre-masking rules

**Files:**
- Create: `src/data/indices.py`
- Test: `tests/data/test_indices.py`

**Interfaces:**
- Produces: `ndvi`, `mndwi`, `pvsi`, `bsi`, `premask` functions operating on `numpy.ndarray` band arrays (any shape, elementwise). Consumed by Task 7 (feature extraction) and Task 9 (post-processing).

- [ ] **Step 1: Write the failing tests**

```python
# tests/data/test_indices.py
import numpy as np
from src.data.indices import ndvi, mndwi, pvsi, bsi, premask

def test_ndvi_known_values():
    nir = np.array([0.5])
    red = np.array([0.1])
    result = ndvi(nir, red)
    np.testing.assert_allclose(result, [(0.5 - 0.1) / (0.5 + 0.1)])

def test_mndwi_known_values():
    green = np.array([0.4])
    swir1 = np.array([0.1])
    result = mndwi(green, swir1)
    np.testing.assert_allclose(result, [(0.4 - 0.1) / (0.4 + 0.1)])

def test_bsi_known_values():
    blue = np.array([0.3])
    red = np.array([0.2])
    result = bsi(blue, red)
    np.testing.assert_allclose(result, [(0.3 - 0.2) / (0.3 + 0.2)])

def test_pvsi_known_values():
    swir1 = np.array([0.3])
    swir2 = np.array([0.2])
    nir = np.array([0.25])
    blue = np.array([0.1])
    red = np.array([0.15])
    ratio_term = (2 * 0.3 - 0.2 - 0.25) / (2 * 0.3 + 0.2 + 0.25)
    expected = ratio_term * (0.1 / 0.15)
    result = pvsi(swir1, swir2, nir, blue, red)
    np.testing.assert_allclose(result, [expected])

def test_indices_handle_zero_denominator_without_nan():
    nir = np.array([0.0])
    red = np.array([0.0])
    result = ndvi(nir, red)
    assert np.isfinite(result[0])

def test_premask_flags_dense_vegetation():
    mask = premask(ndvi_val=np.array([0.6]), mndwi_val=np.array([-0.5]), bsi_val=np.array([0.0]))
    assert mask[0] == False  # noqa: E712 -- masked out (not a PV candidate)

def test_premask_flags_water():
    mask = premask(ndvi_val=np.array([0.0]), mndwi_val=np.array([0.1]), bsi_val=np.array([0.0]))
    assert mask[0] == False  # noqa: E712

def test_premask_flags_bluish_nonpv_material():
    mask = premask(ndvi_val=np.array([0.0]), mndwi_val=np.array([-0.5]), bsi_val=np.array([0.25]))
    assert mask[0] == False  # noqa: E712

def test_premask_keeps_plausible_pv_pixel():
    mask = premask(ndvi_val=np.array([0.1]), mndwi_val=np.array([-0.5]), bsi_val=np.array([0.0]))
    assert mask[0] == True  # noqa: E712
```

- [ ] **Step 2: Run tests to verify they fail**

```powershell
pytest tests/data/test_indices.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'src.data.indices'`.

- [ ] **Step 3: Implement**

```python
# src/data/indices.py
"""Spectral index formulas and pre-masking rules from Shimada & Takeuchi
(2025), as documented in papers/spectral-baseline-specifications.md."""
import numpy as np

_EPS = 1e-9


def ndvi(nir: np.ndarray, red: np.ndarray) -> np.ndarray:
    return (nir - red) / (nir + red + _EPS)


def mndwi(green: np.ndarray, swir1: np.ndarray) -> np.ndarray:
    return (green - swir1) / (green + swir1 + _EPS)


def bsi(blue: np.ndarray, red: np.ndarray) -> np.ndarray:
    return (blue - red) / (blue + red + _EPS)


def pvsi(swir1: np.ndarray, swir2: np.ndarray, nir: np.ndarray,
         blue: np.ndarray, red: np.ndarray) -> np.ndarray:
    ratio_term = (2 * swir1 - swir2 - nir) / (2 * swir1 + swir2 + nir + _EPS)
    return ratio_term * (blue / (red + _EPS))


def premask(ndvi_val: np.ndarray, mndwi_val: np.ndarray, bsi_val: np.ndarray) -> np.ndarray:
    """True where a pixel remains a plausible PV candidate; False where the
    original study's rule-based masks exclude it (dense vegetation,
    water, or bright bluish non-PV material)."""
    dense_vegetation = ndvi_val > 0.5
    water = mndwi_val > 0
    bluish_nonpv = bsi_val > 0.2
    excluded = dense_vegetation | water | bluish_nonpv
    return ~excluded
```

- [ ] **Step 4: Run tests to verify they pass**

```powershell
pytest tests/data/test_indices.py -v
```

Expected: PASS, all 8 tests.

- [ ] **Step 5: Commit**

```powershell
git add src/data/indices.py tests/data/test_indices.py
git commit -m "feat: add spectral index formulas and pre-masking rules"
```

---

## Task 4: Built-up / populated-area mask utility (GEE)

**Files:**
- Create: `src/data/builtup.py`
- Test: `tests/data/test_builtup_smoke.py` (manual smoke test, not part of the automated suite — see Global Constraints on GEE testing philosophy)

**Interfaces:**
- Produces: `get_populated_buffer(aoi: ee.Geometry, buffer_km: float) -> ee.FeatureCollection` and `sample_urban_core_points(aoi: ee.Geometry, n_points: int, built_up_percentile: float = 90) -> ee.FeatureCollection`. Consumed by Task 6 (sampling) and Task 9 (DID-proxy filtering).

**Design note:** the original study's non-PV sampling used a manually delineated 7km buffer around "populated areas" and manually selected urban points (`papers/zenodo-dataset-verification.md` §2). Neither "populated areas" nor "urban" has a literal automatable definition in the public release, so this task substitutes the JRC Global Human Settlement Layer (GHSL) built-up surface (`JRC/GHSL/P2023A/GHS_BUILT_S/2020`, available directly in Earth Engine's public catalog) thresholded to a built-up mask, as a documented, reproducible proxy. State this substitution explicitly in the thesis's Limitations chapter — it is a deviation from manual delineation, not a hidden one.

- [ ] **Step 1: Implement**

```python
# src/data/builtup.py
"""Built-up-area proxy for the original study's manually delineated
"populated areas" / "urban" sampling strata, using the JRC Global Human
Settlement Layer (GHSL) built-up surface product in Earth Engine."""
import ee

GHSL_BUILT_UP_IMAGE = "JRC/GHSL/P2023A/GHS_BUILT_S/2020"


def _built_up_mask(percentile_image: ee.Image, threshold: float) -> ee.Image:
    return percentile_image.gt(threshold)


def get_populated_buffer(aoi: ee.Geometry, buffer_km: float,
                          built_up_threshold_m2: float = 1000) -> ee.FeatureCollection:
    """Vectorized buffer around GHSL-flagged built-up pixels within aoi."""
    built_up = ee.Image(GHSL_BUILT_UP_IMAGE).select("built_surface").clip(aoi)
    mask = _built_up_mask(built_up, built_up_threshold_m2)
    vectors = mask.selfMask().reduceToVectors(
        geometry=aoi, scale=100, geometryType="polygon", maxPixels=1e10
    )
    buffered = vectors.map(lambda f: f.buffer(buffer_km * 1000))
    return buffered


def sample_urban_core_points(aoi: ee.Geometry, n_points: int,
                              built_up_threshold_m2: float = 5000) -> ee.FeatureCollection:
    """Random points within the densest built-up pixels in aoi, as a proxy
    for the original study's manually selected urban non-PV samples."""
    built_up = ee.Image(GHSL_BUILT_UP_IMAGE).select("built_surface").clip(aoi)
    dense_core = _built_up_mask(built_up, built_up_threshold_m2).selfMask()
    return dense_core.stratifiedSample(
        numPoints=n_points, classBand="built_surface", region=aoi,
        scale=100, geometries=True,
    )
```

- [ ] **Step 2: Write the manual smoke-test script (run once, by hand, after `ee.Authenticate()`)**

```python
# tests/data/test_builtup_smoke.py
"""Not collected by pytest automatically (requires live GEE auth + network).
Run manually: python tests/data/test_builtup_smoke.py
"""
import ee
from src.config import load_config
from src.data.builtup import get_populated_buffer, sample_urban_core_points


def main():
    cfg = load_config()
    ee.Initialize(project=cfg.gee_project_id)
    japan_aoi = ee.Geometry.Rectangle([129.0, 31.0, 146.0, 46.0])  # rough Japan bbox

    buffer_fc = get_populated_buffer(japan_aoi, buffer_km=cfg.sampling.buffer_km)
    print("Populated buffer feature count:", buffer_fc.size().getInfo())

    urban_points = sample_urban_core_points(japan_aoi, n_points=10)
    print("Urban core sample points:", urban_points.size().getInfo())
    first = urban_points.first().getInfo()
    print("First point geometry:", first["geometry"])


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run it manually and verify sane output**

```powershell
python tests/data/test_builtup_smoke.py
```

Expected: two non-zero counts printed, and a valid `[lon, lat]` point inside Japan's bounding box. If `Initialize` fails, re-run `ee.Authenticate()` from Task 0 Step 6 and confirm `gee_project_id` in `configs/config.yaml` is correct.

- [ ] **Step 4: Commit**

```powershell
git add src/data/builtup.py tests/data/test_builtup_smoke.py
git commit -m "feat: add GHSL-based built-up-area proxy for populated/urban sampling strata"
```

---

## Task 5: Sentinel-2 L2A autumn-2022 composite (GEE)

**Files:**
- Create: `src/data/sentinel2.py`
- Test: `tests/data/test_sentinel2_smoke.py` (manual smoke test)

**Interfaces:**
- Produces: `get_autumn_2022_composite(aoi: ee.Geometry, cfg: Sentinel2Config) -> ee.Image` with bands renamed to `B2, B3, B4, B8, B11, B12`. Consumed by Task 7 (feature extraction) and Task 12 (patch extraction).

- [ ] **Step 1: Implement**

```python
# src/data/sentinel2.py
"""Sentinel-2 Level-2A surface reflectance composite, autumn 2022, matching
Shimada & Takeuchi (2025)'s acquisition window (reports/paper-draft.md §3.1)."""
import ee
from src.config import Sentinel2Config

S2_COLLECTION = "COPERNICUS/S2_SR_HARMONIZED"
CLOUD_PROB_COLLECTION = "COPERNICUS/S2_CLOUD_PROBABILITY"
CLOUD_PROB_THRESHOLD = 40


def _mask_clouds(image: ee.Image) -> ee.Image:
    cloud_prob = ee.Image(image.get("cloud_probability")).select("probability")
    is_clear = cloud_prob.lt(CLOUD_PROB_THRESHOLD)
    return image.updateMask(is_clear)


def get_autumn_2022_composite(aoi: ee.Geometry, cfg: Sentinel2Config) -> ee.Image:
    s2 = (
        ee.ImageCollection(S2_COLLECTION)
        .filterBounds(aoi)
        .filterDate(cfg.start_date, cfg.end_date)
    )
    cloud_prob = (
        ee.ImageCollection(CLOUD_PROB_COLLECTION)
        .filterBounds(aoi)
        .filterDate(cfg.start_date, cfg.end_date)
    )
    joined = ee.Join.saveFirst("cloud_probability").apply(
        primary=s2, secondary=cloud_prob,
        condition=ee.Filter.equals(leftField="system:index", rightField="system:index"),
    )
    masked = ee.ImageCollection(joined).map(_mask_clouds)
    composite = masked.select(cfg.bands).median().clip(aoi)
    return composite.rename(cfg.bands).multiply(0.0001)  # scale to reflectance [0,1]
```

- [ ] **Step 2: Write the manual smoke-test script**

```python
# tests/data/test_sentinel2_smoke.py
"""Run manually: python tests/data/test_sentinel2_smoke.py"""
import ee
from src.config import load_config
from src.data.sentinel2 import get_autumn_2022_composite


def main():
    cfg = load_config()
    ee.Initialize(project=cfg.gee_project_id)
    small_aoi = ee.Geometry.Rectangle([139.6, 35.5, 139.8, 35.7])  # Tokyo area

    composite = get_autumn_2022_composite(small_aoi, cfg.sentinel2)
    band_names = composite.bandNames().getInfo()
    print("Bands:", band_names)
    assert band_names == cfg.sentinel2.bands, band_names

    stats = composite.reduceRegion(
        reducer=ee.Reducer.mean(), geometry=small_aoi, scale=10, maxPixels=1e9
    ).getInfo()
    print("Mean reflectance per band:", stats)
    for band, value in stats.items():
        assert value is not None and 0.0 <= value <= 1.0, f"{band}={value} out of range"


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run it manually and verify sane output**

```powershell
python tests/data/test_sentinel2_smoke.py
```

Expected: `Bands: ['B2', 'B3', 'B4', 'B8', 'B11', 'B12']`, and mean reflectance values between 0 and 1 for all six bands (surface reflectance is a fraction, not a raw DN — the `0.0001` scale factor is what makes this true; if you see values in the thousands, the multiply step was skipped).

- [ ] **Step 4: Commit**

```powershell
git add src/data/sentinel2.py tests/data/test_sentinel2_smoke.py
git commit -m "feat: add Sentinel-2 L2A autumn-2022 composite fetch"
```

---

## Task 6: Reconstructed point-sampling protocol

**Files:**
- Create: `src/data/sampling.py`
- Test: `tests/data/test_sampling.py`

**Interfaces:**
- Consumes: `get_populated_buffer`, `sample_urban_core_points` (Task 4); PV polygons loaded via `geopandas.read_file(cfg.paths.pv_polygons)` (Task 1); `SamplingConfig` (Task 2).
- Produces: `generate_sampling_squares(pv_gdf, populated_buffer_gdf, cfg: SamplingConfig, seed: int) -> geopandas.GeoDataFrame` (1000 squares, tagged by stratum) and `sample_points_in_squares(squares_gdf, pv_gdf, n_pv: int, n_nonpv: int, seed: int) -> geopandas.GeoDataFrame` (points tagged `label ∈ {pv, non_pv}`). Consumed by Task 7 (feature extraction) and Task 12 (patch extraction).

The geometry logic (square placement, point sampling within squares, stratum proportions) is pure and local — fully unit-testable without GEE, using a synthetic polygon layer standing in for the real one.

- [ ] **Step 1: Write the failing tests**

```python
# tests/data/test_sampling.py
import geopandas as gpd
from shapely.geometry import Polygon, box
from src.config import SamplingConfig
from src.data.sampling import generate_sampling_squares, sample_points_in_squares

def _synthetic_pv_gdf():
    polys = [box(139.70 + i * 0.01, 35.60, 139.71 + i * 0.01, 35.61) for i in range(20)]
    return gpd.GeoDataFrame({"geometry": polys}, crs="EPSG:4326")

def _synthetic_buffer_gdf():
    return gpd.GeoDataFrame({"geometry": [box(139.5, 35.4, 140.0, 35.9)]}, crs="EPSG:4326")

def _cfg():
    return SamplingConfig(
        n_sampling_squares=100, pv_squares=60, buffer_squares=30,
        urban_squares=10, buffer_km=7,
    )

def test_generate_sampling_squares_respects_stratum_counts():
    squares = generate_sampling_squares(
        _synthetic_pv_gdf(), _synthetic_buffer_gdf(), _cfg(), seed=42
    )
    assert len(squares) == 100
    assert (squares["stratum"] == "pv").sum() == 60
    assert (squares["stratum"] == "buffer").sum() == 30
    assert (squares["stratum"] == "urban").sum() == 10

def test_generate_sampling_squares_pv_squares_intersect_pv_polygons():
    pv_gdf = _synthetic_pv_gdf()
    squares = generate_sampling_squares(pv_gdf, _synthetic_buffer_gdf(), _cfg(), seed=42)
    pv_squares = squares[squares["stratum"] == "pv"]
    union = pv_gdf.geometry.union_all()
    assert all(sq.intersects(union) for sq in pv_squares.geometry)

def test_generate_sampling_squares_is_deterministic_given_seed():
    a = generate_sampling_squares(_synthetic_pv_gdf(), _synthetic_buffer_gdf(), _cfg(), seed=1)
    b = generate_sampling_squares(_synthetic_pv_gdf(), _synthetic_buffer_gdf(), _cfg(), seed=1)
    assert a.geometry.equals(b.geometry).all()

def test_sample_points_in_squares_produces_requested_counts_and_labels():
    squares = generate_sampling_squares(
        _synthetic_pv_gdf(), _synthetic_buffer_gdf(), _cfg(), seed=42
    )
    points = sample_points_in_squares(
        squares, _synthetic_pv_gdf(), n_pv=50, n_nonpv=100, seed=42
    )
    assert (points["label"] == "pv").sum() == 50
    assert (points["label"] == "non_pv").sum() == 100
    assert points.geometry.geom_type.eq("Point").all()

def test_sample_points_pv_points_fall_inside_pv_polygons():
    pv_gdf = _synthetic_pv_gdf()
    squares = generate_sampling_squares(pv_gdf, _synthetic_buffer_gdf(), _cfg(), seed=42)
    points = sample_points_in_squares(squares, pv_gdf, n_pv=20, n_nonpv=20, seed=42)
    pv_points = points[points["label"] == "pv"]
    union = pv_gdf.geometry.union_all()
    assert all(p.within(union) for p in pv_points.geometry)
```

- [ ] **Step 2: Run tests to verify they fail**

```powershell
pytest tests/data/test_sampling.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'src.data.sampling'`.

- [ ] **Step 3: Implement**

```python
# src/data/sampling.py
"""Reconstructed version of Shimada & Takeuchi (2025)'s point-sampling
protocol (papers/zenodo-dataset-verification.md §2), built from the
polygon-only public release rather than the original's unreleased point
samples."""
import geopandas as gpd
import numpy as np
from shapely.geometry import Point, box
from src.config import SamplingConfig


def _random_squares_within(bounds_gdf: gpd.GeoDataFrame, n: int, side_deg: float,
                            rng: np.random.Generator) -> list:
    minx, miny, maxx, maxy = bounds_gdf.total_bounds
    squares = []
    while len(squares) < n:
        x = rng.uniform(minx, maxx - side_deg)
        y = rng.uniform(miny, maxy - side_deg)
        squares.append(box(x, y, x + side_deg, y + side_deg))
    return squares


def _squares_centered_on_features(features_gdf: gpd.GeoDataFrame, n: int, side_deg: float,
                                   rng: np.random.Generator) -> list:
    idx = rng.integers(0, len(features_gdf), size=n)
    squares = []
    for i in idx:
        cx, cy = features_gdf.geometry.iloc[i].centroid.coords[0]
        squares.append(box(cx - side_deg / 2, cy - side_deg / 2, cx + side_deg / 2, cy + side_deg / 2))
    return squares


def generate_sampling_squares(pv_gdf: gpd.GeoDataFrame, populated_buffer_gdf: gpd.GeoDataFrame,
                               cfg: SamplingConfig, seed: int, side_deg: float = 0.009
                               ) -> gpd.GeoDataFrame:
    """~1000 1km x 1km squares (side_deg ~= 0.009 deg at Japan's latitude),
    split pv/buffer/urban strata exactly as in the original protocol."""
    rng = np.random.default_rng(seed)
    pv_squares = _squares_centered_on_features(pv_gdf, cfg.pv_squares, side_deg, rng)
    buffer_squares = _random_squares_within(populated_buffer_gdf, cfg.buffer_squares, side_deg, rng)
    urban_squares = _random_squares_within(populated_buffer_gdf, cfg.urban_squares, side_deg, rng)

    geoms = pv_squares + buffer_squares + urban_squares
    strata = (["pv"] * len(pv_squares) + ["buffer"] * len(buffer_squares)
              + ["urban"] * len(urban_squares))
    return gpd.GeoDataFrame({"stratum": strata, "geometry": geoms}, crs=pv_gdf.crs)


def _random_points_in_polygon(poly, n: int, rng: np.random.Generator) -> list[Point]:
    minx, miny, maxx, maxy = poly.bounds
    points = []
    attempts = 0
    max_attempts = n * 200
    while len(points) < n and attempts < max_attempts:
        p = Point(rng.uniform(minx, maxx), rng.uniform(miny, maxy))
        if poly.contains(p):
            points.append(p)
        attempts += 1
    return points


def sample_points_in_squares(squares_gdf: gpd.GeoDataFrame, pv_gdf: gpd.GeoDataFrame,
                              n_pv: int, n_nonpv: int, seed: int) -> gpd.GeoDataFrame:
    rng = np.random.default_rng(seed)
    pv_union = pv_gdf.geometry.union_all()
    squares_union = squares_gdf.geometry.union_all()

    pv_region = squares_union.intersection(pv_union)
    pv_points = _random_points_in_polygon(pv_region, n_pv, rng) if not pv_region.is_empty else []

    nonpv_region = squares_union.difference(pv_union)
    nonpv_points = (_random_points_in_polygon(nonpv_region, n_nonpv, rng)
                     if not nonpv_region.is_empty else [])

    geoms = pv_points + nonpv_points
    labels = ["pv"] * len(pv_points) + ["non_pv"] * len(nonpv_points)
    return gpd.GeoDataFrame({"label": labels, "geometry": geoms}, crs=squares_gdf.crs)
```

- [ ] **Step 4: Run tests to verify they pass**

```powershell
pytest tests/data/test_sampling.py -v
```

Expected: PASS, all 6 tests. Note `test_sample_points_in_squares_produces_requested_counts_and_labels` and the "inside PV polygons" test can be flaky if `_random_points_in_polygon`'s rejection sampling can't find enough points in a very thin region — if that happens with the synthetic fixtures, increase `max_attempts` rather than reducing `n_pv`/`n_nonpv` in the test.

- [ ] **Step 5: Commit**

```powershell
git add src/data/sampling.py tests/data/test_sampling.py
git commit -m "feat: reconstruct stratified point-sampling protocol from PV polygons"
```

---

## Task 7: Feature extraction pipeline

**Files:**
- Create: `src/data/features.py`
- Test: `tests/data/test_features.py`

**Interfaces:**
- Consumes: sampled points GeoDataFrame (Task 6), `get_autumn_2022_composite` (Task 5), `ndvi/mndwi/pvsi/bsi/premask` (Task 3).
- Produces: `build_feature_matrix(points_gdf, band_values_df) -> pandas.DataFrame` with columns `[B2,B3,B4,B8,B11,B12,NDVI,MNDWI,PVSI,valid_candidate,label]`. Consumed by Task 8 (XGBoost) and Task 11 (spatial split).

This task splits network I/O (sampling band values at points via GEE — not unit-tested, manual smoke test only) from the pure feature-assembly logic (fully unit-tested with synthetic band values).

- [ ] **Step 1: Write the failing test for the pure assembly function**

```python
# tests/data/test_features.py
import pandas as pd
from src.data.features import build_feature_matrix

def test_build_feature_matrix_computes_indices_and_validity():
    band_values = pd.DataFrame({
        "B2": [0.10, 0.30], "B3": [0.15, 0.20], "B4": [0.12, 0.25],
        "B8": [0.40, 0.22], "B11": [0.20, 0.10], "B12": [0.15, 0.08],
    })
    labels = pd.Series(["pv", "non_pv"])

    df = build_feature_matrix(band_values, labels)

    assert list(df.columns) == [
        "B2", "B3", "B4", "B8", "B11", "B12", "NDVI", "MNDWI", "PVSI",
        "valid_candidate", "label",
    ]
    assert len(df) == 2
    assert df["label"].tolist() == ["pv", "non_pv"]
    assert df["valid_candidate"].dtype == bool

def test_build_feature_matrix_flags_dense_vegetation_as_invalid():
    band_values = pd.DataFrame({
        "B2": [0.05], "B3": [0.08], "B4": [0.04],
        "B8": [0.50], "B11": [0.10], "B12": [0.08],
    })
    labels = pd.Series(["non_pv"])
    df = build_feature_matrix(band_values, labels)
    assert df["NDVI"].iloc[0] > 0.5
    assert df["valid_candidate"].iloc[0] == False  # noqa: E712
```

- [ ] **Step 2: Run test to verify it fails**

```powershell
pytest tests/data/test_features.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'src.data.features'`.

- [ ] **Step 3: Implement**

```python
# src/data/features.py
"""Assemble the 9-input-feature matrix (6 reflectance bands + NDVI/MNDWI/
PVSI) and pre-masking validity flag, per papers/spectral-baseline-specifications.md."""
import geopandas as gpd
import pandas as pd
from src.data.indices import ndvi, mndwi, pvsi, bsi, premask


def build_feature_matrix(band_values: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
    df = band_values.copy()
    df["NDVI"] = ndvi(df["B8"].values, df["B4"].values)
    df["MNDWI"] = mndwi(df["B3"].values, df["B11"].values)
    df["PVSI"] = pvsi(df["B11"].values, df["B12"].values, df["B8"].values,
                       df["B2"].values, df["B4"].values)
    bsi_val = bsi(df["B2"].values, df["B4"].values)
    df["valid_candidate"] = premask(df["NDVI"].values, df["MNDWI"].values, bsi_val)
    df["label"] = labels.values
    return df[["B2", "B3", "B4", "B8", "B11", "B12", "NDVI", "MNDWI", "PVSI",
               "valid_candidate", "label"]]


def sample_band_values_at_points(composite, points_gdf: gpd.GeoDataFrame,
                                  bands: list[str], scale: int = 10) -> pd.DataFrame:
    """GEE-networked: samples `composite` at each point in points_gdf.
    Not unit-tested (see Task 5's testing-philosophy note) — verify with
    the manual smoke test in Step 4."""
    import ee
    fc = ee.FeatureCollection([
        ee.Feature(ee.Geometry.Point([geom.x, geom.y]))
        for geom in points_gdf.geometry
    ])
    sampled = composite.sampleRegions(collection=fc, scale=scale, geometries=False)
    records = sampled.getInfo()["features"]
    rows = [r["properties"] for r in records]
    return pd.DataFrame(rows)[bands]
```

- [ ] **Step 4: Run test to verify it passes, then write and run a manual smoke test for the networked function**

```powershell
pytest tests/data/test_features.py -v
```

Expected: PASS, both tests.

```python
# tests/data/test_features_smoke.py -- run manually:
# python tests/data/test_features_smoke.py
import ee
from src.config import load_config
from src.data.sentinel2 import get_autumn_2022_composite
from src.data.features import sample_band_values_at_points, build_feature_matrix
import geopandas as gpd
from shapely.geometry import Point

def main():
    cfg = load_config()
    ee.Initialize(project=cfg.gee_project_id)
    aoi = ee.Geometry.Rectangle([139.6, 35.5, 139.8, 35.7])
    composite = get_autumn_2022_composite(aoi, cfg.sentinel2)
    points = gpd.GeoDataFrame(
        {"geometry": [Point(139.70, 35.60), Point(139.75, 35.65)]}, crs="EPSG:4326"
    )
    band_values = sample_band_values_at_points(composite, points, cfg.sentinel2.bands)
    print(band_values)
    labels = ["pv", "non_pv"]
    features = build_feature_matrix(band_values, __import__("pandas").Series(labels))
    print(features)

if __name__ == "__main__":
    main()
```

```powershell
python tests/data/test_features_smoke.py
```

Expected: two rows of plausible reflectance values (roughly 0–0.4 range) and a fully populated feature matrix with no NaNs.

- [ ] **Step 5: Commit**

```powershell
git add src/data/features.py tests/data/test_features.py tests/data/test_features_smoke.py
git commit -m "feat: add feature matrix assembly and GEE band sampling"
```

---

## Task 8: XGBoost baseline classifier

**Files:**
- Create: `src/models/xgboost_baseline.py`
- Test: `tests/models/test_xgboost_baseline.py`

**Interfaces:**
- Consumes: feature DataFrame from Task 7 (columns `B2..PVSI, valid_candidate, label`), `XGBoostConfig` (Task 2).
- Produces: `train_baseline(train_df, cfg: XGBoostConfig) -> xgboost.XGBClassifier` and `predict_proba(model, df) -> numpy.ndarray` (P(PV) per row). Consumed by Task 9 (post-processing) and Task 10 (evaluation).

- [ ] **Step 1: Write the failing tests**

```python
# tests/models/test_xgboost_baseline.py
import numpy as np
import pandas as pd
from src.config import XGBoostConfig
from src.models.xgboost_baseline import train_baseline, predict_proba, FEATURE_COLUMNS

def _synthetic_training_df(n=200, seed=0):
    rng = np.random.default_rng(seed)
    n_pv, n_nonpv = n // 2, n - n // 2
    pv_pvsi = rng.normal(0.6, 0.05, n_pv)
    nonpv_pvsi = rng.normal(0.1, 0.05, n_nonpv)
    pvsi = np.concatenate([pv_pvsi, nonpv_pvsi])
    other = rng.normal(0.2, 0.05, (n, len(FEATURE_COLUMNS) - 1))
    df = pd.DataFrame(other, columns=[c for c in FEATURE_COLUMNS if c != "PVSI"])
    df["PVSI"] = pvsi
    df["label"] = ["pv"] * n_pv + ["non_pv"] * n_nonpv
    return df

def test_train_baseline_uses_fixed_hyperparameters():
    cfg = XGBoostConfig(max_depth=8, eta=0.01717, n_estimators=1000)
    model = train_baseline(_synthetic_training_df(), cfg)
    assert model.get_params()["max_depth"] == 8
    assert abs(model.get_params()["learning_rate"] - 0.01717) < 1e-9
    assert model.get_params()["n_estimators"] == 1000

def test_predict_proba_separates_synthetic_classes():
    cfg = XGBoostConfig(max_depth=4, eta=0.1, n_estimators=50)  # fast for the test
    train_df = _synthetic_training_df(n=400, seed=1)
    model = train_baseline(train_df, cfg)
    test_df = _synthetic_training_df(n=100, seed=2)
    probs = predict_proba(model, test_df)
    pv_probs = probs[test_df["label"] == "pv"]
    nonpv_probs = probs[test_df["label"] == "non_pv"]
    assert pv_probs.mean() > nonpv_probs.mean() + 0.3
```

- [ ] **Step 2: Run tests to verify they fail**

```powershell
pytest tests/models/test_xgboost_baseline.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'src.models.xgboost_baseline'`.

- [ ] **Step 3: Implement**

```python
# src/models/xgboost_baseline.py
"""XGBoost spectral baseline, hyperparameters fixed per
papers/spectral-baseline-specifications.md (Optuna-tuned in the original
study; not re-tuned here — see Global Constraints in plan.md)."""
import numpy as np
import pandas as pd
import xgboost as xgb
from src.config import XGBoostConfig

FEATURE_COLUMNS = ["B2", "B3", "B4", "B8", "B11", "B12", "NDVI", "MNDWI", "PVSI"]


def train_baseline(train_df: pd.DataFrame, cfg: XGBoostConfig) -> xgb.XGBClassifier:
    X = train_df[FEATURE_COLUMNS].values
    y = (train_df["label"] == "pv").astype(int).values
    model = xgb.XGBClassifier(
        max_depth=cfg.max_depth,
        learning_rate=cfg.eta,
        n_estimators=cfg.n_estimators,
        objective="binary:logistic",
        eval_metric="logloss",
        n_jobs=-1,
    )
    model.fit(X, y)
    return model


def predict_proba(model: xgb.XGBClassifier, df: pd.DataFrame) -> np.ndarray:
    X = df[FEATURE_COLUMNS].values
    return model.predict_proba(X)[:, 1]
```

- [ ] **Step 4: Run tests to verify they pass**

```powershell
pytest tests/models/test_xgboost_baseline.py -v
```

Expected: PASS, both tests. `test_predict_proba_separates_synthetic_classes` runs a full 50-tree fit on 400 rows — should complete in well under a second; if it's slow, something's wrong with `n_jobs` or the synthetic data size, not a reason to reduce coverage.

- [ ] **Step 5: Commit**

```powershell
git add src/models/xgboost_baseline.py tests/models/test_xgboost_baseline.py
git commit -m "feat: add XGBoost spectral baseline with fixed replication hyperparameters"
```

---

## Task 9: Post-processing pipeline

**Files:**
- Create: `src/data/postprocessing.py`
- Test: `tests/data/test_postprocessing.py`

**Interfaces:**
- Consumes: a 2D probability raster (`numpy.ndarray`, values in `[0,1]`), optionally a built-up mask raster from Task 4 (as a DID-filtering proxy).
- Produces: `denoise_top_hat(prob_raster) -> numpy.ndarray`, `threshold_probability(prob_raster, threshold=0.6) -> numpy.ndarray` (boolean mask), `segment_polygons(binary_mask, prob_raster) -> list[shapely.geometry.Polygon]` (Felzenszwalb-based), `filter_did_false_positives(polygons, built_up_mask_raster, transform) -> list[shapely.geometry.Polygon]`. Consumed by evaluation (Task 10) when scoring polygon-level output.

- [ ] **Step 1: Write the failing tests**

```python
# tests/data/test_postprocessing.py
import numpy as np
from src.data.postprocessing import denoise_top_hat, threshold_probability, segment_polygons

def test_denoise_top_hat_removes_isolated_single_pixel_noise():
    raster = np.zeros((20, 20))
    raster[10, 10] = 0.9  # isolated single-pixel spike -- should be treated as noise
    raster[3:8, 3:8] = 0.8  # a real 5x5 blob -- should mostly survive
    denoised = denoise_top_hat(raster)
    assert denoised[10, 10] < raster[10, 10]
    assert denoised[5, 5] > 0.3

def test_threshold_probability_matches_expected_pixel_count():
    raster = np.array([[0.9, 0.4], [0.7, 0.1]])
    mask = threshold_probability(raster, threshold=0.6)
    assert mask.tolist() == [[True, False], [True, False]]

def test_segment_polygons_returns_one_polygon_per_blob():
    binary_mask = np.zeros((30, 30), dtype=bool)
    binary_mask[2:6, 2:6] = True
    binary_mask[20:25, 20:25] = True
    prob_raster = binary_mask.astype(float)
    polygons = segment_polygons(binary_mask, prob_raster)
    assert len(polygons) == 2
    for poly in polygons:
        assert poly.area > 0
```

- [ ] **Step 2: Run tests to verify they fail**

```powershell
pytest tests/data/test_postprocessing.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'src.data.postprocessing'`.

- [ ] **Step 3: Implement**

```python
# src/data/postprocessing.py
"""Post-processing pipeline replicating Shimada & Takeuchi (2025):
white top-hat denoise -> probability threshold -> Felzenszwalb segmentation
-> DID (built-up-area) false-positive filtering.
(papers/spectral-baseline-specifications.md §5)"""
import numpy as np
from shapely.geometry import Polygon, shape
from skimage.morphology import white_tophat, disk
from skimage.segmentation import felzenszwalb
from skimage.measure import find_contours


def denoise_top_hat(prob_raster: np.ndarray, radius_px: int = 1) -> np.ndarray:
    footprint = disk(radius_px)
    noise = white_tophat(prob_raster, footprint=footprint)
    return np.clip(prob_raster - noise, 0.0, 1.0)


def threshold_probability(prob_raster: np.ndarray, threshold: float = 0.6) -> np.ndarray:
    return prob_raster >= threshold


def segment_polygons(binary_mask: np.ndarray, prob_raster: np.ndarray) -> list[Polygon]:
    labels = felzenszwalb(prob_raster, scale=50, sigma=0.5, min_size=1)
    labels = np.where(binary_mask, labels, -1)
    polygons = []
    for label_id in np.unique(labels):
        if label_id == -1:
            continue
        blob_mask = (labels == label_id).astype(float)
        contours = find_contours(blob_mask, level=0.5)
        for contour in contours:
            coords = [(x, y) for y, x in contour]
            if len(coords) >= 3:
                poly = Polygon(coords)
                if poly.is_valid and poly.area > 0:
                    polygons.append(poly)
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
```

- [ ] **Step 4: Run tests to verify they pass**

```powershell
pytest tests/data/test_postprocessing.py -v
```

Expected: PASS, all 3 tests.

- [ ] **Step 5: Commit**

```powershell
git add src/data/postprocessing.py tests/data/test_postprocessing.py
git commit -m "feat: add top-hat/threshold/Felzenszwalb/DID-filter post-processing pipeline"
```

---

## Task 10: Evaluation metrics module (shared by baseline and CNN)

**Files:**
- Create: `src/evaluation/metrics.py`
- Test: `tests/evaluation/test_metrics.py`

**Interfaces:**
- Produces: `compute_metrics(y_true: numpy.ndarray, y_pred: numpy.ndarray, y_proba: numpy.ndarray) -> dict` with keys `overall_accuracy, precision, recall, f1, kappa, roc_auc`. Consumed by both the baseline evaluation script and Task 15's CNN training loop.

- [ ] **Step 1: Write the failing tests**

```python
# tests/evaluation/test_metrics.py
import numpy as np
from src.evaluation.metrics import compute_metrics

def test_compute_metrics_perfect_predictions():
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([1, 1, 0, 0])
    y_proba = np.array([0.95, 0.9, 0.05, 0.1])
    m = compute_metrics(y_true, y_pred, y_proba)
    assert m["overall_accuracy"] == 1.0
    assert m["precision"] == 1.0
    assert m["recall"] == 1.0
    assert m["f1"] == 1.0
    assert m["kappa"] == 1.0
    assert m["roc_auc"] == 1.0

def test_compute_metrics_returns_all_expected_keys():
    y_true = np.array([1, 0, 1, 0, 1])
    y_pred = np.array([1, 0, 0, 0, 1])
    y_proba = np.array([0.9, 0.2, 0.4, 0.3, 0.8])
    m = compute_metrics(y_true, y_pred, y_proba)
    assert set(m.keys()) == {
        "overall_accuracy", "precision", "recall", "f1", "kappa", "roc_auc"
    }

def test_compute_metrics_matches_published_baseline_ballpark():
    # Sanity check against the published benchmark
    # (OA 0.984, P 0.999, R 0.969, F1 0.984, Kappa 0.968, ROC-AUC 0.9497)
    # using a confusion-matrix-equivalent synthetic set of the same shape
    # as the real 1500-point verification set (752 pv / 748 non_pv).
    rng = np.random.default_rng(0)
    y_true = np.array([1] * 752 + [0] * 748)
    y_pred = y_true.copy()
    flip_idx = rng.choice(len(y_true), size=24, replace=False)  # ~1.6% error rate
    y_pred[flip_idx] = 1 - y_pred[flip_idx]
    y_proba = np.where(y_pred == 1, rng.uniform(0.6, 1.0, len(y_true)),
                        rng.uniform(0.0, 0.4, len(y_true)))
    m = compute_metrics(y_true, y_pred, y_proba)
    assert m["overall_accuracy"] > 0.95
```

- [ ] **Step 2: Run tests to verify they fail**

```powershell
pytest tests/evaluation/test_metrics.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'src.evaluation.metrics'`.

- [ ] **Step 3: Implement**

```python
# src/evaluation/metrics.py
"""Shared evaluation metrics, matching the metric set Shimada & Takeuchi
(2025) report (reports/paper-draft.md §4.4): overall accuracy, precision,
recall, F1, Cohen's Kappa, ROC-AUC."""
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    cohen_kappa_score, roc_auc_score,
)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> dict:
    return {
        "overall_accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "kappa": cohen_kappa_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_proba),
    }
```

- [ ] **Step 4: Run tests to verify they pass**

```powershell
pytest tests/evaluation/test_metrics.py -v
```

Expected: PASS, all 3 tests.

- [ ] **Step 5: Commit**

```powershell
git add src/evaluation/metrics.py tests/evaluation/test_metrics.py
git commit -m "feat: add shared evaluation metrics module"
```

---

## Task 11: Spatial train/test split

**Files:**
- Create: `src/data/split.py`
- Test: `tests/data/test_split.py`

**Interfaces:**
- Consumes: any GeoDataFrame with a `geometry` column of points (Task 6/7 output).
- Produces: `spatial_train_test_split(gdf, test_fraction: float, block_size_km: float, seed: int) -> tuple[GeoDataFrame, GeoDataFrame]`. Consumed by both baseline and CNN training scripts, and by Task 12 (patch extraction) to keep patch centers aligned with the same split.

**Design note:** partitions space into grid blocks (not individual points) before splitting, so spatially adjacent points cannot land on opposite sides of train/test — directly addressing the leakage risk flagged in `reports/paper-draft.md` §3.3 and §7.

- [ ] **Step 1: Write the failing tests**

```python
# tests/data/test_split.py
import geopandas as gpd
import numpy as np
from shapely.geometry import Point
from src.data.split import spatial_train_test_split

def _grid_points_gdf(n_per_side=20, spacing_deg=0.01):
    pts = [Point(139.0 + i * spacing_deg, 35.0 + j * spacing_deg)
           for i in range(n_per_side) for j in range(n_per_side)]
    return gpd.GeoDataFrame({"geometry": pts}, crs="EPSG:4326")

def test_split_produces_requested_fraction_approximately():
    gdf = _grid_points_gdf()
    train, test = spatial_train_test_split(gdf, test_fraction=0.2, block_size_km=5, seed=0)
    total = len(train) + len(test)
    assert total == len(gdf)
    assert 0.1 < len(test) / total < 0.35  # blocky split won't hit exactly 20%

def test_split_is_deterministic_given_seed():
    gdf = _grid_points_gdf()
    train1, test1 = spatial_train_test_split(gdf, test_fraction=0.2, block_size_km=5, seed=7)
    train2, test2 = spatial_train_test_split(gdf, test_fraction=0.2, block_size_km=5, seed=7)
    assert set(train1.index) == set(train2.index)
    assert set(test1.index) == set(test2.index)

def test_split_no_point_appears_in_both_sets():
    gdf = _grid_points_gdf()
    train, test = spatial_train_test_split(gdf, test_fraction=0.3, block_size_km=5, seed=3)
    assert set(train.index).isdisjoint(set(test.index))

def test_split_points_in_same_small_neighborhood_stay_together():
    # Two points ~50m apart must fall in the same block at block_size_km=5
    gdf = gpd.GeoDataFrame(
        {"geometry": [Point(139.000, 35.000), Point(139.0005, 35.0005)]},
        crs="EPSG:4326",
    )
    # pad with a spread-out grid so the split isn't trivially all-train
    gdf = gpd.GeoDataFrame(
        list(gdf.geometry) + list(_grid_points_gdf().geometry),
        columns=["geometry"], crs="EPSG:4326",
    )
    train, test = spatial_train_test_split(gdf, test_fraction=0.2, block_size_km=5, seed=0)
    both_in_train = 0 in train.index and 1 in train.index
    both_in_test = 0 in test.index and 1 in test.index
    assert both_in_train or both_in_test
```

- [ ] **Step 2: Run tests to verify they fail**

```powershell
pytest tests/data/test_split.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'src.data.split'`.

- [ ] **Step 3: Implement**

```python
# src/data/split.py
"""Block-based spatial train/test split: partitions the AOI into
block_size_km x block_size_km grid cells and assigns whole cells to
train or test, so no two spatially adjacent points can leak across
the split (reports/paper-draft.md §3.3, §7)."""
import geopandas as gpd
import numpy as np

_KM_PER_DEGREE_LAT = 111.0


def _block_id(lon: float, lat: float, block_size_km: float) -> tuple[int, int]:
    km_per_degree_lon = _KM_PER_DEGREE_LAT * np.cos(np.radians(lat))
    col = int(lon * km_per_degree_lon // block_size_km)
    row = int(lat * _KM_PER_DEGREE_LAT // block_size_km)
    return row, col


def spatial_train_test_split(gdf: gpd.GeoDataFrame, test_fraction: float,
                              block_size_km: float, seed: int
                              ) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    blocks = gdf.geometry.apply(lambda p: _block_id(p.x, p.y, block_size_km))
    unique_blocks = sorted(set(blocks))
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(len(unique_blocks))
    n_test_blocks = max(1, int(len(unique_blocks) * test_fraction))
    test_block_set = {unique_blocks[i] for i in shuffled[:n_test_blocks]}

    is_test = blocks.isin(test_block_set)
    return gdf[~is_test], gdf[is_test]
```

- [ ] **Step 4: Run tests to verify they pass**

```powershell
pytest tests/data/test_split.py -v
```

Expected: PASS, all 4 tests.

- [ ] **Step 5: Commit**

```powershell
git add src/data/split.py tests/data/test_split.py
git commit -m "feat: add block-based spatial train/test split to prevent patch leakage"
```

---

## Task 12: Patch extraction for CNN input

**Files:**
- Create: `src/data/patches.py`
- Test: `tests/data/test_patches.py`

**Interfaces:**
- Consumes: a local raster array (`numpy.ndarray`, shape `[n_bands, H, W]`) plus its affine transform, and a points GeoDataFrame (Task 6 output).
- Produces: `extract_patch(raster, transform, point, size_px: int) -> numpy.ndarray | None` (shape `[n_bands, size_px, size_px]`, or `None` if the point is too close to the raster edge) and `extract_all_patches(raster, transform, points_gdf, size_px: int) -> tuple[numpy.ndarray, numpy.ndarray]` (stacked patches + aligned labels, edge-case points dropped). Consumed by Task 13 (PyTorch dataset).

- [ ] **Step 1: Write the failing tests**

```python
# tests/data/test_patches.py
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
from rasterio.transform import from_origin
from src.data.patches import extract_patch, extract_all_patches

def _synthetic_raster(n_bands=6, size=100):
    raster = np.arange(n_bands * size * size, dtype=float).reshape(n_bands, size, size)
    transform = from_origin(139.0, 36.0, 0.0001, 0.0001)  # west, north, xsize, ysize
    return raster, transform

def test_extract_patch_returns_correct_shape():
    raster, transform = _synthetic_raster()
    point = Point(139.005, 35.995)  # well inside the raster
    patch = extract_patch(raster, transform, point, size_px=21)
    assert patch.shape == (6, 21, 21)

def test_extract_patch_returns_none_near_edge():
    raster, transform = _synthetic_raster()
    point = Point(139.0001, 35.9999)  # near the top-left corner
    patch = extract_patch(raster, transform, point, size_px=21)
    assert patch is None

def test_extract_patch_is_centered_on_point():
    raster, transform = _synthetic_raster(n_bands=1, size=101)
    raster[0, 50, 50] = 9999.0  # known center pixel
    from rasterio.transform import xy
    cx, cy = xy(transform, 50, 50)
    patch = extract_patch(raster, transform, Point(cx, cy), size_px=21)
    assert patch[0, 10, 10] == 9999.0  # center of a 21x21 patch is index 10

def test_extract_all_patches_drops_edge_points_and_keeps_labels_aligned():
    raster, transform = _synthetic_raster()
    points = gpd.GeoDataFrame(
        {"label": ["pv", "non_pv"],
         "geometry": [Point(139.005, 35.995), Point(139.0001, 35.9999)]},
        crs="EPSG:4326",
    )
    patches, labels = extract_all_patches(raster, transform, points, size_px=21)
    assert patches.shape == (1, 6, 21, 21)
    assert labels.tolist() == ["pv"]
```

- [ ] **Step 2: Run tests to verify they fail**

```powershell
pytest tests/data/test_patches.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'src.data.patches'`.

- [ ] **Step 3: Implement**

```python
# src/data/patches.py
"""Fixed-size image-patch extraction centered on sample points, for CNN
input (reports/paper-draft.md §3.3). Patch size default (21px at 10m
Sentinel-2 resolution = ~210m context window) is a config-driven design
choice, not hardcoded here -- see configs/config.yaml `patch.size_px`."""
import numpy as np
import geopandas as gpd
from rasterio.transform import rowcol


def extract_patch(raster: np.ndarray, transform, point, size_px: int) -> np.ndarray | None:
    n_bands, height, width = raster.shape
    row, col = rowcol(transform, point.x, point.y)
    half = size_px // 2
    row_start, row_end = row - half, row + half + 1
    col_start, col_end = col - half, col + half + 1
    if row_start < 0 or col_start < 0 or row_end > height or col_end > width:
        return None
    return raster[:, row_start:row_end, col_start:col_end]


def extract_all_patches(raster: np.ndarray, transform, points_gdf: gpd.GeoDataFrame,
                         size_px: int) -> tuple[np.ndarray, np.ndarray]:
    patches, labels = [], []
    for _, row in points_gdf.iterrows():
        patch = extract_patch(raster, transform, row.geometry, size_px)
        if patch is not None:
            patches.append(patch)
            labels.append(row["label"])
    return np.stack(patches), np.array(labels)
```

- [ ] **Step 4: Run tests to verify they pass**

```powershell
pytest tests/data/test_patches.py -v
```

Expected: PASS, all 4 tests.

- [ ] **Step 5: Commit**

```powershell
git add src/data/patches.py tests/data/test_patches.py
git commit -m "feat: add fixed-size patch extraction for CNN input"
```

---

## Task 13: PyTorch Dataset/DataLoader for patches

**Files:**
- Create: `src/data/patch_dataset.py`
- Test: `tests/data/test_patch_dataset.py`

**Interfaces:**
- Consumes: `patches: numpy.ndarray [N, n_bands, H, W]`, `labels: numpy.ndarray [N]` (strings `"pv"`/`"non_pv"`) from Task 12.
- Produces: `PVPatchDataset(torch.utils.data.Dataset)` yielding `(patch_tensor: FloatTensor[n_bands,H,W], label_tensor: LongTensor[])`. Consumed by Task 15 (training loop).

- [ ] **Step 1: Write the failing tests**

```python
# tests/data/test_patch_dataset.py
import numpy as np
import torch
from src.data.patch_dataset import PVPatchDataset

def test_dataset_length_matches_input():
    patches = np.random.rand(10, 6, 21, 21).astype(np.float32)
    labels = np.array(["pv"] * 5 + ["non_pv"] * 5)
    ds = PVPatchDataset(patches, labels)
    assert len(ds) == 10

def test_dataset_returns_correct_types_and_shapes():
    patches = np.random.rand(4, 6, 21, 21).astype(np.float32)
    labels = np.array(["pv", "non_pv", "pv", "non_pv"])
    ds = PVPatchDataset(patches, labels)
    x, y = ds[0]
    assert isinstance(x, torch.Tensor)
    assert x.dtype == torch.float32
    assert x.shape == (6, 21, 21)
    assert isinstance(y, torch.Tensor)
    assert y.dtype == torch.long

def test_dataset_label_encoding_is_consistent():
    patches = np.random.rand(2, 6, 21, 21).astype(np.float32)
    labels = np.array(["pv", "non_pv"])
    ds = PVPatchDataset(patches, labels)
    _, y_pv = ds[0]
    _, y_nonpv = ds[1]
    assert y_pv.item() == 1
    assert y_nonpv.item() == 0

def test_dataset_works_with_dataloader():
    from torch.utils.data import DataLoader
    patches = np.random.rand(16, 6, 21, 21).astype(np.float32)
    labels = np.array(["pv", "non_pv"] * 8)
    ds = PVPatchDataset(patches, labels)
    loader = DataLoader(ds, batch_size=4, shuffle=True)
    batch_x, batch_y = next(iter(loader))
    assert batch_x.shape == (4, 6, 21, 21)
    assert batch_y.shape == (4,)
```

- [ ] **Step 2: Run tests to verify they fail**

```powershell
pytest tests/data/test_patch_dataset.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'src.data.patch_dataset'`.

- [ ] **Step 3: Implement**

```python
# src/data/patch_dataset.py
"""PyTorch Dataset wrapping extracted patches for CNN training."""
import numpy as np
import torch
from torch.utils.data import Dataset

LABEL_TO_INT = {"non_pv": 0, "pv": 1}


class PVPatchDataset(Dataset):
    def __init__(self, patches: np.ndarray, labels: np.ndarray):
        self.patches = patches.astype(np.float32)
        self.labels = np.array([LABEL_TO_INT[label] for label in labels], dtype=np.int64)

    def __len__(self) -> int:
        return len(self.patches)

    def __getitem__(self, idx: int):
        x = torch.from_numpy(self.patches[idx])
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y
```

- [ ] **Step 4: Run tests to verify they pass**

```powershell
pytest tests/data/test_patch_dataset.py -v
```

Expected: PASS, all 4 tests.

- [ ] **Step 5: Commit**

```powershell
git add src/data/patch_dataset.py tests/data/test_patch_dataset.py
git commit -m "feat: add PyTorch dataset wrapper for PV patches"
```

---

## Task 14: CNN architecture

**Files:**
- Create: `src/models/cnn.py`
- Test: `tests/models/test_cnn.py`

**Interfaces:**
- Produces: `PVPatchCNN(torch.nn.Module)`, constructor `(in_channels: int = 6, input_size: int = 21, n_classes: int = 2)`, `forward(x: FloatTensor[B,in_channels,H,W]) -> FloatTensor[B,n_classes]`. Consumed by Task 15 (training loop).

**Design choice, stated explicitly:** a compact from-scratch architecture, not a pretrained ImageNet backbone, is the primary path for this plan. Pretrained backbones (ResNet/EfficientNet) expect 3-channel RGB input; adapting them to 6-channel, non-RGB Sentinel-2 patches means either discarding the SWIR bands or hacking the first conv layer's weights, both of which undercut Ishii et al.'s (2016) finding that spectral-channel completeness matters for this exact detection task (`reports/literature-review.md` #13). A from-scratch 6-channel network keeps that hypothesis testable cleanly and needs less data to avoid overfitting at this patch scale. A pretrained-backbone comparison remains a valid future extension (`reports/paper-draft.md` §4.2) but is out of scope for this plan.

- [ ] **Step 1: Write the failing tests**

```python
# tests/models/test_cnn.py
import torch
from src.models.cnn import PVPatchCNN

def test_forward_pass_output_shape():
    model = PVPatchCNN(in_channels=6, input_size=21, n_classes=2)
    x = torch.rand(8, 6, 21, 21)
    logits = model(x)
    assert logits.shape == (8, 2)

def test_forward_pass_works_with_different_input_size():
    model = PVPatchCNN(in_channels=6, input_size=33, n_classes=2)
    x = torch.rand(4, 6, 33, 33)
    logits = model(x)
    assert logits.shape == (4, 2)

def test_model_has_trainable_parameters():
    model = PVPatchCNN(in_channels=6, input_size=21, n_classes=2)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert n_params > 0

def test_forward_pass_gradients_flow():
    model = PVPatchCNN(in_channels=6, input_size=21, n_classes=2)
    x = torch.rand(2, 6, 21, 21, requires_grad=False)
    logits = model(x)
    loss = logits.sum()
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert all(g is not None for g in grads)
```

- [ ] **Step 2: Run tests to verify they fail**

```powershell
pytest tests/models/test_cnn.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'src.models.cnn'`.

- [ ] **Step 3: Implement**

```python
# src/models/cnn.py
"""Compact from-scratch CNN for PV patch classification. See Task 14's
design-choice note in plan.md for why this is from-scratch rather than a
pretrained ImageNet backbone."""
import torch
import torch.nn as nn


def _conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class PVPatchCNN(nn.Module):
    def __init__(self, in_channels: int = 6, input_size: int = 21, n_classes: int = 2):
        super().__init__()
        self.block1 = _conv_block(in_channels, 32)
        self.block2 = _conv_block(32, 64)
        self.block3 = _conv_block(64, 128)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(128, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.global_pool(x).flatten(1)
        return self.classifier(x)
```

- [ ] **Step 4: Run tests to verify they pass**

```powershell
pytest tests/models/test_cnn.py -v
```

Expected: PASS, all 4 tests. `AdaptiveAvgPool2d(1)` before the classifier is what makes `test_forward_pass_works_with_different_input_size` pass without changing the architecture — if you ever swap it for a `Flatten()` + fixed-size `Linear`, that test will break and the network will stop being patch-size-agnostic.

- [ ] **Step 5: Commit**

```powershell
git add src/models/cnn.py tests/models/test_cnn.py
git commit -m "feat: add compact from-scratch CNN architecture for PV patches"
```

---

## Task 15: CNN training loop

**Files:**
- Create: `src/models/train_cnn.py`
- Create: `scripts/run_cnn_training.py`
- Test: `tests/models/test_train_cnn.py` (smoke test — a real training loop is not meaningfully unit-testable beyond "does it run and reduce loss on trivial data")

**Interfaces:**
- Consumes: `PVPatchDataset` (Task 13), `PVPatchCNN` (Task 14), `compute_metrics` (Task 10).
- Produces: `train_cnn(model, train_loader, val_loader, epochs, lr, device) -> dict` (history of per-epoch train/val loss and val metrics) and `evaluate_cnn(model, loader, device) -> dict` (metrics via Task 10). Consumed by Task 17 (label-efficiency runner).

- [ ] **Step 1: Write the failing smoke test**

```python
# tests/models/test_train_cnn.py
import numpy as np
import torch
from torch.utils.data import DataLoader
from src.data.patch_dataset import PVPatchDataset
from src.models.cnn import PVPatchCNN
from src.models.train_cnn import train_cnn, evaluate_cnn

def _separable_dataset(n=200, seed=0):
    rng = np.random.default_rng(seed)
    n_pv, n_nonpv = n // 2, n - n // 2
    pv_patches = rng.normal(0.8, 0.05, (n_pv, 6, 21, 21)).astype(np.float32)
    nonpv_patches = rng.normal(0.2, 0.05, (n_nonpv, 6, 21, 21)).astype(np.float32)
    patches = np.concatenate([pv_patches, nonpv_patches])
    labels = np.array(["pv"] * n_pv + ["non_pv"] * n_nonpv)
    return PVPatchDataset(patches, labels)

def test_train_cnn_reduces_loss_on_separable_synthetic_data():
    train_ds = _separable_dataset(seed=0)
    val_ds = _separable_dataset(seed=1)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)
    model = PVPatchCNN(in_channels=6, input_size=21, n_classes=2)
    history = train_cnn(model, train_loader, val_loader, epochs=5, lr=1e-3, device="cpu")
    assert history["train_loss"][-1] < history["train_loss"][0]

def test_evaluate_cnn_returns_metrics_dict_with_expected_keys():
    val_ds = _separable_dataset(seed=1)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)
    model = PVPatchCNN(in_channels=6, input_size=21, n_classes=2)
    metrics = evaluate_cnn(model, val_loader, device="cpu")
    assert set(metrics.keys()) == {
        "overall_accuracy", "precision", "recall", "f1", "kappa", "roc_auc"
    }

def test_train_cnn_achieves_high_accuracy_on_trivially_separable_data():
    train_ds = _separable_dataset(seed=0)
    val_ds = _separable_dataset(seed=1)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)
    model = PVPatchCNN(in_channels=6, input_size=21, n_classes=2)
    train_cnn(model, train_loader, val_loader, epochs=15, lr=1e-3, device="cpu")
    metrics = evaluate_cnn(model, val_loader, device="cpu")
    assert metrics["overall_accuracy"] > 0.9
```

- [ ] **Step 2: Run tests to verify they fail**

```powershell
pytest tests/models/test_train_cnn.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'src.models.train_cnn'`.

- [ ] **Step 3: Implement**

```python
# src/models/train_cnn.py
"""Training and evaluation loop for PVPatchCNN."""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.evaluation.metrics import compute_metrics


def train_cnn(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
              epochs: int, lr: float, device: str) -> dict:
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    history = {"train_loss": [], "val_loss": []}

    for _ in range(epochs):
        model.train()
        running_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x.size(0)
        history["train_loss"].append(running_loss / len(train_loader.dataset))

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                val_loss += criterion(logits, y).item() * x.size(0)
        history["val_loss"].append(val_loss / len(val_loader.dataset))

    return history


def evaluate_cnn(model: nn.Module, loader: DataLoader, device: str) -> dict:
    model.to(device)
    model.eval()
    all_y_true, all_y_pred, all_y_proba = [], [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            proba = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            pred = (proba >= 0.5).astype(int)
            all_y_true.append(y.numpy())
            all_y_pred.append(pred)
            all_y_proba.append(proba)
    y_true = np.concatenate(all_y_true)
    y_pred = np.concatenate(all_y_pred)
    y_proba = np.concatenate(all_y_proba)
    return compute_metrics(y_true, y_pred, y_proba)
```

- [ ] **Step 4: Run tests to verify they pass**

```powershell
pytest tests/models/test_train_cnn.py -v
```

Expected: PASS, all 3 tests. If `test_train_cnn_achieves_high_accuracy_on_trivially_separable_data` is flaky, increase `epochs` slightly rather than lowering the `0.9` threshold — the synthetic classes are separated by a 0.6 mean-intensity gap with only 0.05 std, this should converge reliably.

- [ ] **Step 5: Write the training entry-point script**

```python
# scripts/run_cnn_training.py
"""Entry point: train the CNN on the reconstructed sample, evaluate,
print metrics. Assumes Tasks 1-14's outputs already exist on disk."""
import torch
from torch.utils.data import DataLoader
from src.config import load_config
from src.data.patch_dataset import PVPatchDataset
from src.models.cnn import PVPatchCNN
from src.models.train_cnn import train_cnn, evaluate_cnn
import numpy as np


def main():
    cfg = load_config()
    patches = np.load("data/processed/train_patches.npy")
    labels = np.load("data/processed/train_labels.npy", allow_pickle=True)
    val_patches = np.load("data/processed/val_patches.npy")
    val_labels = np.load("data/processed/val_labels.npy", allow_pickle=True)

    train_loader = DataLoader(PVPatchDataset(patches, labels), batch_size=32, shuffle=True)
    val_loader = DataLoader(PVPatchDataset(val_patches, val_labels), batch_size=32, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PVPatchCNN(in_channels=len(cfg.sentinel2.bands), input_size=cfg.patch.size_px)
    history = train_cnn(model, train_loader, val_loader, epochs=30, lr=1e-3, device=device)
    metrics = evaluate_cnn(model, val_loader, device=device)

    print("Final train loss:", history["train_loss"][-1])
    print("Final val loss:", history["val_loss"][-1])
    print("Validation metrics:", metrics)
    torch.save(model.state_dict(), "results/logs/pv_patch_cnn.pt")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Commit**

```powershell
git add src/models/train_cnn.py tests/models/test_train_cnn.py scripts/run_cnn_training.py
git commit -m "feat: add CNN training/evaluation loop and training entry point"
```

---

## Task 16: Failure-case stratification

**Files:**
- Create: `src/evaluation/failure_cases.py`
- Test: `tests/evaluation/test_failure_cases.py`

**Interfaces:**
- Consumes: PV polygon GeoDataFrame with `geometry` (Task 1), optional manual-annotation CSV.
- Produces: `estimate_capacity_kw(polygon_area_m2: numpy.ndarray, kw_per_m2: float = 0.15) -> numpy.ndarray`, `flag_sub_50kw(polygons_gdf, kw_per_m2: float = 0.15) -> pandas.Series[bool]`, `load_manual_failure_flags(csv_path) -> pandas.DataFrame` (columns `point_id, non_crystalline_silicon, spectrally_ambiguous`). Consumed by the Results-stage evaluation script (outside this plan's scope — see plan Note below).

**Design note, stated explicitly:** the public polygon release has no per-installation capacity or panel-material attributes (`papers/zenodo-dataset-verification.md`), so two of the three failure-case subsets cannot be derived automatically:
- **Sub-50 kW** is approximated from polygon area using a capacity-density heuristic (default 0.15 kW/m², i.e. ~150 kW per 1,000 m² — a mid-range published estimate for crystalline-silicon ground-mount systems; treat this as an approximation, not ground truth, and say so explicitly in the thesis's Limitations chapter).
- **Non-crystalline-silicon** and **spectrally ambiguous** panel arrangements have no area-based proxy at all and require manual annotation against Google Earth imagery for a stratified subsample, per `reports/paper-draft.md` §3.5. This task provides the *loader* for that manual annotation (a CSV you fill in by hand), not an automated classifier for it — automating a proxy for something the original study itself identified as needing manual judgment would quietly substitute your assumption for the phenomenon you're trying to measure.

- [ ] **Step 1: Write the failing tests**

```python
# tests/evaluation/test_failure_cases.py
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box
from src.evaluation.failure_cases import (
    estimate_capacity_kw, flag_sub_50kw, load_manual_failure_flags,
)

def test_estimate_capacity_kw_scales_with_area():
    areas = np.array([100.0, 1000.0])
    kw = estimate_capacity_kw(areas, kw_per_m2=0.15)
    np.testing.assert_allclose(kw, [15.0, 150.0])

def test_flag_sub_50kw_identifies_small_installations():
    # areas in m^2, in EPSG:3857-like projected units for the test polygons
    small = box(0, 0, 18, 18)   # 324 m^2 -> ~48.6 kW at 0.15 kW/m^2 -> sub-50kW
    large = box(0, 0, 100, 100)  # 10,000 m^2 -> 1500 kW -> not sub-50kW
    gdf = gpd.GeoDataFrame({"geometry": [small, large]}, crs="EPSG:3857")
    flags = flag_sub_50kw(gdf, kw_per_m2=0.15)
    assert flags.tolist() == [True, False]

def test_load_manual_failure_flags_reads_csv(tmp_path):
    csv_path = tmp_path / "manual_flags.csv"
    csv_path.write_text(
        "point_id,non_crystalline_silicon,spectrally_ambiguous\n"
        "1,True,False\n"
        "2,False,True\n"
    )
    df = load_manual_failure_flags(csv_path)
    assert list(df.columns) == ["point_id", "non_crystalline_silicon", "spectrally_ambiguous"]
    assert df["non_crystalline_silicon"].tolist() == [True, False]
    assert df["spectrally_ambiguous"].tolist() == [False, True]
```

- [ ] **Step 2: Run tests to verify they fail**

```powershell
pytest tests/evaluation/test_failure_cases.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'src.evaluation.failure_cases'`.

- [ ] **Step 3: Implement**

```python
# src/evaluation/failure_cases.py
"""Operational definitions for the three failure-case subsets named in
Shimada & Takeuchi (2025) and reports/paper-draft.md §3.5. See this
task's design note in plan.md for why two of the three require manual
annotation rather than an automated proxy."""
import numpy as np
import pandas as pd
import geopandas as gpd

DEFAULT_KW_PER_M2 = 0.15  # mid-range published capacity density for
                            # crystalline-silicon ground-mount PV; approximation only


def estimate_capacity_kw(polygon_area_m2: np.ndarray, kw_per_m2: float = DEFAULT_KW_PER_M2
                          ) -> np.ndarray:
    return polygon_area_m2 * kw_per_m2


def flag_sub_50kw(polygons_gdf: gpd.GeoDataFrame, kw_per_m2: float = DEFAULT_KW_PER_M2
                   ) -> pd.Series:
    areas_m2 = polygons_gdf.geometry.area.values
    capacity_kw = estimate_capacity_kw(areas_m2, kw_per_m2)
    return pd.Series(capacity_kw < 50.0, index=polygons_gdf.index)


def load_manual_failure_flags(csv_path) -> pd.DataFrame:
    """Loads a hand-annotated CSV (point_id, non_crystalline_silicon,
    spectrally_ambiguous) produced by visually inspecting a stratified
    subsample against Google Earth imagery. This function does not create
    or infer the annotations -- see the design note above."""
    df = pd.read_csv(csv_path)
    expected_cols = {"point_id", "non_crystalline_silicon", "spectrally_ambiguous"}
    if set(df.columns) != expected_cols:
        raise ValueError(f"Expected columns {expected_cols}, got {set(df.columns)}")
    return df
```

- [ ] **Step 4: Run tests to verify they pass**

```powershell
pytest tests/evaluation/test_failure_cases.py -v
```

Expected: PASS, all 3 tests.

- [ ] **Step 5: Commit**

```powershell
git add src/evaluation/failure_cases.py tests/evaluation/test_failure_cases.py
git commit -m "feat: add failure-case stratification (area-based sub-50kW proxy + manual-annotation loader)"
```

---

## Task 17: Label-efficiency experiment runner

**Files:**
- Create: `scripts/run_label_efficiency.py`
- Test: `tests/test_label_efficiency.py`

**Interfaces:**
- Consumes: `train_baseline`/`predict_proba` (Task 8), `PVPatchDataset`/`train_cnn`/`evaluate_cnn` (Task 13/15), `compute_metrics` (Task 10).
- Produces: `run_label_efficiency_curve(train_df, train_patches, train_labels, val_df, val_patches, val_labels, fractions: list[float], cfg) -> pandas.DataFrame` with columns `[fraction, model, overall_accuracy, f1]`. This directly produces the data behind the thesis's secondary research question (`reports/paper-draft.md` §4.5).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_label_efficiency.py
import numpy as np
import pandas as pd
from src.config import XGBoostConfig
from scripts.run_label_efficiency import run_label_efficiency_curve

def _synthetic_baseline_df(n, seed):
    rng = np.random.default_rng(seed)
    n_pv, n_nonpv = n // 2, n - n // 2
    pvsi = np.concatenate([rng.normal(0.6, 0.05, n_pv), rng.normal(0.1, 0.05, n_nonpv)])
    df = pd.DataFrame(rng.normal(0.2, 0.05, (n, 8)),
                       columns=["B2", "B3", "B4", "B8", "B11", "B12", "NDVI", "MNDWI"])
    df["PVSI"] = pvsi
    df["label"] = ["pv"] * n_pv + ["non_pv"] * n_nonpv
    return df

def _synthetic_patches(n, seed):
    rng = np.random.default_rng(seed)
    n_pv, n_nonpv = n // 2, n - n // 2
    pv = rng.normal(0.8, 0.05, (n_pv, 6, 21, 21)).astype(np.float32)
    nonpv = rng.normal(0.2, 0.05, (n_nonpv, 6, 21, 21)).astype(np.float32)
    patches = np.concatenate([pv, nonpv])
    labels = np.array(["pv"] * n_pv + ["non_pv"] * n_nonpv)
    return patches, labels

def test_run_label_efficiency_curve_covers_all_fractions_and_models():
    train_df = _synthetic_baseline_df(300, seed=0)
    val_df = _synthetic_baseline_df(100, seed=1)
    train_patches, train_labels = _synthetic_patches(300, seed=0)
    val_patches, val_labels = _synthetic_patches(100, seed=1)
    cfg = XGBoostConfig(max_depth=4, eta=0.1, n_estimators=30)

    result = run_label_efficiency_curve(
        train_df, train_patches, train_labels,
        val_df, val_patches, val_labels,
        fractions=[0.25, 1.0], cfg=cfg, cnn_epochs=3,
    )

    assert set(result["fraction"]) == {0.25, 1.0}
    assert set(result["model"]) == {"xgboost_baseline", "cnn"}
    assert {"overall_accuracy", "f1"}.issubset(result.columns)
    assert len(result) == 4  # 2 fractions x 2 models
```

- [ ] **Step 2: Run test to verify it fails**

```powershell
pytest tests/test_label_efficiency.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.run_label_efficiency'`.

- [ ] **Step 3: Implement**

```python
# scripts/run_label_efficiency.py
"""Label-efficiency comparison: both models trained on shrinking fractions
of the training set (reports/paper-draft.md §4.5, secondary RQ)."""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from src.config import XGBoostConfig
from src.models.xgboost_baseline import train_baseline, predict_proba, FEATURE_COLUMNS
from src.data.patch_dataset import PVPatchDataset
from src.models.cnn import PVPatchCNN
from src.models.train_cnn import train_cnn, evaluate_cnn
from src.evaluation.metrics import compute_metrics


def run_label_efficiency_curve(train_df, train_patches, train_labels,
                                val_df, val_patches, val_labels,
                                fractions: list[float], cfg: XGBoostConfig,
                                cnn_epochs: int = 20, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_total = len(train_df)
    rows = []

    for fraction in fractions:
        n_subset = max(2, int(n_total * fraction))
        idx = rng.choice(n_total, size=n_subset, replace=False)

        # --- XGBoost baseline ---
        subset_df = train_df.iloc[idx]
        model = train_baseline(subset_df, cfg)
        val_proba = predict_proba(model, val_df)
        val_pred = (val_proba >= 0.5).astype(int)
        val_true = (val_df["label"] == "pv").astype(int).values
        baseline_metrics = compute_metrics(val_true, val_pred, val_proba)
        rows.append({"fraction": fraction, "model": "xgboost_baseline", **baseline_metrics})

        # --- CNN ---
        subset_patches = train_patches[idx]
        subset_labels = train_labels[idx]
        train_loader = DataLoader(
            PVPatchDataset(subset_patches, subset_labels), batch_size=16, shuffle=True
        )
        val_loader = DataLoader(
            PVPatchDataset(val_patches, val_labels), batch_size=16, shuffle=False
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        cnn_model = PVPatchCNN(in_channels=train_patches.shape[1], input_size=train_patches.shape[2])
        train_cnn(cnn_model, train_loader, val_loader, epochs=cnn_epochs, lr=1e-3, device=device)
        cnn_metrics = evaluate_cnn(cnn_model, val_loader, device=device)
        rows.append({"fraction": fraction, "model": "cnn", **cnn_metrics})

    return pd.DataFrame(rows)


def main():
    from src.config import load_config
    cfg = load_config()
    train_df = pd.read_parquet("data/processed/train_features.parquet")
    val_df = pd.read_parquet("data/processed/val_features.parquet")
    train_patches = np.load("data/processed/train_patches.npy")
    train_labels = np.load("data/processed/train_labels.npy", allow_pickle=True)
    val_patches = np.load("data/processed/val_patches.npy")
    val_labels = np.load("data/processed/val_labels.npy", allow_pickle=True)

    result = run_label_efficiency_curve(
        train_df, train_patches, train_labels,
        val_df, val_patches, val_labels,
        fractions=[0.1, 0.25, 0.5, 0.75, 1.0], cfg=cfg.xgboost,
    )
    result.to_csv("results/logs/label_efficiency_curve.csv", index=False)
    print(result)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

```powershell
pytest tests/test_label_efficiency.py -v
```

Expected: PASS. This test trains 2 tiny XGBoost models and 2 tiny 3-epoch CNNs on 21x21x6 synthetic patches — should complete in a few seconds on CPU; if it takes much longer, check `cnn_epochs` wasn't accidentally left at a production value.

- [ ] **Step 5: Commit**

```powershell
git add scripts/run_label_efficiency.py tests/test_label_efficiency.py
git commit -m "feat: add label-efficiency comparison runner for secondary research question"
```

---

## What this plan deliberately does not cover

- **Baseline replication accuracy validation against the published 0.984 OA figure** (`reports/paper-draft.md` §5.1) — that's a Results-chapter analysis step using this plan's Task 8-10 outputs on the real reconstructed data, not a coding task with a pass/fail test.
- **The optional frozen-embedding foundation-model comparison** (`reports/paper-draft.md` §4.3) — explicitly a stretch goal; write a follow-up plan once Tasks 0-17 are working and there's time budget left, per the exposé's own Week 4 framing.
- **Pretrained ResNet/EfficientNet backbone fine-tuning** — deferred per Task 14's design note; a legitimate follow-up plan once patch-dataset size is known from Task 6/12's real output (not the synthetic fixtures used here).
- **DID shapefile sourcing** — Task 9's `filter_did_false_positives` and Task 4's built-up mask both use the GHSL proxy; if you later obtain Japan's actual MLIT/e-Stat DID polygons, swap the proxy for the real boundaries without changing either function's interface.

---

## Self-Review

**Spec coverage:** every methodology element in `reports/paper-draft.md` §4 has a task — §4.1 (baseline) → Tasks 3, 7, 8; §4.2 (CNN, channel choice) → Tasks 12-14; §4.4 (evaluation) → Task 10; §4.5 (label efficiency) → Task 17; §4.6 (compute) → Task 0 Step 3 + CUDA check. `reports/paper-draft.md` §3 (Data) → Tasks 1, 4-6, 11-12. The two resolved `[VERIFY]` documents (`papers/zenodo-dataset-verification.md`, `papers/spectral-baseline-specifications.md`) are threaded through as the source of truth for Tasks 6 and 3/8/9 respectively, cited inline rather than re-derived.

**Placeholder scan:** no `TBD`/`implement later`/bare "add error handling" instructions remain; every code step is complete, runnable code. The two genuinely non-automatable items (manual GEE auth in Task 0 Step 6, manual failure-case annotation in Task 16) are marked as manual by design, not as unfinished plan sections — flagged and given a concrete file format/interface rather than left open.

**Type consistency:** `FEATURE_COLUMNS` (Task 8) matches the 9-column output of `build_feature_matrix` (Task 7) exactly. `PVPatchDataset` (Task 13) consumes the `(patches, labels)` tuple shape produced by `extract_all_patches` (Task 12). `compute_metrics` (Task 10)'s key set (`overall_accuracy, precision, recall, f1, kappa, roc_auc`) is used identically by `evaluate_cnn` (Task 15) and the baseline evaluation call sites. `Config` dataclass fields (Task 2) match every later task's `cfg.<field>` access.
