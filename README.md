# Spatial Context vs. Spectral Signatures
### A CNN Benchmark for Solar PV Detection in Japan

[![tests](https://github.com/jxlianbr/cnn-cv-detection/actions/workflows/tests.yml/badge.svg)](https://github.com/jxlianbr/cnn-cv-detection/actions/workflows/tests.yml)
[![python](https://img.shields.io/badge/python-3.14-blue.svg)](https://www.python.org/)
[![license](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Does a CNN using spatial context detect solar PV better than a pixel-wise
spectral classifier? This replicates the spectral-index/XGBoost baseline of
Shimada & Takeuchi (2025) for Japan, trains a from-scratch CNN patch classifier
on the same Sentinel-2 data, and compares the two.

## Results

Japan-wide sample of 11,703 points from 955 sampling squares, evaluated on a
spatially blocked, held-out test split of 2,136 points. XGBoost is deterministic
and trained once; the CNN is trained with five seeds (mean +- sd):

| model | OA | recall | F1 |
|---|---|---|---|
| XGBoost + spectral indices | 0.967 | 0.958 | 0.965 |
| CNN (6-band, 21x21 patches) | **0.980 +- 0.001** | **0.979 +- 0.005** | **0.979 +- 0.001** |

The difference holds up in every run: an exact McNemar test on the points where
the two models disagree gives p between 0.001 and 0.010 for each of the five
CNN seeds (48-56 points only the CNN gets right, against 20-28 for XGBoost).

The gain is *not* where the spectral method is known to struggle. About four
fifths of the CNN's extra detections are installations above 1 MW. Both models
degrade on the smallest installations (recall 0.79-0.83 below 100 kW on test,
versus ~0.99 above 1000 kW), and below 1 MW which model is ahead depends on the
size bin and does not replicate between validation and test.

### Caveats worth reading before reusing these numbers

- **The labels are the baseline's own output.** The public Zenodo release is
  Shimada & Takeuchi's post-processed detection map, not independent ground
  truth. Installations the spectral baseline missed are absent from it by
  construction, so this measures *agreement with that map*, not absolute
  accuracy. The CNN's edge is plausibly it learning end-to-end what the original
  pipeline did with hand-built morphological post-processing.
- **The sub-50 kW failure case cannot be studied from this data.** Only 171 of
  104,471 released polygons fall under 333 m^2 (~50 kW at 0.15 kW/m^2). An
  independent label source is required -- `scripts/make_annotation_sample.py`
  produces a worksheet that deliberately oversamples small installations.
- **Capacity is estimated from footprint area** at 0.15 kW/m^2, not measured.
- Point sampling is area-weighted, biasing the sample toward large installations
  (sampled median footprint ~9x the population median).

## Setup

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt   # Linux/macOS: .venv/bin/pip
```

PyTorch must come from the CUDA 12.8 index -- the cu121 index has no Python 3.14
wheels:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

Earth Engine access is required. Authenticate once, then point the pipeline at
your own cloud project:

```bash
earthengine authenticate
export GEE_PROJECT_ID=ee-your-project        # Windows: $env:GEE_PROJECT_ID="..."
```

`configs/config.yaml` ships a placeholder project id; the environment variable
overrides it.

## Running

```bash
python scripts/download_pv_polygons.py       # fetch the Zenodo polygon release
python scripts/build_dataset.py              # GEE -> data/processed/ (slow: ~1000 downloads)
python scripts/run_cnn_training.py           # train + evaluate the CNN
python scripts/run_label_efficiency.py --repeats 5
python scripts/run_failure_case_evaluation.py --split val --repeats 5
python scripts/run_failure_case_evaluation.py --split test --repeats 5
python scripts/make_figures.py               # figures + LaTeX tables from results/logs/
python scripts/make_annotation_sample.py     # worksheet for manual annotation
```

`run_failure_case_evaluation.py` trains XGBoost once and the CNN once per seed,
then writes per-point predictions (`results/logs/predictions_<split>.parquet`),
metrics per seed and McNemar tests (`results/logs/mcnemar_<split>.csv`).
`make_figures.py` derives the size-bin analysis from those saved predictions, so
it retrains nothing. (`run_size_detectability.py` is the older, single-run
version of that analysis.)

Validate the chain on a small region before committing to a full run:

```bash
python scripts/build_dataset.py --max-squares 40 --n-pv 400 --n-nonpv 400
```

`build_dataset.py` caches one GeoTIFF per sampling square under `data/raw/`, so
an interrupted run resumes instead of re-downloading.

## Structure

```
src/
  data/        sampling, Sentinel-2 composites, patches, splits, alignment
  models/      XGBoost baseline, CNN architecture, training loop
  evaluation/  metrics, failure-case stratification
scripts/       entry points (see Running)
configs/       experiment configuration
tests/         110 tests
data/          gitignored; reproducible from the scripts
results/       gitignored; figures and logs
```

## Tests

```bash
python -m pytest tests/ -q
```

## License

MIT -- see [LICENSE](LICENSE). Note this covers the code only. The PV polygon
dataset is redistributed by Zenodo under its own terms, and the papers under
`papers/` are not part of this repository.

## Reference

Shimada, S., & Takeuchi, W. (2025). Satellite-based analysis uncovers uneven
solar PV distribution across Japan and its consumption of forest and
agricultural lands. *Scientific Reports*, 15, 26671.

Polygon release: https://doi.org/10.5281/zenodo.10674681
