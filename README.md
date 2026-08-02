# Spatial Context vs. Spectral Signatures
### A CNN Benchmark for Solar PV Detection in Japan

Benchmarking spatial-context CNNs against spectral-signature-based methods for detecting
solar photovoltaic installations in Japanese satellite/aerial imagery.

## Structure

```
data/
  raw/         source imagery & labels, untouched
  processed/   tiled/normalized datasets ready for training
  external/    reference datasets, boundaries, auxiliary layers
notebooks/     exploratory analysis, quick experiments
src/
  data/        loading, tiling, preprocessing
  models/      CNN architectures, spectral baselines
  evaluation/  metrics, benchmarking, comparison scripts
configs/       experiment config files (yaml/json)
results/
  figures/     plots, maps
  logs/        training logs, metrics output
reports/       drafts, writeups (see Expose_Solar_PV_Japan.docx)
```

## Setup

```
pip install -r requirements.txt
```
