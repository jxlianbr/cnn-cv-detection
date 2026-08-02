# Verification Finding: Shimada & Takeuchi (2025) Zenodo Release & Sampling Protocol

## Executive Summary
This document verifies the contents of the Zenodo dataset associated with **Shimada & Takeuchi (2025)** (*Scientific Reports*) [377, 383] and outlines the original study's sampling methodology for training and validation [371, 377].

---

## 1. Zenodo Dataset Verification
* **DOI**: [10.5281/zenodo.10674681](https://zenodo.org/doi/https://doi.org/10.5281/zenodo.10674681) [383]
* **Contents**: The Zenodo repository contains **only the final post-processed solar PV polygon dataset** (covering 571 km² across Japan) [369, 383].
* **Missing Components**: The raw point-level training samples (approx. 60,000 points) and the independent verification point dataset (1,500 points) are **not included** in the public release [377, 383].
* **Implication for Pipeline**: Because the original point-level feature matrices and point coordinates are unavailable, a comparable point sample must be drawn manually from the polygon boundaries and buffer zones, verified against high-resolution imagery [377].

---

## 2. Original Training Sample Protocol (60,000 Points)
To train the XGBoost classifier, Shimada & Takeuchi (2025) constructed 1,000 sampling squares ($1000\text{ m} \times 1000\text{ m}$) across Japan [377]:
* **Spatial Distribution of Centers**:
  * **600 locations**: Randomly placed within a 2020 manually delineated PV dataset [377].
  * **300 locations**: Randomly placed within a 7 km buffer zone surrounding populated areas [377].
  * **100 locations**: Manually selected in urban areas to capture non-PV spectral variations [377].
* **Point Extraction**:
  * Any unannotated PVs inside the 1,000 squares were manually added [377].
  * Random point sampling yielded **20,000 PV points** and **40,000 Non-PV points** [377].
  * After removing 88 points with missing Sentinel-2 imagery values, **19,969 PV points** and **39,943 Non-PV points** were retained [377].
  * An **80–20 train-test split** was used for hyperparameter tuning and model training [377].

---

## 3. Original Independent Verification Sample Protocol (1,500 Points)
To evaluate the final post-processed polygon map, an independent verification dataset was compiled [377]:
* **750 points**: Randomly distributed across the study area [377].
* **750 points**: Manually placed directly within PV sites [377].
* **Final Distribution**: **752 PV points** and **748 Non-PV background points** [377].
* **Performance Benchmark**: The post-processed map achieved an Overall Accuracy of 0.984, Precision of 0.999, Recall of 0.969, F1-Score of 0.984, and Cohen's Kappa of 0.968 [371].

---

## Reference
* **Shimada, S., & Takeuchi, W. (2025).** *Satellite-based analysis uncovers uneven solar PV distribution across Japan and its consumption of forest and agricultural lands.* Scientific Reports, 15, 26671. [369]
