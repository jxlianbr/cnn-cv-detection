# Spectral Baseline Classifier Specifications: Shimada & Takeuchi (2025) Replication

This document contains the exact technical parameters, band selections, index formulas, pre-masking rules, XGBoost hyperparameters, and post-processing steps required to independently replicate the spectral baseline classifier from **Shimada & Takeuchi (2025)** (*Scientific Reports*).

---

## 1. Input Features & Satellite Bands

The baseline classifier uses **9 input features** extracted from **Sentinel-2 Level-2A** imagery:

### 1.1 Surface Reflectance Bands (6 Bands)
* **Blue ($\text{B2}$)**: 490 nm
* **Green ($\text{B3}$)**: 560 nm
* **Red ($\text{B4}$)**: 665 nm
* **Near-Infrared ($\text{NIR}$ / $\text{B8}$)**: 842 nm
* **Short-Wave Infrared 1 ($\text{SWIR1}$ / $\text{B11}$)**: 1610 nm
* **Short-Wave Infrared 2 ($\text{SWIR2}$ / $\text{B12}$)**: 2190 nm

### 1.2 Input Spectral Indices (3 Indices)
* **Normalized Difference Vegetation Index ($\text{NDVI}$)**
* **Modified Normalized Difference Water Index ($\text{MNDWI}$)**
* **Photovoltaic Spectral Index ($\text{PVSI}$)**

*(Note: The Blue Spectral Index, $\text{BSI}$, is calculated separately and used for spectral pre-masking rather than as a direct XGBoost feature).*

---

## 2. Spectral Index Formulas

$$\text{NDVI} = \frac{\text{NIR} - \text{Red}}{\text{NIR} + \text{Red}} = \frac{\text{B8} - \text{B4}}{\text{B8} + \text{B4}}$$

$$\text{MNDWI} = \frac{\text{Green} - \text{SWIR1}}{\text{Green} + \text{SWIR1}} = \frac{\text{B3} - \text{B11}}{\text{B3} + \text{B11}}$$

$$\text{PVSI} = \left( \frac{2 \cdot \text{SWIR1} - \text{SWIR2} - \text{NIR}}{2 \cdot \text{SWIR1} + \text{SWIR2} + \text{NIR}} \right) \times \frac{\text{Blue}}{\text{Red}} = \left( \frac{2 \cdot \text{B11} - \text{B12} - \text{B8}}{2 \cdot \text{B11} + \text{B12} + \text{B8}} \right) \times \frac{\text{B2}}{\text{B4}}$$

$$\text{BSI} = \frac{\text{Blue} - \text{Red}}{\text{Blue} + \text{Red}} = \frac{\text{B2} - \text{B4}}{\text{B2} + \text{B4}}$$

---

## 3. Spectral Pre-Masking Rules

Before passing pixel feature vectors into the XGBoost classifier, rule-based spectral masks are applied to filter out obvious non-PV land cover classes:
* **Dense Vegetation Mask**: $\text{NDVI} > 0.5$
* **Water Body Mask**: $\text{MNDWI} > 0$
* **Bluish Non-PV Material Mask**: $\text{BSI} > 0.2$ (filters out bright blue synthetic roofing and plastics)

---

## 4. XGBoost Model & Hyperparameters

* **Framework**: `xgboost` (Python / C++)
* **Hyperparameter Optimization**: Optuna library (100 iterations on 5-fold cross-validation)
* **Tuned Hyperparameters**:
  * `max_depth`: **8**
  * `eta` (learning rate): **0.01717**
  * `n_estimators`: **1000**
* **Output Mode**: Probabilistic prediction (`predict_proba` for PV class probability $P(\text{PV})$).
* **Feature Importance**: $\text{PVSI}$ was demonstrated as the single most influential feature (measured by Gini gain).

---

## 5. Post-Processing & Segmentation Pipeline

1. **Morphological Noise Subtraction**: A "white top-hat" morphological filter with a 1-pixel radius disk footprint is computed on the raw probability raster and subtracted to reduce salt-and-pepper noise.
2. **Probability Thresholding**: Candidates with median probability $P(\text{PV}) \ge 0.6$ are retained.
3. **Graph-Based Segmentation**: Felzenszwalb efficient graph-based image segmentation is applied to delineate coherent PV facility boundaries.
4. **DID Filtering**: Small polygons located within Densely-Inhabited-Districts (DIDs) are filtered out to eliminate non-PV urban false positives.

---

## 6. Published Reference Benchmark

When evaluated against the 1,500-point independent verification set, the post-processed baseline map achieved:
* **Overall Accuracy**: $0.984$ (98.4%)
* **Precision**: $0.999$
* **Recall**: $0.969$
* **F1-Score**: $0.984$
* **Cohen's Kappa**: $0.968$
* **Test ROC-AUC**: $0.9497$
