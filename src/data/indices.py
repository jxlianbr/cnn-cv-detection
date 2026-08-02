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
