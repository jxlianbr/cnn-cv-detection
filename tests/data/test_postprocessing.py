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
