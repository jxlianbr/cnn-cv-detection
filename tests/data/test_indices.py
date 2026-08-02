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
