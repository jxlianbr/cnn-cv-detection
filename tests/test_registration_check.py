"""The registration estimator recovers a known shift of a synthetic image."""
import numpy as np
from shapely.geometry import box

from scripts.run_registration_check import (estimate_shift, lattice_phase,
                                            metres_per_degree, synthetic_image)

RES = 8.983152841195215e-05
LAT = 36.0


def _scene():
    # a 15 x 12 pixel facility on the lattice, inside a 30 x 30 pixel window
    west, north = 140.0 + 100 * RES, LAT + 30 * RES
    poly = box(west + 8 * RES, north - 22 * RES, west + 20 * RES, north - 7 * RES)
    return [poly], west, north


def test_metres_per_degree_at_36n():
    m_lon, m_lat = metres_per_degree(36.0)
    assert abs(m_lat - 110_959) < 50
    assert abs(m_lon * RES - 8.08) < 0.05  # paper: about 8.1 m at 36 N


def test_lattice_phase_is_zero_on_lattice():
    assert lattice_phase(np.array([1234 * RES, -77 * RES]), RES).max() < 1e-6


def test_recovers_known_continuous_shift():
    geoms, west, north = _scene()
    rng = np.random.default_rng(0)
    img = synthetic_image(geoms, 6.3, -4.2, west, north, RES, LAT, 30, 30,
                          psf_m=5.0, cnr=4.0, rng=rng)
    est = estimate_shift(geoms, {"s": img}, np.ones((30, 30), bool), west, north,
                         RES, LAT, sigmas_m=(5.0,))["s"]
    assert abs(est["dx_m"] - 6.3) < 1.0
    assert abs(est["dy_m"] + 4.2) < 1.0
    assert not est["border"]


def test_blurred_template_is_unbiased_at_zero_shift():
    geoms, west, north = _scene()
    rng = np.random.default_rng(1)
    img = synthetic_image(geoms, 0.0, 0.0, west, north, RES, LAT, 30, 30,
                          psf_m=5.0, cnr=50.0, rng=rng)
    valid = np.ones((30, 30), bool)
    est = estimate_shift(geoms, {"s": img}, valid, west, north, RES, LAT, sigmas_m=(5.0,))["s"]
    assert np.hypot(est["dx_m"], est["dy_m"]) < 0.5
