"""Point gridders: biharmonic spline and nearest neighbor (plan task T24.4)."""

import numpy as np
import pytest
import xarray as xr

from disp_xr.models import Nearest, Spline

from .common import check_estimator


def _bump(e, n):
    return 0.02 * np.exp(
        -((e - 515_000) ** 2 + (n - 4_115_000) ** 2) / (2 * 6_000.0**2)
    )


@pytest.mark.parametrize("damping", [None, 1e-6])
def test_spline_passes_shared_checks(damping):
    check_estimator(lambda: Spline(damping))


def test_nearest_passes_shared_checks():
    check_estimator(Nearest)


def test_spline_reproduces_a_smooth_field():
    rng = np.random.default_rng(3)
    e = 500_000.0 + rng.uniform(0, 30_000, 500)
    n = 4_100_000.0 + rng.uniform(0, 30_000, 500)
    est = Spline().fit((e, n), _bump(e, n))
    like = xr.Dataset(
        coords={
            "y": 4_125_000.0 - 1000.0 * np.arange(21),
            "x": 505_000.0 + 1000.0 * np.arange(21),
        }
    ).rio.write_crs("EPSG:32610")
    grid = est.grid(like=like)
    truth = _bump(*np.meshgrid(like.x.values, like.y.values))
    rms = np.sqrt(np.mean((grid.values - truth) ** 2))
    assert rms < 0.02 * 0.02  # under 2% of the amplitude
    assert grid.rio.crs.to_epsg() == 32610
    # Without damping the spline passes through the data.
    np.testing.assert_allclose(est.predict((e, n)), _bump(e, n), atol=1e-8)


def test_spline_damping_smooths():
    rng = np.random.default_rng(4)
    e = 500_000.0 + rng.uniform(0, 30_000, 300)
    n = 4_100_000.0 + rng.uniform(0, 30_000, 300)
    noisy = _bump(e, n) + 0.002 * rng.standard_normal(e.size)
    exact = Spline().fit((e, n), noisy)
    smooth = Spline(damping=1e-4).fit((e, n), noisy)
    assert np.std(smooth.predict((e, n)) - noisy) > np.std(
        exact.predict((e, n)) - noisy
    )
    assert np.sqrt(np.mean((smooth.predict((e, n)) - _bump(e, n)) ** 2)) < 0.002


def test_nearest_picks_the_closest_point():
    e = np.array([0.0, 10.0, 20.0])
    n = np.zeros(3)
    est = Nearest().fit((e, n), np.array([1.0, 2.0, 3.0]))
    got = est.predict((np.array([1.0, 9.0, 16.0, 100.0]), np.zeros(4)))
    assert got.tolist() == [1.0, 2.0, 3.0, 3.0]
