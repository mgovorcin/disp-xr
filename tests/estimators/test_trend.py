"""Polynomial trends and per-epoch de-ramping (plan task T24)."""

import dask.array as darr
import numpy as np
import pytest
import xarray as xr

import disp_xr  # noqa: F401  (registers .disp)
from disp_xr.models import Trend, deramp, polynomial_terms

from .common import check_estimator, synthetic


@pytest.mark.parametrize("degree", [0, 1, 2])
def test_trend_passes_shared_checks(degree):
    check_estimator(lambda: Trend(degree))


def test_terms():
    assert polynomial_terms(0) == [(0, 0)]
    assert polynomial_terms(2) == [(0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2)]
    with pytest.raises(ValueError):
        polynomial_terms(-1)


def test_trend_recovers_a_quadratic_in_utm():
    (e, n), _ = synthetic()
    de, dn = (e - 515_000) / 1000, (n - 4_115_000) / 1000
    data = 0.01 + 2e-4 * de - 1e-4 * dn + 3e-6 * de**2 - 2e-6 * de * dn + 1e-6 * dn**2
    est = Trend(2).fit((e, n), data)
    np.testing.assert_allclose(est.predict((e, n)), data, rtol=1e-10, atol=1e-14)
    assert est.score((e, n), data) == pytest.approx(1.0)


def test_trend_weights_and_errors():
    (e, n), data = synthetic()
    spiked = data.copy()
    spiked[0] = 10.0
    weights = np.ones_like(data)
    weights[0] = 0.0
    a = Trend(1).fit((e, n), spiked, weights).predict((e, n))
    b = Trend(1).fit((e[1:], n[1:]), data[1:]).predict((e, n))
    np.testing.assert_allclose(a, b)
    with pytest.raises(ValueError, match="points for"):
        Trend(2).fit((e[:4], n[:4]), data[:4])


def _ramp_stack(nt=5, ny=40, nx=50, seed=0, bump=0.0):
    rng = np.random.default_rng(seed)
    x = 500_000.0 + 30.0 * np.arange(nx) + 15
    y = 4_100_000.0 - 30.0 * np.arange(ny) - 15
    xx, yy = np.meshgrid(x, y)
    a, b, c = (
        rng.normal(size=(3, nt, 1, 1))
        * np.array([1e-2, 1e-6, 1e-6])[:, None, None, None]
    )
    ramps = a + b * (xx - 500_000) + c * (yy - 4_100_000)
    signal = bump * np.exp(-((xx - x[35]) ** 2 + (yy - y[20]) ** 2) / (2 * 120.0**2))
    values = ramps + signal
    values[:, :3, :4] = np.nan
    ds = xr.Dataset(
        {"displacement": (("time", "y", "x"), values, {"units": "meters"})},
        coords={"time": np.arange(nt), "y": y, "x": x},
    ).rio.write_crs("EPSG:32610")
    return ds, ramps, signal


def test_deramp_removes_ramps_to_float_precision():
    ds, _, _ = _ramp_stack()
    out = deramp(ds.chunk({"y": 16, "x": 16}), degree=1)
    assert isinstance(out.displacement.data, darr.Array)
    flat = out.displacement.values
    assert np.nanmax(np.abs(flat)) < 1e-12
    assert np.isnan(flat[:, :3, :4]).all()
    coef = out.displacement_ramp_coefficients
    assert coef.dims == ("time", "term") and list(coef.term.values) == ["1", "e", "n"]
    assert "deramp" in out.attrs["history"]


def test_deramp_with_mask_keeps_the_signal():
    ds, ramps, signal = _ramp_stack(bump=0.05)
    yy, xx = np.meshgrid(
        np.arange(ds.sizes["y"]), np.arange(ds.sizes["x"]), indexing="ij"
    )
    away = xr.DataArray(np.hypot(yy - 20, xx - 35) > 20, dims=("y", "x"))
    out = deramp(ds, degree=1, mask=away, stride=1)
    np.testing.assert_allclose(
        out.displacement.values[:, 3:, :],
        np.broadcast_to(signal, ramps.shape)[:, 3:, :],
        atol=1e-4,
    )
    # Without the mask the bump leaks into the fitted plane.
    leaky = deramp(ds, degree=1, stride=1)
    assert np.nanmax(np.abs(leaky.displacement.values - signal)) > 1e-3


def test_deramp_degree2_and_accessor():
    ds, _, _ = _ramp_stack()
    yy = (ds.y - ds.y.mean()) ** 2 * 1e-9
    curved = ds.assign(displacement=ds.displacement + yy)
    out = curved.disp.deramp(2, stride=2)
    assert np.nanmax(np.abs(out.displacement.values)) < 1e-10
    assert out.displacement_ramp_coefficients.sizes["term"] == 6
