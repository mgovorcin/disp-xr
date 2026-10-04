"""Per-pixel model diagnostics and coefficient significance (T32.6b-c)."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from scipy import stats

import disp_xr as dxr
from disp_xr.diagnostics import (
    AD_CRITICAL_5,
    _stats,
    anderson_darling,
    model_diagnostics,
    significance,
)


def test_anderson_darling_matches_scipy():
    rng = np.random.default_rng(0)
    series = rng.normal(size=(6, 60))
    series[1] = rng.exponential(size=60)  # not normal
    series[2, ::5] = np.nan
    ours = anderson_darling(series)
    for k, row in enumerate(series):
        row = row[np.isfinite(row)]
        n = row.size
        expected = stats.anderson(row, "norm", method="interpolate").statistic * (
            1 + 0.75 / n + 2.25 / n**2
        )
        assert ours[k] == pytest.approx(expected, rel=1e-10)
    assert ours[1] > AD_CRITICAL_5 > np.median(np.delete(ours, 1))
    assert np.isnan(anderson_darling(np.full((1, 5), 1.0)))[0]


def test_moments_and_autocorrelation():
    rng = np.random.default_rng(1)
    n = 4000
    white = rng.normal(size=n)
    ar = np.zeros(n)
    for t in range(1, n):
        ar[t] = 0.8 * ar[t - 1] + rng.normal()
    skewed = rng.exponential(size=n)
    out = _stats(np.stack([white, ar, skewed]), lags=(1, 2))
    # columns: n, mean, std, skew, kurt, ac1, ac2, ad
    assert out[0, 3] == pytest.approx(stats.skew(white), rel=1e-9)
    assert out[2, 4] == pytest.approx(stats.kurtosis(skewed), rel=1e-9)
    assert abs(out[0, 5]) < 0.05 and out[1, 5] == pytest.approx(0.8, abs=0.03)
    assert out[1, 6] == pytest.approx(0.64, abs=0.05)


def _stack(noise, nt=120, ny=6, nx=7, seed=2):
    rng = np.random.default_rng(seed)
    times = pd.date_range("2020-01-01", periods=nt, freq="12D")
    years = np.arange(nt) * 12 / 365.25
    if noise == "white":
        eps = 0.002 * rng.standard_normal((nt, ny, nx))
    else:
        eps = np.zeros((nt, ny, nx))
        for t in range(1, nt):
            eps[t] = 0.9 * eps[t - 1] + 0.002 * rng.standard_normal((ny, nx))
    values = (
        0.01 * years[:, None, None] + 0.003 * np.sin(2 * np.pi * years)[:, None, None]
    )
    return xr.DataArray(
        values + eps,
        dims=("time", "y", "x"),
        coords={"time": times, "y": np.arange(ny) * -30.0, "x": np.arange(nx) * 30.0},
        attrs={"units": "meters"},
    )


def test_model_diagnostics_white_and_red():
    for noise, expect_ac in (("white", False), ("red", True)):
        da = _stack(noise).chunk({"y": 3, "x": 4})
        fit = dxr.fit_trajectory(da, dxr.Trajectory(periods=(1.0,)))
        diag = model_diagnostics(da, fit, lags=(1, 3))
        assert diag.anderson_darling.chunks is not None  # lazy
        diag = diag.compute()
        assert set(diag.data_vars) >= {
            "residual_mean",
            "residual_std",
            "skewness",
            "kurtosis",
            "autocorrelation_lag1",
            "autocorrelation_lag3",
            "anderson_darling",
            "n",
        }
        np.testing.assert_allclose(diag.residual_mean, 0.0, atol=1e-9)  # OLS residuals
        lag1 = float(diag.autocorrelation_lag1.median())
        assert (lag1 > 0.5) is expect_ac
        if noise == "white":
            assert float(diag.residual_std.median()) == pytest.approx(0.002, rel=0.2)


def test_significance():
    coef = xr.DataArray([[1.9, 2.5]], dims=("y", "coefficient")).T
    sigma = xr.ones_like(coef)
    fit = xr.Dataset({"coefficients": coef, "sigma": sigma})
    out = significance(fit, alpha=0.05)
    np.testing.assert_allclose(out.z, coef)
    assert float(out.p_value[0, 0]) == pytest.approx(2 * stats.norm.sf(1.9))
    assert bool(out.significant[0, 0]) is False  # p = 0.057
    assert bool(out.significant[1, 0]) is True  # p = 0.012
    np.testing.assert_allclose(out.ci_high - out.ci_low, 2 * 1.959964, rtol=1e-5)
