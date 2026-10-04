"""Temporal denoising: gated Kalman and robust LOWESS (plan task T31).

The injected-signal tests port the study's ``evaluate.score``
(``00_tools/tests/temporal_denoising``): a one-epoch +30 mm spike should be
removed, a +20 mm step kept and a 60-day +15 mm bump kept. They cannot be
gamed by doing nothing (the residual-about-a-line metric can).
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import disp_xr as dxr
from disp_xr.timeseries.denoise import kalman_smooth, lowess, q_from_timescale

SPIKE, STEP, BUMP, BUMP_DAYS = 0.030, 0.020, 0.015, 60.0


def _series(n=70, npix=400, step_days=12, sigma=0.004, seed=0):
    rng = np.random.default_rng(seed)
    times = pd.date_range("2020-01-01", periods=n, freq=f"{step_days}D")
    years = np.arange(n) * step_days / 365.25
    y = 0.01 * years[:, None] + sigma * rng.standard_normal((n, npix))
    da = xr.DataArray(
        y, dims=("time", "pixel"), coords={"time": times}, attrs={"units": "meters"}
    )
    return da, years


def score(apply, da, years):
    """Percent of spike left, step kept, transient kept (study's ``score``)."""
    i = len(years) // 2
    base = apply(da).values
    ys = da.copy()
    ys[i] += SPIKE
    spike = 100 * np.median(apply(ys).values[i] - base[i]) / SPIKE
    yt = da.copy()
    yt[i:] += STEP
    f = apply(yt).values
    step = (
        100
        * np.median(
            np.median(f[i:] - base[i:], axis=0) - np.median(f[:i] - base[:i], axis=0)
        )
        / STEP
    )
    m = np.abs(years - years[i]) * 365.25 <= BUMP_DAYS / 2
    yb = da.copy()
    yb[m] += BUMP
    bump = 100 * np.median(np.median(apply(yb).values[m] - base[m], axis=0)) / BUMP
    return spike, step, bump


def test_kalman_matches_the_study():
    da, years = _series()
    spike, step, bump = score(
        lambda x: dxr.denoise(x, "kalman", tau_days=20, breaks=None), da, years
    )
    # Study (tau 20 d, gate 9): spike 0.2-0.4%, step 100%, transient 93%.
    assert spike < 1.0
    assert step > 90.0
    assert bump > 85.0


def test_lowess_matches_the_study():
    da, years = _series()
    spike, step, bump = score(lambda x: dxr.denoise(x, "lowess"), da, years)
    # Study (bw 60 d, 2 iterations): spike 0.1-0.2%, step 100%, transient 75%.
    assert spike < 2.0
    assert step > 90.0
    assert bump > 60.0


def test_declared_break_keeps_the_step():
    da, years = _series()
    i = len(years) // 2
    yt = da.copy()
    yt[i:] += STEP
    brk = [pd.Timestamp(da.time.values[i]) - pd.Timedelta("1D")]
    for method in ("kalman", "lowess"):
        plain = dxr.denoise(yt, method).values
        broken = dxr.denoise(yt, method, breaks=brk).values
        jump = np.median(broken[i] - broken[i - 1])
        assert jump == pytest.approx(STEP, abs=0.004)
        assert abs(np.median(plain[i] - plain[i - 1]) - STEP) > abs(jump - STEP) - 1e-9


def test_grid_output_refuses_gaps_and_is_lazy():
    da, _ = _series(n=60)
    holey = da.isel(time=np.r_[0:20, 40:60])  # a 240-day hole
    chunked = holey.chunk({"pixel": 100})
    for method in ("kalman", "lowess"):
        out = dxr.denoise(chunked, method, grid="auto")
        assert out.chunks is not None
        out = out.compute()
        assert (np.diff(out.time.values) == np.timedelta64(15, "D")).all()
        refused = dxr.timeseries.gap_mask(holey.time, out.time)
        assert refused.sum() > 5  # inside the 240-day hole
        assert np.isnan(out.values[refused]).all()
        assert np.isfinite(out.values[~refused]).all()


def test_nan_epochs_per_pixel():
    da, years = _series(npix=6)
    values = da.values.copy()
    values[10:14, 0] = np.nan
    values[:, 1] = np.nan
    out = dxr.denoise(da.copy(data=values), "kalman").values
    assert np.isfinite(out[:, 0]).all()  # predicted through its gap
    lo = dxr.denoise(da.copy(data=values), "lowess").values
    assert np.isnan(lo[:, 1]).all()


def test_fixed_lag_converges_to_rts():
    da, years = _series(npix=50)
    full = dxr.denoise(da, "kalman").values
    lag0 = dxr.denoise(da, "kalman", lag=0).values
    lag8 = dxr.denoise(da, "kalman", lag=8).values
    revision = lambda a: np.sqrt(np.mean((a - full) ** 2))  # noqa: E731
    assert revision(lag8) < 0.2 * revision(lag0)
    np.testing.assert_allclose(lag8[-1], full[-1])


def test_denoised_data_are_refused_by_fits_and_recorded():
    da, _ = _series(npix=4)
    ds = da.to_dataset(name="displacement")
    out = ds.disp.denoise("lowess")
    assert out.attrs["denoised"] == "lowess"
    assert "denoise" in out.attrs.get("history", "")
    with pytest.raises(ValueError, match="denoised"):
        dxr.fit_trajectory(out)
    with pytest.raises(ValueError, match="method"):
        dxr.denoise(ds, "median")


def test_helpers():
    assert q_from_timescale(365.25, 1.0) == pytest.approx(3.0)
    t = np.linspace(0, 2, 50)
    y = np.column_stack([np.sin(2 * np.pi * t), t])
    np.testing.assert_allclose(lowess(t, y, t, bandwidth_days=30)[:, 1], t, atol=1e-9)
    disp, rate = kalman_smooth(t, y[:, [1]] * 0 + 0.5 * t[:, None], r=np.array([1e-8]))
    np.testing.assert_allclose(rate[5:-5, 0], 0.5, atol=1e-3)
