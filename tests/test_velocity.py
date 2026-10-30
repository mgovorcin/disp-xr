"""Per-pixel velocity: OLS, Whittle colored noise and MIDAS (plan task T22)."""

import json
import logging
from pathlib import Path

import dask.array as darr
import numpy as np
import pandas as pd
import pytest
import xarray as xr

import disp_xr as dxr
from disp_xr.timeseries.noise import fit_noise
from disp_xr.timeseries.trajectory import Trajectory
from disp_xr.timeseries.velocity import _fill_gaps, _midas_pairs, _select_pairs
from disp_xr.tutorial import make_disp_stack

REFERENCE = json.loads(
    (Path(__file__).parent / "data" / "noise_reference.json").read_text()
)
TERMS = Trajectory(periods=(1.0,))


def _column(series):
    times = pd.to_datetime(series["dates"]).values
    values = np.array(series["values"])
    da = xr.DataArray(
        values[:, None, None], dims=("time", "y", "x"), coords={"time": times}
    )
    return times, values, da


@pytest.mark.parametrize("series", REFERENCE["series"], ids=lambda s: s["name"])
def test_midas_reproduces_geepers(series):
    _, _, da = _column(series)
    ref = series["fits"]["MIDAS"]
    out = dxr.fit_velocity(da, "midas").isel(y=0, x=0)
    assert float(out.velocity) == pytest.approx(ref["velocity"], rel=1e-9)
    assert float(out.velocity_sigma) == pytest.approx(ref["velocity_sigma"], rel=1e-9)
    assert float(out.outlier_fraction) == pytest.approx(
        ref["outlier_fraction"], rel=1e-9
    )
    assert float(out.velocity_scatter) == pytest.approx(
        ref["velocity_scatter"], rel=1e-9
    )


@pytest.mark.parametrize(
    "name", ["insar_12day", "insar_12day_gaps", "gnss_flicker", "gnss_randomwalk"]
)
def test_whittle_matches_single_series(name):
    (series,) = [s for s in REFERENCE["series"] if s["name"] == name]
    times, values, da = _column(series)
    out = dxr.fit_velocity(da, "whittle", terms=TERMS).isel(y=0, x=0)
    one = fit_noise(
        times,
        values,
        terms=TERMS,
        model="PLWN",
        method="whittle",
        sampling_days=series["sampling_days"],
    )
    assert float(out.kappa) == pytest.approx(one.kappa, abs=0.02)
    # A white fraction near zero is "no white noise" for both, however small.
    # Amplitudes come from the Whittle variance of the gap-filled grid, which
    # interpolation lowers (about 10% with 20% gaps); fit_noise measures the
    # variance at the observed epochs.
    total = np.hypot(float(out.sigma_white), float(out.sigma_powerlaw))
    tolerance = 0.15 if "gaps" in name else 0.05
    expected = np.hypot(one.sigma_white, one.sigma_powerlaw)
    assert total == pytest.approx(expected, rel=tolerance)
    # The OLS velocity's sigma under the estimated covariance, against GLS.
    assert float(out.velocity_sigma) == pytest.approx(one.velocity_sigma, rel=0.2)
    ols = dxr.fit_trajectory(da, TERMS).velocity.isel(y=0, x=0)
    assert float(out.velocity) == pytest.approx(float(ols), rel=1e-9)


def test_select_pairs_one_year_apart():
    years = np.arange(0, 3, 12 / 365.25)
    pairs = _select_pairs(years)
    separation = years[pairs[:, 1]] - years[pairs[:, 0]]
    assert separation.min() >= 1 - 0.001 and separation.max() < 1 + 12 / 365.25
    both = _midas_pairs(years)
    assert len(both) == 2 * len(pairs)


def test_midas_is_robust_to_a_step():
    rng = np.random.default_rng(0)
    times = pd.date_range("2017-01-01", periods=180, freq="12D").values
    years = np.arange(180) * 12 / 365.25
    y = 0.004 * years + 1e-3 * rng.standard_normal(180)
    y[90:] += 0.02  # unmodelled 2 cm step
    da = xr.DataArray(y[:, None, None], dims=("time", "y", "x"), coords={"time": times})
    midas = dxr.fit_velocity(da, "midas").velocity.item()
    ols = dxr.fit_velocity(da, "ols").velocity.item()
    assert abs(midas - 0.004) < 0.25 * abs(ols - 0.004)


def test_fill_gaps_matches_np_interp():
    rng = np.random.default_rng(1)
    rows = rng.normal(size=(5, 30))
    rows[rng.random(rows.shape) < 0.3] = np.nan
    rows[:, 0] = np.nan  # leading gap
    filled = _fill_gaps(rows)
    idx = np.arange(30)
    for row, out in zip(rows, filled):
        ok = np.isfinite(row)
        np.testing.assert_allclose(out, np.interp(idx, idx[ok], row[ok]))


@pytest.fixture(scope="module")
def frame(tmp_path_factory):
    root = tmp_path_factory.mktemp("vel")
    truth = make_disp_stack(
        root / "g", "three_ministacks", shape=(64, 64), epochs_per_ministack=30
    )
    return truth, dxr.chain(dxr.open_disp(root / "g"))


@pytest.mark.parametrize("method", ["ols", "whittle", "midas"])
def test_frame(frame, method, caplog):
    truth, ds = frame
    terms = Trajectory(steps=(f"{truth.step_date:%Y-%m-%d}",))
    with caplog.at_level(logging.WARNING, logger="disp_xr"):
        out = dxr.fit_velocity(ds, method, terms=terms)
    assert isinstance(out.velocity.data, darr.Array)
    out = out.compute()
    error = out.velocity - truth.velocity
    if method == "midas":
        # The 10 mm step is not modelled by MIDAS; in a 3-year record a third
        # of the one-year pairs span it, which biases the median (Blewitt et
        # al., 2016). The spatial pattern still follows the truth.
        ok = np.isfinite(out.velocity.values)
        corr = np.corrcoef(out.velocity.values[ok], truth.velocity[ok])[0, 1]
        assert corr > 0.95
    else:
        assert float(np.nanmedian(np.abs(error))) < 1.5e-3  # m/yr
    assert (out.velocity_sigma > 0).all()
    assert out.velocity.attrs["units"] == "meters/year"
    assert out.rio.crs == ds.rio.crs
    if method == "whittle":
        # 91 epochs: below the reliable length, so a warning.
        assert "underestimates" in caplog.text
        assert ((out.kappa >= -3) & (out.kappa <= 0)).all()


def test_frame_is_chunk_independent(frame):
    _, ds = frame
    whole = dxr.fit_velocity(ds, "midas")
    split = dxr.fit_velocity(ds.disp.rechunk("maps"), "midas")
    xr.testing.assert_allclose(whole.velocity, split.velocity)


def test_errors(frame):
    _, ds = frame
    with pytest.raises(ValueError, match="method"):
        dxr.fit_velocity(ds, "kalman")
    with pytest.raises(ValueError, match="linear term"):
        dxr.fit_velocity(ds, "whittle", terms=Trajectory(polynomial=0))
    with pytest.raises(ValueError, match="time"):
        dxr.fit_velocity(ds.displacement.isel(time=0))


@pytest.mark.parametrize("method", ["whittle", "midas", "gls"])
def test_rate_changes_need_ols(method):
    """The other methods report one rate: it would silently be the first."""
    times = pd.date_range("2018-01-01", periods=100, freq="12D")
    da = xr.DataArray(
        np.arange(100.0)[:, None, None], dims=("time", "y", "x"), coords={"time": times}
    )
    with pytest.raises(ValueError, match="rate_changes needs method='ols'"):
        dxr.fit_velocity(da, method, terms=dxr.Trajectory(rate_changes=("2020-01-01",)))


def test_ols_velocity_reports_the_present_rate():
    times = pd.date_range("2018-01-01", periods=200, freq="12D").values
    terms = dxr.Trajectory(periods=(), rate_changes=("2021-06-01",))
    design = terms.design(times)
    y = design @ np.array([1.0, 4.0, -3.0])
    da = xr.DataArray(y[:, None, None], dims=("time", "y", "x"), coords={"time": times})
    out = dxr.fit_velocity(da, "ols", terms=terms)
    np.testing.assert_allclose(out.velocity, 4.0, atol=1e-6)
    np.testing.assert_allclose(out.present_velocity, 1.0, atol=1e-6)


def test_fit_noise_estimates_rate_changes():
    """The route the fit_velocity error names for colored-noise sigmas."""
    times = pd.date_range("2018-01-01", periods=200, freq="12D").values
    terms = Trajectory(periods=(), rate_changes=("2021-06-01",))
    rng = np.random.default_rng(0)
    y = terms.design(times) @ np.array([1.0, 4.0, -3.0]) + rng.normal(0, 0.5, 200)
    result = fit_noise(times, y, terms=terms)
    value, sigma = result.parameters["rate_change_2021-06-01"]
    assert abs(value + 3.0) < 4 * sigma and sigma > 0
