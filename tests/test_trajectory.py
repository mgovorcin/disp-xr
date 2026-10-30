"""Trajectory models and OLS velocity (plan task T20)."""

import dask.array as darr
import numpy as np
import pandas as pd
import pytest
import xarray as xr

import disp_xr as dxr
from disp_xr.timeseries import create_design_matrix, infer_parameters_from_names
from disp_xr.timeseries.trajectory import Trajectory, _solve, predict_trajectory
from disp_xr.tutorial import make_disp_stack


@pytest.fixture(scope="module")
def long_stack(tmp_path_factory):
    """Three ministacks of 30 epochs (about 3 years) with a step mid-record."""
    root = tmp_path_factory.mktemp("traj")
    truth = make_disp_stack(
        root / "g", "three_ministacks", shape=(64, 64), epochs_per_ministack=30
    )
    return truth, dxr.chain(dxr.open_disp(root / "g"))


def _terms(truth):
    return Trajectory(steps=(f"{truth.step_date:%Y-%m-%d}",))


def test_recovers_the_truth(long_stack):
    truth, ds = long_stack
    fit = dxr.fit_trajectory(ds, _terms(truth)).compute()
    error = fit.velocity - truth.velocity
    # White noise of 1 mm: errors within the OLS sigma's statistics.
    normalized = (error / fit.velocity_sigma).values
    assert np.nanmax(np.abs(normalized)) < 5
    assert 0.9 < np.mean(np.abs(normalized) < 2) <= 1.0
    assert np.nanmedian(np.abs(error)) < 1e-3  # m/yr
    np.testing.assert_allclose(
        fit.annual_amplitude.median(), truth.annual_amplitude, rtol=0.05
    )
    step = fit.coefficients.sel(coefficient=f"step_{truth.step_date:%Y-%m-%d}")
    np.testing.assert_allclose(step.median(), truth.step, rtol=0.05)
    assert fit.velocity.attrs["units"] == "meters/year"
    np.testing.assert_allclose(fit.rms.median(), 0.001, rtol=0.2)


def test_matches_per_pixel_lstsq_with_mixed_nan_patterns():
    rng = np.random.default_rng(1)
    times = pd.date_range("2020-01-01", periods=60, freq="12D").values
    data = rng.normal(0, 1e-3, (60, 6, 7)) + np.linspace(0, 0.05, 60)[:, None, None]
    data[rng.random(data.shape) < 0.3] = np.nan  # a different pattern per pixel
    da = xr.DataArray(data, dims=("time", "y", "x"), coords={"time": times})
    terms = Trajectory(periods=(1.0, 0.5))
    fit = dxr.fit_trajectory(da, terms)
    design = terms.design(times)
    for iy, ix in [(0, 0), (3, 4), (5, 6)]:
        y = data[:, iy, ix]
        ok = np.isfinite(y)
        expected, *_ = np.linalg.lstsq(design[ok], y[ok], rcond=None)
        np.testing.assert_allclose(
            fit.coefficients[:, iy, ix], expected, rtol=1e-8, atol=1e-12
        )
        assert int(fit.n_valid[iy, ix]) == ok.sum()


def test_too_few_epochs_is_nan():
    times = pd.date_range("2020-01-01", periods=10, freq="12D").values
    data = np.ones((10, 1, 2))
    data[3:, 0, 0] = np.nan  # 3 epochs for 4 coefficients
    da = xr.DataArray(data, dims=("time", "y", "x"), coords={"time": times})
    fit = dxr.fit_trajectory(da)
    assert np.isnan(fit.velocity[0, 0]) and np.isfinite(fit.velocity[0, 1])
    assert np.isnan(dxr.fit_trajectory(da, min_valid=11).velocity).all()


def test_zero_weight_drops_an_epoch():
    rng = np.random.default_rng(2)
    times = pd.date_range("2020-01-01", periods=40, freq="12D").values
    data = rng.normal(size=(40, 2, 2))
    da = xr.DataArray(data, dims=("time", "y", "x"), coords={"time": times})
    weights = xr.ones_like(da)
    weights[5] = 0
    dropped = da.copy()
    dropped[5] = np.nan
    xr.testing.assert_allclose(
        dxr.fit_trajectory(da, weights=weights).coefficients,
        dxr.fit_trajectory(dropped).coefficients,
    )


def test_steps_and_postseismic_terms_fire():
    # The branch's builder compared absolute onset years with relative time,
    # so these columns were always zero.
    times = pd.date_range("2020-01-01", periods=25, freq="30D").values
    design = create_design_matrix(
        times, 1, [1.0], ["2021-01-01"], {"2021-01-01": [60]}, {"20210101": [30]}
    )
    after = times >= np.datetime64("2021-01-01")
    np.testing.assert_array_equal(design[:, 4], after)
    assert design[after, 5].max() > 0.9 and design[~after, 5].max() == 0
    assert design[after, 6].max() > 2 and design[~after, 6].max() == 0


def test_names_round_trip_and_old_names():
    terms = Trajectory(
        polynomial=2,
        periods=(1.0, 0.5),
        steps=("2021-01-01",),
        postseismic_log=(("2021-01-01", 30),),
        postseismic_exp=(("2021-01-01", 60),),
    )
    params = infer_parameters_from_names(terms.names)
    assert params["polynomial_degree"] == 2 and params["periods"] == [1.0, 0.5]
    assert params["logarithmic_terms"] == {"2021-01-01": [30]}
    assert "postseismic_log_2021-01-01_tau30d" in terms.names
    old = infer_parameters_from_names(["log_term_2021-01-01_tau30d"])
    assert old["logarithmic_terms"] == {"2021-01-01": [30]}


def test_predict_rebuilds_the_model(long_stack):
    truth, ds = long_stack
    fit = dxr.fit_trajectory(ds, _terms(truth))
    model = predict_trajectory(fit, ds.time.values)
    assert isinstance(model.data, darr.Array) and model.dtype == np.float32
    design = _terms(truth).design(ds.time.values)
    expected = np.einsum("tp,pyx->tyx", design, fit.coefficients.values)
    np.testing.assert_allclose(model.values, expected, rtol=1e-5, atol=1e-6)


def test_lazy_and_chunk_independent(long_stack):
    truth, ds = long_stack
    fit = dxr.fit_trajectory(ds, _terms(truth))
    assert isinstance(fit.velocity.data, darr.Array)
    split = dxr.fit_trajectory(ds.disp.rechunk("maps"), _terms(truth))
    xr.testing.assert_allclose(fit.velocity, split.velocity)
    assert fit.rio.crs == ds.rio.crs
    assert "fit_trajectory" in fit.attrs["history"]


def test_solver_on_constant_series_has_zero_sigma():
    design = np.column_stack([np.ones(20), np.arange(20.0)])
    data = (2.0 + 0.5 * np.arange(20.0))[None, :]
    coef, sigma, rms, count = _solve(data, np.ones_like(data), design, 0)
    np.testing.assert_allclose(coef[0], [2.0, 0.5])
    np.testing.assert_allclose(sigma[0], 0, atol=1e-12)
    assert count[0] == 20


def test_grouped_and_batched_paths_agree(monkeypatch):
    from disp_xr.timeseries import trajectory

    rng = np.random.default_rng(3)
    times = pd.date_range("2020-01-01", periods=50, freq="12D").values
    data = rng.normal(0, 1e-3, (200, 50)) + np.linspace(0, 0.03, 50)
    data[:40, 7] = np.nan  # a second shared pattern
    data[150:, rng.integers(0, 50, 50)] = np.nan  # rare patterns
    design = Trajectory(periods=(1.0,)).design(times)
    weights = np.ones_like(data)
    monkeypatch.setattr(trajectory, "GROUP_MIN_PIXELS", 10)
    grouped = trajectory._solve(data, weights, design, 0)
    monkeypatch.setattr(trajectory, "GROUP_MIN_PIXELS", 10**9)
    batched = trajectory._solve(data, weights, design, 0)
    for a, b in zip(grouped, batched):
        np.testing.assert_allclose(a, b, rtol=1e-9, atol=1e-15)


# --- rate changes -----------------------------------------------------------


def _hinge_series(
    times, change="2021-06-01", before=3.0, delta=-8.0, noise=0.0, seed=0
):
    """Offset + two-segment trend + annual cycle, in years from the first epoch."""
    terms = Trajectory(rate_changes=(change,))
    design = terms.design(times)
    truth = {
        "constant": 5.0,
        "linear_trend": before,
        "annual_sin": 1.5,
        "annual_cos": -0.5,
    }
    truth[f"rate_change_{change}"] = delta
    beta = np.array([truth[n] for n in terms.names])
    rng = np.random.default_rng(seed)
    return terms, (
        design @ beta + rng.normal(0, noise, len(times)) if noise else design @ beta
    )


def _times(n=200):
    return pd.date_range("2018-01-01", periods=n, freq="12D").values


def test_rate_change_column_is_a_hinge():
    times = _times()
    terms = Trajectory(periods=(), rate_changes=("2021-06-01",))
    assert terms.names == ["constant", "linear_trend", "rate_change_2021-06-01"]
    hinge = terms.design(times)[:, -1]
    after = times >= np.datetime64("2021-06-01")
    assert (hinge[~after] == 0).all() and (hinge[after] > 0).all()
    # continuous, slope 1 per year after the change
    days = (
        (times[after] - np.datetime64("2021-06-01"))
        .astype("timedelta64[D]")
        .astype(float)
    )
    np.testing.assert_allclose(hinge[after], days / 365.25, atol=2e-3)


def test_rate_changes_come_last_so_other_terms_keep_their_positions():
    plain = Trajectory(steps=("2020-01-01",))
    terms = Trajectory(steps=("2020-01-01",), rate_changes=("2021-06-01",))
    assert terms.names[: len(plain.names)] == plain.names
    times = _times()
    np.testing.assert_array_equal(
        terms.design(times)[:, : len(plain.names)], plain.design(times)
    )


def test_rate_change_names_round_trip():
    terms = Trajectory(rate_changes=("2020-01-01", "2022-03-04"))
    params = infer_parameters_from_names(terms.names)
    assert params["rate_change_dates"] == ["2020-01-01", "2022-03-04"]


def test_fit_recovers_rates_and_predict_rebuilds_them():
    times = _times()
    terms, y = _hinge_series(times)
    data = np.broadcast_to(y, (4, 4, y.size)).transpose(2, 0, 1).astype(np.float64)
    da = xr.DataArray(
        data, dims=("time", "y", "x"), coords={"time": times}, attrs={"units": "mm"}
    )
    fit = dxr.fit_trajectory(da, terms)
    np.testing.assert_allclose(fit.velocity, 3.0, atol=1e-6)
    np.testing.assert_allclose(fit.present_velocity, -5.0, atol=1e-6)
    np.testing.assert_allclose(
        fit.coefficients.sel(coefficient="rate_change_2021-06-01"), -8.0, atol=1e-6
    )
    assert fit.present_velocity.attrs["units"] == "mm/year"
    np.testing.assert_allclose(
        predict_trajectory(fit, times, dtype="float64").isel(y=0, x=0), y, atol=1e-6
    )


def test_present_velocity_sigma_includes_the_covariance():
    times = _times()
    terms, y = _hinge_series(times, noise=1.0)
    design = terms.design(times)
    inv = np.linalg.inv(design.T @ design)
    c = np.array(
        [n == "linear_trend" or n.startswith("rate_change") for n in terms.names], float
    )
    beta = inv @ design.T @ y
    s2 = ((y - design @ beta) ** 2).sum() / (y.size - design.shape[1])
    da = xr.DataArray(y[:, None, None], dims=("time", "y", "x"), coords={"time": times})
    fit = dxr.fit_trajectory(da, terms)
    expected = np.sqrt(c @ inv @ c * s2)
    np.testing.assert_allclose(fit.present_velocity_sigma.item(), expected, rtol=1e-10)
    # with the two terms' negative covariance it is not their quadrature sum
    quadrature = np.hypot(
        fit.sigma.sel(coefficient="linear_trend").item(),
        fit.sigma.sel(coefficient="rate_change_2021-06-01").item(),
    )
    assert expected < quadrature


def test_grouped_and_batched_contrast_sigma_agree(monkeypatch):
    from disp_xr.timeseries import trajectory

    times = _times()
    terms, y = _hinge_series(times, noise=1.0)
    rng = np.random.default_rng(1)
    data = y + rng.normal(0, 1.0, (70, y.size))
    weights = np.ones_like(data)
    design = terms.design(times)
    c = np.array(
        [n == "linear_trend" or n.startswith("rate_change") for n in terms.names], float
    )
    grouped = trajectory._solve(data, weights, design, 0, c)
    monkeypatch.setattr(trajectory, "GROUP_MIN_PIXELS", 10**6)
    batched = trajectory._solve(data, weights, design, 0, c)
    np.testing.assert_allclose(grouped[4], batched[4], rtol=1e-9)
    assert np.isfinite(grouped[4]).all()


def test_no_epochs_after_the_change_gives_nan_not_a_wrong_rate():
    times = _times(50)  # ends in 2019
    terms, _ = _hinge_series(times, change="2021-06-01")
    y = np.arange(50.0)
    da = xr.DataArray(y[:, None, None], dims=("time", "y", "x"), coords={"time": times})
    fit = dxr.fit_trajectory(da, terms)
    assert np.isnan(fit.present_velocity.item()) and np.isnan(fit.velocity.item())


def test_no_present_velocity_without_a_linear_trend_or_rate_changes():
    times = _times()
    da = xr.DataArray(
        np.arange(200.0)[:, None, None], dims=("time", "y", "x"), coords={"time": times}
    )
    assert "present_velocity" not in dxr.fit_trajectory(da, Trajectory())
    quadratic = Trajectory(polynomial=2, rate_changes=("2021-06-01",))
    assert "present_velocity" not in dxr.fit_trajectory(da, quadratic)
