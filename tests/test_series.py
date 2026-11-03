"""Per-series geepers models as maps (disp_xr.timeseries.series)."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import disp_xr as dxr
from disp_xr.timeseries.series import _bins, _median_windows

pytest.importorskip("geepers.smooth")

N_EPOCHS = 240  # 8 years at 12 days


def _cube(signal, noise_mm=1.0, n=10, seed=0, units="meters"):
    """A chained, re-referenced Dataset; ``signal(years)`` is in mm."""
    times = pd.date_range("2017-01-01", periods=N_EPOCHS, freq="12D")
    years = np.arange(N_EPOCHS) * 12 / 365.25
    rng = np.random.default_rng(seed)
    data = signal(years)[:, None, None] + rng.normal(0, noise_mm, (N_EPOCHS, n, n))
    scale = 1e-3 if units == "meters" else 1.0
    da = xr.DataArray(
        (data * scale).astype("float32"),
        dims=("time", "y", "x"),
        coords={
            "time": times,
            "y": np.arange(n) * -30.0 + 3000.0,
            "x": np.arange(n) * 30.0,
        },
        attrs={"units": units},
    )
    return xr.Dataset(
        {"displacement": da},
        attrs={"reference_date": str(times[0]), "reference": '{"method": "median"}'},
    )


def _line(years):
    return 4.0 * years


def _hinge(years):
    return 2.0 * years - 12.0 * np.maximum(0.0, years - 4.0)


def _sigmoid(years):
    return 200.0 / (1.0 + np.exp(-(years - 4.0) * 2.0))


def test_median_windows():
    block = np.arange(16.0).reshape(1, 4, 4)
    block[0, 0, 0] = np.nan
    out = _median_windows(block, 2)
    np.testing.assert_allclose(out[0, 0, 0], np.nanmedian(block[0, :2, :2]))
    np.testing.assert_allclose(out[0, 1, 1], np.median(block[0, 2:, 2:]))


def test_bins_start_at_the_first_epoch():
    times = pd.date_range("2020-01-01", periods=7, freq="12D").values
    index, starts = _bins(times, 30.0)
    assert index.tolist() == [0, 0, 0, 1, 1, 2, 2]
    assert starts[0] == np.datetime64("2020-01-01") and starts.size == 3


def test_smooth_on_a_line_gives_its_rate_on_the_multilooked_grid():
    maps = dxr.fit_series(_cube(_line))
    assert maps.sizes["y"] == maps.sizes["x"] == 2  # 10 pixels, 5 x 5 windows
    np.testing.assert_allclose(maps.x, [60.0, 210.0])  # window centers
    assert (maps.trend_shape == 0).all()
    np.testing.assert_allclose(maps.rate, 4e-3, atol=4e-4)  # back in meters per year
    assert maps.rate.attrs["units"] == "meters/year"
    assert maps.rates.dims == ("time", "y", "x")
    assert maps.attrs["status"] == "provisional"
    assert [s["op"] for s in maps.disp.history][-1] == "fit_series"


def test_smooth_follows_a_signal_that_is_not_piecewise():
    maps = dxr.fit_series(_cube(_sigmoid, noise_mm=0.5))
    assert (maps.trend_shape == 2).all()
    series = maps.rates.isel(y=0, x=0)
    # the rate peaks near the middle of the record, far above its ends
    peak = pd.Timestamp(series.idxmax().item())
    assert abs((peak - pd.Timestamp("2021-01-01")).days) < 365
    assert series.max() > 5 * abs(series.isel(time=-1))


def test_millimeters_give_the_same_rates_as_meters():
    meters = dxr.fit_series(_cube(_line))
    millimeters = dxr.fit_series(_cube(_line, units="mm"))
    np.testing.assert_allclose(millimeters.rate, meters.rate * 1000, rtol=1e-3)


def test_lazy_equals_eager_with_chunks_that_split_windows():
    ds = _cube(_line)
    eager = dxr.fit_series(ds)
    lazy = dxr.fit_series(ds.chunk({"time": 1, "y": 3, "x": 7}))
    assert lazy.rate.chunks is not None
    np.testing.assert_allclose(lazy.rate.compute(), eager.rate, rtol=1e-3)


def test_short_or_empty_cells_are_nan_with_code_minus_one():
    ds = _cube(_line)
    ds["displacement"][:, :5, :5] = np.nan
    maps = dxr.fit_series(ds)
    assert maps.trend_shape.values[0, 0] == -1 and np.isnan(maps.rate.values[0, 0])
    assert maps.trend_shape.values[1, 1] == 0


def test_linearity_is_provisional_and_does_not_warn():
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        maps = dxr.fit_series(_cube(_line), "linearity", n_simulations=100)
    assert maps.attrs["status"].startswith("provisional")
    assert (maps.linearity_model == 0).all()
    np.testing.assert_allclose(maps.velocity, 4e-3, atol=4e-4)
    assert (maps.validity_years > 0).all()


def test_linearity_finds_a_rate_change():
    maps = dxr.fit_series(_cube(_hinge), "linearity", n_simulations=100)
    assert (maps.linearity_model != 0).all()
    assert (maps.departure > 0).all()
    breakpoint_ = pd.Timestamp(maps.breakpoint.values[0, 0])
    assert abs((breakpoint_ - pd.Timestamp("2021-01-01")).days) < 365


def test_significant_is_the_gate_on_p_and_departure():
    hinge = dxr.fit_series(_cube(_hinge), "linearity", n_simulations=100)
    line = dxr.fit_series(_cube(_line), "linearity", n_simulations=100)
    high = dxr.fit_series(
        _cube(_hinge), "linearity", n_simulations=100, min_departure_mm=1e6
    )
    assert (hinge.significant == 1).all()
    assert (line.significant == 0).all()
    assert (high.significant == 0).all()
    assert (high.linearity_model != 0).all()  # the verdict is not the gate's


def test_rate_noise_comes_from_a_second_fit():
    from geepers.linearity import linearity_test

    cube = _cube(_line, n=5)
    two = dxr.fit_series(cube, "linearity", n_simulations=100)
    one = dxr.fit_series(cube, "linearity", n_simulations=100, uncertainty_noise=None)
    assert two.velocity_sigma.attrs["noise_model"] == "RWFNWN"
    assert one.velocity_sigma.attrs["noise_model"] == "FNWN"
    index, starts = _bins(cube.time.values, 30.0)
    series = cube.displacement.isel(y=slice(0, 5), x=slice(0, 5)).median(("y", "x"))
    y = pd.Series(series.values * 1000.0).groupby(index).median().to_numpy(dtype=float)
    direct = linearity_test(
        starts, y, sampling_days=30.0, noise_model="RWFNWN", n_simulations=1
    )
    np.testing.assert_allclose(
        two.velocity_sigma.values[0, 0], direct.trend.velocity_uncertainty / 1000.0
    )
    # The verdict's columns do not depend on the second fit
    np.testing.assert_array_equal(two.p_quadratic, one.p_quadratic)
    np.testing.assert_array_equal(two.departure, one.departure)


def test_needs_a_chained_series_and_known_units():
    ds = _cube(_line).assign_coords(
        reference_time=("time", np.repeat(pd.date_range("2017", periods=2), 120))
    )
    del ds.attrs["reference_date"]
    with pytest.raises(ValueError, match="chain"):
        dxr.fit_series(ds)
    with pytest.raises(ValueError, match="meters or millimeters"):
        dxr.fit_series(_cube(_line).displacement.assign_attrs(units="radians"))
    with pytest.raises(ValueError, match="'smooth' or 'linearity'"):
        dxr.fit_series(_cube(_line), "spline")


def test_warns_without_a_reference():
    ds = _cube(_line)
    del ds.attrs["reference"]
    with pytest.warns(UserWarning, match="re-referenced"):
        dxr.fit_series(ds)
