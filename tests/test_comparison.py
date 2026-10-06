"""DISP against GNSS with geepers metrics (plan task T32)."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import disp_xr as dxr

pytest.importorskip("geepers.validation")  # geepers >= 0.4
from fixtures.fake_gnss import stations_from_disp, vertical_los  # noqa: E402
from geepers.analysis import epoch_rmse, pairwise_differential_rmse  # noqa: E402

from disp_xr.residuals import _merged_tables, compare  # noqa: E402

PIXELS = [(8, 10), (20, 40), (35, 15), (50, 50), (12, 55)]


@pytest.fixture
def frame(stack_dir):
    return dxr.stack_disp(dxr.open_disp(stack_dir)).disp.rereference("median")


@pytest.fixture
def gps(frame, tmp_path):
    src = stations_from_disp(frame, PIXELS, tmp_path / "cache", extra_rate={3: 0.05})
    return dxr.gnss.load(frame, source=src)


def test_compare_at_stations(frame, gps):
    result = compare(frame, gps, vertical_los(frame))
    series = result.series
    assert series.residual.dims == ("station", "time")
    good = series.residual.drop_sel(station="S003")
    centered = good - good.mean("time")
    assert float(np.abs(centered).max()) < 0.005  # GNSS follows DISP, 1 mm noise
    stations = result.stations
    assert list(stations.index[stations.outlier]) == ["S003"]
    assert stations.loc["S003", "velocity_residual"] == pytest.approx(-0.05, abs=0.01)
    assert len(result.pairs) == 10  # 5 stations, every pair
    assert not result.profile.empty and not result.semivariogram.empty
    assert "flag" in result.epochs.columns
    assert set(result.summary) >= {"bias", "mad", "rmse", "r2", "n"}


def test_datum_offset_is_removed(frame, tmp_path):
    # GNSS 1 cm/yr off DISP everywhere (DISP relative, GNSS absolute), S003
    # another 5 cm/yr off.
    rates = dict.fromkeys(range(len(PIXELS)), 0.01) | {3: 0.06}
    src = stations_from_disp(frame, PIXELS, tmp_path / "cache", extra_rate=rates)
    result = compare(frame, dxr.gnss.load(frame, source=src), vertical_los(frame))
    stations = result.stations
    assert result.summary["datum_offset"] == pytest.approx(-0.01, abs=0.002)
    assert stations.loc["S003", "relative_residual"] == pytest.approx(-0.05, abs=0.01)
    assert list(stations.index[stations.outlier]) == ["S003"]


def test_disp_noise_does_not_flag_good_stations(frame, tmp_path):
    # Stations that follow DISP, DISP noise of 1 cm per epoch, GNSS sigmas of
    # 1 mm: with the GNSS sigma alone in chi2 every station was flagged
    # (F08882, plan task T34.0).
    src = stations_from_disp(frame, PIXELS, tmp_path / "cache")
    gps = dxr.gnss.load(frame, source=src)
    rng = np.random.default_rng(1)
    noisy = frame.copy()
    noisy["displacement"] = frame.displacement + rng.normal(
        0, 0.01, frame.displacement.shape
    ).astype("float32")
    result = compare(noisy, gps, vertical_los(frame))
    assert result.summary["disp_sigma"] == pytest.approx(0.01, rel=0.3)
    # chi2 only: 11 epochs of 1 cm noise also make the velocities noisy
    assert (np.sqrt(result.stations.chi2) < 3).all()
    gnss_only = compare(noisy, gps, vertical_los(frame), disp_sigma=0.0)
    assert gnss_only.summary["disp_sigma"] == 0.0
    assert (np.sqrt(gnss_only.stations.chi2) > 3).all()
    assert gnss_only.stations.outlier.all()


def test_metrics_are_geepers(frame, gps):
    result = compare(frame, gps, vertical_los(frame))
    tables = _merged_tables(result.series)
    coords = {
        s: (float(lo), float(la))
        for s, lo, la in zip(gps.station.values, gps.lon.values, gps.lat.values)
    }
    expected = pairwise_differential_rmse(tables, coords)
    np.testing.assert_allclose(result.pairs.rmse, expected.rmse)
    np.testing.assert_allclose(result.epochs.rmse, epoch_rmse(tables).rmse)


def test_requirement_line(frame, gps):
    result = compare(
        frame, gps, vertical_los(frame), requirement=lambda d: 0.003 + 0 * d
    )
    assert {"requirement", "fraction_passing"} <= set(result.profile.columns)


def test_opera_requirements_by_default(frame, gps):
    from disp_xr.residuals import DISP_S1_DISPLACEMENT, DISP_S1_VELOCITY

    result = compare(frame, gps, vertical_los(frame))
    profile = result.profile
    # The displacement curve 3 (1 + sqrt(L)) mm, in meters like the data.
    centers = profile["requirement"].to_numpy()
    assert np.all((centers >= 0.003) & (centers < 0.03))
    pairs = result.velocity_pairs
    assert (
        len(pairs)
        == len(result.stations.dropna(subset=["velocity_residual"]))
        * (len(result.stations.dropna(subset=["velocity_residual"])) - 1)
        // 2
    )
    np.testing.assert_allclose(pairs["requirement"], 0.002)
    expected = (pairs["difference"].abs() <= 0.002)[pairs["applies"]].mean()
    assert result.summary["velocity_fraction_passing"] == pytest.approx(expected)
    assert result.summary["requirement"] == DISP_S1_DISPLACEMENT.name
    assert result.summary["velocity_requirement"] == DISP_S1_VELOCITY.name
    none = compare(
        frame, gps, vertical_los(frame), requirement=None, velocity_requirement=None
    )
    assert none.velocity_pairs is None
    assert "fraction_passing" not in none.profile.columns


def test_requirement_curve():
    from disp_xr.residuals import DISP_S1_DISPLACEMENT, Requirement

    np.testing.assert_allclose(
        DISP_S1_DISPLACEMENT([0.0, 1.0, 49.0]), [0.003, 0.006, 0.024]
    )
    flat = Requirement("flat", 2.0, units="mm/yr")
    np.testing.assert_array_equal(
        flat.applies([0.05, 0.1, 50.0, 51.0]), [False, True, True, False]
    )


def test_compare_on_cells(frame, gps):
    cells = dxr.to_points(dxr.downsample(frame, factor=3))
    on_cells = compare(cells, gps, vertical_los(frame))
    assert on_cells.series.insar.notnull().any()
    tree = dxr.quadtree(frame, threshold=1.0, min_level=1, max_level=3)
    on_leaves = compare(tree, gps, vertical_los(frame))
    assert on_leaves.series.sizes["station"] == 5
    assert list(on_leaves.stations.index[on_leaves.stations.outlier]) == ["S003"]


def test_plots(frame, gps):
    import matplotlib

    matplotlib.use("Agg")
    result = compare(frame, gps, vertical_los(frame))
    assert result.plot_velocities() is not None
    assert result.plot_semivariogram() is not None


def _residuals(n_station=5, nt=90, seed=3):
    rng = np.random.default_rng(seed)
    times = pd.date_range("2020-01-01", periods=nt, freq="12D")
    years = np.arange(nt) * 12 / 365.25
    common = 0.004 * np.sin(2 * np.pi * 2 * years)  # semiannual, all stations
    values = common[None] + 0.001 * rng.standard_normal((n_station, nt))
    return (
        xr.DataArray(
            values,
            dims=("station", "time"),
            coords={"station": [f"S{k}" for k in range(n_station)], "time": times},
        ),
        common,
    )


def test_spectra_find_the_semiannual_signal():
    from disp_xr.residuals import residual_spectra

    resid, _ = _residuals()
    power = residual_spectra(resid)
    peak = power.frequency[int(power.mean("station").argmax("frequency"))]
    assert float(peak) == pytest.approx(2.0, abs=0.1)
    assert power.dims == ("station", "frequency")


def test_common_mode_recovers_the_shared_signal():
    from disp_xr.residuals import common_mode

    resid, common = _residuals()
    out = common_mode(resid)
    assert float(out.explained_variance[0]) > 0.8
    r = np.corrcoef(out.common_mode.isel(component=0), common)[0, 1]
    assert abs(r) > 0.98
    assert float(np.abs(out.cleaned).max()) < 0.006
