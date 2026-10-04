"""DISP against GNSS with geepers metrics (plan task T32)."""

import numpy as np
import pytest

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
