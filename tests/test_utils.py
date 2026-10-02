"""Characterization tests for ``disp_xr.utils`` (plan task T2.6)."""

import random

import numpy as np
import pytest
import xarray as xr

from disp_xr import utils


@pytest.fixture
def ps_grid():
    pct_mask = np.zeros((8, 8))
    pct_ps = np.zeros((8, 8))
    for iy, ix in [(1, 1), (2, 5), (6, 3)]:
        pct_ps[iy, ix] = 100
    return pct_mask, pct_ps


def test_find_reference_point_picks_a_stable_pixel(ps_grid):
    random.seed(0)
    assert utils.find_reference_point(*ps_grid) in {(1, 1), (2, 5), (6, 3)}


def test_find_reference_point_excludes_phase_jumps(ps_grid):
    jumps = np.zeros((8, 8))
    jumps[1, 1] = jumps[2, 5] = 3
    random.seed(0)
    assert utils.find_reference_point(*ps_grid, phase_jumps=jumps) == (6, 3)


@pytest.mark.filterwarnings("ignore:All-NaN slice encountered")
def test_find_reference_point_raises_without_candidates(ps_grid):
    pct_mask, pct_ps = ps_grid
    with pytest.raises(ValueError, match="No valid reference point found"):
        utils.find_reference_point(np.full((8, 8), 100.0), pct_ps)


def test_find_reference_point_is_deterministic(ps_grid):
    """Regression: bug 7, the pick was random."""
    random.seed(0)
    first = utils.find_reference_point(*ps_grid)
    random.seed(5)
    assert utils.find_reference_point(*ps_grid) == first


def test_get_burst_ids_from_runconfig(two_ministacks):
    assert utils.get_burst_ids(two_ministacks.paths[0]) == [
        "T042_088905_IW1",
        "T042_088906_IW1",
    ]


def test_latlon_utm_round_trip():
    x, y = utils.latlon_to_utm(37.0, -122.0, 32610)
    lat, lon = utils.utm_to_latlon(x, y, 32610)
    assert (lat, lon) == pytest.approx((37.0, -122.0))


def test_get_extent_from_geotransform(two_ministacks):
    ds = xr.open_dataset(two_ministacks.paths[0], engine="h5netcdf")
    x0, x1, y1, y0 = utils.get_extent(ds)
    assert (x0, y0) == (two_ministacks.x[0] - 15, two_ministacks.y[0] + 15)
    assert x1 - x0 == 64 * 30 and y0 - y1 == 64 * 30


def test_get_chunks_indices_covers_every_chunk(two_ministacks):
    ds = xr.open_dataset(two_ministacks.paths[0], engine="h5netcdf").chunk(
        {"time": -1, "y": 32, "x": 32}
    )
    assert len(utils.get_chunks_indices(ds)) == 4


def test_get_extent_is_deprecated_in_favor_of_rio_bounds(two_ministacks):
    import rioxarray  # noqa: F401

    from disp_xr.geometry import get_extent

    ds = xr.open_dataset(
        two_ministacks.paths[0], engine="h5netcdf", decode_coords="all"
    )
    with pytest.warns(DeprecationWarning, match="ds.rio.bounds"):
        x0, x1, y1, y0 = get_extent(ds)
    assert ds.rio.bounds() == (x0, y1, x1, y0)
