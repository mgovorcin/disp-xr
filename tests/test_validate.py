"""validate_disp and the grid helper."""

import numpy as np
import pytest
import xarray as xr

import disp_xr as dxr
from disp_xr.geometry import grid
from disp_xr.validate import DispValidationError, validate_disp


@pytest.fixture
def ds(stack_dir):
    return dxr.open_disp(stack_dir, layers=["displacement"])


def test_grid(two_ministacks, ds):
    g = grid(ds)
    assert g.crs.to_epsg() == 32610
    assert g.shape == (64, 64)
    assert (g.transform.c, g.transform.a) == (two_ministacks.x[0] - 15, 30.0)


def test_valid_stack_passes(ds):
    validate_disp(ds, variables=["displacement"], require_reference_time=True)


def test_all_problems_are_listed(ds):
    bad = ds.drop_vars(["spatial_ref", "reference_time"]).isel(time=0)
    with pytest.raises(DispValidationError) as err:
        validate_disp(bad, variables=["coherence"], require_reference_time=True)
    message = str(err.value)
    assert "missing dims ['time']" in message
    assert "missing variables ['coherence']" in message
    assert "missing reference_time" in message


def test_missing_crs(ds):
    with pytest.raises(DispValidationError, match="no CRS"):
        validate_disp(ds.drop_vars("spatial_ref"))


def test_unsorted_coordinate(ds):
    shuffled = ds.isel(x=np.r_[1, 0, np.arange(2, 64)])
    with pytest.raises(DispValidationError, match="x coordinate is not monotonic"):
        validate_disp(shuffled)


def test_per_pixel_warns_on_split_time(stack_dir):
    maps = dxr.open_disp(stack_dir, layers=["displacement"], chunks="maps")
    with pytest.warns(UserWarning, match="time is split across chunks"):
        validate_disp(maps, per_pixel=True)


def test_dataarray(ds):
    validate_disp(ds.displacement, variables=["displacement"])
    with pytest.raises(DispValidationError):
        validate_disp(xr.DataArray([1.0], dims="t"))
