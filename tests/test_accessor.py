"""The .disp accessor (plan task T9)."""

import subprocess
import sys

import dask
import numpy as np
import pytest
import xarray as xr

import disp_xr as dxr
from disp_xr.validate import DispValidationError


@pytest.fixture
def ds(stack_dir):
    return dxr.open_disp(stack_dir, layers=["displacement", "temporal_coherence"])


def test_chain_matches_functions(ds):
    via_accessor = ds.disp.stack().disp.to_units("mm")
    via_functions = dxr.to_units(dxr.stack_disp(ds), "mm")
    xr.testing.assert_identical(via_accessor, via_functions)


def test_input_is_not_mutated(ds):
    before = ds.copy(deep=True)
    ds.disp.stack().disp.to_units("mm")
    xr.testing.assert_identical(ds, before)
    assert ds.attrs == before.attrs


def test_history_and_crs_survive_the_chain(ds):
    out = ds.disp.stack().disp.to_units("cm")
    assert [h["op"] for h in out.disp.history] == [
        "open_disp",
        "stack_disp",
        "to_units",
    ]
    assert out.disp.grid.crs.to_epsg() == 32610
    assert out.rio.crs.to_epsg() == 32610


def test_chain_is_lazy(ds):
    def fail(*_args, **_kwargs):
        raise AssertionError("the accessor computed")

    with dask.config.set(scheduler=fail):
        out = (
            ds.disp.stack()
            .disp.to_units("mm")
            .disp.rechunk("timeseries")
            .disp.validate(per_pixel=True)
        )
    assert isinstance(out.displacement.data, dask.array.Array)


def test_validate_returns_object_or_raises(ds):
    assert ds.disp.validate() is ds
    with pytest.raises(DispValidationError):
        ds.drop_vars("spatial_ref").disp.validate()


def test_dataarray_accessor(ds):
    da = ds.displacement.disp.to_units("mm")
    np.testing.assert_allclose(da.values, ds.displacement.values * 1000)
    assert da.disp.grid.shape == (64, 64)


def test_registration_overrides_nothing():
    code = "import warnings; warnings.simplefilter('error'); import rioxarray, disp_xr"
    subprocess.run([sys.executable, "-c", code], check=True)
