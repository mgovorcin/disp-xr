"""Length unit conversion."""

import numpy as np
import pytest
import xarray as xr

import disp_xr as dxr
from disp_xr._history import history
from disp_xr.units import to_units


def test_dataset_converts_lengths_only(stack_dir):
    ds = dxr.open_disp(stack_dir, layers=["displacement", "temporal_coherence"])
    mm = to_units(ds, "millimeters")
    np.testing.assert_allclose(mm.displacement.values, ds.displacement.values * 1000)
    assert mm.displacement.attrs["units"] == "mm"
    assert mm.displacement.attrs["grid_mapping"] == "spatial_ref"
    xr.testing.assert_identical(mm.temporal_coherence, ds.temporal_coherence)
    assert ds.displacement.attrs["units"] == "meters"  # input unchanged


def test_round_trip_and_history():
    da = xr.DataArray([1.5], dims="t", attrs={"units": "m"})
    back = to_units(to_units(da, "cm"), "m")
    assert float(back[0]) == pytest.approx(1.5)
    assert [h["op"] for h in history(back)] == ["to_units", "to_units"]


def test_is_lazy(stack_dir):
    ds = dxr.open_disp(stack_dir, layers=["displacement"])
    assert to_units(ds, "mm").displacement.chunks is not None


def test_unknown_unit_raises():
    with pytest.raises(ValueError, match="Unsupported unit 'inch'"):
        to_units(xr.DataArray([1.0], dims="t"), "inch")
