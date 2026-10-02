"""open_disp: a lazy (time, y, x) dataset of one frame."""

import dask
import numpy as np
import pytest
import xarray as xr

from disp_xr.open import open_disp
from disp_xr.product import MixedProductionError, find_products


def _no_compute(_dsk, _keys, **_kwargs):
    raise AssertionError("open_disp computed a dask graph")


@pytest.fixture
def ds(stack_dir):
    return open_disp(stack_dir)


def test_open_reads_no_pixels(stack_dir):
    with dask.config.set(scheduler=_no_compute):
        ds = open_disp(stack_dir, corrections=True)
    assert all(isinstance(v.data, dask.array.Array) for v in ds.data_vars.values())


def test_dims_and_coords(two_ministacks, ds):
    assert dict(ds.sizes) == {"time": 10, "y": 64, "x": 64}
    np.testing.assert_array_equal(ds.x, two_ministacks.x)
    np.testing.assert_array_equal(ds.y, two_ministacks.y)
    assert list(ds.time.values) == list(two_ministacks.times[1:].values)
    assert list(ds.reference_time.values) == list(two_ministacks.reference_times.values)


def test_crs_and_transform(two_ministacks, ds):
    assert ds.rio.crs.to_epsg() == 32610
    t = ds.rio.transform()
    assert (t.c, t.f, t.a, t.e) == (
        two_ministacks.x[0] - 15,
        two_ministacks.y[0] + 15,
        30.0,
        -30.0,
    )
    assert all(ds[v].attrs["grid_mapping"] == "spatial_ref" for v in ds.data_vars)


def test_displacement_is_raw_granule_values(two_ministacks, ds):
    k = 7
    expected = xr.open_dataset(two_ministacks.paths[k], engine="h5netcdf")
    np.testing.assert_array_equal(
        ds.displacement.isel(time=k).values, expected.displacement.values
    )


def test_integer_layers_keep_dtype(ds):
    assert ds.recommended_mask.dtype == np.uint8
    assert ds.connected_component_labels.dtype == np.uint16
    assert ds.shp_counts.dtype == np.int16
    assert ds.recommended_mask.attrs["_FillValue"] == 255


def test_layer_selection(stack_dir):
    ds = open_disp(stack_dir, layers=["displacement", "temporal_coherence"])
    assert set(ds.data_vars) == {"displacement", "temporal_coherence"}


def test_unknown_layer_raises(stack_dir):
    with pytest.raises(ValueError, match=r"Unknown DISP-S1 layers: \['nope'\]"):
        open_disp(stack_dir, layers=["nope"])


def test_corrections_merge(stack_dir):
    ds = open_disp(stack_dir, layers=["displacement"], corrections=True)
    assert {"ionospheric_delay", "solid_earth_tide"} <= set(ds.data_vars)
    assert float(ds.solid_earth_tide.isel(time=0, y=0, x=0)) == pytest.approx(-0.001)


def test_chunk_presets(stack_dir):
    assert open_disp(stack_dir).displacement.chunks[0] == (10,)
    assert open_disp(stack_dir, chunks="maps").displacement.chunks[0] == (1,) * 10
    custom = open_disp(stack_dir, chunks={"time": -1, "y": 32, "x": 32})
    assert custom.displacement.chunks[1] == (32, 32)


def test_accepts_find_products_table(stack_dir):
    df = find_products(stack_dir)
    assert open_disp(df.iloc[:3]).sizes["time"] == 3


def test_attrs(ds):
    assert ds.attrs["product"] == "DISP-S1"
    assert ds.attrs["frame_id"] == 11116
    assert ds.attrs["product_versions"] == ["1.0"]


def test_refuses_mixed_productions(disp_stacks):
    with pytest.raises(MixedProductionError):
        open_disp(disp_stacks["mixed_dtype"].paths[0].parent)


def test_opens_nisar_sas_output(disp_stacks):
    ds = open_disp(disp_stacks["nisar"].paths[0].parent)
    assert ds.attrs["product"] == "DISP-NISAR"
    assert ds.sizes["time"] == 10


def test_granule_on_another_grid_raises(two_ministacks, tmp_path):
    import shutil

    import h5py

    for p in two_ministacks.paths[:2]:
        shutil.copy(p, tmp_path / p.name)
    with h5py.File(tmp_path / two_ministacks.paths[1].name, "a") as f:
        f["x"][...] = f["x"][...] + 30.0
    with pytest.raises(ValueError, match="cannot align|cannot be aligned|exact"):
        open_disp(tmp_path)
