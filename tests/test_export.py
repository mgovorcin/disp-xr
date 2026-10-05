"""Export: Zarr, NetCDF, GeoTIFF and provenance (plan task T14)."""

import json

import numpy as np
import pytest
import rasterio
import xarray as xr
import zarr

import disp_xr as dxr
from disp_xr.export.zarr import chunk_scheme, round_mantissa, zarr_encoding


@pytest.fixture
def ds(stack_dir):
    return dxr.stack_disp(dxr.open_disp(stack_dir))


def _equal_values(a, b):
    xr.testing.assert_equal(a.reset_coords(drop=True), b.reset_coords(drop=True))


# --- chunk scheme ----------------------------------------------------------------


@pytest.mark.parametrize("preset", ["timeseries", "maps", "ministack"])
def test_shards_are_whole_multiples_of_chunks(preset):
    import dask.array as darr

    # Lazy: only the sizes matter.
    big = xr.Dataset(
        {"d": (("time", "y", "x"), darr.zeros((300, 7000, 9000), dtype="f4"))}
    )
    if preset == "ministack":
        big = big.isel(time=slice(0, 15))
    chunks, shards = chunk_scheme(big, preset)
    if preset == "ministack":
        assert chunks == {"time": 15, "y": 512, "x": 512}
        assert shards == {"time": 15, "y": 1024, "x": 1024}
    for dim in ("y", "x"):
        assert shards[dim] % chunks[dim] == 0
    if preset == "timeseries":
        assert chunks == {"time": 300, "y": 64, "x": 64}
        assert shards == {"time": 300, "y": 256, "x": 256}
    elif preset == "maps":
        assert chunks == {"time": 1, "y": 1024, "x": 1024}
        assert shards["y"] == 5120  # 4864 rounded to five 1024 chunks
    inner_mb = np.prod(list(chunks.values())) * 4 / 1e6
    assert 4 <= inner_mb <= 20


def test_small_array_is_one_chunk(ds):
    chunks, shards = chunk_scheme(ds, "timeseries")
    assert chunks["y"] == shards["y"] == 64 and chunks["time"] == ds.sizes["time"]


def test_round_mantissa():
    z = np.array([1.2345678, np.nan, -3.0], "float32")
    out = round_mantissa(z, 7)
    assert out[0] == 1.234375 and np.isnan(out[1]) and out[2] == -3.0
    assert z[0] == np.float32(1.2345678)  # input untouched
    np.testing.assert_array_equal(round_mantissa(z, 23), z)
    with pytest.raises(TypeError):
        round_mantissa(np.arange(3), 4)


# --- Zarr --------------------------------------------------------------------------


@pytest.mark.parametrize("preset", ["timeseries", "maps"])
def test_zarr_round_trip(ds, tmp_path, preset):
    path = tmp_path / "out.zarr"
    dxr.to_zarr(ds, path, preset=preset)
    back = xr.open_zarr(path, mask_and_scale=False, decode_coords="all")
    for name in ds.data_vars:
        _equal_values(back[name], ds[name])
        assert back[name].dtype == ds[name].dtype
    np.testing.assert_array_equal(back.reference_time, ds.reference_time)
    assert back.rio.crs == ds.rio.crs
    assert back.rio.transform() == ds.rio.transform()


def test_zarr_layout_and_fill_values(ds, tmp_path):
    path = tmp_path / "out.zarr"
    dxr.to_zarr(ds, path)
    group = zarr.open_group(path, mode="r")
    disp = group["displacement"]
    assert disp.shards is not None and disp.chunks[0] == ds.sizes["time"]
    assert np.isnan(disp.metadata.fill_value)
    mask = group["recommended_mask"]
    assert mask.dtype == np.uint8 and mask.metadata.fill_value == 255
    assert mask.attrs["_FillValue"] == 255
    assert mask.attrs["flag_meanings"] == "bad good"
    assert group["x"].compressors == ()
    attrs = group.attrs.asdict()
    assert attrs["proj:code"] == f"EPSG:{ds.rio.crs.to_epsg()}"
    assert attrs["spatial:transform"] == list(ds.rio.transform())[:6]


def test_zarr_masked_read_and_cf(ds, tmp_path):
    import cf_xarray  # noqa: F401

    path = tmp_path / "out.zarr"
    dxr.to_zarr(ds, path)
    back = xr.open_zarr(path, decode_coords="all")
    assert np.isnan(back.recommended_mask.isel(time=0)).all()  # zero epoch nodata
    assert back.displacement.encoding["grid_mapping"] == "spatial_ref"
    assert "spatial_ref" in back.cf.grid_mapping_names.get("transverse_mercator", [])


def test_zarr_provenance(ds, tmp_path):
    path = tmp_path / "out.zarr"
    dxr.to_zarr(ds.disp.rereference("point", row=40, col=40), path)
    attrs = xr.open_zarr(path).attrs
    assert attrs["disp_xr_version"] == dxr.__version__
    assert set(json.loads(attrs["software_versions"])) >= {"xarray", "zarr", "dask"}
    assert len(json.loads(attrs["input_granules"])) == 10
    ops = [json.loads(line)["op"] for line in attrs["history"].splitlines()]
    assert ops[-2:] == ["stack_disp", "rereference"]


def test_zarr_refuses_existing_store(ds, tmp_path):
    path = tmp_path / "out.zarr"
    dxr.to_zarr(ds, path)
    with pytest.raises(Exception, match="(?i)exist|contains"):
        dxr.to_zarr(ds, path)
    dxr.to_zarr(ds, path, mode="w")


def test_zarr_keep_bits_is_opt_in(ds, tmp_path):
    exact, rounded = tmp_path / "exact.zarr", tmp_path / "rounded.zarr"
    dxr.to_zarr(ds, exact)
    dxr.to_zarr(ds, rounded, keep_bits=10)
    a = xr.open_zarr(exact).displacement.values
    b = xr.open_zarr(rounded).displacement.values
    np.testing.assert_array_equal(a, ds.displacement.values)
    assert not np.array_equal(a, b)
    np.testing.assert_allclose(a, b, rtol=2**-10)
    assert xr.open_zarr(rounded).attrs["mantissa_bits_kept"] == 10


def test_source_encoding_is_dropped(stack_dir, tmp_path):
    raw = dxr.open_disp(stack_dir)
    assert raw.temporal_coherence.encoding["zlib"]  # gzip from the granules
    dxr.to_zarr(raw, tmp_path / "raw.zarr")
    codecs = zarr.open_group(tmp_path / "raw.zarr")["temporal_coherence"].compressors
    assert [type(c).__name__ for c in codecs] == ["ZstdCodec"]
    assert "chunks" in zarr_encoding(raw)["temporal_coherence"]


# --- NetCDF ----------------------------------------------------------------------


def test_netcdf_round_trip(ds, tmp_path):
    path = tmp_path / "out.nc"
    dxr.to_netcdf(ds, path)
    back = xr.open_dataset(
        path, engine="h5netcdf", mask_and_scale=False, decode_coords="all"
    )
    for name in ds.data_vars:
        _equal_values(back[name], ds[name])
    assert back.recommended_mask.attrs["_FillValue"] == 255
    assert back.rio.crs == ds.rio.crs
    assert back.attrs["disp_xr_version"] == dxr.__version__
    assert back.displacement.encoding["chunksizes"] == (ds.sizes["time"], 64, 64)
    assert back.displacement.encoding["zlib"]


# --- GeoTIFF ---------------------------------------------------------------------


def test_geotiff_round_trip(ds, tmp_path):
    layer = ds.displacement.isel(time=-1, drop=True)
    (path,) = dxr.to_geotiff(layer, tmp_path / "last.tif")
    with rasterio.open(path) as src:
        assert src.driver == "GTiff" and src.overviews(1) is not None
        assert src.crs == ds.rio.crs and src.transform == ds.rio.transform()
        assert src.tags()["DISP_XR_VERSION"] == dxr.__version__
        np.testing.assert_array_equal(src.read(1), layer.values)
        assert np.isnan(src.nodata)
        assert src.profile["compress"] == "deflate" and src.profile["tiled"]
    assert not list(tmp_path.glob(".*staging*"))


def test_geotiff_integer_nodata(ds, tmp_path):
    (path,) = dxr.to_geotiff(ds.water_mask.isel(time=1, drop=True), tmp_path / "w.tif")
    with rasterio.open(path) as src:
        assert src.nodata == 255 and src.dtypes[0] == "uint8"


def test_geotiff_per_epoch(ds, tmp_path):
    paths = dxr.to_geotiff(ds.displacement, tmp_path / "disp.tif", per_epoch=True)
    assert len(paths) == ds.sizes["time"]
    assert paths[0].name == f"disp_{str(ds.time.values[0])[:10].replace('-', '')}.tif"


def test_geotiff_checks(ds, tmp_path):
    with pytest.raises(ValueError, match=r"needs \(y, x\)"):
        dxr.to_geotiff(ds.displacement, tmp_path / "x.tif")
    with (
        dxr.config.set({"max_load_bytes": 10}),
        pytest.raises(ValueError, match="max_load_bytes"),
    ):
        dxr.to_geotiff(ds.displacement.isel(time=0), tmp_path / "x.tif")


def test_geotiff_from_dataset_is_a_cog_with_provenance(tmp_path):
    from disp_xr.tutorial import make_disp_stack

    make_disp_stack(tmp_path / "g", shape=(600, 600), epochs_per_ministack=2)
    ds = dxr.stack_disp(dxr.open_disp(tmp_path / "g"))
    (path,) = dxr.to_geotiff(ds.isel(time=-1), tmp_path / "a.tif", layer="displacement")
    with rasterio.open(path) as src:
        tags = src.tags()
        assert src.overviews(1)  # above 512 pixels GDAL adds overviews
        assert src.profile["tiled"]
        assert src.tags(ns="IMAGE_STRUCTURE").get("LAYOUT") == "COG"
    assert tags["DISP_XR_VERSION"] == dxr.__version__
    assert len(json.loads(tags["INPUT_GRANULES"])) == 4
    assert json.loads(tags["HISTORY"].splitlines()[-1])["op"] == "stack_disp"
    assert set(json.loads(tags["SOFTWARE_VERSIONS"])) >= {"xarray", "zarr"}
    with pytest.raises(ValueError, match="No layer"):
        dxr.to_geotiff(ds, tmp_path / "b.tif", layer="nope")


def test_provenance_of_a_reopened_store(ds, tmp_path):
    # Stores hold input_granules as a JSON string; re-exporting must not split it.
    dxr.to_zarr(ds, tmp_path / "a.zarr")
    dxr.to_zarr(xr.open_zarr(tmp_path / "a.zarr"), tmp_path / "b.zarr")
    granules = json.loads(xr.open_zarr(tmp_path / "b.zarr").attrs["input_granules"])
    assert len(granules) == 10 and granules[0].startswith("OPERA_L3_DISP")


def test_maps_store_feeds_per_pixel_and_cell_functions(ds, tmp_path):
    # The store keeps one chunk per epoch for the data but one chunk for the
    # per-epoch coordinate reference_time; functions that align chunks must
    # use the data's chunks, not DataArray.chunksizes (which raises here).
    path = tmp_path / "maps.zarr"
    dxr.to_zarr(ds.disp.rereference("median"), path, preset="maps")
    back = xr.open_zarr(path)
    assert len(back.displacement.chunks[0]) > 1
    assert len(back.reference_time.chunks[0]) == 1
    vel = dxr.fit_velocity(back, method="ols")
    assert np.isfinite(vel.velocity.values).any()
    coarse = dxr.downsample(back, factor=5, layers=["displacement"])
    assert np.isfinite(coarse.displacement.values).any()
    tree = dxr.quadtree(back, n_points=20, min_level=1, max_level=4)
    assert tree.points.sizes["point"] > 0


def test_map_outputs_carry_grid_mapping(ds):
    # Readers that look up the CRS per variable (GDAL, xpublish-tiles) need
    # grid_mapping on every (y, x) variable.
    chained = ds.disp.rereference("median")
    outputs = {
        "fit_velocity": dxr.fit_velocity(chained, method="ols"),
        "fit_trajectory": dxr.fit_trajectory(chained),
        "downsample": dxr.downsample(chained, factor=8),
    }
    for name, out in outputs.items():
        for var, da in out.data_vars.items():
            if {"y", "x"} <= set(da.dims):
                assert da.attrs.get("grid_mapping") == "spatial_ref", (name, var)
