"""Chunk sizing from a byte target."""

import numpy as np
import pytest
import xarray as xr

import disp_xr  # noqa: F401  (registers .disp)
from disp_xr._chunks import (
    FILE_CHUNK,
    clamp_chunks,
    rechunk,
    resolve_chunks,
    suggest_chunks,
)
from disp_xr.config import config


@pytest.mark.parametrize("nt", [50, 300, 1000])
def test_timeseries_chunk_stays_near_target(nt):
    c = suggest_chunks((nt, 7000, 9000), "float32", "timeseries")
    assert c["time"] == nt
    nbytes = nt * c["y"] * c["x"] * 4
    assert nbytes <= config.get("chunks.target_mb") * 1e6
    assert c["y"] == c["x"]


def test_spatial_chunk_aligns_to_file_chunks_when_large():
    c = suggest_chunks((20, 7000, 9000), "float32", "timeseries")
    assert c["y"] % FILE_CHUNK == 0


def test_float64_gets_smaller_chunks():
    c32 = suggest_chunks((300, 7000, 9000), "float32")
    c64 = suggest_chunks((300, 7000, 9000), "float64")
    assert c64["y"] < c32["y"]


def test_maps_layout_is_one_epoch():
    assert suggest_chunks((300, 7000, 9000), "float32", "maps")["time"] == 1


def test_small_stack_is_clamped():
    assert suggest_chunks((10, 64, 64)) == {"time": 10, "y": 64, "x": 64}


def test_target_from_config():
    with config.set({"chunks.target_mb": 10}):
        small = suggest_chunks((300, 7000, 9000))
    assert small["y"] < suggest_chunks((300, 7000, 9000))["y"]


def test_unknown_layout_raises():
    with pytest.raises(ValueError, match="layout must be"):
        suggest_chunks((1, 1, 1), layout="cube")


def test_clamp_and_minus_one():
    assert clamp_chunks({"time": -1, "y": 500}, (10, 64, 64)) == {
        "time": 10,
        "y": 64,
        "x": 64,
    }


def test_resolve_accepts_preset_dict_or_none():
    shape = (300, 7000, 9000)
    assert resolve_chunks(None, shape, np.float32) == suggest_chunks(
        shape, layout="native"
    )
    assert resolve_chunks("maps", shape, np.float32)["time"] == 1
    assert resolve_chunks({"y": 100}, shape, np.float32)["y"] == 100


def test_native_preset_is_the_file_layout():
    from disp_xr.config import config

    block = config.get("chunks.read_block")
    assert suggest_chunks((300, 7000, 9000), layout="native") == {
        "time": 1,
        "y": block,
        "x": block,
    }


def test_rechunk_and_time_contiguous(stack_dir):
    import disp_xr as dxr
    from disp_xr._chunks import time_contiguous

    ds = dxr.open_disp(stack_dir)
    assert ds.displacement.chunks[0] == (1,) * 10
    ts = dxr.rechunk(ds, "timeseries")
    assert ts.displacement.chunks[0] == (10,)
    assert ds.disp.rechunk({"time": 5}).displacement.chunks[0] == (5, 5)
    assert time_contiguous(ds.displacement).chunks[0] == (10,)
    whole = ts.displacement
    assert time_contiguous(whole) is whole  # already one time chunk
    assert time_contiguous(ds.displacement.load()).chunks is None  # not dask


def test_time_contiguous_merges_within_blocks():
    """Time is merged inside the input's blocks, halved to fit; no shuffle."""
    import dask
    import dask.array as darr
    import numpy as np
    import xarray as xr

    from disp_xr._chunks import time_contiguous

    data = darr.zeros((389, 2100, 3000), dtype="float32", chunks=(1, 1024, 1024))
    da = xr.DataArray(data, dims=("time", "y", "x"))
    with dask.config.set({"array.rechunk.method": "p2p"}):  # as on a cluster
        out = time_contiguous(da)
    ny, nx = out.data.chunksize[1:]
    assert out.data.chunksize[0] == 389
    assert 1024 % ny == 0 and 1024 % nx == 0
    assert 389 * 4 * ny * nx <= 100e6
    assert not any("p2p" in name for name in out.data.dask.layers)

    irregular = xr.DataArray(
        darr.zeros((4, 100, 100), chunks=((1,) * 4, (30, 70), (50, 50))),
        dims=("time", "y", "x"),
    )
    assert time_contiguous(irregular).chunks[0] == (4,)
    np.testing.assert_array_equal(time_contiguous(irregular).values, 0)


def test_bounded_threads():
    import dask

    from disp_xr.config import bounded_threads, config

    @bounded_threads
    def workers():
        return dask.config.get("num_workers", None)

    assert workers() == config.get("num_workers")
    with config.set({"num_workers": 4}):
        assert workers() == 4
    with config.set({"num_workers": None}):
        assert workers() is None
    with dask.config.set(num_workers=7):  # the user's own setting wins
        assert workers() == 7


def test_rechunk_without_a_layout_is_deprecated():
    da = xr.DataArray(np.zeros((4, 8, 8)), dims=("time", "y", "x")).chunk(1)
    with pytest.warns(DeprecationWarning, match="rechunk without a layout"):
        out = rechunk(da)
    assert out.chunks == rechunk(da, "timeseries").chunks
    ds = da.to_dataset(name="displacement")
    with pytest.warns(DeprecationWarning, match="rechunk without a layout"):
        ds.disp.rechunk()
