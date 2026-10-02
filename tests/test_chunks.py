"""Chunk sizing from a byte target."""

import numpy as np
import pytest

from disp_xr._chunks import FILE_CHUNK, clamp_chunks, resolve_chunks, suggest_chunks
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
    assert resolve_chunks(None, shape, np.float32) == suggest_chunks(shape)
    assert resolve_chunks("maps", shape, np.float32)["time"] == 1
    assert resolve_chunks({"y": 100}, shape, np.float32)["y"] == 100
