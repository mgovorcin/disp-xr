"""Icechunk export: commits, appends, tags and provenance (plan task T14.2b)."""

import json

import numpy as np
import pytest
import xarray as xr

import disp_xr as dxr

icechunk = pytest.importorskip("icechunk")


@pytest.fixture
def ds(stack_dir):
    return dxr.chain(dxr.open_disp(stack_dir))


@pytest.fixture
def repo():
    return icechunk.Repository.create(icechunk.in_memory_storage())


def _equal_values(a, b):
    xr.testing.assert_equal(a.reset_coords(drop=True), b.reset_coords(drop=True))


def _assert_same(back, ds):
    for name in ds.data_vars:
        _equal_values(back[name], ds[name])
        assert back[name].dtype == ds[name].dtype
    np.testing.assert_array_equal(back.reference_time, ds.reference_time)
    assert back.rio.crs == ds.rio.crs
    assert back.rio.transform() == ds.rio.transform()


def test_round_trip(ds, repo):
    snapshot = dxr.to_icechunk(ds, repo)
    back = dxr.open_icechunk(repo)
    _assert_same(back, ds)
    assert back.attrs["source_snapshot_id"] == snapshot
    assert back.displacement.encoding["shards"] is not None


def test_local_path(ds, tmp_path):
    path = tmp_path / "repo"
    dxr.to_icechunk(ds, path)
    _assert_same(dxr.open_icechunk(path), ds)


def test_commit_metadata(ds, repo):
    snapshot = dxr.to_icechunk(ds, repo, message="first stack")
    info = repo.lookup_snapshot(snapshot)
    assert info.message == "first stack"
    meta = info.metadata
    assert meta["disp_xr_version"] == dxr.__version__
    assert len(meta["input_granules"]) == 10  # granules, not epochs
    assert meta["frame_id"] == ds.attrs["frame_id"]
    assert meta["product"] == "DISP-S1"
    assert json.loads(meta["history"].splitlines()[-1])["op"] == "chain"


def test_refuses_existing_data(ds, repo):
    first = dxr.to_icechunk(ds.isel(time=slice(0, 5)), repo)
    with pytest.raises(FileExistsError):
        dxr.to_icechunk(ds, repo)
    dxr.to_icechunk(ds, repo, mode="w")
    assert dxr.open_icechunk(repo).sizes["time"] == ds.sizes["time"]
    # The replaced snapshot stays readable.
    assert dxr.open_icechunk(repo, snapshot_id=first).sizes["time"] == 5


def test_append_one_ministack_per_commit(ds, repo):
    first, second = ds.isel(time=slice(0, 5)), ds.isel(time=slice(5, None))
    s1 = dxr.to_icechunk(first, repo, append_dim="time")
    s2 = dxr.to_icechunk(second, repo, append_dim="time")
    back = dxr.open_icechunk(repo)
    _assert_same(back, ds)
    # The first write fixes the layout: one ministack per time chunk.
    assert back.displacement.encoding["chunks"][0] == 5
    assert back.displacement.encoding["shards"][0] == 5
    # Each commit keeps its own state.
    _assert_same(dxr.open_icechunk(repo, snapshot_id=s1), first)
    assert repo.lookup_snapshot(s2).message.startswith("Append 6 epochs")
    assert len(json.loads(back.attrs["input_granules"])) == 10


def test_append_uneven_ministacks(ds, repo):
    dxr.to_icechunk(ds.isel(time=slice(0, 3)), repo, append_dim="time")
    dxr.to_icechunk(ds.isel(time=slice(3, 8)), repo, append_dim="time")
    dxr.to_icechunk(ds.isel(time=slice(8, None)), repo, append_dim="time")
    _assert_same(dxr.open_icechunk(repo), ds)


def test_append_checks_grid(ds, repo):
    dxr.to_icechunk(ds.isel(time=slice(0, 5)), repo, append_dim="time")
    later = ds.isel(time=slice(5, None))
    with pytest.raises(ValueError, match="x coordinate"):
        dxr.to_icechunk(later.assign_coords(x=later.x + 30), repo, append_dim="time")
    with pytest.raises(ValueError, match="variables differ"):
        dxr.to_icechunk(later.drop_vars("water_mask"), repo, append_dim="time")
    with pytest.raises(ValueError, match="after the store's last epoch"):
        dxr.to_icechunk(ds.isel(time=slice(3, None)), repo, append_dim="time")
    other = "float64" if ds.displacement.dtype == "float32" else "float32"
    with pytest.raises(ValueError, match=f"displacement is {other}"):
        dxr.to_icechunk(
            later.assign(displacement=later.displacement.astype(other)),
            repo,
            append_dim="time",
        )
    # Refused appends leave the store unchanged.
    assert dxr.open_icechunk(repo).sizes["time"] == 5


def test_timeseries_preset_cannot_append(ds, repo):
    with pytest.raises(ValueError, match="timeseries"):
        dxr.to_icechunk(ds, repo, append_dim="time", preset="timeseries")
    with pytest.raises(ValueError, match="'time'"):
        dxr.to_icechunk(ds, repo, append_dim="x")


def test_tag_and_branch(ds, repo):
    s1 = dxr.to_icechunk(ds.isel(time=slice(0, 5)), repo, tag="calval-v1")
    assert repo.lookup_tag("calval-v1") == s1
    s2 = dxr.to_icechunk(ds, repo, branch="variant", mode="w")
    assert dxr.open_icechunk(repo, tag="calval-v1").sizes["time"] == 5
    assert dxr.open_icechunk(repo, branch="variant").sizes["time"] == ds.sizes["time"]
    assert dxr.open_icechunk(repo).attrs["source_snapshot_id"] == s1
    assert dxr.open_icechunk(repo, branch="variant").attrs["source_snapshot_id"] == s2
    with pytest.raises(ValueError, match="at most one"):
        dxr.open_icechunk(repo, branch="main", tag="calval-v1")


def test_snapshot_id_reaches_downstream_provenance(ds, repo, tmp_path):
    snapshot = dxr.to_icechunk(ds, repo)
    path = tmp_path / "out.zarr"
    dxr.to_zarr(dxr.open_icechunk(repo), path)
    assert xr.open_zarr(path).attrs["source_snapshot_id"] == snapshot
