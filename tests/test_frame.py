"""Frame repositories: virtual granules plus stored chain offsets (disp_xr.frame)."""

import shutil

import numpy as np
import pytest
import xarray as xr

import disp_xr as dxr

icechunk = pytest.importorskip("icechunk")
pytest.importorskip("virtualizarr")

LAYERS = ("displacement", "short_wavelength_displacement")


@pytest.fixture
def three(disp_stacks, tmp_path):
    # A private copy: the virtual frame references these files.
    out = tmp_path / "granules"
    shutil.copytree(disp_stacks["three_ministacks"].paths[0].parent, out)
    return out


def _assert_same(frame: xr.Dataset, reference: xr.Dataset) -> None:
    np.testing.assert_array_equal(frame["time"].values, reference["time"].values)
    for name in LAYERS:
        assert frame[name].dtype == reference[name].dtype
        np.testing.assert_array_equal(frame[name].values, reference[name].values)


def test_open_frame_equals_chain(three, tmp_path):
    repo = tmp_path / "frame"
    dxr.build_virtual_frame(three, repo, max_workers=1)
    snapshot = dxr.add_chain_offsets(repo, tag="v1")
    assert snapshot is not None
    frame = dxr.open_frame(repo, tag="v1")
    _assert_same(frame, dxr.chain(dxr.open_disp(three)))
    assert frame.attrs["source_snapshot_id"] == snapshot
    assert [step["op"] for step in frame.disp.history][-1] == "open_frame"
    flat = dxr.open_frame(repo, zero_epoch=False)
    _assert_same(flat, dxr.chain(dxr.open_disp(three), zero_epoch=False))
    # every epoch is read as itself and one offset: the result stays lazy
    assert frame.displacement.chunks is not None


def test_offsets_extend_with_new_ministacks(three, tmp_path):
    repo = tmp_path / "frame"
    granules = dxr.find_products(three)
    last = granules.reference_time.max()
    dxr.build_virtual_frame(
        granules[granules.reference_time < last], repo, max_workers=1
    )
    dxr.add_chain_offsets(repo)
    assert dxr.add_chain_offsets(repo) is None  # nothing new

    dxr.build_virtual_frame(
        granules[granules.reference_time == last], repo, max_workers=1
    )
    with pytest.raises(ValueError, match="No chain offsets for 1 ministacks"):
        dxr.open_frame(repo)
    snapshot = dxr.add_chain_offsets(repo)
    storage = icechunk.local_filesystem_storage(str(repo))
    info = next(icechunk.Repository.open(storage).ancestry(snapshot_id=snapshot))
    assert len(info.metadata["chain_ministacks"]) == 1  # only the new one
    _assert_same(dxr.open_frame(repo), dxr.chain(dxr.open_disp(three)))


def test_one_group_per_production(disp_stacks, tmp_path):
    """Granules stored at another precision go in their own group; the frame
    reads as open_disp reads mixed granules (cast to float32)."""
    src = tmp_path / "granules"
    shutil.copytree(disp_stacks["mixed_dtype"].paths[0].parent, src)
    granules = dxr.find_products(src)
    repo = tmp_path / "frame"
    for i, ref in enumerate(sorted(granules.reference_time.unique())):
        part = granules[granules.reference_time == ref]
        dxr.build_virtual_frame(part, repo, group=f"production/p{i}", max_workers=1)
    dxr.add_chain_offsets(repo)
    frame = dxr.open_frame(repo)
    with pytest.warns(dxr.product.MixedPrecisionWarning):
        reference = dxr.chain(dxr.open_disp(src))
    _assert_same(frame, reference)
    assert "displacement" in frame.attrs["mixed_precision"]


def test_nan_policy_omit(three, tmp_path):
    repo = tmp_path / "frame"
    dxr.build_virtual_frame(three, repo, max_workers=1)
    dxr.add_chain_offsets(repo, nan_policy="omit")
    frame = dxr.open_frame(repo)
    assert frame.attrs["nan_policy"] == "omit"
    _assert_same(frame, dxr.chain(dxr.open_disp(three), nan_policy="omit"))
    with pytest.raises(ValueError, match="nan_policy='omit'"):
        dxr.add_chain_offsets(repo, nan_policy="propagate")


def test_open_frame_needs_offsets(three, tmp_path):
    repo = tmp_path / "frame"
    dxr.build_virtual_frame(three, repo, max_workers=1)
    with pytest.raises(ValueError, match="no chain offsets"):
        dxr.open_frame(repo)


def test_offsets_written_in_several_shards(three, tmp_path, monkeypatch):
    """Full frames span several shards: dask chunks must match the shards so
    parallel writes never share one (64 x 64 frames fit in one by default)."""
    from disp_xr import frame as frame_module

    monkeypatch.setattr(frame_module, "OFFSET_READ_CHUNK", (1, 16, 16))
    monkeypatch.setattr(frame_module, "OFFSET_SHARD", (1, 32, 32))
    repo = tmp_path / "frame"
    granules = dxr.find_products(three)
    last = granules.reference_time.max()
    dxr.build_virtual_frame(
        granules[granules.reference_time < last], repo, max_workers=1
    )
    dxr.add_chain_offsets(repo)
    dxr.build_virtual_frame(
        granules[granules.reference_time == last], repo, max_workers=1
    )
    dxr.add_chain_offsets(repo)  # the append path, also over several shards
    session = icechunk.Repository.open(
        icechunk.local_filesystem_storage(str(repo))
    ).readonly_session("main")
    stored = xr.open_zarr(session.store, group="chain", consolidated=False)
    assert stored.displacement_offset.encoding["shards"] == (1, 32, 32)
    # reference dates fall at any time of day: whole seconds, not "days since"
    assert stored["ministack"].encoding["units"] == "seconds since 1970-01-01"
    _assert_same(dxr.open_frame(repo), dxr.chain(dxr.open_disp(three)))
