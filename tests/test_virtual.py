"""Virtual frames in Icechunk: build, open, validate (plan task T14.2c)."""

import json
import shutil

import h5py
import numpy as np
import pytest
import xarray as xr

import disp_xr as dxr
from disp_xr.product import MixedProductionError
from disp_xr.virtual import (
    _container_prefix,
    build_virtual_frame,
    open_virtual,
    validate_virtual,
)

icechunk = pytest.importorskip("icechunk")
pytest.importorskip("virtualizarr")


@pytest.fixture
def granules(two_ministacks, tmp_path):
    # A private copy: some tests change a granule.
    out = tmp_path / "granules"
    shutil.copytree(two_ministacks.paths[0].parent, out)
    return out


def _raw(stack_dir):
    return dxr.open_disp(stack_dir, chunks={"time": -1, "y": -1, "x": -1})


def test_build_and_open(granules, tmp_path):
    repo = tmp_path / "repo"
    snapshots = build_virtual_frame(granules, repo, max_workers=1)
    assert len(snapshots) == 2  # one commit per ministack
    back = open_virtual(repo)
    raw = _raw(granules)
    for name in raw.data_vars:
        np.testing.assert_array_equal(back[name].values, raw[name].values)
        assert back[name].dtype == raw[name].dtype
    np.testing.assert_array_equal(back.time, raw.time)
    np.testing.assert_array_equal(back.reference_time, raw.reference_time)
    assert back.rio.crs == raw.rio.crs
    assert back.rio.transform() == raw.rio.transform()
    assert back.displacement.encoding["grid_mapping"] == "spatial_ref"
    assert back.attrs["source_snapshot_id"] == snapshots[-1]


def test_chunks_are_references(granules, tmp_path):
    repo = tmp_path / "repo"
    build_virtual_frame(granules, repo, max_workers=1)
    # The repository holds metadata and the loaded coordinates only.
    size = sum(p.stat().st_size for p in (repo / "chunks").rglob("*") if p.is_file())
    data = sum(p.stat().st_size for p in granules.glob("*.nc"))
    assert size < data / 10


def test_commit_metadata(granules, tmp_path):
    repo = tmp_path / "repo"
    first, second = build_virtual_frame(granules, repo, max_workers=1)
    r = icechunk.Repository.open(icechunk.local_filesystem_storage(str(repo)))
    meta = r.lookup_snapshot(second).metadata
    assert meta["virtual"] is True
    assert meta["product"] == "DISP-S1"
    assert len(meta["input_granules"]) == 5  # this ministack's granules
    assert meta["disp_xr_version"] == dxr.__version__
    assert r.lookup_snapshot(first).message.startswith("Reference 5 granules")
    attrs = open_virtual(repo).attrs
    assert len(json.loads(attrs["input_granules"])) == 10


def test_layers_and_snapshot(granules, tmp_path):
    repo = tmp_path / "repo"
    first, _ = build_virtual_frame(
        granules, repo, layers=["displacement", "recommended_mask"], max_workers=1
    )
    assert set(open_virtual(repo).data_vars) == {"displacement", "recommended_mask"}
    assert open_virtual(repo, snapshot_id=first).sizes["time"] == 5
    with pytest.raises(ValueError, match="Unknown"):
        build_virtual_frame(granules, tmp_path / "r2", layers=["nope"], max_workers=1)


def test_extends_existing_frame(granules, tmp_path):
    repo = tmp_path / "repo"
    df = dxr.find_products(granules)
    build_virtual_frame(df.iloc[:5], repo, max_workers=1)
    build_virtual_frame(df.iloc[5:], repo, max_workers=1)
    assert open_virtual(repo).sizes["time"] == 10
    with pytest.raises(ValueError, match="after the store's last epoch"):
        build_virtual_frame(df.iloc[5:], repo, max_workers=1)


def test_refuses_mixed_productions(disp_stacks, tmp_path):
    with pytest.raises(MixedProductionError):
        build_virtual_frame(
            disp_stacks["mixed_dtype"].paths[0].parent, tmp_path / "r", max_workers=1
        )


def test_process_pool(granules, tmp_path):
    repo = tmp_path / "repo"
    build_virtual_frame(granules, repo, max_workers=2)
    validate_virtual(repo, granules)


def test_validate(granules, tmp_path):
    repo = tmp_path / "repo"
    build_virtual_frame(granules, repo, max_workers=1)
    report = validate_virtual(repo, granules, n_samples=5)
    assert report["n_granules"] == 10 and len(report["pixels"]) == 5


def test_validate_finds_changed_granule(granules, tmp_path):
    repo = tmp_path / "repo"
    build_virtual_frame(granules, repo, max_workers=1)
    # The references point at the granule bytes, so replacing a granule after
    # the build makes the store differ from what h5py reads.
    path = sorted(granules.glob("*.nc"))[3]
    with h5py.File(path, "r+") as f:
        f["displacement"][...] = f["displacement"][()] + 1
    with pytest.raises(ValueError, match="differs"):
        validate_virtual(repo, granules, n_samples=20)


def test_validate_checks_dates(granules, tmp_path):
    repo = tmp_path / "repo"
    df = dxr.find_products(granules)
    build_virtual_frame(df.iloc[:5], repo, max_workers=1)
    with pytest.raises(ValueError, match="epochs"):
        validate_virtual(repo, granules)


def test_container_prefix():
    assert _container_prefix(["s3://b/x/a.nc", "s3://b/y/b.nc"]) == "s3://b/"
    assert _container_prefix(["file:///d/a.nc", "file:///d/b.nc"]) == "file:///d/"
    assert _container_prefix(["file:///d/a.nc"]) == "file:///d/"
    with pytest.raises(ValueError, match="buckets"):
        _container_prefix(["s3://a/x.nc", "s3://b/y.nc"])
    with pytest.raises(ValueError, match="https"):
        _container_prefix(["https://h/a.nc"])


def test_open_needs_repository(tmp_path):
    with pytest.raises(FileNotFoundError):
        open_virtual(tmp_path / "missing")


def test_existing_repository_needs_container(granules):
    repo = icechunk.Repository.create(icechunk.in_memory_storage())
    with pytest.raises(ValueError, match="virtual chunk container"):
        build_virtual_frame(granules, repo, max_workers=1)


def test_native_copy_from_virtual(granules, tmp_path):
    # The documented path for per-pixel work: virtual index, then native store.
    repo = tmp_path / "repo"
    build_virtual_frame(granules, repo, max_workers=1)
    path = tmp_path / "native.zarr"
    dxr.to_zarr(open_virtual(repo), path)
    np.testing.assert_array_equal(
        xr.open_zarr(path, mask_and_scale=False).displacement.values,
        _raw(granules).displacement.values,
    )


def test_s3_container_with_static_credentials(tmp_path):
    """Regression: icechunk 2.3 refused bare S3 static credentials."""
    from disp_xr._credentials import AWSCredentials
    from disp_xr.virtual import _open_repository

    repo = _open_repository(
        tmp_path / "repo",
        prefix="s3://bucket/granules/",
        credentials=AWSCredentials("key", "secret", "token"),
    )
    assert "s3://bucket/granules/" in repo.config.virtual_chunk_containers
