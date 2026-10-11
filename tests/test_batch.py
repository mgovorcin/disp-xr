"""Region batch runner (plan task T37.4)."""

import numpy as np
import pytest
import xarray as xr

import disp_xr as dxr
from disp_xr.batch import (
    RegionJob,
    failed_jobs,
    init_store,
    plan_jobs,
    read_manifest,
    run_region,
    write_manifest,
)


def chain(ds, job):  # noqa: ARG001 (process signature)
    return dxr.chain(ds)[["displacement"]]


@pytest.fixture(scope="module")
def frame(tmp_path_factory):
    from disp_xr.tutorial import make_disp_stack

    # 256 x 256 so that small shards split the frame into 16 regions.
    folder = tmp_path_factory.mktemp("batch") / "granules"
    make_disp_stack(folder, "two_ministacks", shape=(256, 256))
    granules = sorted(str(p) for p in folder.glob("*.nc"))
    template = chain(dxr.open_disp(granules), None)
    return granules, template


def test_regions_rebuild_the_frame(frame, tmp_path):
    granules, template = frame
    store = tmp_path / "frame.zarr"
    shards = init_store(template, store, target_mb=0.01)
    jobs = plan_jobs(template, store, granules, shards)
    assert len(jobs) == 16
    covered = np.zeros((template.sizes["y"], template.sizes["x"]), dtype=int)
    for job in jobs:
        covered[job.slices["y"], job.slices["x"]] += 1
    assert (covered == 1).all()  # every pixel in exactly one region
    results = [run_region(job, chain) for job in jobs]
    assert all(r.status == "ok" for r in results), [r.error for r in results]
    built = xr.open_zarr(store)
    np.testing.assert_allclose(
        built.displacement.values, template.displacement.values, equal_nan=True
    )


def test_failures_are_reported_and_resubmitted(frame, tmp_path):
    granules, template = frame
    store = tmp_path / "frame.zarr"
    shards = init_store(template, store, target_mb=0.01)
    jobs = plan_jobs(template, store, granules, shards)
    broken = RegionJob.from_json(jobs[0].to_json())
    broken.granules = ["/does/not/exist.nc"]
    results = [run_region(broken, chain)] + [run_region(j, chain) for j in jobs[1:]]
    assert results[0].status == "failed" and "exist" in results[0].error
    manifest = tmp_path / "manifest.jsonl"
    write_manifest(results, manifest)
    retry = failed_jobs(jobs, read_manifest(manifest))
    assert [j.job_id for j in retry] == [jobs[0].job_id]
    write_manifest([run_region(job, chain) for job in retry], manifest)
    assert failed_jobs(jobs, read_manifest(manifest)) == []
    built = xr.open_zarr(store)
    np.testing.assert_allclose(
        built.displacement.values, template.displacement.values, equal_nan=True
    )


def test_job_round_trips_json(frame, tmp_path):
    granules, template = frame
    job = plan_jobs(template, tmp_path / "s.zarr", granules, {"y": 32, "x": 32})[0]
    assert RegionJob.from_json(job.to_json()) == job
    assert job.job_id.startswith(f"F{template.attrs['frame_id']:05d}-")
