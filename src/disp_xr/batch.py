"""Build a frame cube with independent region jobs (PRD C6, plan task T37.4).

Modeled on earth-mover's serverless-datacube-demo. A coordinator:

1. builds a lazy *template* of the output (for example
   ``dxr.chain(dxr.open_disp(granules))``), which reads no pixel;
2. writes the store's metadata once with :func:`init_store`;
3. splits the frame into regions aligned to the shards with
   :func:`plan_jobs`; each :class:`RegionJob` serializes to JSON, one per AWS
   Batch array index or dask task;

and each worker runs :func:`run_region`: it opens the granules on its region,
applies a processing function, overwrites its own shards and returns a
:class:`RegionResult`. Shards are the unit of safe parallel writing, and a
job owns whole shards, so a retry overwrites them cleanly. The results form
a manifest, and :func:`failed_jobs` lists the jobs to resubmit.

The granule list is resolved once by the coordinator and travels in the
jobs. Workers reading ``s3://`` granules get temporary ASF credentials,
cached and renewed after 50 minutes, so each job starts with valid ones.

Processing must be local in space: masking, corrections, chaining and
per-pixel fits work region by region; a spatial reference or a deramp
needs the whole frame, so apply it after the cube is assembled (lazily, on
the store).

Examples
--------
>>> template = dxr.chain(dxr.open_disp(granules))  # doctest: +SKIP
>>> shards = init_store(template, "frame.zarr")  # doctest: +SKIP
>>> jobs = plan_jobs(template, "frame.zarr", granules, shards)  # doctest: +SKIP
>>> results = [run_region(job, chain) for job in jobs]  # doctest: +SKIP
>>> retry = failed_jobs(jobs, results)  # doctest: +SKIP

"""

from __future__ import annotations

import json
import time
import traceback
import warnings
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

import xarray as xr

from .export.zarr import Preset, _prepare, _with_zarr_conventions, zarr_encoding

__all__ = [
    "RegionJob",
    "RegionResult",
    "failed_jobs",
    "init_store",
    "plan_jobs",
    "plan_regions",
    "read_manifest",
    "run_region",
    "write_manifest",
]

SPATIAL = ("y", "x")


@dataclass
class RegionJob:
    """One region of a frame cube: what to read, where to write it.

    Examples
    --------
    >>> job = RegionJob(
    ...     job_id="F11116-y0-x0",
    ...     frame=11116,
    ...     store="frame.zarr",
    ...     region={"y": [0, 1024], "x": [0, 1024]},
    ...     granules=["g1.nc", "g2.nc"],
    ... )
    >>> RegionJob.from_json(job.to_json()) == job
    True
    >>> job.slices["y"]
    slice(0, 1024, None)

    """

    job_id: str
    frame: int
    store: str
    region: dict[str, list[int]]
    """``{"y": [start, stop], "x": [start, stop]}`` in pixels."""
    granules: list[str]
    parameters: dict[str, Any] = field(default_factory=dict)

    @property
    def slices(self) -> dict[str, slice]:
        """The region as slices."""
        return {dim: slice(start, stop) for dim, (start, stop) in self.region.items()}

    def to_json(self) -> str:
        """Return the job as JSON (one line)."""
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, text: str) -> RegionJob:
        """Return the job encoded by :meth:`to_json`."""
        return cls(**json.loads(text))


@dataclass
class RegionResult:
    """What a region job reports: success or the error, and how long it took.

    Examples
    --------
    >>> result = run_region(job, chain)  # doctest: +SKIP
    >>> result.status, result.seconds  # doctest: +SKIP

    """

    job_id: str
    status: str
    """``"ok"`` or ``"failed"``."""
    seconds: float
    region: dict[str, list[int]]
    error: Optional[str] = None


def init_store(
    template: xr.Dataset,
    store: Union[str, Path],
    *,
    preset: Preset = "timeseries",
    target_mb: Optional[float] = None,
    compression_level: int = 3,
) -> dict[str, int]:
    """Write the metadata of ``template`` to a new Zarr v3 store, no data.

    Coordinates are written; data variables are created empty with the
    encoding :func:`disp_xr.to_zarr` would use. Returns the shard sizes by
    dimension, which :func:`plan_jobs` aligns the regions to.

    Examples
    --------
    >>> template = dxr.chain(dxr.open_disp(granules))  # doctest: +SKIP
    >>> shards = init_store(template, "frame.zarr")  # doctest: +SKIP

    """
    out, shards = _prepare(template, preset, None, target_mb, json_native=True)
    out = _with_zarr_conventions(out)
    encoding = zarr_encoding(
        out, preset, compression_level=compression_level, target_mb=target_mb
    )
    for var in out.data_vars.values():
        var.encoding.pop("_FillValue", None)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Consolidated metadata")
        out.to_zarr(store, mode="w", compute=False, encoding=encoding, zarr_format=3)
    return shards


def plan_regions(
    template: xr.Dataset, shards: dict[str, int]
) -> list[dict[str, list[int]]]:
    """Return regions of whole shards in ``y`` and ``x`` (all epochs each).

    Examples
    --------
    >>> import numpy as np, xarray as xr
    >>> template = xr.Dataset(coords={"y": np.arange(300), "x": np.arange(500)})
    >>> len(plan_regions(template, {"y": 256, "x": 256}))
    4

    """
    regions = []
    for y0 in range(0, template.sizes["y"], shards["y"]):
        for x0 in range(0, template.sizes["x"], shards["x"]):
            regions.append(
                {
                    "y": [y0, min(y0 + shards["y"], template.sizes["y"])],
                    "x": [x0, min(x0 + shards["x"], template.sizes["x"])],
                }
            )
    return regions


def plan_jobs(
    template: xr.Dataset,
    store: Union[str, Path],
    granules: Sequence[Union[str, Path]],
    shards: dict[str, int],
    *,
    parameters: Optional[dict[str, Any]] = None,
) -> list[RegionJob]:
    r"""Return one :class:`RegionJob` per region of :func:`plan_regions`.

    Examples
    --------
    >>> jobs = plan_jobs(template, "frame.zarr", granules, shards)  # doctest: +SKIP
    >>> lines = [job.to_json() + "\n" for job in jobs]  # doctest: +SKIP
    >>> open("jobs.jsonl", "w").writelines(lines)  # doctest: +SKIP

    """
    frame = int(template.attrs.get("frame_id", 0))
    files = [str(g) for g in granules]
    return [
        RegionJob(
            job_id=f"F{frame:05d}-y{region['y'][0]}-x{region['x'][0]}",
            frame=frame,
            store=str(store),
            region=region,
            granules=files,
            parameters=dict(parameters or {}),
        )
        for region in plan_regions(template, shards)
    ]


def run_region(
    job: RegionJob, process: Callable[[xr.Dataset, RegionJob], xr.Dataset]
) -> RegionResult:
    """Open the job's region, process it and overwrite its shards.

    ``process(ds, job)`` gets the region of ``open_disp(job.granules)`` and
    returns the variables of the template on that region. Errors are caught
    and reported in the result, so a coordinator can collect every job.

    Examples
    --------
    >>> def chain(ds, job):
    ...     return dxr.chain(dxr.mask(ds, "recommended"))
    >>> result = run_region(job, chain)  # doctest: +SKIP

    """
    from . import open_disp

    start = time.perf_counter()
    try:
        ds = open_disp(job.granules).isel(job.slices)
        out = process(ds, job)
        # Region writes take only variables on the region's dimensions; the
        # coordinates and metadata were written by init_store.
        keep = [n for n, v in out.data_vars.items() if set(SPATIAL) <= set(v.dims)]
        out = out[keep].drop_vars(
            [c for c in out.coords if not set(SPATIAL) <= set(out[c].dims)]
        )
        out = out.drop_vars([c for c in SPATIAL if c in out.coords])
        # One dask chunk per region: a region is one shard, small by design.
        out = out.drop_encoding().chunk(-1)
        for var in out.data_vars.values():
            # Attributes, fill values included, were written by init_store.
            var.attrs = {}
        out.to_zarr(job.store, region=job.slices, mode="r+", zarr_format=3)
        return RegionResult(job.job_id, "ok", time.perf_counter() - start, job.region)
    except Exception:
        return RegionResult(
            job.job_id,
            "failed",
            time.perf_counter() - start,
            job.region,
            error=traceback.format_exc(limit=3),
        )


def failed_jobs(
    jobs: Sequence[RegionJob], results: Sequence[RegionResult]
) -> list[RegionJob]:
    """Return the jobs without an ``"ok"`` result, to resubmit.

    Examples
    --------
    >>> retry = failed_jobs(jobs, read_manifest("manifest.jsonl"))  # doctest: +SKIP

    """
    done = {r.job_id for r in results if r.status == "ok"}
    return [job for job in jobs if job.job_id not in done]


def write_manifest(results: Sequence[RegionResult], path: Union[str, Path]) -> None:
    """Append results to a JSON-lines manifest.

    Examples
    --------
    >>> write_manifest([result], "manifest.jsonl")  # doctest: +SKIP

    """
    with open(path, "a") as f:
        for result in results:
            f.write(json.dumps(asdict(result)) + "\n")


def read_manifest(path: Union[str, Path]) -> list[RegionResult]:
    """Return the results in a JSON-lines manifest (later lines win per job).

    Examples
    --------
    >>> results = read_manifest("manifest.jsonl")  # doctest: +SKIP

    """
    latest: dict[str, RegionResult] = {}
    with open(path) as f:
        for line in f:
            if line.strip():
                result = RegionResult(**json.loads(line))
                latest[result.job_id] = result
    return list(latest.values())
