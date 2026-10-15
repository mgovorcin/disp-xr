# Large frames and batch jobs

A DISP-S1 frame is about 70 million pixels by hundreds of epochs, so how dask
runs matters as much as the code. These settings come from running full
frames on a 224-core, 1 TB node.

## Processes, not threads

Use a dask cluster of local processes for full frames:

```python
from distributed import Client, LocalCluster

with LocalCluster(n_workers=24, threads_per_worker=1, memory_limit="8GB") as cluster, Client(cluster):
    ds = dxr.open_disp("F08882/")
    ...
```

Dask's default threaded scheduler runs at about one core on two common
steps:

- **Reading granules.** The HDF5 library holds a process-wide lock, so
  threads decompress NetCDF granules one at a time.
- **Reading Zarr.** zarr decodes chunks in one thread per process.

On a full frame, `rereference("auto")` took half an hour on threads, because
it reads every quality layer of every epoch.

## Memory and graph size

Measured on full DISP-S1 frames (2026-10-05):

- **Workers:** 8 GB per worker was enough for every step of a full-frame
  study (open, chain, write, velocity, downsampling, quadtrees) with 16 to
  24 workers; no worker paused, spilled or restarted.
- **Coordinator:** the process that builds the graph and, with
  `LocalCluster`, runs the scheduler, holds the task graph. A frame
  prepared for writing is about 200,000 tasks and 35 MB pickled, and the
  process stays near 1 GB. (Before disp-xr 0.3 a frame-sized array of
  default weights was embedded in velocity graphs and pushed it to
  80-100 GB.)
- **Opening:** `open_disp` reads each granule's metadata, about 0.2 s per
  granule, so a 400-granule frame takes over a minute to open. Build a
  virtual frame once when you open the same frame repeatedly.
- **Task counts** grow with epochs times spatial blocks times layers: one
  epoch per chunk by default, so downsampling a 389-epoch frame is about
  2 million tasks, which the distributed scheduler handles.

Memory budget (decided 2026-10-06, plan task T37.1): 8 GB per worker and
4 GB for the coordinator. No worker of the full-frame T29 runs paused,
spilled or restarted at 8 GB, and a frame prepared for writing peaks at
about 1.2 GB on the coordinator.

### Every core operation on a full frame

Profiled with `benchmarks/profile_frame.py` (plan task T37.3, 2026-10-13
and 14): F11116, 389 epochs of 6898 x 9618, opened from the NetCDF
granules, masked, chained and re-referenced to the median, in one graph per
operation; `LocalCluster` with 16 workers of 2 threads and 8 GB. Peak
resident memory; times on a shared node (load 30 to 120) are indicative.

| Operation | Tasks | Worker peak | Coordinator peak | Compute |
|---|---:|---:|---:|---:|
| open | 131,000 | 0.7 GB | 2.0 GB | 47 s |
| chain | 360,000 | 0.7 GB | 2.1 GB | 2.6 min |
| mask | 409,000 | 0.4 GB | 1.9 GB | 25 min |
| rereference median | 531,000 | 5.7 GB | 2.5 GB | 35 min |
| fit_velocity OLS | 628,000 | 6.5 GB | 8.1 GB | 72 min |
| fit_velocity MIDAS | 620,000 | 7.1 GB, paused | 8.0 GB | 1.9 h |
| downsample 1 km | 2,828,000 | 5.9 GB | 19 GB | 3.5 h |
| quadtree | 4,304,000 | 5.9 GB | 122 GB | 3.4 h |
| denoise Kalman | 633,000 | 7.3 GB, paused | (111 GB, after quadtree) | 40 min |
| fit_velocity GLS | | over 8 GB: 71 restarts | | stopped |

- **Workers** fit 8 GB for every operation but GLS. MIDAS and the Kalman
  filter come close (paused, not restarted); give them one thread per
  worker or 10 GB. GLS needs more than 8 GB per worker on a full frame (a
  first run also crashed the coordinator process during its noise
  sampling); fit it on a region of the frame, or on 16 GB workers.
- **The coordinator** grows with the task count, 5 to 28 KB per task with
  `LocalCluster` (graph and scheduler in one process): over the 4 GB
  budget from a few hundred thousand tasks. A whole frame read from 389 granules
  is 130,000 tasks before any processing, and downsampling and quadtrees
  multiply that by levels and layers.
- The Kalman row ran after quadtree in the same process, which had not
  returned quadtree's memory; its coordinator figure is not its own.

What to do on full frames, in order of effect:

1. **Build a store first, then process it.** Write the masked, chained
   cube once with region jobs (below) or `to_zarr`, then run velocity,
   downsampling and quadtrees from the store: the graph no longer carries
   389 granules, the mask, the chain and the reference.
2. **Fewer, larger chunks.** `config.set({"chunks.read_block": 2048})`
   should quarter the task count of an opened frame (four times fewer
   blocks; not profiled yet); a `"timeseries"` store has
   one chunk per spatial block for all epochs.
3. **Run the scheduler apart from the client** (`dask scheduler` or a
   deployed cluster) when graphs pass a few hundred thousand tasks, and
   budget 8 to 16 GB for it; with `LocalCluster` it shares the client
   process.
4. **Split per-pixel methods by region** (GLS, and MIDAS or Kalman at 8 GB)
   with `disp_xr.batch`: each job holds one region's series.

## disp-xr settings

Each setting can be given in code (`disp_xr.config.set({...})`) or as an
environment variable for batch jobs (`DISP_XR_` plus the key in capitals,
with `__` for nested keys).

| Setting | Default | What it does |
| --- | --- | --- |
| `max_load_bytes` | 2 GB | Largest array that export and plotting load into memory. Larger ones raise `LoadTooLargeError`, which names the size and what to do, before anything is read |
| `num_workers` | 32 | Dask threads when disp-xr computes (plots, exports), unless dask's own `num_workers` is set. One per core (224) exhausted memory maps when reading a pixel series from hundreds of granules |
| `numba_threads` | 1 | numba threads inside each dask task. numba starts one per core in every task, which crashed a 224-core node |
| `threads` | None | BLAS threads inside each task for compiled kernels |
| `chunks.read_block` | 1024 | Spatial block read from each granule |
| `chunks.target_mb` | 100 | Target size of one dask chunk and Zarr shard |

## Environment for batch jobs

These multiply with dask's own parallelism, so set them per job:

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export ZARR_ASYNC__CONCURRENCY=4      # zarr's default is 10 per thread
export DISP_XR_MAX_LOAD_BYTES=8000000000
```

`ZARR_ASYNC__CONCURRENCY` caps zarr's concurrent requests. Its default of 10
applies per dask thread, which floods object stores and local disks alike
on large machines.

## Shared nodes: check memory first

On a busy shared node, fresh memory can stall in the kernel (transparent
huge page compaction). Touching 2 GB then takes 40 s instead of half a
second, and every step of a run slows by as much. Before timing anything,
or when a run is unexpectedly slow, check:

```bash
python -c "import numpy as np, time; t = time.perf_counter(); np.ones(2**28); print(time.perf_counter() - t)"
```

A healthy node takes about 0.2 s. Ten times that means the run measures the
node, not the code.

## Laziness

Building a result reads nothing. `tests/test_lazy.py` builds the core
operations under a dask scheduler that refuses to compute. A few steps read
small data on purpose:

- `open_disp` reads `reference_time`;
- `to_points` reads which cells hold data;
- `quadtree` computes its split scores;
- `rereference(check_point=True)` reads the reference pixel.

## Building a frame with region jobs

`disp_xr.batch` splits the build of a frame cube into independent jobs, one
per region of whole shards, for AWS Batch array jobs or a pool of dask
workers. The coordinator writes the store's metadata once and plans the
jobs:

```python
import json
import disp_xr as dxr
from disp_xr.batch import init_store, plan_jobs

granules = list(search_table["filename"])           # resolved once, here
template = dxr.chain(dxr.open_disp(granules))  # lazy: reads no pixel
shards = init_store(template, "s3://bucket/F11116.zarr")
jobs = plan_jobs(template, "s3://bucket/F11116.zarr", granules, shards)
with open("jobs.jsonl", "w") as f:
    f.writelines(job.to_json() + "\n" for job in jobs)
```

Each worker (AWS Batch array index `i`) runs one job and reports a result:

```python
import os
from disp_xr.batch import RegionJob, run_region, write_manifest

line = open("jobs.jsonl").readlines()[int(os.environ["AWS_BATCH_JOB_ARRAY_INDEX"])]
job = RegionJob.from_json(line)
result = run_region(job, lambda ds, job: dxr.chain(dxr.mask(ds, "recommended")))
write_manifest([result], f"manifest-{job.job_id}.jsonl")
```

A job owns whole shards and overwrites them, so a retry is safe. Collect
the manifests and resubmit what failed with `failed_jobs(jobs, results)`.

Processing in a job must be local in space: masking, corrections, chaining
and per-pixel fits work region by region. A spatial reference or a deramp
needs the whole frame, so apply it afterwards on the assembled store,
lazily:

```python
cube = dxr.open_zarr("s3://bucket/F11116.zarr").disp.rereference("median")
velocity = dxr.fit_velocity(cube, "midas")
cells = dxr.downsample(cube, factor=33)
report = dxr.residuals.compare(cells, dxr.gnss.load(cube), dxr.open_static(cube))
```
