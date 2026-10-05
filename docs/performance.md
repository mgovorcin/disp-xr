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
