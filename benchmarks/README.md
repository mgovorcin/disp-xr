# Benchmarks

Scripts that time disp-xr on synthetic data. Shared nodes vary a lot, so each
result records the machine load; compare columns of one run, not runs.

## `bench_trajectory.py` (plan task T20.5)

`fit_trajectory` on a 1000 x 1000 x 100 float32 stack, six coefficients
(offset, velocity, annual and semiannual pairs). Tile columns: one 256 x 256
tile on one core, BLAS limited to one thread. Whole stack: 16 tiles, dask
threaded scheduler.

Run on 2026-10-02, Aurora node, 224 cores, load 62:

| NaN % | patterns per tile | hybrid (s) | batched only (s) | by pattern only (s) | whole stack (s) |
|---:|---:|---:|---:|---:|---:|
| 0 | 1 | 1.36 | 1.19 | 0.64 | 36.2 |
| 1 | 9,725 | 1.44 | 7.10 | 17.66 | 57.7 |
| 20 | 65,536 | 3.23 | 7.30 | 78.07 | 143.4 |

- **hybrid** is what `fit_trajectory` uses: pixels sharing a valid-epoch
  pattern (at least 64) are solved with one factorization, the rest with
  batched per-pixel normal equations. It is the fastest whenever a tile has
  NaNs; with none, plain grouping is about twice as fast (the pattern
  bookkeeping costs more than the one solve).
- **Thread scaling is poor.** On 8 tiles: 9.7 s single-threaded, 4.3 s with
  the threaded scheduler (2.3x). NumPy's `einsum`, `unique` and small-matrix
  `linalg` loops hold the GIL. For frame-scale fits use a dask distributed
  cluster with worker processes:

  ```python
  from dask.distributed import Client
  client = Client(n_workers=32, threads_per_worker=1)
  fit = dxr.fit_trajectory(ds, terms).compute()
  ```

Next optimizations, not done: a no-NaN fast path for whole tiles, and
GIL-free kernels (BLAS `matmul` in place of `einsum`, or numba).
