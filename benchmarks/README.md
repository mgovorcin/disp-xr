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

## `bench_velocity.py` (plan task T22.4)

`fit_velocity` on a 1000 x 1000 x 200 float32 stack (12-day epochs, flicker
plus white noise, 2% NaNs). One tile: 256 x 256 on one core. Whole stack: 16
tiles, dask threaded scheduler. Run on 2026-10-03, load 76 (the Whittle row
overlapped a test run):

| method | one tile (s) | whole stack (s) |
|---|---:|---:|
| ols | 2.0 | 52.8 |
| midas | 0.6 | 4.9 |
| whittle | 6.6 | 186.1 |

- MIDAS uses numbagg's NaN median, about 250x faster than `np.nanmedian`
  (the first version took 16.5 s per tile and 307 s for the stack). numbagg
  compiles on first use (about 10 s), excluded above.
- Whittle scores candidates with matrix products: a coarse grid shared by all
  pixels, then finer grids shared by the pixels of each coarse cell, with each
  lattice point's spectrum computed once. A per-pixel local grid took 83 s per
  tile.
- Like `fit_trajectory`, OLS and Whittle hold the GIL, so threads scale
  poorly; use a dask distributed cluster with worker processes for frames.

## `whittle_bias.py` (plan task T22)

Whittle against exact RMLE (`fit_noise`, PLWN) on 12-day series of a trend,
flicker (kappa -1) and white noise, 30 realizations per length:

| epochs | years | kappa exact | kappa Whittle | sigma Whittle / exact | velocity error / sigma, exact | same, Whittle |
|---:|---:|---:|---:|---:|---:|---:|
| 60 | 2.0 | -1.03 | -0.19 | 0.59 | 1.06 | 1.86 |
| 100 | 3.3 | -0.92 | -0.75 | 0.93 | 1.21 | 1.47 |
| 200 | 6.6 | -0.94 | -0.81 | 0.99 | 0.94 | 1.01 |
| 400 | 13.1 | -1.04 | -1.00 | 1.01 | 0.84 | 0.81 |
| 800 | 26.3 | -0.93 | -0.94 | 0.99 | 0.90 | 0.94 |

Whittle is reliable from about 200 epochs; below, it underestimates `|kappa|`
and its sigmas are too small. `fit_velocity(method="whittle")` warns below
200 epochs.

## `bench_gls.py` (plan task T23.6)

`fit_velocity(method="gls")` on a 10.2-million-pixel frame (3200 x 3200 x 200
epochs, 2% NaNs, generated lazily), five coherence groups, 16 exact noise fits
per group. Run on 2026-10-03, load 160 to 190 on 224 cores, dask threaded
scheduler limited to 32 threads:

| step | time |
|---|---:|
| noise shape per group (80 exact fits, plus generating the blocks that hold the 240 candidate pixels) | 165 s |
| exact GLS of every pixel | 1,090 s |

Per 256 x 256 tile on one core (`_gls_block`): 0.7 s with no NaNs, 0.9 s with
2%, 6.8 s with 20% (more missing epochs per pixel, larger Schur blocks).

- The threaded run used about 300% CPU: most of the solve holds the GIL, and
  the node was busy. The real cost is about 0.3 ms per pixel of CPU time, so
  a dask distributed cluster with worker processes should take minutes, not
  measured here:

  ```python
  from dask.distributed import Client
  client = Client(n_workers=32, threads_per_worker=1)
  vel = dxr.fit_velocity(ds, "gls", terms=terms).compute()
  ```

- Two problems found on the way, both fixed: picking sample pixels counted
  the valid epochs of every pixel, reading the whole frame (428 s for the
  noise step before); and a tile's temporaries reached about 1 GB, so dask's
  default of one thread per core (224) exhausted memory. Pixels are now solved
  in batches of 8,192.

## `bench_downsample.py` (plan task T27.6)

`downsample` of a 100 x 1000 x 1000 stack (random values, 10% NaNs,
per-epoch weights) to factor 33, dask limited to 32 threads, including the
generation of the random data. Run on 2026-10-03, load 151:

| stat | time |
|---|---:|
| wmean | 16.3 s |
| wmedian | 28.8 s |

### Hexagons and a correction (plan task T27.8)

Wall-clock timings on the shared node were unreliable on 2026-10-03
afternoon: the same `downsample` code took 16 s in the morning, 260 s in
the afternoon and 1.2 s minutes later. The first compute of a process spends
seconds getting memory from the busy node, so whichever method ran first
looked slow. The numbers first reported for hexagons (11 s against 260 s for
squares, and flox 15 to 23 s against 0.1 s for a bincount sum) had this
order effect and are withdrawn.

`bench_downsample.py` now warms up, alternates the order and keeps the best
of two runs. Its afternoon result (100 x 1000 x 1000, load 52):

| grid | wmean | wmedian |
|---|---:|---:|
| squares, factor 33 | 1.2 s | 1.8 s |
| hexagons, 1,064 m | 1.5 s | 51.2 s |

The hexagon median sorts each epoch of the whole frame (the cells cross
chunk borders), which the square median avoids; it is the slow path.
Single-thread CPU time also drifted by 3x between runs, so only
deterministic measures are recorded for the two optimizations:

- **One pass for the five per-cell sums** (Σw, Σw², Σwv, Σwv², valid
  count) per dask block, instead of five full-size temporaries and five
  padded block sums: the graph for 20 x 1000 x 1000 shrank from 1,943 to
  1,376 tasks (squares) and from 720 to 443 (hexagons).
- **Windowed hexagon partial sums**: each block returns sums for the cells
  of its rows of hexagons only. For a 7,000 x 9,000 frame with 200 epochs,
  one block's partial is 10.5 MB instead of 138.5 MB at 1,064 m (13x),
  2.9 MB instead of 35.9 MB at 2,128 m.

**Why timings swung on 2026-10-03.** Touching 400 MB of new memory took
17 s on the node that afternoon (about 0.1 s normally): 789 of 1,007 GB
were in use and transparent huge pages are set to `always`, so page faults
stalled in the kernel. Any step that allocates (concatenating chunks,
temporaries) was slowed by up to 100x, depending on the moment. Check the
node before timing anything:

```python
import numpy as np, time
t = time.perf_counter(); z = np.zeros(50_000_000); z[::512] = 1
print(time.perf_counter() - t)  # well under 1 s on a healthy node
```

**Hexagon median by bands.** The weighted median (and the mode) of a
hexagon needs all its pixels in one task, and hexagons cross chunk borders.
It used to take one whole epoch of the frame per task: values, weights and
labels, 1.5 GB for a 7,000 x 9,000 frame before sorting temporaries, times
the number of dask threads. Each task now holds one band of rows (the cells
whose center lies in it, plus a halo of one hexagon radius) for a few
epochs, sized to about 128 MB. Results are identical to the whole-frame
computation (tested). Its speed was not measured, for the reason above.

**Quadtree (plan task T28.5).** `bench_downsample.py` also times
`quadtree` (levels 3 to 8). Not run yet: on 2026-10-03 evening the page-touch
check above took 61 s for 400 MB. The quadtree reads the data twice (the
finest level's sums, from which every level and score follow, then the
leaves' values), whatever the number of levels.
