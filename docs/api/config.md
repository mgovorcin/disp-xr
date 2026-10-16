# Settings and chunks

## Settings

`disp_xr.config` holds the runtime settings. Change them for a block of code
with `config.set`, or for a batch job with `DISP_XR_*` environment variables
(nested keys use a double underscore).

```python
from disp_xr.config import config

with config.set({"chunks.target_mb": 50}):
    ...
```

| Key | Default | Meaning |
|---|---|---|
| `chunks.target_mb` | 100 | Uncompressed size of one dask chunk (and Zarr shard) for the `timeseries` and `maps` presets, in MB |
| `chunks.read_block` | 1024 | Spatial block read from each granule; the `native` chunk size. 2048 gives 3.5 times fewer tasks when writing a full-frame store (`performance.md`) |
| `max_load_bytes` | 2 GiB | Largest array export and plotting load into memory |
| `cache.dir` | `None` | Downloaded static layers; `None` means `$XDG_CACHE_HOME/disp_xr` or `~/.cache/disp_xr` |
| `threads` | `None` | Threads per worker for compiled kernels |
| `num_workers` | 32 | Dask threads when disp-xr computes (plots, exports), unless dask's own `num_workers` is set |
| `numba_threads` | 1 | Threads of numba kernels (numbagg) inside each dask task, set as `NUMBA_NUM_THREADS` on import unless that variable is set or numba is already loaded; numba's default of one per core made every dask thread open 224 threads on a large node and crash |

## Chunks

::: disp_xr._chunks
    options:
      members: false
      show_root_heading: false

::: disp_xr.rechunk
