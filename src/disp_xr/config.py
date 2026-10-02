"""Runtime settings for disp-xr.

One :class:`donfig.Config` named ``disp_xr``. Values come from the defaults
below, then from ``DISP_XR_*`` environment variables (nested keys use a double
underscore, for example ``DISP_XR_CHUNKS__TARGET_MB=50``), then from
``config.set``. Batch jobs set environment variables instead of changing code.

Examples
--------
>>> from disp_xr.config import config
>>> config.get("chunks.target_mb")
100
>>> with config.set({"chunks.target_mb": 50}):
...     config.get("chunks.target_mb")
50

"""

import functools
from collections.abc import Callable
from typing import Any, TypeVar

from donfig import Config

F = TypeVar("F", bound=Callable[..., Any])

DEFAULTS = {
    "chunks": {
        # Target uncompressed size of one dask chunk (and Zarr shard), in MB.
        "target_mb": 100,
        # Spatial block read from each granule before rechunking to a preset.
        # Measured on a 209-granule frame: whole layers open in 15 s but make
        # a one-pixel series read every full layer (247 s); 256 (the HDF5
        # chunk) builds a graph of millions of tasks (87 s to open); 1024 opens
        # in 19 s and reads a pixel series in 6 s.
        "read_block": 1024,
    },
    # Largest array, in bytes, that export and plotting load into memory.
    "max_load_bytes": 2 * 1024**3,
    # Downloaded static layers; None means $XDG_CACHE_HOME/disp_xr or
    # ~/.cache/disp_xr.
    "cache": {"dir": None},
    # Threads per worker for compiled kernels; None leaves the library default.
    "threads": None,
    # Dask threads when disp-xr computes (plots, exports), unless dask's own
    # num_workers is set. Dask's default is one per core: 224 on the Aurora
    # nodes, where reading a pixel series from 209 granules needed about 650
    # memory maps per thread and hit vm.max_map_count (65,530), so thread
    # creation failed. 32 read the series in 16 s. None uses dask's default.
    "num_workers": 32,
}

config = Config("disp_xr", defaults=[DEFAULTS])


def bounded_threads(func: F) -> F:
    """Run ``func`` with at most ``config["num_workers"]`` dask threads.

    Applied to the functions that compute (plots, exports). A ``num_workers``
    set in dask's own config, or a non-threaded scheduler, takes precedence.
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        import dask

        workers = config.get("num_workers")
        if workers is None or dask.config.get("num_workers", None) is not None:
            return func(*args, **kwargs)
        with dask.config.set(num_workers=int(workers)):
            return func(*args, **kwargs)

    return wrapper  # type: ignore[return-value]
