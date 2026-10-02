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

from donfig import Config

DEFAULTS = {
    "chunks": {
        # Target uncompressed size of one dask chunk (and Zarr shard), in MB.
        "target_mb": 100,
    },
    # Largest array, in bytes, that export and plotting load into memory.
    "max_load_bytes": 2 * 1024**3,
    # Threads per worker for compiled kernels; None leaves the library default.
    "threads": None,
}

config = Config("disp_xr", defaults=[DEFAULTS])
