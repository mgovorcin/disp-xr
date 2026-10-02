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
}

config = Config("disp_xr", defaults=[DEFAULTS])
