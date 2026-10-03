"""Benchmark downsample on a 1000 x 1000 x 100 stack (plan tasks T27.6, T27.8).

Weighted mean and weighted median at factor 33 (990 m squares), and with
``downsample_hex`` at the equal-area spacing (1,064 m hexagons), per-epoch
random weights, 10% NaNs, dask limited to ``config["num_workers"]`` threads.
Run::

    pixi run -e test python benchmarks/bench_downsample.py
"""

import os
import time

import dask
import dask.array as da
import numpy as np
import xarray as xr

import disp_xr as dxr
from disp_xr.config import config

SHAPE, CHUNK = (100, 1000, 1000), 264  # 8 blocks of 33 per chunk


def main() -> None:
    """Print the benchmark table."""
    rng = da.random.default_rng(0)
    values = rng.standard_normal(SHAPE, chunks=(-1, CHUNK, CHUNK))
    values = da.where(
        rng.random(SHAPE, chunks=(-1, CHUNK, CHUNK)) < 0.1, np.nan, values
    )
    weights = rng.random(SHAPE, chunks=(-1, CHUNK, CHUNK))
    coords = {
        "time": np.arange(SHAPE[0]),
        "y": 4.1e6 - 30.0 * np.arange(SHAPE[1]),
        "x": 5.0e5 + 30.0 * np.arange(SHAPE[2]),
    }
    ds = xr.Dataset({"displacement": (("time", "y", "x"), values)}, coords=coords)
    w = xr.DataArray(weights, dims=("time", "y", "x"), coords=coords)
    print(
        f"stack {SHAPE}, factor 33, {os.cpu_count()} cores, "
        f"load {os.getloadavg()[0]:.0f}"
    )
    with dask.config.set(num_workers=config.get("num_workers")):
        for stat in ("wmean", "wmedian"):
            start = time.perf_counter()
            out = dxr.downsample(ds, factor=33, stat=stat, weights=w).compute()
            print(
                f"{stat:>8}: {time.perf_counter() - start:6.1f} s -> {dict(out.sizes)}",
                flush=True,
            )
        spacing = 990.0 * np.sqrt(2 / np.sqrt(3))  # same area as a 990 m square
        for stat in ("wmean", "wmedian"):
            start = time.perf_counter()
            out = dxr.downsample_hex(ds, spacing=spacing, stat=stat, weights=w)
            out = out.compute()
            print(
                f"hex {stat:>8}: {time.perf_counter() - start:6.1f} s -> "
                f"{dict(out.sizes)}",
                flush=True,
            )


if __name__ == "__main__":
    main()
