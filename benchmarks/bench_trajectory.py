"""Benchmark fit_trajectory on a 1000 x 1000 x 100 stack (plan task T20.5).

On one 256 x 256 tile, compares the solver ``fit_trajectory`` uses (pixels
grouped by NaN pattern, rare patterns batched) with each approach alone:
batched per-pixel normal equations, and one ``lstsq`` per pattern. Then times
the whole stack. Three NaN fractions: none, 1% and 20% of values at random,
the last giving nearly one pattern per pixel. Run::

    pixi run -e test python benchmarks/bench_trajectory.py

and record the printed table in ``benchmarks/README.md``. The machine load is
printed too: shared nodes make absolute times vary.
"""

import os
import time

import numpy as np
import pandas as pd
import xarray as xr
from threadpoolctl import threadpool_limits

import disp_xr as dxr
from disp_xr.timeseries import trajectory
from disp_xr.timeseries.trajectory import Trajectory, _solve

SHAPE = (100, 1000, 1000)
TERMS = Trajectory(periods=(1.0, 0.5))


def stack(nan_fraction: float, seed: int = 0) -> xr.DataArray:
    """Return a linear trend plus 1 mm noise with ``nan_fraction`` NaNs."""
    rng = np.random.default_rng(seed)
    times = pd.date_range("2020-01-01", periods=SHAPE[0], freq="12D").values
    years = np.arange(SHAPE[0]) * 12 / 365.25
    data = (0.01 * years[:, None, None] + rng.normal(0, 1e-3, SHAPE)).astype("f4")
    data[rng.random(SHAPE) < nan_fraction] = np.nan
    return xr.DataArray(data, dims=("time", "y", "x"), coords={"time": times})


def by_pattern(block: np.ndarray, design: np.ndarray) -> np.ndarray:
    """Solve one lstsq per NaN pattern over all pixels sharing it."""
    n_time = block.shape[0]
    y = block.reshape(n_time, -1)
    valid = np.isfinite(y)
    keys, inverse = np.unique(np.packbits(valid, axis=0), axis=1, return_inverse=True)
    coef = np.full((design.shape[1], y.shape[1]), np.nan)
    for k in range(keys.shape[1]):
        cols = np.flatnonzero(inverse.ravel() == k)
        rows = valid[:, cols[0]]
        if rows.sum() > design.shape[1]:
            fit = np.linalg.lstsq(design[rows], y[rows][:, cols], rcond=None)
            coef[:, cols] = fit[0]
    return coef


def timed(func, *args) -> float:
    """Return the run time of ``func(*args)`` in seconds."""
    start = time.perf_counter()
    func(*args)
    return time.perf_counter() - start


def main() -> None:
    """Print the benchmark table."""
    design = TERMS.design(stack(0).time.values)
    load = os.getloadavg()[0]
    print(
        f"stack {SHAPE}, {len(TERMS.names)} coefficients, "
        f"{os.cpu_count()} cores, load {load:.0f}"
    )
    print(
        f"{'NaN %':>6} {'patterns':>9} {'hybrid (s)':>11} {'batched (s)':>12} "
        f"{'by pattern (s)':>15} {'whole stack (s)':>16}"
    )
    for nan_fraction in (0.0, 0.01, 0.2):
        da = stack(nan_fraction)
        tile = da.values[:, :256, :256]
        flat = np.isfinite(tile).reshape(SHAPE[0], -1)
        n_patterns = np.unique(np.packbits(flat, axis=0), axis=1).shape[1]
        pixels = np.moveaxis(tile, 0, -1)  # (y, x, time), the apply_ufunc layout
        ones = np.ones_like(pixels)
        hybrid = timed(_solve, pixels, ones, design, 0)
        default, trajectory.GROUP_MIN_PIXELS = trajectory.GROUP_MIN_PIXELS, 10**9
        batched = timed(_solve, pixels, ones, design, 0)
        trajectory.GROUP_MIN_PIXELS = default
        with threadpool_limits(limits=1):
            grouped = timed(by_pattern, tile, design)
        chunked = da.chunk({"time": -1, "y": 256, "x": 256})
        whole = timed(lambda: dxr.fit_trajectory(chunked, TERMS).velocity.compute())
        print(
            f"{nan_fraction * 100:6.0f} {n_patterns:9d} {hybrid:11.2f} "
            f"{batched:12.2f} {grouped:15.2f} {whole:16.1f}"
        )
    print("tile columns: one 256 x 256 tile, one core, BLAS limited to one thread")


if __name__ == "__main__":
    main()
