"""Benchmark fit_velocity on a 1000 x 1000 x 200 stack (plan task T22.4).

Times the three methods on one 256 x 256 tile (one core) and on the whole
stack with the dask threaded scheduler. 200 epochs of 12 days with 2% NaNs:
the length from which Whittle is reliable. Run::

    pixi run -e test python benchmarks/bench_velocity.py
"""

import os
import time

import numpy as np
import pandas as pd
import xarray as xr

import disp_xr as dxr
from disp_xr.timeseries.noise import _fractional_diff_coeffs

SHAPE = (200, 1000, 1000)


def stack(seed: int = 0) -> xr.DataArray:
    """Return a trend plus flicker and white noise with 2% NaNs."""
    rng = np.random.default_rng(seed)
    n = SHAPE[0]
    times = pd.date_range("2017-01-01", periods=n, freq="12D").values
    years = np.arange(n) * 12 / 365.25
    psi = _fractional_diff_coeffs(0.5, n).astype("f4")
    white = rng.standard_normal(SHAPE, dtype="f4")
    flicker = np.apply_along_axis(lambda w: np.convolve(w, psi)[:n], 0, white)
    data = 0.005 * years[:, None, None] + 2e-3 * flicker
    data += 2e-3 * rng.standard_normal(SHAPE, dtype="f4")
    data[rng.random(SHAPE) < 0.02] = np.nan
    return xr.DataArray(data.astype("f4"), dims=("time", "y", "x"), coords={"time": times})


def main() -> None:
    """Print the benchmark table."""
    da = stack()
    load = os.getloadavg()[0]
    print(f"stack {SHAPE}, {os.cpu_count()} cores, load {load:.0f}")
    print(f"{'method':>8} {'one tile (s)':>13} {'whole stack (s)':>16}")
    tile = da.isel(y=slice(0, 256), x=slice(0, 256))
    chunked = da.chunk({"time": -1, "y": 256, "x": 256})
    # numbagg compiles its median on first use (about 10 s); not timed.
    dxr.fit_velocity(da.isel(y=slice(0, 2), x=slice(0, 2)), "midas").velocity.compute()
    for method in ("ols", "midas", "whittle"):
        start = time.perf_counter()
        dxr.fit_velocity(tile, method).velocity.compute(scheduler="synchronous")
        one = time.perf_counter() - start
        start = time.perf_counter()
        dxr.fit_velocity(chunked, method).velocity.compute()
        whole = time.perf_counter() - start
        print(f"{method:>8} {one:13.1f} {whole:16.1f}", flush=True)


if __name__ == "__main__":
    main()
