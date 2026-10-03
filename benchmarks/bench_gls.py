"""Benchmark grouped GLS on a 10-million-pixel frame (plan task T23.6).

3200 x 3200 pixels x 200 epochs (12 days), 2% NaNs, generated lazily in dask
blocks (trend, random-walk and white noise; the GLS cost does not depend on
the noise realization), with five coherence groups. Times the noise
estimation (exact fits of 16 sampled pixels per group) and the per-pixel GLS
separately, with dask limited to ``config["num_workers"]`` threads. Run::

    pixi run -e test python benchmarks/bench_gls.py
"""

import os
import time

import dask
import dask.array as da
import numpy as np
import pandas as pd
import xarray as xr

import disp_xr as dxr
from disp_xr.config import config
from disp_xr.timeseries.gls import coherence_groups, estimate_group_noise, gls_velocity

N_TIME, NY, NX, CHUNK = 200, 3200, 3200, 256


def frame() -> xr.Dataset:
    """Return the lazy synthetic frame with a coherence layer."""
    years = np.arange(N_TIME) * 12 / 365.25

    def block(block_info=None):
        loc = block_info[None]["array-location"]
        shape = [b - a for a, b in loc]
        rng = np.random.default_rng(loc[1][0] * NX + loc[2][0])
        walk = 5e-4 * rng.standard_normal(shape).cumsum(axis=0)
        data = 0.005 * years[:, None, None] + walk + 2e-3 * rng.standard_normal(shape)
        data[rng.random(shape) < 0.02] = np.nan
        return data.astype("f4")

    chunks = (
        (N_TIME,),
        (CHUNK,) * (NY // CHUNK) + ((NY % CHUNK,) if NY % CHUNK else ()),
        (CHUNK,) * (NX // CHUNK) + ((NX % CHUNK,) if NX % CHUNK else ()),
    )
    data = da.map_blocks(block, chunks=chunks, dtype="f4")
    coherence = da.broadcast_to(
        da.from_array(np.linspace(0.3, 0.95, NX, dtype="f4"), chunks=CHUNK),
        (NY, NX),
        chunks=(CHUNK, CHUNK),
    )
    times = pd.date_range("2017-01-01", periods=N_TIME, freq="12D").values
    return xr.Dataset(
        {
            "displacement": (("time", "y", "x"), data, {"units": "meters"}),
            "temporal_coherence": (("y", "x"), coherence),
        },
        coords={"time": times, "y": -30.0 * np.arange(NY), "x": 30.0 * np.arange(NX)},
    )


def main() -> None:
    """Print the timings."""
    ds = frame()
    terms = dxr.Trajectory(periods=(1.0,))
    print(
        f"frame {N_TIME} x {NY} x {NX} ({NY * NX / 1e6:.1f} M pixels), "
        f"{os.cpu_count()} cores, load {os.getloadavg()[0]:.0f}",
        flush=True,
    )
    labels = coherence_groups(ds, 5)
    start = time.perf_counter()
    shapes = estimate_group_noise(ds.displacement, labels, terms=terms, n_samples=16)
    print(
        f"noise per group ({len(shapes)} groups x 16 exact fits): "
        f"{time.perf_counter() - start:.1f} s",
        flush=True,
    )
    start = time.perf_counter()
    out = gls_velocity(ds, terms=terms, groups=labels, shapes=shapes)
    # Each tile's solve allocates about 1 GB of temporaries; dask's default of
    # one thread per core (224 here) exhausted memory, so use disp-xr's cap.
    with dask.config.set(num_workers=config.get("num_workers")):
        sigma = out.velocity_sigma.compute()
    total = time.perf_counter() - start
    print(
        f"GLS of every pixel (shapes above): {total:.1f} s; "
        f"median sigma {float(sigma.median()) * 1000:.3f} mm/yr",
        flush=True,
    )


if __name__ == "__main__":
    main()
