"""Benchmark suite of the core operations (plan task T37.2).

Times open, chain, mask, re-reference, fit_velocity (OLS, MIDAS, GLS),
downsample, quadtree and denoise on synthetic granules, end to end (each
operation is computed). Every run:

- records the versions of disp-xr, xarray, zarr, dask and numpy, the host,
  core count, load and the time to touch fresh memory (a busy shared node
  stalls on page faults; a run on such a node measures the node, see
  docs/performance.md);
- warms every operation up once, untimed (numba compiles on first use);
- runs the trials interleaved: trial 1 of every operation, then trial 2,
  so a slow phase of the node spreads over all operations instead of
  landing on one;
- writes the raw rows (one per operation and trial) as CSV and prints the
  median and spread.

Run::

    pixi run -e test python benchmarks/suite.py --size tile --trials 5

``--size`` is ``smoke`` (seconds, used by the tests), ``tile`` (a
1024 x 1024 tile, 60 epochs) or ``frame`` (4096 x 4096, 120 epochs, a
sixteenth of a DISP-S1 frame by area).
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.metadata
import os
import platform
import socket
import statistics
import tempfile
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

import disp_xr as dxr
from disp_xr.tutorial import make_disp_stack

SIZES = {
    # (ny, nx), epochs per ministack (three ministacks)
    "smoke": ((64, 64), 12),
    "tile": ((1024, 1024), 20),
    "frame": ((4096, 4096), 40),
}
PACKAGES = ("disp_xr", "xarray", "zarr", "dask", "numpy")


def page_touch_seconds(nbytes: int = 2**28) -> float:
    """Return the time to allocate and touch ``nbytes`` of fresh memory."""
    start = time.perf_counter()
    block = np.ones(nbytes // 8)
    elapsed = time.perf_counter() - start
    del block
    return elapsed


def environment() -> dict[str, object]:
    """Return the versions and machine state recorded with every row."""
    versions = {}
    for name in PACKAGES:
        try:
            versions[f"{name}_version"] = importlib.metadata.version(
                name.replace("_", "-")
            )
        except importlib.metadata.PackageNotFoundError:
            versions[f"{name}_version"] = "unknown"
    return {
        **versions,
        "host": socket.gethostname(),
        "machine": platform.machine(),
        "cores": os.cpu_count(),
    }


def operations(folder: Path) -> dict[str, Callable[[], object]]:
    """Return the timed operations; each computes its result."""
    ds = dxr.open_disp(folder)
    chained = dxr.chain(dxr.mask(ds, "recommended")).disp.rereference("median")
    return {
        "open": lambda: dxr.open_disp(folder).displacement.isel(time=-1).compute(),
        "chain": lambda: dxr.chain(ds).displacement.isel(time=-1).compute(),
        "mask": lambda: dxr.mask(ds, "recommended").displacement.compute(),
        "rereference median": (
            lambda: dxr.rereference(dxr.chain(ds), "median").displacement.compute()
        ),
        "fit_velocity ols": lambda: dxr.fit_velocity(chained, "ols").velocity.compute(),
        "fit_velocity midas": (
            lambda: dxr.fit_velocity(chained, "midas").velocity.compute()
        ),
        "fit_velocity gls": lambda: dxr.fit_velocity(chained, "gls").velocity.compute(),
        "downsample 1 km": (
            lambda: dxr.downsample(
                chained, factor=33, layers=["displacement"]
            ).displacement.compute()
        ),
        "quadtree": (
            lambda: dxr.quadtree(
                chained, threshold=0.003, min_level=3, max_level=7
            ).points.displacement.compute()
        ),
        "denoise kalman": lambda: dxr.denoise(chained, "kalman").compute(),
    }


def run(size: str, trials: int, out: Path | None) -> pd.DataFrame:
    """Run the suite and return one row per operation and trial."""
    shape, epochs = SIZES[size]
    folder = Path(tempfile.mkdtemp()) / "granules"
    make_disp_stack(
        folder, "three_ministacks", shape=shape, epochs_per_ministack=epochs
    )
    ops = operations(folder)
    for fn in ops.values():
        fn()  # warm-up: numba compiles, files enter the page cache
    env = environment()
    started = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    rows = []
    for trial in range(trials):
        touch = page_touch_seconds()
        for name, fn in ops.items():
            start = time.perf_counter()
            fn()
            rows.append(
                {
                    "operation": name,
                    "trial": trial,
                    "seconds": time.perf_counter() - start,
                    "size": size,
                    "shape": f"{3 * epochs + 1}x{shape[0]}x{shape[1]}",
                    "load": os.getloadavg()[0],
                    "page_touch_s": touch,
                    "started": started,
                    **env,
                }
            )
    table = pd.DataFrame(rows)
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(out, index=False)
    return table


def summary(table: pd.DataFrame) -> pd.DataFrame:
    """Return the median, spread and count of each operation's trials."""
    grouped = table.groupby("operation", sort=False)["seconds"]
    return pd.DataFrame(
        {
            "median_s": grouped.median(),
            "min_s": grouped.min(),
            "max_s": grouped.max(),
            "spread": grouped.apply(
                lambda s: (
                    statistics.pstdev(s) / statistics.median(s) if len(s) > 1 else 0.0
                )
            ),
            "trials": grouped.size(),
        }
    ).round(3)


def main() -> None:
    """Parse the arguments, run the suite and print the summary."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--size", choices=sorted(SIZES), default="tile")
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--out", type=Path, help="CSV of the raw rows")
    args = parser.parse_args()
    out = args.out or Path(__file__).parent / "results" / (
        f"{dt.date.today()}_{socket.gethostname()}_{args.size}.csv"
    )
    table = run(args.size, args.trials, out)
    touch = table["page_touch_s"].max()
    print(f"{args.size}: {table['shape'].iloc[0]}, wrote {out}")
    if touch > 2.0:
        print(f"warning: touching 256 MB took {touch:.1f} s; the node is stalling")
    print(summary(table).to_string())


if __name__ == "__main__":
    main()
