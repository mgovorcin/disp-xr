r"""Memory and graph size of the core operations on a full frame (plan task T37.3).

Runs each core operation on a real DISP-S1 frame under a dask
``LocalCluster`` with the memory budget of ``docs/performance.md`` (8 GB per
worker) and records, per operation:

- the time to build the lazy result and its task count (a build that reads
  data shows up as a long build);
- the wall time of the compute;
- the peak resident memory of every worker (sampled every 0.1 s) and of the
  coordinator, which with ``LocalCluster`` also runs the scheduler;
- worker log lines that report pausing, spilling or a restart, and whether
  a worker was replaced.

Operations whose result is a full cube are reduced with ``sum`` so the
result stays small; the reduction adds one small task per chunk. Timings on
a shared node are indicative only; the memory figures are not affected.

Run::

    pixi run -e test python benchmarks/profile_frame.py GRANULE_DIR \\
        --workers 16 --threads 2 --memory 8GB --out frame_profile.csv

``--read-block`` sets ``chunks.read_block`` for the whole run. ``--store``
profiles the recommended full-frame path (issue #93): the masked, chained,
re-referenced cube is written once to a Zarr store (operation ``write
store``, skipped when the store exists) and the processing operations read
from it instead of the granules.
"""

from __future__ import annotations

import argparse
import statistics
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd
import psutil
import xarray as xr
from distributed import Client, LocalCluster

import disp_xr as dxr

LOG_FLAGS = ("Pausing", "spill", "Restarting", "memory limit", "Unmanaged memory")


class PeakSampler:
    """Track the peak resident memory of this process in a thread."""

    def __init__(self, interval: float = 0.1):
        """Start sampling every ``interval`` seconds."""
        self.interval = interval
        self.process = psutil.Process()
        self.peak = self.process.memory_info().rss
        self._stop = threading.Event()
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self) -> None:
        while not self._stop.is_set():
            self.peak = max(self.peak, self.process.memory_info().rss)
            time.sleep(self.interval)

    def reset(self) -> int:
        """Return the peak since the last reset and start again."""
        peak, self.peak = self.peak, self.process.memory_info().rss
        return peak


def _start_sampler(dask_worker) -> None:
    dask_worker._peak_sampler = PeakSampler()


def _reset_sampler(dask_worker) -> int:
    sampler = getattr(dask_worker, "_peak_sampler", None)
    if sampler is None:  # a worker that replaced one that died
        _start_sampler(dask_worker)
        return 0
    return sampler.reset()


class _Call:
    """A side-effecting call profiled like a lazy result (one graph, one run)."""

    def __init__(self, graph_of: Any, run: Callable[[], None]):
        self._graph_of, self._run = graph_of, run

    def __dask_graph__(self):
        return self._graph_of.__dask_graph__()

    def compute(self) -> None:
        self._run()


def _processing(chained: xr.Dataset) -> dict[str, Callable[[], Any]]:
    """Operations on a masked, chained, re-referenced cube."""
    return {
        "fit_velocity ols": lambda: dxr.fit_velocity(chained, "ols").velocity,
        "fit_velocity midas": lambda: dxr.fit_velocity(chained, "midas").velocity,
        "fit_velocity gls": lambda: dxr.fit_velocity(chained, "gls").velocity,
        "downsample 1 km": lambda: (
            dxr.downsample(chained, factor=33, layers=["displacement"]).displacement
        ),
        "quadtree": lambda: (
            dxr.quadtree(
                chained, threshold=0.003, min_level=3, max_level=7
            ).points.displacement
        ),
        "denoise kalman": lambda: dxr.denoise(chained, "kalman").displacement.sum(),
    }


def store_operations(folder: Path, store: Path) -> dict[str, Callable[[], Any]]:
    """Write the chained cube once (unless the store exists), then read it."""
    ops: dict[str, Callable[[], Any]] = {}
    if not store.exists():
        ds = dxr.open_disp(folder)
        chained = dxr.chain(dxr.mask(ds, "recommended")).disp.rereference("median")
        ops["write store"] = lambda: _Call(chained, lambda: dxr.to_zarr(chained, store))
    # opened lazily, after the write has run
    stored = {}

    def from_store(name: str) -> Callable[[], Any]:
        def build():
            if "ds" not in stored:
                stored["ds"] = dxr.open_zarr(store)
            return _processing(stored["ds"])[name]()

        return build

    ops.update({name: from_store(name) for name in _processing(xr.Dataset())})
    return ops


def operations(folder: Path) -> dict[str, Callable[[], Any]]:
    """Return the lazy builders of each operation's (small) result."""
    ds = dxr.open_disp(folder)
    chained = dxr.chain(dxr.mask(ds, "recommended")).disp.rereference("median")
    return {
        "open": lambda: dxr.open_disp(folder).displacement.isel(time=-1),
        "chain": lambda: dxr.chain(ds).displacement.isel(time=-1),
        "mask": lambda: dxr.mask(ds, "recommended").displacement.sum(),
        "rereference median": (
            lambda: dxr.rereference(dxr.chain(ds), "median").displacement.sum()
        ),
        **_processing(chained),
    }


def flagged_lines(client: Client, seen: set[str]) -> list[str]:
    """Return worker log lines with a memory flag not reported before."""
    lines = []
    for logs in client.get_worker_logs().values():
        for _, line in logs:
            if line not in seen and any(f in line for f in LOG_FLAGS):
                seen.add(line)
                lines.append(line)
    return lines


def profile(
    folder: Path, client: Client, only: list[str] | None, store: Path | None = None
) -> pd.DataFrame:
    """Build and compute each operation; return one row per operation."""
    coordinator = PeakSampler()
    client.run(_start_sampler)
    seen: set[str] = set()
    rows = []
    builders = operations(folder) if store is None else store_operations(folder, store)
    for name, build in builders.items():
        if only and name not in only:
            continue
        workers_before = set(client.scheduler_info()["workers"])
        client.run(_reset_sampler)
        coordinator.reset()
        start = time.perf_counter()
        result = build()
        tasks = len(result.__dask_graph__())
        build_s = time.perf_counter() - start
        start = time.perf_counter()
        result.compute()
        compute_s = time.perf_counter() - start
        peaks = [p / 1e9 for p in client.run(_reset_sampler).values()]
        flags = flagged_lines(client, seen)
        workers_after = set(client.scheduler_info()["workers"])
        row = {
            "operation": name,
            "build_s": round(build_s, 1),
            "tasks": tasks,
            "compute_s": round(compute_s, 1),
            "worker_peak_gb_max": round(max(peaks), 2),
            "worker_peak_gb_median": round(statistics.median(peaks), 2),
            "coordinator_peak_gb": round(coordinator.reset() / 1e9, 2),
            "flagged_log_lines": len(flags),
            "workers_replaced": len(workers_before - workers_after),
        }
        rows.append(row)
        print(row, flush=True)
        for line in flags[:5]:
            print("   ", line[:200], flush=True)
    return pd.DataFrame(rows)


def main() -> None:
    """Start the cluster, profile, print and write the rows."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("granules", type=Path, help="folder of one frame's granules")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--memory", default="8GB", help="memory limit per worker")
    parser.add_argument("--only", nargs="*", help="operations to run")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--read-block", type=int, help="chunks.read_block")
    parser.add_argument("--store", type=Path, help="Zarr store of the chained cube")
    args = parser.parse_args()
    settings = {} if args.read_block is None else {"chunks.read_block": args.read_block}
    with (
        dxr.config.set(settings),
        LocalCluster(
            n_workers=args.workers,
            threads_per_worker=args.threads,
            memory_limit=args.memory,
        ) as cluster,
        Client(cluster) as client,
    ):
        print(f"dashboard {client.dashboard_link}", flush=True)
        table = profile(args.granules, client, args.only, args.store)
    table.to_csv(args.out, index=False)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
