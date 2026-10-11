"""Build the study stacks of the T29 frames (T29.2, input step).

For each DISP-S1 frame in ``frames.yaml`` this follows ``disp_xr stack``
(product corrections, recommended mask, chaining, re-reference) on the
displacement and the three quality layers only, keeps every ``STRIDE``-th
epoch (cumulative displacement stays valid on a subset of epochs) and writes
``<out>/<frame>/stack.zarr`` in the maps layout the quadtree reads. It then
writes ``velocity.zarr``: an OLS velocity, the 30 m reference for the
velocity reconstruction error and the input of the velocity split metric.

Two departures from ``disp_xr stack``, both for speed on full frames:

- the reference is the median of a small box around a GNSS station near the
  frame center (``reference_box``), not ``auto``, whose stable-point search
  reads every quality layer of every epoch first. The study compares methods
  with each other and removes a plane before the GNSS comparison, so the
  reference does not change its results;
- dask runs on local processes, one thread each: reads of the NetCDF
  granules hold a process-wide HDF5 lock, so threads decompress one at a
  time.

The product's quality thresholds (``similarity_threshold`` and
``temporal_coherence_threshold`` on ``recommended_mask``, the default weight
floors) differ between granules; the most common values are recorded on the
stack as ``study_floors`` (``product_floors``).

Run with ``pixi run -e test-py312 python studies/split_metric/prepare.py
[FRAME ...]``; the output directory is ``DISP_XR_STUDY_OUT`` (default
``/u/aurora-r0/govorcin/01_OPERA/DISP/data/split_metric``).
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import xarray as xr
import yaml
from stations import HERE, frame_files

import disp_xr as dxr
from disp_xr.product import PRODUCTS

STRIDE = 3
LAYERS = [
    "displacement",
    "phase_similarity",
    "temporal_coherence",
    "timeseries_inversion_residuals",
    "recommended_mask",
    "water_mask",
]
# "masked": the recommended mask, as `disp_xr stack` does. "unmasked": only
# water is removed and the quality weights do the rest (the primary arm since
# 2026-10-05, the user's decision: downsample unmasked data, weighted by
# quality).
VARIANT = os.environ.get("DISP_XR_STUDY_VARIANT", "masked")
assert VARIANT in ("masked", "unmasked"), VARIANT
REFERENCE_HALF_BOX = 32  # pixels: a 2 km box
# Two frames can run side by side with DISP_XR_STUDY_WORKERS=16.
WORKERS = int(os.environ.get("DISP_XR_STUDY_WORKERS", "24"))
OUT = Path(
    os.environ.get(
        "DISP_XR_STUDY_OUT", "/u/aurora-r0/govorcin/01_OPERA/DISP/data/split_metric"
    )
)


def frame_out(frame: str) -> Path:
    """Return the folder of a frame's study inputs for ``VARIANT``."""
    return OUT / frame if VARIANT == "masked" else OUT / frame / VARIANT


def result_name(frame: str) -> str:
    """Return the result file stem of a frame for ``VARIANT``."""
    return frame if VARIANT == "masked" else f"{frame}_{VARIANT}"


def product_floors(files: list[Path]) -> dict[str, float]:
    """Return the most common quality thresholds recorded in the granules."""
    names = {
        "phase_similarity": "similarity_threshold",
        "temporal_coherence": "temporal_coherence_threshold",
    }
    seen: dict[str, Counter] = {layer: Counter() for layer in names}
    for path in files:
        with h5py.File(path, "r") as f:
            attrs = f["recommended_mask"].attrs
            for layer, key in names.items():
                value = attrs[key]
                seen[layer][
                    float(value.decode() if isinstance(value, bytes) else value)
                ] += 1
    return {layer: count.most_common(1)[0][0] for layer, count in seen.items()}


def reference_box(ds: xr.Dataset, frame: str) -> tuple[slice, slice, str]:
    """Return the reference box rows and columns, and what it is centered on.

    The used GNSS station nearest the frame center (from
    ``stations/<frame>_los.csv``) or, without one, the frame center.
    """
    ys, xs = ds["y"].values, ds["x"].values
    cy, cx = ys.mean(), xs.mean()
    where = "frame center"
    table = HERE / "stations" / f"{frame}_los.csv"
    if table.exists():
        stations = pd.read_csv(table)
        k = int(np.hypot(stations["x"] - cx, stations["y"] - cy).argmin())
        cy, cx = stations["y"][k], stations["x"][k]
        where = f"GNSS station {stations['id'][k]}"
    row, col = int(np.abs(ys - cy).argmin()), int(np.abs(xs - cx).argmin())
    h = REFERENCE_HALF_BOX
    return (
        slice(max(row - h, 0), row + h + 1),
        slice(max(col - h, 0), col + h + 1),
        where,
    )


def rereference_box(ds: xr.Dataset, frame: str) -> xr.Dataset:
    """Subtract the per-epoch median of the reference box (reads only the box)."""
    rows, cols, where = reference_box(ds, frame)
    box = ds["displacement"].isel(y=rows, x=cols).chunk({"y": -1, "x": -1})
    offset = box.median(("y", "x"), skipna=True)
    out = ds.assign(displacement=ds["displacement"] - offset)
    out["displacement"].attrs = ds["displacement"].attrs
    out.attrs["reference"] = json.dumps(
        {
            "method": "box median",
            "rows": [rows.start, rows.stop],
            "cols": [cols.start, cols.stop],
            "center": where,
        }
    )
    return out


def build(frame: str, entry: dict) -> None:
    """Write the stack and the velocity of one frame."""
    out = frame_out(frame)
    out.mkdir(parents=True, exist_ok=True)
    files = frame_files(frame, entry)
    floors = product_floors(files)
    ds = dxr.open_disp(files, layers=LAYERS, corrections=True)
    spec = PRODUCTS[ds.attrs["product"]]
    which = [
        c for c in spec.applicable_corrections if c not in spec.pre_applied_corrections
    ]
    if which:
        ds = dxr.correct(ds, which)
    ds = (
        dxr.mask(ds, "recommended")
        if VARIANT == "masked"
        else dxr.mask(ds, None, water=True)
    )
    ds = dxr.chain(ds)
    ds = rereference_box(ds, frame)
    keep = list(range(0, ds.sizes["time"], STRIDE))
    if keep[-1] != ds.sizes["time"] - 1:
        keep.append(ds.sizes["time"] - 1)
    keep_vars = [n for n in LAYERS[:4] if n in ds]  # masks are applied, not kept
    ds = ds.isel(time=keep)[keep_vars]
    ds.attrs["study_stride"] = STRIDE
    ds.attrs["study_floors"] = json.dumps(floors)
    print(f"{frame}: floors {floors}, reference {ds.attrs['reference']}", flush=True)
    t0 = time.perf_counter()
    dxr.to_zarr(ds, out / "stack.zarr", preset="maps", mode="w")
    print(f"{frame}: stack, {len(keep)} epochs, {time.perf_counter() - t0:.0f} s")

    stack = xr.open_zarr(out / "stack.zarr")
    t0 = time.perf_counter()
    vel = dxr.fit_velocity(stack, method="ols")
    vel.to_zarr(out / "velocity.zarr", mode="w")
    print(f"{frame}: velocity, {time.perf_counter() - t0:.0f} s")


def main(names: list[str]) -> None:
    """Build every frame named on the command line, or all of them."""
    from distributed import Client, LocalCluster

    frames = yaml.safe_load((HERE / "frames.yaml").read_text())["disp_s1"]
    with (
        LocalCluster(
            n_workers=WORKERS, threads_per_worker=1, memory_limit="8GB"
        ) as cluster,
        Client(cluster),
    ):
        for frame in names or list(frames):
            build(frame, frames[frame])


if __name__ == "__main__":
    main(sys.argv[1:])
