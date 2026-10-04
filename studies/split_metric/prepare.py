"""Build the study stacks of the T29 frames (T29.2, input step).

For each DISP-S1 frame in ``frames.yaml`` this follows ``disp_xr stack``
(product corrections, recommended mask, chaining, re-reference) on the
displacement and the three quality layers only, keeps every ``STRIDE``-th
epoch (cumulative displacement stays valid on a subset of epochs) and writes
``<out>/<frame>/stack.zarr`` in the maps layout the quadtree reads. It then
writes ``velocity.zarr``: an OLS velocity, the 30 m reference for the
velocity reconstruction error and the input of the velocity split metric.

Run with ``pixi run -e test-py312 python studies/split_metric/prepare.py
[FRAME ...]``; the output directory is ``DISP_XR_STUDY_OUT`` (default
``/u/aurora-r0/govorcin/01_OPERA/DISP/data/split_metric``).
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import xarray as xr
import yaml
from stations import HERE, frame_dir

import disp_xr as dxr
from disp_xr.product import PRODUCTS

STRIDE = 3
LAYERS = [
    "displacement",
    "phase_similarity",
    "temporal_coherence",
    "timeseries_inversion_residuals",
    "recommended_mask",
    # Read by the "auto" reference (stable point search), dropped after it.
    "persistent_scatterer_mask",
]
OUT = Path(
    os.environ.get(
        "DISP_XR_STUDY_OUT", "/u/aurora-r0/govorcin/01_OPERA/DISP/data/split_metric"
    )
)


def build(frame: str, entry: dict) -> None:
    """Write the stack and the velocity of one frame."""
    out = OUT / frame
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(frame_dir(frame, entry).glob("*.nc"))
    ds = dxr.open_disp(files, layers=LAYERS, corrections=True)
    spec = PRODUCTS[ds.attrs["product"]]
    which = [
        c for c in spec.applicable_corrections if c not in spec.pre_applied_corrections
    ]
    if which:
        ds = dxr.correct(ds, which)
    ds = dxr.mask(ds, "recommended")
    ds = dxr.stack_disp(ds)
    ds = dxr.rereference(ds, "auto")
    keep = list(range(0, ds.sizes["time"], STRIDE))
    if keep[-1] != ds.sizes["time"] - 1:
        keep.append(ds.sizes["time"] - 1)
    keep_vars = [n for n in LAYERS[:4] if n in ds]
    ds = ds.isel(time=keep)[keep_vars]
    ds.attrs["study_stride"] = STRIDE
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
    frames = yaml.safe_load((HERE / "frames.yaml").read_text())["disp_s1"]
    for frame in names or list(frames):
        build(frame, frames[frame])


if __name__ == "__main__":
    main(sys.argv[1:])
