"""Run the T29 split-metric comparison (T29.2, T29.3).

For each frame, cell size (1, 2 and 10 km) and method this downsamples the
study stack written by ``prepare.py`` and scores the result:

- ``grid``: the regular grid of :func:`disp_xr.downsample` at that size. Its
  number of cells with data is the point budget of the size;
- ``displacement``, ``velocity``, ``structure``: :func:`disp_xr.quadtree`
  with that split metric and the same budget (``n_points``), so the methods
  are compared at equal point counts.

Criteria written now: reconstruction error against the 30 m field (each
method's cell values painted back on the 30 m grid, nearest; RMSE over
reliable pixels) for displacement (RMS over epochs, and the last epoch) and
velocity, the cell count and the runtime. GNSS agreement per cell (T32) and
gradient preservation (T29.4) are added by their own steps.

Writes ``results/<frame>.csv``, one row per size and method. Run with
``pixi run -e test-py312 python studies/split_metric/run.py [FRAME ...]``.
"""

from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd
import xarray as xr
from prepare import OUT
from stations import HERE

import disp_xr as dxr

SIZES_KM = {1: 33, 2: 67, 10: 333}
METRICS = ("displacement", "velocity", "structure")
MIN_LEVEL, MAX_LEVEL = 3, 10


def grid_leaf_id(shape: tuple[int, int], factor: int) -> np.ndarray:
    """Return the cell index of each pixel for a regular grid of ``factor``."""
    rows = np.arange(shape[0]) // factor
    cols = np.arange(shape[1]) // factor
    n_cols = -(-shape[1] // factor)
    return rows[:, None] * n_cols + cols[None, :]


def paint_rmse(
    leaf_id: np.ndarray, values: np.ndarray, field: np.ndarray, reliable: np.ndarray
) -> float:
    """Return the RMSE of cell values painted on the grid against ``field``.

    ``values`` holds one value per cell; ``leaf_id`` -1 marks pixels in no
    cell, which count as missing.
    """
    painted = np.append(values, np.nan)[leaf_id]
    diff = (painted - field)[reliable]
    diff = diff[np.isfinite(diff)]
    return float(np.sqrt(np.mean(diff**2))) if diff.size else np.nan


def cell_mean(
    leaf_id: np.ndarray, field: np.ndarray, weight: np.ndarray, n: int
) -> np.ndarray:
    """Return the weighted mean of ``field`` in each of ``n`` cells."""
    ok = (leaf_id >= 0) & np.isfinite(field) & (weight > 0)
    ids = leaf_id[ok]
    sum_w = np.bincount(ids, weight[ok], minlength=n)
    sum_wv = np.bincount(ids, weight[ok] * field[ok], minlength=n)
    with np.errstate(invalid="ignore", divide="ignore"):
        return sum_wv / sum_w


def score_all(
    parts: list[dict],
    ds: xr.Dataset,
    velocity: np.ndarray,
    static: np.ndarray,
    reliable: np.ndarray,
) -> None:
    """Add the reconstruction errors to every partition, reading each epoch once."""
    per_epoch = np.full((len(parts), ds.sizes["time"]), np.nan)
    for t in range(ds.sizes["time"]):
        field = ds["displacement"].isel(time=t).values
        for i, part in enumerate(parts):
            per_epoch[i, t] = paint_rmse(
                part["leaf_id"], part["cell_disp"][t], field, reliable
            )
    for i, part in enumerate(parts):
        leaf_id = part.pop("leaf_id")
        part.pop("cell_disp")
        cell_vel = cell_mean(leaf_id, velocity, static, int(leaf_id.max()) + 1)
        part["disp_rmse_mm"] = 1e3 * float(np.sqrt(np.nanmean(per_epoch[i] ** 2)))
        part["disp_rmse_last_mm"] = 1e3 * per_epoch[i, -1]
        part["vel_rmse_mm_yr"] = 1e3 * paint_rmse(leaf_id, cell_vel, velocity, reliable)


def run_frame(frame: str) -> pd.DataFrame:
    """Score every size and method on one frame."""
    ds = xr.open_zarr(OUT / frame / "stack.zarr")
    velocity_da = xr.open_zarr(OUT / frame / "velocity.zarr")["velocity"]
    quality = [
        n
        for n in (
            "phase_similarity",
            "temporal_coherence",
            "timeseries_inversion_residuals",
        )
        if n in ds
    ]
    static = dxr.quality_weights(ds, quality).static_weight.fillna(0.0).values
    velocity = velocity_da.values
    reliable = (static > 0) & ds["displacement"].notnull().any("time").values
    parts = []
    for size_km, factor in SIZES_KM.items():
        t0 = time.perf_counter()
        coarse = dxr.downsample(ds, factor=factor, layers=["displacement"]).compute()
        cell_disp = coarse["displacement"].values.reshape(coarse.sizes["time"], -1)
        n_points = int(np.isfinite(cell_disp).any(axis=0).sum())
        parts.append(
            {
                "frame": frame,
                "size_km": size_km,
                "method": "grid",
                "n_points": n_points,
                "threshold": np.nan,
                "seconds": time.perf_counter() - t0,
                "levels": "",
                "leaf_id": grid_leaf_id(reliable.shape, factor).astype(np.int32),
                "cell_disp": cell_disp,
            }
        )
        print({k: v for k, v in parts[-1].items() if k[0] != "l" and k[0] != "c"})
        for metric in METRICS:
            t0 = time.perf_counter()
            tree = dxr.quadtree(
                ds,
                n_points=n_points,
                metric=metric,
                min_level=MIN_LEVEL,
                max_level=MAX_LEVEL,
                velocity=velocity_da if metric == "velocity" else None,
            )
            cell_disp = tree.points["displacement"].transpose("time", "point").values
            levels = tree.points["level"].to_series().value_counts().sort_index()
            parts.append(
                {
                    "frame": frame,
                    "size_km": size_km,
                    "method": metric,
                    "n_points": tree.points.sizes["point"],
                    "threshold": tree.points.attrs["quadtree_threshold"],
                    "seconds": time.perf_counter() - t0,
                    "levels": " ".join(f"{k}:{v}" for k, v in levels.items()),
                    "leaf_id": tree.leaf_id.values.astype(np.int32),
                    "cell_disp": cell_disp,
                }
            )
            print({k: v for k, v in parts[-1].items() if k[0] != "l" and k[0] != "c"})
    score_all(parts, ds, velocity, static, reliable)
    return pd.DataFrame(parts)


def main(names: list[str]) -> None:
    """Run every frame named on the command line."""
    out = HERE / "results"
    out.mkdir(exist_ok=True)
    for frame in names:
        run_frame(frame).to_csv(out / f"{frame}.csv", index=False)


if __name__ == "__main__":
    main(sys.argv[1:])
