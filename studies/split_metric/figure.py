"""Figure of the T29 partitions: 30 m velocity, regular grid and quadtree.

On a window of a study stack (unmasked arm), the regular 2 km grid and a
quadtree with the same number of cells (``structure`` metric by default),
each cell painted with the weighted mean of the 30 m velocity and outlined.

Run::

    DISP_XR_STUDY_VARIANT=unmasked pixi run -e test-py312 python figure.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from prepare import WORKERS, frame_out
from pyproj import Transformer
from run import MIN_LEVEL, cached_weights, cell_mean, grid_leaf_id, reliable_pixels

import disp_xr as dxr
from disp_xr.quadtree import MetricName

HERE = Path(__file__).parent
# (frame, title, west, south, east, north)
WINDOWS = [
    (
        "F11116",
        "creeping San Andreas near Slack Canyon",
        -120.95,
        35.95,
        -120.45,
        36.32,
    ),
    ("F08882", "northwest Harris County, Houston", -95.85, 29.75, -95.55, 30.05),
]
FACTOR = 67  # 2 km


def window(frame: str, bounds: tuple[float, float, float, float]):
    """Return the stack, velocity, weights and reliable mask in ``bounds``."""
    ds = xr.open_zarr(frame_out(frame) / "stack.zarr")
    velocity = xr.open_zarr(frame_out(frame) / "velocity.zarr")["velocity"]
    keep = xr.DataArray(
        reliable_pixels(ds, frame), dims=("y", "x"), coords={"y": ds.y, "x": ds.x}
    )
    weights, static = cached_weights(ds, frame, "default", {})
    to_grid = Transformer.from_crs("EPSG:4326", ds.rio.crs, always_xy=True)
    west, south, east, north = bounds
    (x0, x1), (y0, y1) = to_grid.transform([west, east], [south, north])
    sel = {"x": slice(x0, x1), "y": slice(y1, y0)}
    ds = ds.sel(sel)
    keep = keep.sel(sel)
    ds = ds.assign(displacement=ds["displacement"].where(keep))
    # In memory: the n_points search builds the tree several times, and a
    # lazy window would be read from Zarr each time.
    return (
        ds[["displacement"]].load(),
        velocity.sel(sel).where(keep).values,
        weights.sel(sel).load(),
        static.sel(sel).values,
        keep.values,
    )


def outline(ax, leaf_id: np.ndarray) -> None:
    """Draw each cell as a rectangle around its pixels (pixel coordinates)."""
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Rectangle

    rows, cols = np.nonzero(leaf_id >= 0)
    ids = leaf_id[rows, cols]
    n = int(ids.max()) + 1
    top = np.full(n, leaf_id.shape[0])
    left = np.full(n, leaf_id.shape[1])
    bottom = np.full(n, -1)
    right = np.full(n, -1)
    np.minimum.at(top, ids, rows)
    np.minimum.at(left, ids, cols)
    np.maximum.at(bottom, ids, rows)
    np.maximum.at(right, ids, cols)
    used = bottom >= 0
    boxes = [
        Rectangle((c0 - 0.5, r0 - 0.5), c1 - c0 + 1, r1 - r0 + 1)
        for r0, c0, r1, c1 in zip(top[used], left[used], bottom[used], right[used])
    ]
    ax.add_collection(
        PatchCollection(boxes, facecolor="none", edgecolor="0.15", linewidth=0.5)
    )


def partitions(
    frame: str, bounds: tuple[float, float, float, float], metric: MetricName
) -> dict[str, np.ndarray]:
    """Return the 30 m field and both partitions, cached next to the stack."""
    cache = frame_out(frame) / f"figure_{metric}.npz"
    if cache.exists():
        return dict(np.load(cache))
    ds, velocity, weights, static, keep = window(frame, bounds)
    reliable = keep & (static > 0)
    field = np.where(reliable, velocity, np.nan)
    grid_id = grid_leaf_id(field.shape, FACTOR)
    grid_vel = cell_mean(grid_id, field, static, int(grid_id.max()) + 1)
    tree = dxr.quadtree(
        ds,
        n_points=int(np.isfinite(grid_vel).sum()),
        metric=metric,
        min_level=MIN_LEVEL,
        max_level=8,
        weights=weights,
    )
    tree_id = np.asarray(tree.leaf_id.values, dtype=np.int64)
    out = {
        "field": field,
        "grid_id": grid_id,
        "grid_vel": grid_vel,
        "tree_id": tree_id,
        "tree_vel": cell_mean(tree_id, field, static, int(tree_id.max()) + 1),
        "tree_level": tree.points["level"].values,
    }
    np.savez(cache, **out)
    return out


def painted(ids: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Paint one value per cell back on the pixels (mm/yr); -1 is no cell."""
    return 1e3 * np.where(ids >= 0, np.append(values, np.nan)[ids], np.nan)


def main(metric: MetricName = "structure") -> None:
    """Draw one row of three panels per window."""
    fig, axes = plt.subplots(
        len(WINDOWS), 3, figsize=(15, 5 * len(WINDOWS)), layout="constrained"
    )
    for row, (frame, title, west, south, east, north) in zip(axes, WINDOWS):
        parts = partitions(frame, (west, south, east, north), metric)
        field = parts["field"]
        n_grid = int(np.isfinite(parts["grid_vel"]).sum())
        levels = dict(zip(*np.unique(parts["tree_level"], return_counts=True)))
        limit = float(np.nanpercentile(np.abs(1e3 * field), 98))
        panels = [
            ("30 m velocity (reliable pixels)", 1e3 * field, None),
            (
                f"regular 2 km grid, {n_grid} cells",
                painted(parts["grid_id"], parts["grid_vel"]),
                parts["grid_id"],
            ),
            (
                f"quadtree ({metric}), {len(parts['tree_level'])} cells",
                painted(parts["tree_id"], parts["tree_vel"]),
                parts["tree_id"],
            ),
        ]
        for ax, (name, image, ids) in zip(row, panels):
            shown = ax.imshow(
                image, cmap="RdBu", vmin=-limit, vmax=limit, interpolation="nearest"
            )
            if ids is not None:
                outline(ax, ids)
            ax.set_title(f"{frame}, {title}\n{name}", loc="left", fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
        fig.colorbar(shown, ax=row, label="LOS velocity (mm/yr)", shrink=0.8)
        print(f"{frame}: {n_grid} grid cells, quadtree cells per level {levels}")
    out = HERE.parents[1] / "docs" / "studies" / "img" / f"quadtree_{metric}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    print(out)


if __name__ == "__main__":
    from distributed import Client, LocalCluster

    # Processes: zarr decodes chunks in one thread per process, and the study
    # stacks hold one epoch per chunk.
    with (
        LocalCluster(
            n_workers=WORKERS, threads_per_worker=1, memory_limit="8GB"
        ) as cluster,
        Client(cluster),
    ):
        main(*sys.argv[1:])  # type: ignore[arg-type]
