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
from prepare import frame_out
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


def outline(ax, leaf_id: np.ndarray, extent: list[float]) -> None:
    """Draw cell boundaries: pixels whose right or lower neighbor is another cell."""
    edge = np.zeros(leaf_id.shape, dtype=bool)
    edge[:, :-1] |= leaf_id[:, :-1] != leaf_id[:, 1:]
    edge[:-1, :] |= leaf_id[:-1, :] != leaf_id[1:, :]
    ax.imshow(
        np.where(edge, 1.0, np.nan),
        extent=extent,
        cmap="Greys",
        vmin=0,
        vmax=1,
        interpolation="nearest",
    )


def main(metric: MetricName = "structure") -> None:
    """Draw one row of three panels per window."""
    fig, axes = plt.subplots(
        len(WINDOWS), 3, figsize=(15, 5 * len(WINDOWS)), layout="constrained"
    )
    for row, (frame, title, west, south, east, north) in zip(axes, WINDOWS):
        ds, velocity, weights, static, keep = window(frame, (west, south, east, north))
        reliable = keep & (static > 0)
        field = np.where(reliable, velocity, np.nan)
        grid_id = grid_leaf_id(field.shape, FACTOR)
        grid_vel = cell_mean(grid_id, field, static, int(grid_id.max()) + 1)
        n_points = int(np.isfinite(grid_vel).sum())
        tree = dxr.quadtree(
            ds,
            n_points=n_points,
            metric=metric,
            min_level=MIN_LEVEL,
            max_level=8,
            weights=weights,
        )
        tree_id = np.asarray(tree.leaf_id.values, dtype=np.int64)
        tree_vel = cell_mean(tree_id, field, static, int(tree_id.max()) + 1)
        extent = [float(ds.x[0]), float(ds.x[-1]), float(ds.y[-1]), float(ds.y[0])]
        limit = float(np.nanpercentile(np.abs(1e3 * field), 98))
        style = {"cmap": "RdBu", "vmin": -limit, "vmax": limit, "extent": extent}
        panels = [
            ("30 m velocity (reliable pixels)", 1e3 * field, None),
            (
                f"regular 2 km grid, {n_points} cells",
                1e3 * np.append(grid_vel, np.nan)[grid_id],
                grid_id,
            ),
            (
                f"quadtree ({metric}), {int(tree.points.sizes['point'])} cells",
                1e3
                * np.where(tree_id >= 0, np.append(tree_vel, np.nan)[tree_id], np.nan),
                tree_id,
            ),
        ]
        for ax, (name, image, ids) in zip(row, panels):
            shown = ax.imshow(image, interpolation="nearest", **style)
            if ids is not None:
                outline(ax, ids, extent)
            ax.set_title(f"{frame}, {title}\n{name}", loc="left", fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
        fig.colorbar(shown, ax=row, label="LOS velocity (mm/yr)", shrink=0.8)
        print(f"{frame}: {n_points} cells, quadtree levels:")
        print(tree.points["level"].to_series().value_counts().sort_index().to_dict())
    out = HERE.parents[1] / "docs" / "studies" / "img" / f"quadtree_{metric}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    print(out)


if __name__ == "__main__":
    main(*sys.argv[1:])  # type: ignore[arg-type]
