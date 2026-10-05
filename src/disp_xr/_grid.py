"""Tag map layers with their grid mapping (CF ``grid_mapping``)."""

from __future__ import annotations

import xarray as xr


def with_grid_mapping(ds: xr.Dataset) -> xr.Dataset:
    """Set ``grid_mapping`` on every ``(y, x)`` variable when ``spatial_ref`` is there.

    Readers that look for the CRS per variable (GDAL through rioxarray,
    xpublish-tiles) find it only through this attribute.

    Examples
    --------
    >>> import numpy as np
    >>> ds = xr.Dataset(
    ...     {"v": (("y", "x"), np.zeros((2, 2)))}, coords={"spatial_ref": 0}
    ... )
    >>> with_grid_mapping(ds)["v"].attrs["grid_mapping"]
    'spatial_ref'

    """
    if "spatial_ref" not in ds.coords:
        return ds
    for var in ds.data_vars.values():
        if {"y", "x"} <= set(var.dims):
            var.attrs.setdefault("grid_mapping", "spatial_ref")
    return ds


def coordinate_grids(ds: xr.Dataset, like: xr.DataArray) -> tuple[xr.DataArray, ...]:
    """Return the ``x`` and ``y`` coordinates broadcast to 2D, chunked like ``like``.

    Lazy when ``like`` is: a frame-sized NumPy grid would be embedded in the
    task graph (about 500 MB each on a DISP-S1 frame) and shipped to the
    scheduler.
    """
    x = ds["x"].astype("float64")
    y = ds["y"].astype("float64")
    if like.chunks is not None:
        chunks = dict(zip(like.dims, like.chunks))
        x = x.chunk({"x": chunks.get("x", -1)})
        y = y.chunk({"y": chunks.get("y", -1)})
    xx, yy = xr.broadcast(x, y)
    return xx, yy
