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
