"""Deprecated helpers kept for backward compatibility.

Functions moved to :mod:`disp_xr.geometry` and :mod:`disp_xr.reference`; the
old names still resolve here with a ``DeprecationWarning``.
"""

import importlib
import warnings
from typing import Any

import numpy as np
import xarray as xr

_MOVED = {
    "latlon_to_utm": "disp_xr.geometry",
    "utm_to_latlon": "disp_xr.geometry",
    "get_extent": "disp_xr.geometry",
    "get_burst_ids": "disp_xr.geometry",
    "find_reference_point": "disp_xr.reference",
}


def __getattr__(name: str) -> Any:
    if name in _MOVED:
        target = _MOVED[name]
        warnings.warn(
            f"disp_xr.utils.{name} moved to {target}.{name}",
            DeprecationWarning,
            stacklevel=2,
        )
        return getattr(importlib.import_module(target), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def get_chunks_indices(xr_array: xr.Dataset) -> list:
    """Get the indices for chunked slices of an xarray Dataset.

    Determines the chunk boundaries of the given xarray and returns a list
    of slice objects representing these chunks.

    Parameters
    ----------
    xr_array : xr.Dataset
        The input xarray Dataset.

    Returns
    -------
    list
        A list of slice objects representing the chunked slices.

    """
    chunks = xr_array.chunks
    _, iy, ix = chunks["time"], chunks["y"], chunks["x"]

    idx = [sum(ix[:i]) for i in range(len(ix) + 1)]
    idy = [sum(iy[:i]) for i in range(len(iy) + 1)]

    slices = []

    for i in range(len(idy) - 1):  # Y-axis slices for idy
        for j in range(len(idx) - 1):  # X-axis slices for idx
            # Create a slice using the ranges of idt, idy, and idx
            # skip first date
            slice_ = np.s_[:, idy[i] : idy[i + 1], idx[j] : idx[j + 1]]
            slices.append(slice_)
    return slices
