"""Check that a dataset has the structure disp-xr operations expect."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Union

import xarray as xr

from .geometry import grid

__all__ = ["DispValidationError", "validate_disp"]


class DispValidationError(ValueError):
    """A dataset is missing something a disp-xr operation needs."""


def validate_disp(
    obj: Union[xr.Dataset, xr.DataArray],
    *,
    variables: Sequence[str] = (),
    require_reference_time: bool = False,
    per_pixel: bool = False,
) -> None:
    """Raise if ``obj`` is not a usable DISP stack; list every problem found.

    Checks dims ``(time, y, x)``, monotonic ``x`` and ``y`` coordinates, a CRS,
    the requested ``variables`` and, optionally, the ``reference_time``
    coordinate. With ``per_pixel=True``, warns when ``time`` is split across
    dask chunks, since per-pixel operations then rechunk (slow on big frames).

    Raises
    ------
    DispValidationError
        If any check fails.

    Examples
    --------
    >>> dxr.validate_disp(ds, variables=["displacement"])  # doctest: +SKIP

    """
    problems = []
    missing_dims = [d for d in ("time", "y", "x") if d not in obj.dims]
    if missing_dims:
        problems.append(f"missing dims {missing_dims}")
    for dim in ("x", "y"):
        if dim in obj.coords and not (
            obj.indexes[dim].is_monotonic_increasing
            or obj.indexes[dim].is_monotonic_decreasing
        ):
            problems.append(f"{dim} coordinate is not monotonic")
    if not missing_dims:
        try:
            if grid(obj).crs is None:
                problems.append("no CRS (expected a spatial_ref coordinate)")
        except Exception as e:  # rioxarray raises several error types
            problems.append(f"cannot read the grid: {e}")
    names = obj.data_vars if isinstance(obj, xr.Dataset) else [obj.name]
    absent = [v for v in variables if v not in names]
    if absent:
        problems.append(f"missing variables {absent}")
    if require_reference_time and "reference_time" not in obj.coords:
        problems.append("missing reference_time coordinate")
    if problems:
        raise DispValidationError("Not a usable DISP stack: " + "; ".join(problems))

    if per_pixel:
        arrays = obj.data_vars.values() if isinstance(obj, xr.Dataset) else [obj]
        split = [
            str(a.name)
            for a in arrays
            if a.chunks is not None
            and "time" in a.dims
            and len(a.chunks[a.dims.index("time")]) > 1
        ]
        if split:
            warnings.warn(
                f"time is split across chunks in {split}; per-pixel operations will "
                "rechunk each time. Before several of them, rechunk once with "
                "disp_xr.rechunk(obj, 'timeseries').",
                stacklevel=2,
            )
