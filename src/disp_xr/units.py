"""Length units for DISP layers.

disp-xr works in meters (DISP products store ``units = "meters"``) and converts
only for output. Velocities (``m/yr``) are added with the time-series fits.

Examples
--------
>>> import xarray as xr
>>> da = xr.DataArray([0.01], dims="t", attrs={"units": "meters"})
>>> mm = to_units(da, "mm")
>>> float(mm[0]), mm.attrs["units"]
(10.0, 'mm')

"""

from __future__ import annotations

from typing import TypeVar

import xarray as xr

from ._history import recorded

__all__ = ["to_units"]

XarrayObj = TypeVar("XarrayObj", xr.Dataset, xr.DataArray)

_SCALE = {"m": 1.0, "cm": 1e-2, "mm": 1e-3}
_ALIASES = {"m": "m", "meter": "m", "meters": "m", "metre": "m", "metres": "m"}
_ALIASES.update({"cm": "cm", "centimeters": "cm", "mm": "mm", "millimeters": "mm"})


def _canonical(unit: str) -> str | None:
    return _ALIASES.get(unit.strip().lower())


def _convert(da: xr.DataArray, target: str) -> xr.DataArray:
    source = _canonical(str(da.attrs.get("units", "")))
    if source is None:
        return da
    out = da * (_SCALE[source] / _SCALE[target])
    out.attrs = {**da.attrs, "units": target}
    out.encoding = da.encoding
    return out


@recorded
def to_units(obj: XarrayObj, unit: str) -> XarrayObj:
    """Convert every length variable of ``obj`` to ``unit`` (m, cm or mm).

    Variables whose ``units`` attribute is not a length (``unitless``,
    ``radians``) are returned unchanged.

    Raises
    ------
    ValueError
        If ``unit`` is not a supported length unit.

    Examples
    --------
    >>> import numpy as np, xarray as xr
    >>> da = xr.DataArray(np.array([0.01]), attrs={"units": "meters"})
    >>> float(to_units(da, "mm")[0])
    10.0

    """
    target = _canonical(unit)
    if target is None:
        raise ValueError(f"Unsupported unit {unit!r}; use one of {sorted(_SCALE)}")
    if isinstance(obj, xr.DataArray):
        return _convert(obj, target)
    out = obj.copy()
    for name in obj.data_vars:
        out[name] = _convert(obj[name], target)
    return out
