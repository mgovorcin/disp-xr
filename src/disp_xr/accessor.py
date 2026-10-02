"""The ``.disp`` accessor on xarray objects.

It only delegates to disp-xr functions, so ``ds.disp.stack()`` and
``dxr.stack_disp(ds)`` are the same operation. Registered when ``disp_xr`` is
imported.

Examples
--------
>>> import disp_xr  # noqa: F401
>>> import xarray as xr
>>> hasattr(xr.Dataset(), "disp")
True

"""

from __future__ import annotations

from typing import Any, Union

import xarray as xr

from ._history import history
from .geometry import Grid, grid
from .stack import stack_disp
from .units import to_units
from .validate import validate_disp

__all__ = ["DispArrayAccessor", "DispDatasetAccessor"]


class _Common:
    _obj: Union[xr.Dataset, xr.DataArray]

    @property
    def history(self) -> list[dict]:
        """disp-xr operations recorded on this object, oldest first."""
        return history(self._obj)

    @property
    def grid(self) -> Grid:
        """CRS, transform and ``(y, x)`` shape."""
        return grid(self._obj)

    def to_units(self, unit: str) -> Any:
        """Convert length variables to ``unit`` (see :func:`disp_xr.units.to_units`)."""
        return to_units(self._obj, unit)

    def validate(self, **kwargs: Any) -> Any:
        """Raise if the object is not a usable DISP stack; else return it.

        See :func:`disp_xr.validate.validate_disp` for the keyword arguments.
        """
        validate_disp(self._obj, **kwargs)
        return self._obj


@xr.register_dataset_accessor("disp")
class DispDatasetAccessor(_Common):
    """``ds.disp``: disp-xr operations on a Dataset."""

    def __init__(self, ds: xr.Dataset) -> None:
        """Wrap ``ds``; xarray creates the accessor on first ``ds.disp`` use."""
        self._obj = ds

    def stack(self, **kwargs: Any) -> xr.Dataset:
        """Chain ministacks (see :func:`disp_xr.stack_disp`)."""
        return stack_disp(self._obj, **kwargs)


@xr.register_dataarray_accessor("disp")
class DispArrayAccessor(_Common):
    """``da.disp``: disp-xr operations on a DataArray."""

    def __init__(self, da: xr.DataArray) -> None:
        """Wrap ``da``; xarray creates the accessor on first ``da.disp`` use."""
        self._obj = da
