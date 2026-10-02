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

from typing import Any, Optional, Union

import xarray as xr

from ._chunks import ChunkSpec, rechunk
from ._history import history
from .corrections import correct, remove_plate_motion
from .geometry import Grid, grid
from .plot import plot_map, plot_point
from .quality import mask, quality_summary
from .reference import rereference
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

    def rechunk(self, chunks: ChunkSpec = "timeseries") -> Any:
        """Rechunk for one kind of work (see :func:`disp_xr.rechunk`)."""
        return rechunk(self._obj, chunks)

    def plot_map(self, time: Any = None, **kwargs: Any) -> Any:
        """Map one epoch (see :func:`disp_xr.plot.plot_map`)."""
        return plot_map(self._obj, time, **kwargs)

    def plot_point(
        self, lon: Optional[float] = None, lat: Optional[float] = None, **kwargs: Any
    ) -> Any:
        """Plot a point time series (see :func:`disp_xr.plot.plot_point`)."""
        return plot_point(self._obj, lon, lat, **kwargs)

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

    def mask(self, preset: str | None = "recommended", **criteria: Any) -> xr.Dataset:
        """Mask displacement by quality (see :func:`disp_xr.quality.mask`)."""
        return mask(self._obj, preset, **criteria)

    def correct(self, *args: Any, **kwargs: Any) -> xr.Dataset:
        """Subtract corrections (see :func:`disp_xr.corrections.correct`)."""
        return correct(self._obj, *args, **kwargs)

    def remove_plate_motion(self, static: xr.Dataset, plate: str) -> xr.Dataset:
        """Remove rigid plate motion (see :func:`disp_xr.remove_plate_motion`)."""
        return remove_plate_motion(self._obj, static, plate)

    def rereference(self, method: str = "auto", **kwargs: Any) -> xr.Dataset:
        """Re-reference displacement (see :func:`disp_xr.reference.rereference`)."""
        return rereference(self._obj, method, **kwargs)

    def quality_summary(self, **kwargs: Any) -> xr.Dataset:
        """Per-pixel quality summary (see :func:`disp_xr.quality.quality_summary`)."""
        return quality_summary(self._obj, **kwargs)


@xr.register_dataarray_accessor("disp")
class DispArrayAccessor(_Common):
    """``da.disp``: disp-xr operations on a DataArray."""

    def __init__(self, da: xr.DataArray) -> None:
        """Wrap ``da``; xarray creates the accessor on first ``da.disp`` use."""
        self._obj = da
