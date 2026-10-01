"""Deprecated: moved to :mod:`disp_xr.export.raster`."""

import importlib
import warnings
from typing import Any

_new = importlib.import_module("disp_xr.export.raster")

warnings.warn(
    "disp_xr.io moved to disp_xr.export.raster",
    DeprecationWarning,
    stacklevel=2,
)


def __getattr__(name: str) -> Any:
    return getattr(_new, name)
