"""Deprecated: moved to :mod:`disp_xr.export.mintpy`."""

import importlib
import warnings
from typing import Any

_new = importlib.import_module("disp_xr.export.mintpy")

warnings.warn(
    "disp_xr.mintpy moved to disp_xr.export.mintpy",
    DeprecationWarning,
    stacklevel=2,
)


def __getattr__(name: str) -> Any:
    return getattr(_new, name)
