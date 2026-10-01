"""Deprecated: moved to :mod:`disp_xr.geometry.static` (needs the aws extra)."""

import importlib
import warnings
from typing import Any

_new = importlib.import_module("disp_xr.geometry.static")

warnings.warn(
    "disp_xr.static_layers moved to disp_xr.geometry.static",
    DeprecationWarning,
    stacklevel=2,
)


def __getattr__(name: str) -> Any:
    return getattr(_new, name)
