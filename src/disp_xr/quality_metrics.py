"""Deprecated: moved to :mod:`disp_xr.quality`."""

import importlib
import warnings
from typing import Any

_new = importlib.import_module("disp_xr.quality")

warnings.warn(
    "disp_xr.quality_metrics moved to disp_xr.quality",
    DeprecationWarning,
    stacklevel=2,
)


def __getattr__(name: str) -> Any:
    return getattr(_new, name)
