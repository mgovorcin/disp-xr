"""Deprecated: moved to :mod:`disp_xr.search` (needs the aws extra)."""

import importlib
import warnings
from typing import Any

_new = importlib.import_module("disp_xr.search")

warnings.warn(
    "disp_xr.download moved to disp_xr.search",
    DeprecationWarning,
    stacklevel=2,
)


def __getattr__(name: str) -> Any:
    return getattr(_new, name)
