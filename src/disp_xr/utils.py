"""Deprecated helpers kept for backward compatibility.

Functions moved to :mod:`disp_xr.geometry` and :mod:`disp_xr.reference`; the
old names still resolve here with a ``DeprecationWarning``.
"""

import importlib
import warnings
from typing import Any

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
