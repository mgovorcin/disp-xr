"""Record what disp-xr did to a dataset, in ``attrs["history"]``.

Each operation appends one JSON line: the disp-xr version, the operation name
and its parameters. No timestamps, so the same inputs give the same output
(golden-dataset comparisons stay stable).

Examples
--------
>>> import xarray as xr
>>> ds = record(xr.Dataset(), "demo", factor=2)
>>> history(ds)[-1]["op"], history(ds)[-1]["params"]
('demo', {'factor': 2})

"""

from __future__ import annotations

import functools
import inspect
import json
from collections.abc import Callable
from typing import Any, TypeVar

import xarray as xr

from ._version import __version__

__all__ = ["history", "record", "recorded"]

XarrayObj = TypeVar("XarrayObj", xr.Dataset, xr.DataArray)


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (xr.Dataset, xr.DataArray)):
        return f"<{type(value).__name__} dims={dict(value.sizes)}>"
    return str(value)


def record(obj: XarrayObj, op: str, /, **params: Any) -> XarrayObj:
    """Return ``obj`` with one history line appended (the input is not changed)."""
    out = obj.copy(deep=False)
    out.attrs = dict(obj.attrs)
    line = json.dumps(
        {"disp_xr": __version__, "op": op, "params": _jsonable(params)},
        sort_keys=True,
    )
    previous = out.attrs.get("history", "")
    out.attrs["history"] = f"{previous}\n{line}" if previous else line
    return out


def history(obj: xr.Dataset | xr.DataArray) -> list[dict]:
    """Return the recorded disp-xr operations, oldest first.

    Lines that are not disp-xr records (other tools also write ``history``)
    are skipped.
    """
    out = []
    for line in str(obj.attrs.get("history", "")).splitlines():
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(entry, dict) and "op" in entry:
            out.append(entry)
    return out


def recorded(func: Callable[..., XarrayObj]) -> Callable[..., XarrayObj]:
    """Decorate a public operation so its result records the call.

    The first positional argument (the input data) is not recorded.
    """
    signature = inspect.signature(func)

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> XarrayObj:
        result = func(*args, **kwargs)
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        params = dict(list(bound.arguments.items())[1:])
        return record(result, func.__name__, **params)

    return wrapper
