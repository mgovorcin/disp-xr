"""Dask chunk sizes for DISP stacks.

Chunks are sized from a byte target (``config["chunks.target_mb"]``), not a
fixed shape, and aligned to the 256 x 256 HDF5 chunks of DISP granules so a dask
chunk never reads part of a file chunk. Three presets:

``"native"`` (the :func:`disp_xr.open_disp` default)
    One epoch and one read block (``config["chunks.read_block"]``, 1,024 x
    1,024) per chunk: the layout of the files. Every selection reads only the
    epochs and blocks it needs, so maps, points and chaining stay cheap.
``"timeseries"``
    The whole time axis in each chunk, for per-pixel work (fits, filters).
    Functions that need it rechunk to it themselves (:func:`time_contiguous`,
    which merges time within the input's spatial blocks instead of
    reshuffling them).
``"maps"``
    One epoch per chunk, sized to the byte target, for whole-map work.

:func:`rechunk` switches a dataset between them when one kind of access will be
repeated, for example ``ds.disp.rechunk("timeseries")`` before many point reads
of a store that fits in memory.

Clamping is adapted from opera-utils (src/opera_utils/disp/_utils.py,
``_clamp_chunk_dict``, commit 97f77d9), Copyright (c) 2022 California Institute
of Technology, BSD-3-Clause. See THIRD_PARTY_NOTICES.md.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Mapping
from typing import Any, Literal, Optional, Union

import numpy as np
import xarray as xr

from .config import config

# Spatial chunk size of DISP granules (both productions, both sensors).
FILE_CHUNK = 256
MIN_CHUNK = 32

Preset = Literal["native", "timeseries", "maps"]
PRESETS = ("native", "timeseries", "maps")
ChunkSpec = Union[Preset, Mapping[str, int], None]


def suggest_chunks(
    shape: tuple[int, int, int],
    dtype: np.dtype | str = "float32",
    layout: Preset = "timeseries",
    target_mb: float | None = None,
) -> dict[str, int]:
    """Return ``{"time", "y", "x"}`` chunk sizes for a ``(time, y, x)`` stack.

    Parameters
    ----------
    shape : tuple[int, int, int]
        ``(time, y, x)`` size of the stack.
    dtype : numpy dtype
        Data type, for the byte count.
    layout : {"native", "timeseries", "maps"}
        File layout (one epoch, one read block), whole time axis per chunk,
        or one epoch per chunk.
    target_mb : float, optional
        Target chunk size in MB; defaults to ``config["chunks.target_mb"]``.

    Returns
    -------
    dict[str, int]
        Chunk sizes, square in space, clamped to ``shape``.

    Examples
    --------
    >>> suggest_chunks((300, 7000, 9000), "float32", "timeseries")
    {'time': 300, 'y': 256, 'x': 256}
    >>> suggest_chunks((300, 7000, 9000), "float32", "maps")
    {'time': 1, 'y': 4864, 'x': 4864}
    >>> suggest_chunks((300, 7000, 9000), "float32", "native")
    {'time': 1, 'y': 1024, 'x': 1024}

    """
    nt, ny, nx = shape
    if layout not in PRESETS:
        raise ValueError(f"layout must be one of {PRESETS}, not {layout!r}")
    if layout == "native":
        block = int(config.get("chunks.read_block"))
        return clamp_chunks({"time": 1, "y": block, "x": block}, shape)
    target = (target_mb or config.get("chunks.target_mb")) * 1e6
    itemsize = np.dtype(dtype).itemsize
    depth = nt if layout == "timeseries" else 1
    side = math.sqrt(target / (depth * itemsize))
    step = FILE_CHUNK if side >= FILE_CHUNK else MIN_CHUNK
    side = max(MIN_CHUNK, int(side // step) * step)
    return clamp_chunks({"time": depth, "y": side, "x": side}, shape)


def clamp_chunks(
    chunks: Mapping[str, int] | None, shape: tuple[int, int, int]
) -> dict[str, int]:
    """Clamp ``(time, y, x)`` chunk sizes to the array shape; ``-1`` means all."""
    out = {**(chunks or {})}
    for dim, size in zip(("time", "y", "x"), shape):
        requested = out.get(dim, size)
        out[dim] = size if requested == -1 else min(requested, size)
    return out


def resolve_chunks(
    chunks: ChunkSpec, shape: tuple[int, int, int], dtype: np.dtype | str
) -> dict[str, int]:
    """Turn a preset name, a dict or None (``"native"``) into chunk sizes."""
    if chunks is None or isinstance(chunks, str):
        return suggest_chunks(shape, dtype, chunks or "native")  # type: ignore[arg-type]
    return clamp_chunks(chunks, shape)


def _stack_shape(obj: Any) -> tuple[tuple[int, int, int], np.dtype]:
    sizes = obj.sizes
    shape = (sizes.get("time", 1), sizes.get("y", 1), sizes.get("x", 1))
    arrays = [obj] if not hasattr(obj, "data_vars") else list(obj.data_vars.values())
    floats = [a.dtype for a in arrays if a.dtype.kind == "f"] or [np.dtype("float32")]
    return shape, max(floats, key=lambda d: d.itemsize)


def rechunk(obj: Any, chunks: Optional[ChunkSpec] = None) -> Any:
    """Return ``obj`` (Dataset or DataArray) with dask chunks for one kind of work.

    Parameters
    ----------
    obj : xr.Dataset or xr.DataArray
        DISP data with ``time``, ``y``, ``x`` dimensions (any subset).
    chunks : {"native", "timeseries", "maps"} or dict
        A preset (see the module docstring) or explicit ``{"time", "y", "x"}``
        sizes, ``-1`` for a whole axis. Required: leaving it out still means
        ``"timeseries"`` but warns, and is an error from 1.0 (so every call
        says which layout it asks for).

    Returns
    -------
    Same type as ``obj``, lazily rechunked. Nothing is read.

    Examples
    --------
    >>> import xarray as xr, numpy as np
    >>> da = xr.DataArray(np.zeros((4, 8, 8)), dims=("time", "y", "x")).chunk(1)
    >>> rechunk(da, "timeseries").chunks[0]
    (4,)

    """
    if chunks is None:
        warnings.warn(
            "rechunk without a layout is deprecated; pass one, for example "
            'rechunk(ds, "timeseries") (an error from 1.0)',
            DeprecationWarning,
            stacklevel=2,
        )
        chunks = "timeseries"
    shape, dtype = _stack_shape(obj)
    spec = resolve_chunks(chunks, shape, dtype)
    return obj.chunk({d: n for d, n in spec.items() if d in obj.dims})


def data_chunks(da: xr.DataArray) -> dict[str, tuple[int, ...]]:
    """Return the chunks of a DataArray's data by dimension (empty if eager).

    Unlike ``DataArray.chunksizes`` this ignores the coordinates: per-epoch
    coordinates read from a store (``reference_time``) are often one chunk
    while the data have one chunk per epoch.

    Examples
    --------
    >>> import xarray as xr, numpy as np
    >>> da = xr.DataArray(np.zeros((4, 8)), dims=("time", "x")).chunk(time=1)
    >>> data_chunks(da)["time"]
    (1, 1, 1, 1)

    """
    return dict(zip(da.dims, da.chunks)) if da.chunks else {}


def time_contiguous(obj: Any) -> Any:
    """Return ``obj`` with the whole time axis in each chunk, for per-pixel work.

    Unchanged if it is not a dask array or ``time`` is already one chunk.
    Otherwise time is merged within the input's spatial blocks, each halved
    until a chunk fits ``config["chunks.target_mb"]``, so every output chunk
    comes from one input block per epoch (concatenation, not a shuffle), with
    dask's in-memory ``"tasks"`` rechunk. On 32 full-frame epochs this fitted
    OLS in 366 s against 811 s with the ``"timeseries"`` preset through the
    distributed ``"p2p"`` shuffle (``benchmarks/README.md``). Irregular
    spatial chunks fall back to the preset.
    """
    import dask

    if isinstance(obj, xr.DataArray):
        chunks = data_chunks(obj)
    else:
        try:
            chunks = dict(getattr(obj, "chunksizes", None) or {})
        except ValueError:  # variables chunked differently along a dimension
            return rechunk(obj, "timeseries")
    if "time" not in chunks or len(chunks["time"]) == 1:
        return obj
    shape, dtype = _stack_shape(obj)
    sides = _block_sides(chunks, shape[0] * np.dtype(dtype).itemsize)
    if sides is None:
        return rechunk(obj, "timeseries")
    with dask.config.set({"array.rechunk.method": "tasks"}):
        return obj.chunk({"time": -1, **sides})


def _block_sides(
    chunks: Mapping[str, tuple[int, ...]], bytes_per_pixel: int
) -> dict[str, int] | None:
    """Return spatial chunk sizes that divide the input blocks and fit the target.

    ``None`` if a spatial dimension is chunked irregularly.
    """
    sides = {}
    for dim in ("y", "x"):
        if dim not in chunks:
            continue
        sizes = chunks[dim]
        if any(n != sizes[0] for n in sizes[:-1]) or sizes[-1] > sizes[0]:
            return None
        sides[dim] = sizes[0]
    target = config.get("chunks.target_mb") * 1e6
    while bytes_per_pixel * math.prod(sides.values()) > target:
        halvable = [d for d, n in sides.items() if n % 2 == 0 and n // 2 >= MIN_CHUNK]
        if not halvable:
            break
        largest = max(halvable, key=lambda d: sides[d])
        sides[largest] //= 2
    return sides
