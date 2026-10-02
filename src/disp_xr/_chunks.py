"""Dask chunk sizes for DISP stacks.

Chunks are sized from a byte target (``config["chunks.target_mb"]``), not a
fixed shape, and aligned to the 256 x 256 HDF5 chunks of DISP granules so a dask
chunk never reads part of a file chunk. Two presets:

``"timeseries"``
    The whole time axis in each chunk, for per-pixel work (fits, filters).
``"maps"``
    One epoch per chunk, for maps and exports.

Clamping is adapted from opera-utils (src/opera_utils/disp/_utils.py,
``_clamp_chunk_dict``, commit dff4b45), Copyright (c) 2022 California Institute
of Technology, BSD-3-Clause. See THIRD_PARTY_NOTICES.md.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Literal, Union

import numpy as np

from .config import config

# Spatial chunk size of DISP granules (both productions, both sensors).
FILE_CHUNK = 256
MIN_CHUNK = 32

Preset = Literal["timeseries", "maps"]
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
    layout : {"timeseries", "maps"}
        Whole time axis per chunk, or one epoch per chunk.
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

    """
    nt, ny, nx = shape
    if layout not in ("timeseries", "maps"):
        raise ValueError(f"layout must be 'timeseries' or 'maps', not {layout!r}")
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
    """Turn a preset name, a dict or None (``"timeseries"``) into chunk sizes."""
    if chunks is None or isinstance(chunks, str):
        return suggest_chunks(shape, dtype, chunks or "timeseries")
    return clamp_chunks(chunks, shape)
