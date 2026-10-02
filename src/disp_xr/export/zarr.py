"""Write DISP datasets to Zarr v3 and NetCDF following the PRD storage rules.

One chunk grid for every variable, sized from byte targets: the shard (the
unit of writing) is one dask chunk from :func:`disp_xr._chunks.suggest_chunks`,
and the chunk inside it (the unit of reading) is 64 x 64 over the whole time
axis for ``"timeseries"`` or 1,024 x 1,024 of one epoch for ``"maps"``.
``"ministack"`` is for stores appended to along time (Icechunk): chunk and
shard hold all epochs of the first write, and the chunk is sized to about
``MINISTACK_READ_MB`` in space.

``round_mantissa`` and the per-variable encoding idea are adapted from
opera-utils (src/opera_utils/disp/_utils.py and _reformat.py, commit dff4b45),
Copyright (c) 2022 California Institute of Technology, BSD-3-Clause. See
THIRD_PARTY_NOTICES.md. Changes: one chunk grid derived from byte targets,
shards always a whole multiple of chunks, Zstd (a Zarr v3 codec) instead of
Blosc, explicit fill values, and rounding opt-in.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal, Optional, Union

import numpy as np
import xarray as xr

from .._chunks import suggest_chunks
from ._provenance import with_provenance

__all__ = ["chunk_scheme", "round_mantissa", "to_netcdf", "to_zarr", "zarr_encoding"]

Preset = Literal["timeseries", "maps", "ministack"]
READ_CHUNK = {"timeseries": 64, "maps": 1024}
MINISTACK_READ_MB = 16
DIMS = ("time", "y", "x")


def round_mantissa(z: np.ndarray, keep_bits: int) -> np.ndarray:
    """Return a copy of ``z`` with all but ``keep_bits`` mantissa bits zeroed.

    Lossy; it makes float data compress better. Rounds to nearest, ties to
    even, as numcodecs ``BitRound``. NaN stays NaN.

    Examples
    --------
    >>> round_mantissa(np.array([1.2345678], "float32"), 7)
    array([1.234375], dtype=float32)

    """
    z = np.array(z, copy=True)
    if z.dtype.kind != "f":
        raise TypeError("Only float arrays can be bit-rounded")
    bits = {2: 10, 4: 23, 8: 52}[z.dtype.itemsize]
    if not 0 <= keep_bits <= bits:
        raise ValueError(f"keep_bits must be in [0, {bits}] for {z.dtype}")
    if keep_bits == bits:
        return z
    int_dtype = np.dtype(z.dtype.str.replace("f", "i"))
    as_int = z.view(int_dtype)
    drop = bits - keep_bits
    half = int_dtype.type(1 << (drop - 1))
    mask = ~int_dtype.type((1 << drop) - 1)
    # Ties to even: add half minus one, plus the lowest kept bit.
    as_int += half - 1 + ((as_int >> drop) & 1)
    as_int &= mask
    return z


def chunk_scheme(
    ds: xr.Dataset, preset: Preset = "timeseries", target_mb: Optional[float] = None
) -> tuple[dict[str, int], dict[str, int]]:
    """Return ``(chunks, shards)`` sizes per dimension for writing ``ds``.

    Shards are a whole multiple of chunks in every dimension. The shard is
    sized for the widest dtype among the ``(time, y, x)`` variables, so all
    variables share one grid.
    """
    if preset not in (*READ_CHUNK, "ministack"):
        raise ValueError(
            f"preset must be 'timeseries', 'maps' or 'ministack', not {preset!r}"
        )
    shape = tuple(ds.sizes.get(d, 1) for d in DIMS)
    itemsize = max(
        (v.dtype.itemsize for v in ds.data_vars.values() if v.ndim >= 2), default=4
    )
    if preset == "ministack":
        # The time axis of this write is one ministack: "timeseries" sizing
        # over it, with a read chunk of a few MB instead of 64 x 64.
        shard = suggest_chunks(shape, f"f{itemsize}", "timeseries", target_mb)
        read = suggest_chunks(shape, f"f{itemsize}", "timeseries", MINISTACK_READ_MB)
        read_side = read["y"]
    else:
        shard = suggest_chunks(shape, f"f{itemsize}", preset, target_mb)
        read_side = READ_CHUNK[preset]
    chunks: dict[str, int] = {}
    shards: dict[str, int] = {}
    for dim, size in zip(DIMS, shape):
        if dim == "time":
            chunks[dim] = shards[dim] = shard["time"]
            continue
        chunk = min(read_side, size)
        # Round the shard to a whole number of chunks; it may pass the array
        # edge, which Zarr allows (the last shard is partial).
        multiple = max(1, round(min(shard[dim], size) / chunk))
        multiple = min(multiple, math.ceil(size / chunk))
        chunks[dim], shards[dim] = chunk, chunk * multiple
    return chunks, shards


def _fill_value(var: xr.DataArray) -> Any:
    if var.dtype.kind == "f":
        return np.nan
    return var.attrs.get("_FillValue", var.encoding.get("_FillValue"))


def zarr_encoding(
    ds: xr.Dataset,
    preset: Preset = "timeseries",
    *,
    compression_level: int = 3,
    target_mb: Optional[float] = None,
) -> dict[str, dict[str, Any]]:
    """Build the Zarr v3 encoding of every variable and coordinate of ``ds``."""
    from zarr.codecs import ZstdCodec

    chunks, shards = chunk_scheme(ds, preset, target_mb)
    encoding: dict[str, dict[str, Any]] = {}
    for name, var in ds.data_vars.items():
        if var.ndim == 0:
            continue
        enc: dict[str, Any] = {
            "chunks": tuple(chunks[d] for d in var.dims),
            "shards": tuple(shards[d] for d in var.dims),
            "compressors": [ZstdCodec(level=compression_level)],
        }
        fill = _fill_value(var)
        if fill is not None:
            enc["_FillValue"] = fill
            enc["fill_value"] = fill
        encoding[str(name)] = enc
    for name, coord in ds.coords.items():
        if coord.ndim:
            encoding[str(name)] = {"compressors": None, "chunks": coord.shape}
    return encoding


def _prepare(
    ds: xr.Dataset, preset: Preset, keep_bits: Optional[int], target_mb: Optional[float]
) -> tuple[xr.Dataset, dict[str, int]]:
    out = with_provenance(ds).drop_encoding()
    _, shards = chunk_scheme(out, preset, target_mb)
    for name, var in out.data_vars.items():
        # The fill value goes into the encoding; xarray refuses it in both.
        if "_FillValue" in var.attrs:
            var.encoding["_FillValue"] = var.attrs.pop("_FillValue")
        if keep_bits is not None and var.dtype.kind == "f":
            out[name] = xr.apply_ufunc(
                round_mantissa,
                var,
                kwargs={"keep_bits": keep_bits},
                dask="parallelized",
                output_dtypes=[var.dtype],
                keep_attrs=True,
            )
            out[name].encoding = var.encoding
    for coord in out.coords.values():
        # reference_time carries a NaN _FillValue from the granules, which
        # cannot be encoded once datetimes are stored as integers.
        if coord.dtype.kind != "f":
            coord.attrs.pop("_FillValue", None)
    if keep_bits is not None:
        out.attrs["mantissa_bits_kept"] = keep_bits
    # Align dask chunks to shards so every shard is written by one task.
    out = out.chunk({d: s for d, s in shards.items() if d in out.dims})
    for name in ("time", "reference_time", "x", "y"):
        if name in out.coords and out[name].chunks is not None:
            out[name] = out[name].compute()
    return out, shards


def _with_zarr_conventions(ds: xr.Dataset) -> xr.Dataset:
    """Add the Zarr ``proj:`` and ``spatial:`` attributes next to CF ``spatial_ref``."""
    if ds.rio.crs is None:
        return ds
    from rioxarray._convention.zarr import (
        add_convention_declaration,
        format_spatial_transform,
    )

    attrs = dict(ds.attrs)
    add_convention_declaration(attrs, "proj:")
    add_convention_declaration(attrs, "spatial:")
    attrs["proj:wkt2"] = ds.rio.crs.to_wkt()
    epsg = ds.rio.crs.to_epsg()
    if epsg is not None:
        attrs["proj:code"] = f"EPSG:{epsg}"
    attrs["spatial:transform"] = format_spatial_transform(ds.rio.transform())
    attrs["spatial:dimensions"] = ["y", "x"]
    attrs["spatial:shape"] = [ds.sizes["y"], ds.sizes["x"]]
    out = ds.copy()
    out.attrs = attrs
    return out


def to_zarr(
    ds: xr.Dataset,
    store: Union[str, Path, Any],
    *,
    preset: Preset = "timeseries",
    mode: Literal["w", "w-"] = "w-",
    keep_bits: Optional[int] = None,
    compression_level: int = 3,
    target_mb: Optional[float] = None,
    consolidated: bool = True,
    storage_options: Optional[Mapping[str, Any]] = None,
) -> None:
    """Write ``ds`` to a Zarr v3 store.

    Parameters
    ----------
    ds : xr.Dataset
        DISP dataset with ``(time, y, x)`` and ``(y, x)`` variables.
    store : str, Path or zarr store
        Local path, URL (with ``storage_options``) or any zarr ``Store``.
    preset : {"timeseries", "maps", "ministack"}
        Chunk layout; see the module docstring.
    mode : {"w-", "w"}
        Fail if the store exists (default), or overwrite it.
    keep_bits : int, optional
        Mantissa bits to keep in float variables (lossy). ``None`` keeps the
        data exact, as Cal/Val outputs need.
    compression_level : int
        Zstd level.
    target_mb : float, optional
        Shard size target; defaults to ``config["chunks.target_mb"]``.
    consolidated : bool
        Write consolidated metadata (readers need not use it).
    storage_options : mapping, optional
        fsspec options for a URL store.

    """
    out, _ = _prepare(ds, preset, keep_bits, target_mb)
    out = _with_zarr_conventions(out)
    encoding = zarr_encoding(
        out, preset, compression_level=compression_level, target_mb=target_mb
    )
    for var in out.data_vars.values():
        var.encoding.pop("_FillValue", None)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Consolidated metadata")
        out.to_zarr(
            store,
            mode=mode,
            encoding=encoding,
            zarr_format=3,
            consolidated=consolidated,
            storage_options=dict(storage_options) if storage_options else None,
        )


def to_netcdf(
    ds: xr.Dataset,
    path: Union[str, Path],
    *,
    preset: Preset = "timeseries",
    keep_bits: Optional[int] = None,
    compression_level: int = 4,
    target_mb: Optional[float] = None,
) -> None:
    """Write ``ds`` to NetCDF-4 (h5netcdf) with zlib and the read-chunk layout.

    NetCDF has no shards, so the HDF5 chunk is the read chunk of ``preset``.
    """
    out, _ = _prepare(ds, preset, keep_bits, target_mb)
    chunks, _ = chunk_scheme(out, preset, target_mb)
    encoding: dict[str, dict[str, Any]] = {}
    for name, var in out.data_vars.items():
        if var.ndim == 0:
            continue
        enc: dict[str, Any] = {
            "zlib": True,
            "complevel": compression_level,
            "chunksizes": tuple(min(chunks[d], var.sizes[d]) for d in var.dims),
        }
        fill = var.encoding.pop("_FillValue", None)
        enc["_FillValue"] = np.nan if var.dtype.kind == "f" else fill
        encoding[str(name)] = enc
    out.to_netcdf(path, engine="h5netcdf", encoding=encoding)
