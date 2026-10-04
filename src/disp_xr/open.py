"""Open a frame of DISP granules as one lazy ``(time, y, x)`` dataset."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Optional, Union

import h5py
import numpy as np
import pandas as pd
import rioxarray  # noqa: F401  (registers the .rio accessor)
import xarray as xr

from ._chunks import ChunkSpec, resolve_chunks
from ._history import recorded
from ._remote import is_remote_url, open_file, open_h5
from .config import config
from .product import PRODUCTS, check_granules, find_products

logger = logging.getLogger(__name__)

__all__ = ["open_disp", "open_disp_tree"]

PathsLike = Union[str, Path, Iterable[Union[str, Path]], pd.DataFrame]

# Each DISP layer is (y, x) in the file and gains the time dimension on
# concatenation. join="exact" makes a granule on a different grid an error
# instead of NaN padding; compat="override" takes scalar metadata from the
# first granule.
_COMBINE = {
    "combine": "nested",
    "concat_dim": "time",
    "data_vars": "all",
    "coords": "minimal",
    "compat": "override",
    "join": "exact",
}


@recorded
def open_disp(
    paths: PathsLike,
    *,
    layers: Optional[Sequence[str]] = None,
    corrections: bool = False,
    chunks: ChunkSpec = "native",
    frame_id: Optional[int] = None,
    min_version: Optional[str] = None,
    check: bool = True,
) -> xr.Dataset:
    """Open DISP granules of one frame as a lazy dataset.

    Displacement is returned as stored: relative to each granule's reference
    date (``reference_time``). Chaining across ministacks is a separate step.

    Parameters
    ----------
    paths : str, Path, iterable of paths, or DataFrame
        A directory, files, or the output of
        :func:`disp_xr.product.find_products`.
    layers : sequence of str, optional
        Layers to keep. Defaults to all displacement and quality layers.
    corrections : bool
        Also open the ``/corrections`` group layers.
    chunks : {"native", "timeseries", "maps"} or dict
        Dask chunks. ``"native"`` (default) keeps the file layout, one epoch
        and one read block per chunk, so any selection reads only what it
        needs; functions that work along time rechunk for themselves.
        ``"timeseries"`` and ``"maps"`` are sized from
        ``config["chunks.target_mb"]``; a dict gives ``{"time", "y", "x"}``
        sizes (``-1`` means the whole axis). See :func:`disp_xr.rechunk`.
    frame_id : int, optional
        Frame to keep when ``paths`` holds several.
    min_version : str, optional
        Lowest product version to keep.
    check : bool
        Refuse granules of mixed productions (see
        :func:`disp_xr.product.check_granules`). Productions that differ only
        in the precision of a float layer (DISP-S1 displacement is float32
        before November 2025 and float64 after) are read together with a
        :class:`~disp_xr.product.MixedPrecisionWarning`; those layers are cast
        to float32 and listed in ``attrs["mixed_precision"]``.

    Returns
    -------
    xr.Dataset
        Dims ``(time, y, x)``; coordinates ``time``, ``reference_time``, ``x``,
        ``y`` and ``spatial_ref`` (times to the second, as in granule
        names); one variable per layer, each with
        ``grid_mapping``. No pixel is read until computed. Layers keep their
        stored dtype: float layers use NaN for no data, integer layers (masks,
        labels, counts) keep the product's nodata value, given in their
        ``_FillValue`` attribute.

    Raises
    ------
    ValueError
        If a requested layer does not exist in the product.

    """
    df = (
        paths
        if isinstance(paths, pd.DataFrame)
        else find_products(paths, frame_id=frame_id, min_version=min_version)
    )
    mixed: list[str] = []
    if check:
        mixed = check_granules(df, allow_mixed_precision=True).attrs["mixed_precision"]
    (product,) = df["product"].unique()
    spec = PRODUCTS[product]
    available = spec.displacement_layers + spec.quality_layers
    keep = tuple(layers) if layers is not None else available
    unknown = set(keep) - set(available)
    if unknown:
        raise ValueError(f"Unknown {product} layers: {sorted(unknown)}")

    files = list(df["path"])
    ds = _open(files, drop=[v for v in available if v not in keep], group=None)
    if corrections:
        corr = _open(files, drop=[], group="corrections")
        corr = corr[[v for v in spec.correction_layers if v in corr]]
        ds = xr.merge([ds, corr], join="exact", compat="override")

    for name in mixed:
        if name in ds:
            ds[name] = ds[name].astype(np.float32)
    # Whole seconds, as in granule names. Productions encode time differently
    # (seconds since the acquisition, or float seconds since 2010, which
    # decodes nanoseconds off), and ministacks are linked by exact dates.
    ds["time"] = ds["time"].dt.floor("s")
    ds["reference_time"] = ds["reference_time"].dt.floor("s")
    ds = ds.set_coords("reference_time")
    first = ds[keep[0]] if keep else next(iter(ds.data_vars.values()))
    ds = ds.chunk(
        resolve_chunks(
            chunks, (ds.sizes["time"], ds.sizes["y"], ds.sizes["x"]), first.dtype
        )
    )
    # Small, and needed eagerly to group ministacks.
    ds["reference_time"] = ds["reference_time"].compute(scheduler="synchronous")
    for name in ds.data_vars:
        ds[name].attrs.setdefault("grid_mapping", "spatial_ref")
    ds.attrs.update(
        product=product,
        frame_id=int(df["frame_id"].iloc[0]),
        product_versions=sorted(set(df["version"])),
        # Provenance for exports: granule names, not paths (paths are local).
        input_granules=[Path(str(f)).name for f in files],
    )
    if mixed:
        ds.attrs["mixed_precision"] = mixed
    logger.info(f"Opened {len(files)} {product} granules, frame {ds.attrs['frame_id']}")
    return ds


def _open(files: list, drop: list[str], group: Optional[str]) -> xr.Dataset:
    block = config.get("chunks.read_block")
    # Remote granules are passed to xarray as fsspec file objects.
    sources = [open_file(f) if is_remote_url(f) else f for f in files]
    return xr.open_mfdataset(
        sources,
        engine="h5netcdf",
        group=group,
        decode_coords="all",
        # Keep integer layers integer; float layers already store NaN as fill.
        mask_and_scale=False,
        drop_variables=drop,
        chunks={"y": block, "x": block},
        **_COMBINE,
    )


def open_disp_tree(
    paths: PathsLike,
    *,
    layers: Optional[Sequence[str]] = None,
    chunks: ChunkSpec = "native",
    frame_id: Optional[int] = None,
    min_version: Optional[str] = None,
    check: bool = True,
) -> xr.DataTree:
    """Open a frame as a DataTree that mirrors the granule groups.

    Nodes: the root holds the displacement and quality layers (as
    :func:`open_disp`), ``/corrections`` the correction layers, and
    ``/identification`` and ``/metadata`` hold the first granule's scalar
    values as attributes (xarray cannot decode their byte strings). Values that
    change per granule, such as the dates, are in the ``time`` and
    ``reference_time`` coordinates.

    Parameters
    ----------
    paths, layers, chunks, frame_id, min_version, check
        As for :func:`open_disp`.

    Returns
    -------
    xr.DataTree

    """
    df = (
        paths
        if isinstance(paths, pd.DataFrame)
        else find_products(paths, frame_id=frame_id, min_version=min_version)
    )
    ds = open_disp(df, layers=layers, corrections=True, chunks=chunks, check=check)
    (product,) = df["product"].unique()
    correction_names = [v for v in PRODUCTS[product].correction_layers if v in ds]
    first = df["path"].iloc[0]
    return xr.DataTree.from_dict(
        {
            "/": ds.drop_vars(correction_names),
            "/corrections": ds[correction_names],
            "/identification": xr.Dataset(attrs=_scalar_attrs(first, "identification")),
            "/metadata": xr.Dataset(attrs=_scalar_attrs(first, "metadata")),
        }
    )


def _scalar_attrs(path: str, group: str) -> dict:
    out: dict = {}
    with open_h5(path) as f:
        if group not in f:
            return out
        for name, item in f[group].items():
            if isinstance(item, h5py.Dataset) and item.shape == ():
                value = item[()]
                out[name] = value.decode() if isinstance(value, bytes) else value.item()
    return out
