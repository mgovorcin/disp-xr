"""Multiscale pyramids of downsampled levels in Zarr (PRD C4, plan task T27.7).

A pyramid is one Zarr group whose children are the levels, as sibling groups
``0`` (the input, 30 m), ``1``, ``2``, ... (each from
:func:`disp_xr.downsample` of level ``0``). Every level is self-contained: CF
``spatial_ref``, the Zarr ``proj:`` and ``spatial:`` attributes with its own
``spatial:transform`` and ``spatial:shape``, and the provenance attributes.
The root carries the multiscales ``layout``: per level its ``asset``,
``derived_from``, the relative ``transform`` (scale = factor, translation 0,
since cell edges start at the first pixel edge), the absolute
``spatial:transform`` and ``spatial:shape``, the resampling method, and the
reduction of each variable under ``disp_xr:resampling``.

The multiscales, ``spatial:`` and ``proj:`` conventions are at pilot maturity
(multiscales v0.1, breaking changes expected before v1), so their ids and
URLs live only in this module and the version is recorded in the root
attributes. :func:`open_pyramid` finds levels from the ``layout``, never from
group names.

Conventions: https://github.com/zarr-conventions/multiscales (v0.1),
https://github.com/zarr-conventions/spatial, https://github.com/zarr-conventions/proj.
"""

from __future__ import annotations

import json
import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, Optional, Union

import xarray as xr

from ..config import config
from ._provenance import provenance_attrs
from .zarr import Preset, to_zarr

__all__ = ["MULTISCALES_CONVENTION", "build_overviews", "open_pyramid", "to_pyramid"]

MULTISCALES_VERSION = "v0.1"
MULTISCALES_CONVENTION = {
    "schema_url": (
        "https://raw.githubusercontent.com/zarr-conventions/multiscales/refs/tags/"
        f"{MULTISCALES_VERSION}/schema.json"
    ),
    "spec_url": (
        "https://github.com/zarr-conventions/multiscales/blob/"
        f"{MULTISCALES_VERSION}/README.md"
    ),
    "uuid": "d35379db-88df-4056-af3a-620245f8e347",
    "name": "multiscales",
    "description": "Multiscale layout of zarr datasets",
}
# disp-xr names of its reductions in the layout.
RESAMPLING = {
    "wmean": "weighted_mean",
    "wmedian": "weighted_median",
    "mode": "mode",
    "any": "any",
}


def _resampling(level: xr.Dataset) -> dict[str, str]:
    """Return the reduction of each variable of a downsampled level."""
    out = {}
    for name, var in level.data_vars.items():
        how = var.attrs.get("aggregation", var.attrs.get("reduction"))
        out[str(name)] = RESAMPLING.get(how, "cell_statistic")
    return out


def _transform(level: xr.Dataset) -> list[float]:
    from rioxarray._convention.zarr import format_spatial_transform

    return format_spatial_transform(level.rio.transform())


def _layout(levels: Mapping[str, xr.Dataset], factors: Sequence[int]) -> list[dict]:
    base = next(iter(levels))
    layout: list[dict[str, Any]] = []
    for (asset, level), factor in zip(levels.items(), (1, *factors)):
        entry: dict[str, Any] = {
            "asset": asset,
            "transform": {"scale": [float(factor)] * 2, "translation": [0.0, 0.0]},
            "spatial:transform": _transform(level),
            "spatial:shape": [level.sizes["y"], level.sizes["x"]],
            "disp_xr:factor": int(factor),
        }
        if asset != base:
            entry["derived_from"] = base
            entry["resampling_method"] = RESAMPLING[level.attrs["downsample_stat"]]
            entry["disp_xr:resampling"] = _resampling(level)
        layout.append(entry)
    return layout


def _root_attrs(ds: xr.Dataset, layout: list[dict], levels: Mapping) -> dict:
    from rioxarray._convention.zarr import add_convention_declaration

    attrs: dict[str, Any] = {"zarr_conventions": [dict(MULTISCALES_CONVENTION)]}
    if ds.rio.crs is not None:
        add_convention_declaration(attrs, "proj:")
        add_convention_declaration(attrs, "spatial:")
        epsg = ds.rio.crs.to_epsg()
        if epsg is not None:
            attrs["proj:code"] = f"EPSG:{epsg}"
        attrs["proj:wkt2"] = ds.rio.crs.to_wkt()
        attrs["spatial:dimensions"] = ["y", "x"]
    coarse = [lv for k, lv in levels.items() if k != layout[0]["asset"]]
    attrs["multiscales"] = {
        "layout": layout,
        "resampling_method": layout[1]["resampling_method"] if coarse else "none",
    }
    attrs["disp_xr:multiscales_convention"] = MULTISCALES_VERSION
    if coarse:
        attrs["disp_xr:weights"] = str(coarse[0].attrs["downsample_weights"])
    attrs.update(provenance_attrs(ds))
    return attrs


def build_overviews(
    ds: xr.Dataset, factors: Sequence[int] = (33, 67, 333), **downsample_kwargs: Any
) -> xr.DataTree:
    """Return a pyramid of ``ds`` and its downsampled levels as a DataTree.

    Parameters
    ----------
    ds : xr.Dataset
        Chained, re-referenced data at full resolution; level ``0``.
    factors : sequence of int
        Downsampling factors of levels ``1``, ``2``, ... relative to ``ds``,
        increasing. Every level is computed from ``ds`` (a weighted median
        cannot be built from a coarser level).
    **downsample_kwargs
        Passed to :func:`disp_xr.downsample` (``stat``, ``weights``,
        ``variogram``, ...). ``layers`` defaults to every numeric layer of
        ``ds``, including the quality layers used for the weights, so every
        level holds the variables of level ``0`` (plus the cell statistics).

    Returns
    -------
    xr.DataTree
        Children ``"0"``, ``"1"``, ... and the multiscales attributes on the
        root; lazy if ``ds`` is.

    """
    from ..downsample import CATEGORICAL_DEFAULTS, downsample

    factors = [int(f) for f in factors]
    if not factors or factors != sorted(set(factors)) or factors[0] < 2:
        raise ValueError(f"factors must be increasing integers >= 2, not {factors}")
    if "layers" not in downsample_kwargs:
        # Every level holds the variables of level 0 (readers such as bowser
        # switch levels by zoom for the same variable), so the quality layers
        # used for the weights are aggregated too.
        categorical = {
            **CATEGORICAL_DEFAULTS,
            **dict(downsample_kwargs.get("categorical") or {}),
        }
        downsample_kwargs["layers"] = [
            str(n)
            for n, v in ds.data_vars.items()
            if v.dtype.kind in "fiu"
            and {"y", "x"} <= set(v.dims)
            and n not in categorical
        ]
    levels = {"0": ds}
    for i, f in enumerate(factors, start=1):
        levels[str(i)] = downsample(ds, factor=f, **downsample_kwargs)
    layout = _layout(levels, factors)
    root = xr.Dataset(attrs=_root_attrs(ds, layout, levels))
    return xr.DataTree.from_dict({"/": root, **{f"/{k}": v for k, v in levels.items()}})


def _level_preset(level: xr.Dataset, base: Preset, small_mb: float) -> Preset:
    largest = max((v.nbytes for v in level.data_vars.values()), default=0)
    return "whole" if largest <= small_mb * 1e6 else base


def to_pyramid(
    data: Union[xr.Dataset, xr.DataTree],
    store: Union[str, Path, Any],
    *,
    factors: Sequence[int] = (33, 67, 333),
    mode: Literal["w-", "w"] = "w-",
    preset: Preset = "timeseries",
    small_mb: Optional[float] = None,
    storage_options: Optional[Mapping[str, Any]] = None,
    **downsample_kwargs: Any,
) -> xr.DataTree:
    """Write a multiscale pyramid to a Zarr v3 store.

    Parameters
    ----------
    data : xr.Dataset or xr.DataTree
        Full-resolution data (levels are built with :func:`build_overviews`
        and ``factors``), or a tree from :func:`build_overviews`.
    store : str, Path or zarr store
        Where to write.
    factors : sequence of int
        Downsampling factors when ``data`` is a Dataset.
    mode : {"w-", "w"}
        Fail if the store exists (default), or overwrite it.
    preset : {"timeseries", "maps", "ministack", "whole"}
        Chunk layout of level ``0`` and of levels too large for one chunk
        (see :func:`disp_xr.to_zarr`).
    small_mb : float, optional
        Levels whose largest variable is at most this many MB are stored as
        one chunk over all of time (a 1 km frame is about 250 x 300 cells);
        default ``config["chunks.target_mb"]``.
    storage_options : mapping, optional
        fsspec options for a URL store.
    **downsample_kwargs
        Passed to :func:`disp_xr.downsample` when ``data`` is a Dataset.

    Returns
    -------
    xr.DataTree
        The pyramid that was written.

    Examples
    --------
    >>> dxr.to_pyramid(ds, "frame_pyramid.zarr", factors=(33, 67))  # doctest: +SKIP
    >>> tree = dxr.open_pyramid("frame_pyramid.zarr")  # doctest: +SKIP

    """
    import zarr

    tree = (
        data
        if isinstance(data, xr.DataTree)
        else build_overviews(data, factors, **downsample_kwargs)
    )
    small = config.get("chunks.target_mb") if small_mb is None else small_mb
    options = dict(storage_options) if storage_options else None
    root = zarr.open_group(store, mode=mode, zarr_format=3, storage_options=options)
    layout = tree.attrs["multiscales"]["layout"]
    for i, entry in enumerate(layout):
        level = tree[entry["asset"]].to_dataset()
        to_zarr(
            level,
            store,
            preset=preset if i == 0 else _level_preset(level, preset, small),
            mode="w",
            consolidated=False,
            storage_options=options,
            group=entry["asset"],
        )
    root = zarr.open_group(store, mode="r+", storage_options=options)
    root.attrs.update(tree.attrs)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Consolidated metadata")
        zarr.consolidate_metadata(root.store)
    return tree


def open_pyramid(
    store: Union[str, Path, Any],
    *,
    storage_options: Optional[Mapping[str, Any]] = None,
) -> xr.DataTree:
    """Open a pyramid written by :func:`to_pyramid`, lazily.

    Levels are read in the order of the root ``multiscales`` layout, from
    each entry's ``asset``; groups the layout does not list are ignored.

    Returns
    -------
    xr.DataTree
        One child per level, named by its ``asset``, with the root attributes
        (``multiscales`` and provenance).

    """
    import zarr

    options = dict(storage_options) if storage_options else None
    root = zarr.open_group(store, mode="r", storage_options=options)
    attrs = dict(root.attrs)
    if "multiscales" not in attrs:
        raise ValueError("store has no multiscales layout at its root")
    multiscales = attrs["multiscales"]
    if isinstance(multiscales, str):
        multiscales = attrs["multiscales"] = json.loads(multiscales)
    nodes: dict[str, xr.Dataset] = {"/": xr.Dataset(attrs=attrs)}
    for entry in multiscales["layout"]:
        nodes["/" + entry["asset"]] = xr.open_zarr(
            store,
            group=entry["asset"],
            consolidated=False,
            chunks={},
            storage_options=options,
        )
    return xr.DataTree.from_dict(nodes)
