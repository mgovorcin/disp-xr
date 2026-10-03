"""GeoZarr stores that bowser opens directly (PRD C4, plan task T27.9).

``to_geozarr(ds, store)`` writes a GeoZarr store: the multiscale pyramid of
:func:`disp_xr.to_pyramid` (or one flat group), with CF and Zarr ``proj:`` /
``spatial:`` metadata on every level. ``profile="bowser"`` (the default)
adds what `bowser <https://github.com/opera-adt/bowser>`_ uses to display the
data, without changing anything a strict GeoZarr reader relies on:

- display levels by factor 2 until the shorter side is under 256 cells
  (bowser picks the level whose pixel matches the tile pixel, so analysis
  factors such as 33 alone leave mid zooms reading 30 m);
- ``bowser_uses_spatial_ref`` on every variable: true only for displacement
  and velocity. Otherwise bowser infers it from the units and subtracts a
  reference value from sigmas (meters) and inversion residuals (radians);
- ``long_name`` on every variable (the colorbar title);
- ``reference_lonlat`` from :func:`disp_xr.rereference` (a point
  reference), shown as the reference marker;
- the line-of-sight attributes bowser reads (``heading_angle_deg``,
  ``incidence_angle_deg`` and ``los_enu_ground_to_satellite``, near, center
  and far) from the static layers, when given;
- ``bowser_histogram`` per epoch on level ``0`` (bins and percentiles), so the
  histogram panel does not read a full-resolution epoch per request;
- float32 data, and chunks of one epoch by 256 x 256 in shards of 4 epochs
  by 1,024 x 1,024 (the ``"tiles"`` preset, bowser's tuned layout).

Checked against bowser ``ops`` (``0a61c6a``; zarr 3.1.1, xarray 2025.8,
rioxarray 0.19) with ``tests/bowser_check.py``: the per-level ``proj:`` and
``spatial:`` attributes, the multiscales ``transform``, consolidated metadata,
``spatial_ref`` as a coordinate, integer masks with a ``_FillValue`` (decoded
to NaN) and one-chunk small levels do not change what bowser reads or shows.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, Optional, Union

import numpy as np
import xarray as xr

from ..config import bounded_threads, config
from .multiscales import build_overviews, to_pyramid
from .zarr import to_zarr

__all__ = [
    "bowser_attrs",
    "display_factors",
    "histogram_stats",
    "los_attrs",
    "reference_lonlat",
    "to_geozarr",
]

Profile = Literal["bowser", "standard"]
# Variables bowser should offer a moving reference point for (relative
# measurements); everything else is shown as stored.
RELATIVE_LAYERS = ("displacement", "velocity")
HISTOGRAM_BINS = 100


def display_factors(
    shape: tuple[int, int], min_size: int = 256, max_levels: int = 6
) -> list[int]:
    """Return factor-2 display levels, as bowser's own converter builds them.

    Factors 2, 4, 8, ... while the shorter side stays at least ``min_size``
    cells, at most ``max_levels`` of them.
    """
    out: list[int] = []
    factor = 2
    while min(shape) // factor >= min_size and len(out) < max_levels:
        out.append(factor)
        factor *= 2
    return out


def bowser_attrs(ds: xr.Dataset) -> xr.Dataset:
    """Return ``ds`` with float32 data and the variable attributes bowser reads."""
    out = ds.copy()
    for name, var in ds.data_vars.items():
        if not {"y", "x"} <= set(var.dims):
            continue
        if var.dtype.kind == "f" and var.dtype != np.float32:
            out[name] = var.astype(np.float32)
        out[name].attrs = {
            **var.attrs,
            "long_name": var.attrs.get("long_name", str(name).replace("_", " ")),
            "bowser_uses_spatial_ref": str(name) in RELATIVE_LAYERS,
        }
    return out


def reference_lonlat(ds: xr.Dataset) -> Optional[list[float]]:
    """Return ``[lon, lat]`` of a point reference recorded by ``rereference``.

    None when the data were referenced to a median or an area, or not at all.
    """
    record = ds.attrs.get("reference")
    if not record:
        return None
    record = json.loads(record) if isinstance(record, str) else record
    if "x" not in record or ds.rio.crs is None:
        return None
    from pyproj import Transformer

    to_lonlat = Transformer.from_crs(ds.rio.crs, "EPSG:4326", always_xy=True)
    lon, lat = to_lonlat.transform(record["x"], record["y"])
    return [float(lon), float(lat)]


def los_attrs(
    static: xr.Dataset, look_side: Literal["right", "left"] = "right"
) -> dict[str, Any]:
    """Return bowser's line-of-sight attributes from the static layers.

    Near, center and far are the 5th, 50th and 95th percentiles of the
    incidence angle; the ENU vectors are rebuilt from those incidences and
    the median azimuth. The ground-track heading is the azimuth of the
    ground-to-satellite vector plus 90 degrees for a right-looking radar
    (Sentinel-1), minus 90 for a left-looking one.

    Parameters
    ----------
    static : xr.Dataset
        ``los_east`` and ``los_north`` (and ``los_up``), as from
        :func:`disp_xr.open_static`.
    look_side : {"right", "left"}
        Radar look direction.

    """
    from ..geometry.los import incidence_azimuth_to_los, los_to_incidence_azimuth

    east = np.asarray(static["los_east"].values, dtype=np.float64).ravel()
    north = np.asarray(static["los_north"].values, dtype=np.float64).ravel()
    ok = np.isfinite(east) & np.isfinite(north)
    incidence, azimuth = los_to_incidence_azimuth(east[ok], north[ok])
    # Circular median of the azimuth (it can straddle 0/360).
    az = float(np.degrees(np.angle(np.median(np.exp(1j * np.radians(azimuth))))))
    az %= 360
    near, center, far = (float(v) for v in np.percentile(incidence, [5, 50, 95]))
    heading = (az + (90.0 if look_side == "right" else -90.0)) % 360

    def enu(inc: float) -> dict[str, float]:
        e, n, u = incidence_azimuth_to_los(inc, az)
        return {"east": float(e), "north": float(n), "up": float(u)}

    return {
        "heading_angle_deg": heading,
        "incidence_angle_deg": {"near": near, "center": center, "far": far},
        "los_enu_ground_to_satellite": {
            "near": enu(near),
            "center": enu(center),
            "far": enu(far),
        },
    }


def _stats(values: np.ndarray) -> dict[str, Any]:
    valid = values.ravel().astype(np.float64)
    valid = valid[np.isfinite(valid)]
    keys = ("min", "max", "p2", "p98", "p16", "p84", "p23", "p977")
    if valid.size == 0:
        return dict.fromkeys(keys, 0.0) | {"bins": [], "counts": []}
    counts, edges = np.histogram(valid, bins=HISTOGRAM_BINS)
    pct = np.percentile(valid, [2, 98, 16, 84, 2.275, 97.725])
    return {
        "bins": edges.tolist(),
        "counts": counts.tolist(),
        "min": float(valid.min()),
        "max": float(valid.max()),
        **dict(zip(keys[2:], (float(p) for p in pct))),
    }


def histogram_stats(da: xr.DataArray) -> list[dict[str, Any]]:
    """Return bowser's per-epoch histogram entries for one variable.

    One dict per epoch (one for a 2D layer) with 100 bins, the counts, the
    extremes and the 2/98, 16/84 and 2.275/97.725 percentiles, matching
    bowser's ``_compute_histogram_stats``. Epochs are computed in parallel,
    each read once.
    """
    import dask

    if "time" in da.dims:
        slices = [da.isel(time=i) for i in range(da.sizes["time"])]
    else:
        slices = [da]
    tasks = [dask.delayed(_stats)(s.data) for s in slices]
    with dask.config.set(num_workers=config.get("num_workers")):
        return list(dask.compute(*tasks))


@bounded_threads
def to_geozarr(
    ds: xr.Dataset,
    store: Union[str, Path, Any],
    *,
    profile: Profile = "bowser",
    pyramid: bool = True,
    factors: Sequence[int] = (),
    static: Optional[xr.Dataset] = None,
    look_side: Literal["right", "left"] = "right",
    histogram: bool = True,
    min_size: int = 256,
    mode: Literal["w-", "w"] = "w-",
    storage_options: Optional[Mapping[str, Any]] = None,
    **downsample_kwargs: Any,
) -> Optional[xr.DataTree]:
    """Write ``ds`` as a GeoZarr store, by default one that bowser opens.

    Parameters
    ----------
    ds : xr.Dataset
        Chained, re-referenced data with ``(time, y, x)`` layers.
    store : str, Path or zarr store
        Where to write; give it a ``.zarr`` suffix for bowser.
    profile : {"bowser", "standard"}
        ``"bowser"`` adds display levels, bowser attributes, histograms and
        the tile chunk layout (see the module docstring); ``"standard"``
        writes the plain :func:`disp_xr.to_pyramid` layout.
    pyramid : bool
        Write a multiscale pyramid (groups ``0``, ``1``, ...); otherwise one
        flat group at the root.
    factors : sequence of int
        Analysis levels (for example 33, 67, 333) next to the display levels
        of the bowser profile; the only levels of the standard profile
        (default 33, 67, 333 there).
    static : xr.Dataset, optional
        Static layers with ``los_east`` and ``los_north``
        (:func:`disp_xr.open_static`), for the line-of-sight attributes.
    look_side : {"right", "left"}
        Radar look direction, for the heading (Sentinel-1 looks right).
    histogram : bool
        Store per-epoch histograms of the level-0 float layers (bowser
        profile).
    min_size : int
        Smallest side, in cells, of a display level (bowser uses 256).
    mode : {"w-", "w"}
        Fail if the store exists (default), or overwrite it.
    storage_options : mapping, optional
        fsspec options for a URL store.
    **downsample_kwargs
        Passed to :func:`disp_xr.downsample` (``weights``, ...).

    Returns
    -------
    xr.DataTree or None
        The pyramid written, or None for a flat store.

    Examples
    --------
    >>> dxr.to_geozarr(ds, "frame.zarr", static=dxr.open_static(ds))  # doctest: +SKIP

    Then ``bowser run --stack-file frame.zarr``.

    """
    if profile not in ("bowser", "standard"):
        raise ValueError(f"profile must be 'bowser' or 'standard', not {profile!r}")
    if profile == "standard":
        if pyramid:
            return to_pyramid(
                ds,
                store,
                factors=tuple(factors) or (33, 67, 333),
                mode=mode,
                storage_options=storage_options,
                **downsample_kwargs,
            )
        to_zarr(ds, store, mode=mode, storage_options=storage_options)
        return None

    base = bowser_attrs(ds)
    extra: dict[str, Any] = {}
    if (lonlat := reference_lonlat(ds)) is not None:
        extra["reference_lonlat"] = lonlat
    if static is not None:
        extra |= los_attrs(static, look_side)
    if histogram:
        for name, var in base.data_vars.items():
            if var.dtype.kind == "f" and {"y", "x"} <= set(var.dims):
                base[name].attrs["bowser_histogram"] = histogram_stats(var)
    base.attrs |= extra
    if not pyramid:
        to_zarr(base, store, preset="tiles", mode=mode, storage_options=storage_options)
        return None
    levels = sorted(
        set(display_factors((ds.sizes["y"], ds.sizes["x"]), min_size))
        | {int(f) for f in factors}
    )
    if not levels:
        raise ValueError("frame too small for display levels; pass pyramid=False")
    downsample_kwargs.setdefault("stat", "wmean")
    tree = build_overviews(base, levels, **downsample_kwargs)
    nodes = {"/": tree.to_dataset().assign_attrs(extra)}
    for name, node in tree.children.items():
        level = node.to_dataset()
        if name != "0":
            level = bowser_attrs(level).assign_attrs(extra)
            # Histograms describe level 0 (bowser reads them there only).
            for var in level.data_vars.values():
                var.attrs.pop("bowser_histogram", None)
        nodes[f"/{name}"] = level
    return to_pyramid(
        xr.DataTree.from_dict(nodes),
        store,
        mode=mode,
        preset="tiles",
        storage_options=storage_options,
    )
