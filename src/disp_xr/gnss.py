"""GNSS stations next to DISP, with geepers as the engine (PRD F10, task T25).

geepers (https://github.com/opera-adt/geepers, the ``gnss`` extra) loads the
GNSS series: its sources (UNR, UNR grid, sideshow) download, cache and read
station files, and its sampler reads DISP around each station. This module
adapts them to xarray and to DISP:

- :func:`get_source` picks a geepers source; :func:`stations` lists the stations
  over a frame;
- :func:`load` returns a ``(station, time)`` Dataset of east, north and up
  motion with sigmas and correlations, and station coordinates in lon/lat
  and in the frame's CRS;
- :func:`to_los` projects them onto the line of sight with the static layers
  (sigma from geepers ``get_sigma_los``);
- :func:`sample` reads DISP at the stations (median in a window, geepers
  ``sample_insar``);
- :func:`align` puts GNSS on the DISP epochs and on the DISP reference date.
- :func:`centred` stamps daily solutions at the centre of their UTC day.

Typical use::

    gps = dxr.gnss.load(ds, start="2017-01-01")
    gps_los = dxr.gnss.to_los(gps, dxr.open_static(ds))
    insar = dxr.gnss.sample(ds, gps, buffer_m=100)
    gps_aligned = dxr.gnss.align(gps_los["los"], insar.time)
"""

from __future__ import annotations

import inspect
from typing import Any, Literal, Optional, Union

import numpy as np
import pandas as pd
import xarray as xr

__all__ = ["align", "centred", "get_source", "load", "sample", "stations", "to_los"]

SOURCES = {
    "unr": "UnrSource",
    "unr_grid": "UnrGridSource",
    "sideshow": "SideshowSource",
}
COMPONENTS = ("east", "north", "up")
UNCERTAINTY = ("sigma_east", "sigma_north", "sigma_up", "corr_en", "corr_eu", "corr_nu")

Bbox = tuple[float, float, float, float]


def _geepers() -> Any:
    try:
        import geepers.gps_sources as sources
    except ImportError as err:
        raise ImportError(
            "disp_xr.gnss needs geepers: pip install 'disp-xr[gnss]'"
        ) from err
    return sources


def get_source(name: Union[str, Any] = "unr", cache_dir: Optional[str] = None) -> Any:
    """Return a geepers GNSS source.

    Parameters
    ----------
    name : {"unr", "unr_grid", "sideshow"} or a geepers source
        A source name, or an instance of a geepers ``BaseGpsSource``
        subclass (returned as is).
    cache_dir : str, optional
        Where geepers caches downloads (its default otherwise).

    Examples
    --------
    >>> source = get_source("unr", cache_dir="gnss-cache")  # doctest: +SKIP

    """
    if not isinstance(name, str):
        return name
    sources = _geepers()
    if name not in SOURCES:
        raise ValueError(f"source must be one of {sorted(SOURCES)}, not {name!r}")
    return getattr(sources, SOURCES[name])(cache_dir=cache_dir)


def _bbox(target: Union[xr.Dataset, xr.DataArray, Bbox]) -> Bbox:
    if isinstance(target, (xr.Dataset, xr.DataArray)):
        return tuple(target.rio.transform_bounds("EPSG:4326"))  # type: ignore[return-value]
    return tuple(target)  # type: ignore[return-value]


def _inside(gdf, ds: Union[xr.Dataset, xr.DataArray]):
    """Keep stations inside the frame's grid (its lon/lat box has corners)."""
    from pyproj import Transformer

    to_grid = Transformer.from_crs("EPSG:4326", ds.rio.crs, always_xy=True)
    x, y = to_grid.transform(gdf["lon"].to_numpy(), gdf["lat"].to_numpy())
    xs, ys = ds["x"].values, ds["y"].values
    keep = (x >= xs.min()) & (x <= xs.max()) & (y >= ys.min()) & (y <= ys.max())
    return gdf[keep].assign(x=x[keep], y=y[keep])


def stations(
    target: Union[xr.Dataset, xr.DataArray, Bbox],
    *,
    source: Union[str, Any] = "unr",
    cache_dir: Optional[str] = None,
):
    """Return the GNSS stations over a frame or a lon/lat box.

    Parameters
    ----------
    target : xr.Dataset, xr.DataArray or (west, south, east, north)
        A DISP dataset (stations inside its grid, with ``x``, ``y`` in its
        CRS) or a box in degrees.
    source : str or geepers source
        See :func:`get_source`.
    cache_dir : str, optional
        geepers cache directory.

    Returns
    -------
    geopandas.GeoDataFrame
        geepers station table (``id``, ``lon``, ``lat``, ``alt``, geometry).

    Examples
    --------
    >>> table = dxr.gnss.stations(ds)  # doctest: +SKIP
    >>> table = dxr.gnss.stations((-121.5, 35.5, -120.0, 37.0))  # doctest: +SKIP

    """
    src = get_source(source, cache_dir)
    gdf = src.stations(bbox=_bbox(target))
    if isinstance(target, (xr.Dataset, xr.DataArray)):
        gdf = _inside(gdf, target)
    return gdf.reset_index(drop=True)


def load(
    target: Union[xr.Dataset, xr.DataArray, Bbox, list[str]],
    *,
    source: Union[str, Any] = "unr",
    start: Optional[str] = None,
    end: Optional[str] = None,
    cache_dir: Optional[str] = None,
    max_workers: int = 8,
) -> xr.Dataset:
    """Load GNSS series as a ``(station, time)`` Dataset.

    Parameters
    ----------
    target : xr.Dataset, (west, south, east, north) or list of str
        A DISP dataset (its stations, with ``x``, ``y`` in its CRS), a box in
        degrees, or station ids.
    source : str or geepers source
        See :func:`get_source`.
    start, end : str, optional
        Dates (ISO) to keep.
    cache_dir : str, optional
        geepers cache directory.
    max_workers : int
        Stations downloaded in parallel by geepers.

    Returns
    -------
    xr.Dataset
        ``east``, ``north``, ``up`` (meters, as geepers returns them) and
        ``sigma_*``, ``corr_*``; coordinates ``station``, ``time``, ``lon``,
        ``lat``, ``alt`` (and ``x``, ``y`` for a dataset target).

    Examples
    --------
    >>> gps = dxr.gnss.load(ds, start="2017-01-01")  # doctest: +SKIP
    >>> gps = dxr.gnss.load(["P539", "P565"])  # doctest: +SKIP

    """
    src = get_source(source, cache_dir)
    is_ds = isinstance(target, (xr.Dataset, xr.DataArray))
    if is_ds or (isinstance(target, tuple) and len(target) == 4):
        table = stations(target, source=src)  # type: ignore[arg-type]
        ids = table["id"].tolist()
    else:
        ids = list(target)  # type: ignore[arg-type]
        table = src.stations().set_index("id").loc[ids].reset_index()
    if not ids:
        raise ValueError("no GNSS station in the target area")
    long = src.timeseries_many(
        ids=ids,
        frame="ENU",
        start_date=start,
        end_date=end,
        zero_by="start",
        max_workers=max_workers,
    )
    frame = pd.DataFrame(long.drop(columns="geometry", errors="ignore"))
    frame["date"] = pd.to_datetime(frame["date"])
    names = [c for c in (*COMPONENTS, *UNCERTAINTY) if c in frame]
    cube = (
        frame.set_index(["id", "date"])[names]
        .to_xarray()
        .rename({"id": "station", "date": "time"})
    )
    meta = table.set_index("id").loc[cube["station"].values]
    coords: dict[str, Any] = {
        name: ("station", meta[name].to_numpy(np.float64))
        for name in ("lon", "lat", "alt")
        if name in meta
    }
    if is_ds:
        coords["x"] = ("station", meta["x"].to_numpy(np.float64))
        coords["y"] = ("station", meta["y"].to_numpy(np.float64))
    cube = cube.assign_coords(coords)
    for name in names:
        cube[name].attrs["units"] = "meters" if not name.startswith("corr") else "1"
    cube.attrs["gnss_source"] = type(src).__name__
    return cube


def to_los(gnss: xr.Dataset, static: xr.Dataset) -> xr.Dataset:
    """Project GNSS east, north and up onto the line of sight.

    The LOS unit vector is read at each station's pixel (nearest) in the
    static layers; the LOS sigma propagates the E/N/U sigmas and
    correlations (geepers ``get_sigma_los``).

    Parameters
    ----------
    gnss : xr.Dataset
        Output of :func:`load` for a dataset target (with ``x``, ``y``).
    static : xr.Dataset
        ``los_east``, ``los_north``, ``los_up`` on the frame grid
        (:func:`disp_xr.open_static`).

    Returns
    -------
    xr.Dataset
        ``gnss`` with ``los`` and ``sigma_los`` (positive toward the
        satellite) and the LOS vector components per station.

    Examples
    --------
    >>> los = dxr.gnss.to_los(gps, dxr.open_static(ds))  # doctest: +SKIP
    >>> los["los"].sel(station="P539").plot()  # doctest: +SKIP

    """
    from geepers.uncertainty import get_sigma_los

    from .geometry.los import enu_to_los

    if "x" not in gnss.coords:
        raise ValueError("gnss needs x and y station coordinates (load with a dataset)")
    at = {"x": gnss["x"], "y": gnss["y"]}
    vectors = static[["los_east", "los_north", "los_up"]].sel(at, method="nearest")
    vectors = vectors.drop_vars(["x", "y", "spatial_ref"], errors="ignore").compute()
    out = gnss.copy()
    out["los"] = enu_to_los(gnss["east"], gnss["north"], gnss["up"], vectors)
    out["los"].attrs = {
        "units": "meters",
        "long_name": "GNSS motion toward the satellite",
    }
    if all(name in gnss for name in UNCERTAINTY):
        sigma = np.full(gnss["east"].shape, np.nan)
        order = gnss["east"].dims.index("station")
        for k in range(gnss.sizes["station"]):
            vec = np.array(
                [float(vectors[c][k]) for c in ("los_east", "los_north", "los_up")]
            )
            row = gnss.isel(station=k)
            value = get_sigma_los(vec, *(row[name].values for name in UNCERTAINTY))
            index: list[Any] = [slice(None)] * sigma.ndim
            index[order] = k
            sigma[tuple(index)] = value
        out["sigma_los"] = (gnss["east"].dims, sigma, {"units": "meters"})
    for name in ("los_east", "los_north", "los_up"):
        out = out.assign_coords({name: ("station", vectors[name].values)})
    return out


def _pixel_size(ds: Union[xr.Dataset, xr.DataArray]) -> float:
    return abs(float(ds["x"][1] - ds["x"][0]))


def sample(
    ds: xr.Dataset,
    gnss: Union[xr.Dataset, Any],
    *,
    buffer_m: float = 0.0,
    layer: str = "displacement",
) -> xr.DataArray:
    """Read DISP at the stations: the median in a window around each one.

    Uses geepers' ``XarrayReader.read_window`` (as its ``sample_insar``
    does), with the station's pixel found by flooring (geepers rounds, which
    reads the neighboring pixel for half the stations). With ``buffer_m``,
    geepers releases that have ``buffer_meters`` take a circular window of
    that radius; older ones (0.3) a square window of the same half-width in
    pixels.

    Parameters
    ----------
    ds : xr.Dataset
        DISP data with a CRS.
    gnss : xr.Dataset or DataFrame
        Output of :func:`load` (``lon``, ``lat`` station coordinates), or a
        table with ``lon``, ``lat`` (and ``id``) columns.
    buffer_m : float
        Window radius in meters (0 reads the station's pixel).
    layer : str
        Layer to read.

    Returns
    -------
    xr.DataArray
        ``(station, time)`` DISP at each station.

    Examples
    --------
    >>> insar = dxr.gnss.sample(ds, gps, buffer_m=90)  # doctest: +SKIP

    """
    import math
    import warnings

    from geepers.io import XarrayReader

    if isinstance(gnss, xr.Dataset):
        table = pd.DataFrame(
            {
                "id": gnss["station"].values,
                "lon": gnss["lon"].values,
                "lat": gnss["lat"].values,
            }
        )
    else:
        table = pd.DataFrame(gnss)
    da = ds[layer]
    if not da.attrs.get("units"):
        raise ValueError(
            f"{layer!r} has no units attribute (arithmetic can drop it); "
            "set it, for example ds[layer].attrs['units'] = 'meters'"
        )
    reader = XarrayReader(da)
    kwargs: dict[str, Any] = {}
    if "buffer_meters" in inspect.signature(reader.read_window).parameters and buffer_m:
        pixels, kwargs["buffer_meters"] = 0, buffer_m
    else:
        pixels = int(round(buffer_m / _pixel_size(ds)))
    # geepers rounds (col, row) from the corner-anchored inverse transform,
    # which picks the next pixel whenever a station sits in the right or lower
    # half of its pixel; the pixel holding a point is the floor.
    windows = reader.read_window(
        table["lon"].to_numpy(),
        table["lat"].to_numpy(),
        pixels,
        op=math.floor,
        **kwargs,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="All-NaN slice")
        sampled = xr.concat(
            [w.median(dim=("x", "y"), skipna=True) for w in windows], dim="pixel"
        ).compute()
    sampled = sampled.rename({"pixel": "station"})
    sampled = sampled.assign_coords(
        station=table["id"].to_numpy() if "id" in table else np.arange(len(table))
    )
    sampled = sampled.drop_vars(
        [c for c in sampled.coords if c not in ("station", "time")], errors="ignore"
    )
    dims = [d for d in ("station", "time") if d in sampled.dims]
    out = sampled.transpose(*dims, ...)
    out.attrs = {**da.attrs, "gnss_buffer_m": float(buffer_m)}
    return out.rename(layer)


def _interp_linear(gnss: xr.DataArray, times: pd.DatetimeIndex) -> xr.DataArray:
    """Linear interpolation in time per station, NaN outside each record."""
    source = gnss["time"].values.astype("datetime64[ns]").astype(np.int64).astype(float)
    target = times.values.astype("datetime64[ns]").astype(np.int64).astype(float)

    def one(series: np.ndarray) -> np.ndarray:
        ok = np.isfinite(series)
        if ok.sum() < 2:
            return np.full(target.shape, np.nan)
        return np.interp(target, source[ok], series[ok], left=np.nan, right=np.nan)

    out = xr.apply_ufunc(
        one,
        gnss,
        input_core_dims=[["time"]],
        output_core_dims=[["new_time"]],
        vectorize=True,
    )
    return out.rename(new_time="time").assign_coords(time=times).transpose(*gnss.dims)


def centred(gnss: xr.DataArray) -> xr.DataArray:
    """Stamp daily solutions at the centre of the UTC day they cover.

    Daily solutions (UNR and most others) are dated at midnight but average
    the whole UTC day. Matched by nearest timestamp, an acquisition after
    12:00 UTC would take the next day's solution, so a series whose stamps
    are all at midnight is moved to noon. Sub-daily series are returned
    unchanged.

    Parameters
    ----------
    gnss : xr.DataArray
        Series with a ``time`` coordinate.

    Returns
    -------
    xr.DataArray
        ``gnss`` with its ``time`` moved by 12 h when it is daily.

    Examples
    --------
    >>> on_noon = dxr.gnss.centred(los["los"])  # doctest: +SKIP

    """
    stamps = pd.DatetimeIndex(gnss["time"].values)
    if len(stamps) and (stamps == stamps.normalize()).all():
        return gnss.assign_coords(time=stamps + pd.Timedelta(hours=12))
    return gnss


def align(
    gnss: xr.DataArray,
    times: Union[xr.DataArray, pd.DatetimeIndex, np.ndarray],
    *,
    reference: Optional[Any] = None,
    method: Literal["nearest", "linear"] = "nearest",
    tolerance: str = "1D",
) -> xr.DataArray:
    """Put a GNSS series on the DISP epochs and the DISP reference date.

    Parameters
    ----------
    gnss : xr.DataArray
        ``(station, time)`` GNSS series, for example ``to_los(...)["los"]``.
    times : array of datetimes
        DISP epochs (``ds.time``).
    reference : date-like, optional
        DISP reference date; default the first epoch (a chained stack is
        relative to it).
    method : {"nearest", "linear"}
        Nearest GNSS solution within ``tolerance``, or linear interpolation.
        Daily solutions count as stamped at noon (:func:`centred`), so an
        epoch takes the solution of its own UTC day.
    tolerance : str
        Largest distance in time for ``"nearest"`` (pandas offset).

    Returns
    -------
    xr.DataArray
        GNSS at ``times`` minus its value at ``reference``; NaN where the
        GNSS series does not reach an epoch. A station with no solution at
        ``reference`` (installed later, or a gap that day) is referenced to
        its first epoch instead, so its residual against DISP carries a
        constant offset, which rates and mean-removed statistics ignore.
        The ``gnss_reference`` coordinate gives each station's reference
        epoch.

    Examples
    --------
    >>> on_epochs = dxr.gnss.align(los["los"], ds.time)  # doctest: +SKIP

    """
    times = pd.DatetimeIndex(np.asarray(times))
    reference = times[0] if reference is None else pd.Timestamp(reference)
    gnss = centred(gnss)
    if method == "nearest":
        on = gnss.reindex(
            time=times, method="nearest", tolerance=pd.Timedelta(tolerance)
        )
        at_ref = gnss.reindex(
            time=[reference], method="nearest", tolerance=pd.Timedelta(tolerance)
        ).isel(time=0, drop=True)
    elif method == "linear":
        on = _interp_linear(gnss, times)
        at_ref = _interp_linear(gnss, pd.DatetimeIndex([reference])).isel(
            time=0, drop=True
        )
    else:
        raise ValueError(f"method must be 'nearest' or 'linear', not {method!r}")
    first = np.isfinite(on).argmax("time").compute()
    at_first = on.isel(time=first)
    missing = at_ref.isnull()
    out = on - at_ref.where(~missing, at_first)
    reference_epoch = (
        on["time"]
        .isel(time=first)
        .where(missing, np.datetime64(reference, "ns"))
        .drop_vars("time")
    )
    out = out.assign_coords(gnss_reference=reference_epoch)
    out.attrs = {**gnss.attrs, "reference_time": str(reference.date())}
    return out
