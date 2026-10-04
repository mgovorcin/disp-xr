"""Quick-look maps and point time series (``plot`` extra).

Plotting is one of the two places disp-xr computes (with export): a map loads
one epoch, a point loads one pixel series. Both convert lengths to ``units``
(millimeters by default) and label the axes with them.

Displacement is signed (positive toward the satellite), so maps use a
diverging blue-gray-red scale centered on zero, with limits symmetric about
it. Point plots draw a dashed rule at every reference-date change, where a
chained series is joined or an unchained one restarts at zero.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Optional, Union

import numpy as np
import pandas as pd
import xarray as xr

from .config import bounded_threads
from .geometry import grid, lonlat_to_rowcol
from .units import _canonical, to_units

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.colors import Colormap

__all__ = ["diverging_cmap", "plot_hex", "plot_map", "plot_point"]

# Diverging pair (blue <-> red) around a neutral gray midpoint, and the
# series color, from the disp-xr reference palette. The outer steps darken
# each arm so the extremes stay distinct from the mid tones.
DIVERGING = ("#123d73", "#2a78d6", "#f0efec", "#e34948", "#7a1716")
SERIES = "#2a78d6"
RULE = "#8a8984"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"


def _pyplot() -> Any:
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError(
            "Plotting needs matplotlib: pip install 'disp_xr[plot]'"
        ) from e
    return plt


def diverging_cmap() -> Colormap:
    """Return the disp-xr diverging colormap (blue negative, red positive).

    Examples
    --------
    >>> from matplotlib.colors import to_hex
    >>> cmap = diverging_cmap()
    >>> to_hex(cmap(0.0)), to_hex(cmap(1.0))
    ('#123d73', '#7a1716')

    """
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("disp_xr_diverging", DIVERGING, N=256)


def _in_units(da: xr.DataArray, units: Optional[str]) -> xr.DataArray:
    """Convert a length layer to ``units``; other layers are left as they are."""
    if units is None or _canonical(str(da.attrs.get("units", ""))) is None:
        return da
    return to_units(da, units)


def _layer(obj: Union[xr.Dataset, xr.DataArray], layer: str) -> xr.DataArray:
    if isinstance(obj, xr.DataArray):
        return obj
    if layer not in obj:
        raise ValueError(f"No layer {layer!r}; have {sorted(obj.data_vars)}")
    # A shallow copy: the attrs dict of ds[layer] is the variable's own.
    da = obj[layer].copy(deep=False)
    if "reference_date" in obj.attrs:
        da.attrs = {**da.attrs, "reference_date": obj.attrs["reference_date"]}
    return da


def _style(ax: Axes) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)


def _date(value: Any) -> str:
    return f"{pd.Timestamp(value):%Y-%m-%d}"


@bounded_threads
def plot_map(
    obj: Union[xr.Dataset, xr.DataArray],
    time: Optional[Any] = None,
    *,
    layer: str = "displacement",
    units: Optional[str] = "mm",
    vmax: Optional[float] = None,
    percentile: float = 98.0,
    cmap: Optional[Union[str, Colormap]] = None,
    max_pixels: int = 2000,
    ax: Optional[Axes] = None,
    title: Optional[str] = None,
) -> Axes:
    """Map one epoch of a layer (or a 2D layer such as velocity).

    Parameters
    ----------
    obj : xr.Dataset or xr.DataArray
        A DISP dataset (``layer`` is plotted) or one layer, ``(time, y, x)``
        or ``(y, x)``, with a CRS.
    time : date-like, optional
        Epoch to plot, the nearest one to this date. Defaults to the last.
    layer : str
        Layer of a Dataset to plot.
    units : str, optional
        Length unit for the color scale (``"mm"``, ``"cm"``, ``"m"``);
        ``None`` keeps the stored unit. Non-length layers are not converted.
    vmax : float, optional
        Color limit; the scale runs from ``-vmax`` to ``vmax``. Defaults to
        the ``percentile`` of the absolute values, so outliers do not wash
        out the map.
    percentile : float
        Percentile of ``|value|`` used when ``vmax`` is not given.
    cmap : str or Colormap, optional
        Defaults to :func:`diverging_cmap`.
    max_pixels : int
        Larger maps are block-averaged so that neither side exceeds this many
        pixels before loading.
    ax : matplotlib Axes, optional
        Axes to draw on; a new figure otherwise.
    title : str, optional
        Defaults to the layer, the epoch and the reference date.

    Returns
    -------
    matplotlib.axes.Axes

    """
    plt = _pyplot()
    da = _layer(obj, layer)
    ref = da.attrs.get("reference_date")
    if "time" in da.dims:
        da = da.isel(time=-1) if time is None else da.sel(time=time, method="nearest")
    if set(da.dims) != {"y", "x"}:
        raise ValueError(f"plot_map needs (y, x) after selecting time, not {da.dims}")
    factor = max(1, math.ceil(max(da.sizes["y"], da.sizes["x"]) / max_pixels))
    if factor > 1:
        da = da.coarsen(y=factor, x=factor, boundary="trim").mean(keep_attrs=True)
    da = _in_units(da.astype("float64", keep_attrs=True), units).compute()

    if vmax is None:
        finite = np.abs(da.values[np.isfinite(da.values)])
        vmax = float(np.percentile(finite, percentile)) if finite.size else 1.0
        vmax = vmax or 1.0
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6), layout="constrained")
    image = da.plot.imshow(
        ax=ax,
        x="x",
        y="y",
        cmap=cmap or diverging_cmap(),
        vmin=-vmax,
        vmax=vmax,
        add_colorbar=True,
        cbar_kwargs={"label": f"{da.name or layer} ({da.attrs.get('units', '')})"},
        interpolation="nearest",
    )
    image.colorbar.outline.set_visible(False)
    image.colorbar.ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)
    ax.set_aspect("equal")
    # xarray titles the axes with the selected coordinates; ours replaces it.
    ax.set_title("")
    ax.ticklabel_format(style="plain", useOffset=False)
    crs = grid(da).crs
    epsg = crs.to_epsg() if crs is not None else None
    suffix = f" [EPSG:{epsg}]" if epsg else ""
    ax.set_xlabel(f"x (m){suffix}", color=TEXT_SECONDARY)
    ax.set_ylabel(f"y (m){suffix}", color=TEXT_SECONDARY)
    _style(ax)
    if title is None:
        parts = [str(da.name or layer)]
        if "time" in da.coords and da["time"].ndim == 0:
            parts.append(_date(da["time"].values))
        if ref:
            parts.append(f"relative to {_date(ref)}")
        title = ", ".join(parts)
    ax.set_title(title, loc="left", fontsize=11)
    return ax


def plot_hex(
    hexes: xr.Dataset,
    time: Optional[Any] = None,
    *,
    layer: str = "displacement",
    units: Optional[str] = "mm",
    vmax: Optional[float] = None,
    percentile: float = 98.0,
    cmap: Optional[Union[str, Colormap]] = None,
    ax: Optional[Axes] = None,
    title: Optional[str] = None,
) -> Axes:
    """Map one epoch of a layer of :func:`disp_xr.downsample_hex` cells.

    Draws each hexagon as a polygon (``PolyCollection``); empty cells are
    left blank. Parameters as in :func:`plot_map`.

    Returns
    -------
    matplotlib.axes.Axes

    """
    from matplotlib.collections import PolyCollection

    from .hexagons import cell_polygons

    plt = _pyplot()
    da = hexes[layer]
    if "time" in da.dims:
        da = da.isel(time=-1) if time is None else da.sel(time=time, method="nearest")
    if da.dims != ("point",):
        raise ValueError(f"plot_hex needs (point,) after selecting time, not {da.dims}")
    da = _in_units(da.astype("float64", keep_attrs=True), units).compute()
    values = da.values
    finite = np.isfinite(values)
    if vmax is None:
        vmax = (
            float(np.percentile(np.abs(values[finite]), percentile))
            if finite.any()
            else 1.0
        )
        vmax = vmax or 1.0
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6), layout="constrained")
    outlines = cell_polygons(hexes)
    polygons = [outlines[i] for i in np.flatnonzero(finite)]
    collection = PolyCollection(
        polygons, array=values[finite], cmap=cmap or diverging_cmap(), linewidths=0
    )
    collection.set_clim(-vmax, vmax)
    ax.add_collection(collection)
    ax.autoscale_view()
    colorbar = plt.colorbar(
        collection, ax=ax, label=f"{layer} ({da.attrs.get('units', '')})"
    )
    colorbar.outline.set_visible(False)
    colorbar.ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)
    ax.set_aspect("equal")
    ax.ticklabel_format(style="plain", useOffset=False)
    ax.set_xlabel("x (m)", color=TEXT_SECONDARY)
    ax.set_ylabel("y (m)", color=TEXT_SECONDARY)
    _style(ax)
    if title is None:
        if "h3_resolution" in hexes.attrs:
            size = f"H3 resolution {hexes.attrs['h3_resolution']}"
        else:
            size = f"{hexes.attrs['hex_spacing']:g} m hexagons"
        parts = [layer, size]
        if "time" in da.coords and da["time"].ndim == 0:
            parts.append(_date(da["time"].values))
        title = ", ".join(parts)
    ax.set_title(title, loc="left", fontsize=11)
    return ax


def _pixel_window(
    da: xr.DataArray, row: int, col: int, buffer_m: float
) -> xr.DataArray:
    if buffer_m <= 0:
        return da.isel(y=row, x=col)
    res = abs(float(grid(da).transform.a))
    half = math.ceil(buffer_m / res)
    window = da.isel(
        y=slice(max(row - half, 0), row + half + 1),
        x=slice(max(col - half, 0), col + half + 1),
    )
    return window.median(dim=("y", "x"), skipna=True, keep_attrs=True)


@bounded_threads
def plot_point(
    obj: Union[xr.Dataset, xr.DataArray],
    lon: Optional[float] = None,
    lat: Optional[float] = None,
    *,
    row: Optional[int] = None,
    col: Optional[int] = None,
    buffer_m: float = 0.0,
    layer: str = "displacement",
    units: Optional[str] = "mm",
    ax: Optional[Axes] = None,
    title: Optional[str] = None,
) -> Axes:
    """Plot the time series at one point, with reference-change markers.

    Parameters
    ----------
    obj : xr.Dataset or xr.DataArray
        A DISP dataset (``layer`` is plotted) or one ``(time, y, x)`` layer.
    lon, lat : float, optional
        Point location; or give ``row`` and ``col``.
    row, col : int, optional
        Pixel indices, instead of ``lon``/``lat``.
    buffer_m : float
        Plot the median of the pixels within this distance (a square window)
        instead of one pixel.
    layer : str
        Layer of a Dataset to plot.
    units : str, optional
        Length unit of the y axis; ``None`` keeps the stored unit.
    ax : matplotlib Axes, optional
        Axes to draw on; a new figure otherwise.
    title : str, optional
        Defaults to the layer and the point.

    Returns
    -------
    matplotlib.axes.Axes

    Raises
    ------
    OutOfBoundsError
        If the point is outside the grid.

    """
    plt = _pyplot()
    da = _layer(obj, layer)
    if "time" not in da.dims:
        raise ValueError(f"plot_point needs a time dimension, not {da.dims}")
    if lon is not None and lat is not None:
        row, col = lonlat_to_rowcol(da, lon, lat)
        where = f"lon {lon:.4f}, lat {lat:.4f}"
    elif row is not None and col is not None:
        where = f"row {row}, col {col}"
    else:
        raise ValueError("Give lon and lat, or row and col")
    series = _in_units(_pixel_window(da, row, col, buffer_m), units).compute()

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 3.5), layout="constrained")
    ax.plot(
        series["time"].values,
        series.values,
        color=SERIES,
        linewidth=2,
        marker="o",
        markersize=4,
        markeredgecolor="white",
        markeredgewidth=1,
    )
    if "reference_time" in series.coords:
        refs = pd.unique(series["reference_time"].values)
        for ref in refs[1:]:
            ax.axvline(
                ref,
                color=RULE,
                linewidth=1,
                linestyle=(0, (4, 3)),
            )
        if len(refs) > 1:
            ax.text(
                refs[1],
                1.0,
                " reference change",
                transform=ax.get_xaxis_transform(),
                va="top",
                fontsize=8,
                color=TEXT_SECONDARY,
            )
    ax.axhline(0, color=GRID, linewidth=1, zorder=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_ylabel(
        f"{series.name or layer} ({series.attrs.get('units', '')})",
        color=TEXT_SECONDARY,
    )
    _style(ax)
    if title is None:
        extra = f", median within {buffer_m:g} m" if buffer_m > 0 else ""
        title = f"{series.name or layer} at {where}{extra}"
    ax.set_title(title, loc="left", fontsize=11)
    return ax
