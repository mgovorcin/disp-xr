"""Re-reference displacement to a point, an area or the most stable pixel.

``ReferenceMethod`` and :func:`get_reference_values` are adapted from opera-utils
(src/opera_utils/disp/_enums.py and src/opera_utils/disp/_reference.py, commit
dff4b45), and the center-of-mass pick in :func:`find_stable_point` from
``find_reference_point`` in src/opera_utils/disp/rebase_reference.py. Copyright
(c) 2022 California Institute of Technology, BSD-3-Clause. See
THIRD_PARTY_NOTICES.md. Changes: the point is bounds-checked, the median honors
the good-pixel mask, ``high_coherence`` requires a mask, border pixels are
selected without ``dropna`` (which forces a compute), and the area and auto
methods are new.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from enum import Enum
from typing import Any, Optional, Union

import numpy as np
import xarray as xr
from scipy import ndimage

from ._history import recorded
from .geometry import OutOfBoundsError, grid, lonlat_to_rowcol

logger = logging.getLogger(__name__)

__all__ = [
    "OutOfBoundsError",
    "ReferenceMethod",
    "find_reference_point",
    "find_stable_point",
    "get_reference_values",
    "rereference",
]

DISPLACEMENT_LAYERS = ("displacement", "short_wavelength_displacement")


class ReferenceMethod(str, Enum):
    """How the per-epoch reference value is chosen."""

    none = "none"
    """No re-referencing."""
    point = "point"
    """The value of one pixel."""
    median = "median"
    """Median of the good pixels of the whole scene."""
    border = "border"
    """Median of the good pixels in a border strip."""
    high_coherence = "high_coherence"
    """Median of the pixels of a required good-pixel mask."""
    area = "area"
    """Median of the good pixels inside a bbox or polygon."""
    auto = "auto"
    """The value of the pixel chosen by :func:`find_stable_point`."""

    def __str__(self) -> str:
        return self.value


def _check_point(da: xr.DataArray, row: int, col: int) -> tuple[int, int]:
    ny, nx = da.sizes["y"], da.sizes["x"]
    if not (0 <= row < ny and 0 <= col < nx):
        raise OutOfBoundsError(
            f"Point (row={row}, col={col}) is outside the {(ny, nx)} grid"
        )
    return row, col


def _spatial_median(da: xr.DataArray, good: Optional[xr.DataArray]) -> xr.DataArray:
    if good is not None:
        da = da.where(good)
    stacked = da.stack(pixels=("y", "x"))
    if stacked.chunks is not None:
        stacked = stacked.chunk({"pixels": -1})
    return stacked.median("pixels", skipna=True).drop_vars(
        ["pixels", "y", "x"], errors="ignore"
    )


def _border(da: xr.DataArray, n: int) -> xr.DataArray:
    """Border strips of width ``n`` as a ``(..., y, x)``-free ``pixels`` axis."""
    ny = da.sizes["y"]
    strips = [
        da.isel(y=slice(None, n)),
        da.isel(y=slice(ny - n, None)),
        da.isel(y=slice(n, ny - n), x=slice(None, n)),
        da.isel(y=slice(n, ny - n), x=slice(-n, None)),
    ]
    stacked = [
        s.stack(pixels=("y", "x")).drop_vars(["pixels", "y", "x"]) for s in strips
    ]
    return xr.concat(stacked, dim="pixels")


def get_reference_values(
    da: xr.DataArray,
    *,
    method: Union[str, ReferenceMethod],
    row: Optional[int] = None,
    col: Optional[int] = None,
    lon: Optional[float] = None,
    lat: Optional[float] = None,
    border_pixels: int = 3,
    good_pixel_mask: Optional[Any] = None,
) -> xr.DataArray:
    """Return the value to subtract from ``da`` at each epoch.

    Parameters
    ----------
    da : xr.DataArray
        Displacement, dims ``(time, y, x)``.
    method : {"point", "median", "border", "high_coherence"}
        See :class:`ReferenceMethod`. ``area`` and ``auto`` are handled by
        :func:`rereference`.
    row, col : int, optional
        Pixel for ``point``.
    lon, lat : float, optional
        Point in degrees for ``point``, converted with the grid of ``da``.
    border_pixels : int
        Border width for ``border``.
    good_pixel_mask : array-like of bool, optional
        True where a pixel may be used. Required for ``high_coherence``.

    Returns
    -------
    xr.DataArray
        Dims ``(time,)``.

    Raises
    ------
    OutOfBoundsError
        If the point is outside the grid.
    ValueError
        If the point or the ``high_coherence`` mask is missing.

    """
    method = ReferenceMethod(method)
    good = None
    if good_pixel_mask is not None:
        good = (
            good_pixel_mask
            if isinstance(good_pixel_mask, xr.DataArray)
            else xr.DataArray(
                np.asarray(good_pixel_mask), dims=da.dims[-good_pixel_mask.ndim :]
            )
        )

    if method is ReferenceMethod.point:
        if row is None or col is None:
            if lon is None or lat is None:
                raise ValueError("Need (row, col) or (lon, lat)")
            row, col = lonlat_to_rowcol(da, lon, lat)
        row, col = _check_point(da, row, col)
        return da.isel(y=row, x=col)
    if method is ReferenceMethod.median:
        return _spatial_median(da, good)
    if method is ReferenceMethod.border:
        border = _border(da.where(good) if good is not None else da, border_pixels)
        if border.chunks is not None:
            border = border.chunk({"pixels": -1})
        return border.median("pixels", skipna=True)
    if method is ReferenceMethod.high_coherence:
        if good is None:
            raise ValueError("high_coherence needs good_pixel_mask")
        return _spatial_median(da, good)
    raise ValueError(f"get_reference_values does not handle method {str(method)!r}")


def _area_mask(
    ds: xr.Dataset,
    bbox: Optional[Sequence[float]],
    geometry: Optional[Any],
) -> xr.DataArray:
    if (bbox is None) == (geometry is None):
        raise ValueError("area needs exactly one of bbox or geometry")
    import shapely  # in the aws extra
    from pyproj import Transformer
    from shapely.ops import transform

    if bbox is not None:
        geometry = shapely.box(*bbox)

    crs = grid(ds).crs
    if crs is None:
        raise ValueError("The dataset has no CRS; cannot place a lon/lat area")
    to_grid = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform
    projected = transform(to_grid, geometry)
    xx, yy = np.meshgrid(ds["x"].values, ds["y"].values)
    inside = shapely.contains_xy(projected, xx, yy)
    if not inside.any():
        raise OutOfBoundsError("The area holds no pixel centers of the grid")
    return xr.DataArray(inside, dims=("y", "x"), coords={"y": ds["y"], "x": ds["x"]})


def _phase_jump_epochs(residuals: xr.DataArray) -> xr.DataArray:
    # The L1 inversion residual of a cycle slip is about a multiple of pi, so
    # anything above half of pi counts as a jump.
    return (residuals >= np.pi / 2).sum("time")


def find_stable_point(
    ds: xr.Dataset,
    *,
    max_quality: float = 0.95,
    min_quality: float = 0.5,
    step: float = 0.05,
) -> tuple[int, int]:
    """Choose a stable reference pixel, deterministically.

    Candidates are pixels never masked by ``recommended_mask``, with no phase
    jump in ``timeseries_inversion_residuals``, whose harmonic-mean temporal
    coherence is at least a threshold. The threshold steps down from
    ``max_quality`` to ``min_quality`` until there are candidates. Persistent
    scatterers are preferred when any candidate is one. The pick is the
    candidate closest to the candidates' center of mass, so it sits inside the
    good area rather than on an isolated pixel.

    The quality layers are computed (2D maps); displacement is not read.

    Raises
    ------
    ValueError
        If no pixel reaches ``min_quality``.

    """
    from .quality import quality_summary

    summary = quality_summary(ds).compute()
    candidates_base = (summary["pct_masked"] == 0).values
    if "timeseries_inversion_residuals" in ds:
        jumps = _phase_jump_epochs(ds["timeseries_inversion_residuals"]).compute()
        candidates_base &= (jumps == 0).values
    coherence = summary["temporal_coherence_hmean"].fillna(0).values
    ps = (summary["pct_ps"] == 100).values

    for threshold in np.arange(max_quality, min_quality - step / 2, -step):
        candidates = candidates_base & (coherence >= threshold - 1e-9)
        if not candidates.any():
            continue
        center = ndimage.center_of_mass(candidates)
        if (candidates & ps).any():
            candidates &= ps
        rows, cols = np.nonzero(candidates)
        idx = np.argmin((rows - center[0]) ** 2 + (cols - center[1]) ** 2)
        return int(rows[idx]), int(cols[idx])
    raise ValueError(f"No stable pixel with harmonic-mean coherence >= {min_quality}")


@recorded
def rereference(
    ds: xr.Dataset,
    method: Union[str, ReferenceMethod] = "auto",
    *,
    row: Optional[int] = None,
    col: Optional[int] = None,
    lon: Optional[float] = None,
    lat: Optional[float] = None,
    bbox: Optional[Sequence[float]] = None,
    geometry: Optional[Any] = None,
    border_pixels: int = 3,
    good_pixels: Optional[xr.DataArray] = None,
    check_point: bool = True,
) -> xr.Dataset:
    """Subtract a per-epoch reference value from the displacement layers.

    Parameters
    ----------
    ds : xr.Dataset
        DISP dataset, usually after :func:`disp_xr.stack_disp`.
    method : str or ReferenceMethod
        ``"point"``, ``"median"``, ``"border"``, ``"high_coherence"`` (needs
        ``good_pixels``), ``"area"`` or ``"auto"`` (:func:`find_stable_point`).
    row, col : int, optional
        Pixel for ``point``.
    lon, lat : float, optional
        Point in degrees for ``point``, used when ``row``/``col`` are not given.
    bbox : sequence of float, optional
        ``(west, south, east, north)`` in degrees for ``area``.
    geometry : shapely geometry, optional
        Polygon in lon/lat for ``area``. ``area`` needs shapely (``aws`` extra).
    border_pixels : int
        Border width for ``border``.
    good_pixels : xr.DataArray of bool, optional
        Pixels the medians may use, for example
        ``disp_xr.valid_mask(ds, "strict")``.
    check_point : bool
        For ``point`` and ``auto``, read the reference pixel's displacement and
        raise if it is NaN at every epoch.

    Returns
    -------
    xr.Dataset
        ``displacement`` and ``short_wavelength_displacement`` re-referenced;
        ``attrs["reference"]`` records the method and, for a point, its row,
        column and map coordinates.

    Raises
    ------
    OutOfBoundsError
        If the point is outside the grid or masked, or the area is empty.

    """
    method = ReferenceMethod(method)
    if method is ReferenceMethod.none:
        return ds
    record: dict[str, Any] = {"method": str(method)}
    kwargs: dict[str, Any] = {
        "border_pixels": border_pixels,
        "good_pixel_mask": good_pixels,
    }

    if method is ReferenceMethod.auto:
        row, col = find_stable_point(ds)
        method = ReferenceMethod.point
    if method is ReferenceMethod.point:
        if row is None or col is None:
            if lon is None or lat is None:
                raise ValueError("Need (row, col) or (lon, lat)")
            row, col = lonlat_to_rowcol(ds, lon, lat)
        row, col = _check_point(ds["displacement"], row, col)
        if check_point and bool(ds["displacement"].isel(y=row, x=col).isnull().all()):
            raise OutOfBoundsError(f"Reference pixel (row={row}, col={col}) is masked")
        record.update(
            row=row,
            col=col,
            x=float(ds["x"][col]),
            y=float(ds["y"][row]),
        )
        kwargs.update(row=row, col=col)
    if method is ReferenceMethod.area:
        area = _area_mask(ds, bbox, geometry)
        kwargs["good_pixel_mask"] = area if good_pixels is None else area & good_pixels
        method = ReferenceMethod.median
        if bbox is not None:
            record["bbox"] = list(bbox)
        else:
            record["geometry"] = geometry.wkt  # type: ignore[union-attr]

    out = ds.copy()
    for name in DISPLACEMENT_LAYERS:
        if name in ds:
            ref = get_reference_values(ds[name], method=method, **kwargs)
            # Non-index coordinates (spatial_ref, reference_time) come from ds[name]
            # already; comparing them again could compute dask coordinates.
            out[name] = ds[name] - ref.reset_coords(drop=True)
            out[name].attrs = ds[name].attrs
    out.attrs["reference"] = json.dumps(record)
    logger.info(f"Re-referenced displacement: {record}")
    return out


def find_reference_point(
    pct_mask: np.ndarray,
    pct_ps: np.ndarray,
    phase_jumps: Optional[np.ndarray] = None,
    percentile: int = 99,
    max_jumps: int = 1,
) -> tuple[int, int]:
    """Select a reference pixel from PS and mask percentages (legacy API).

    Candidates are pixels whose PS percentage is at or above the
    ``percentile`` of pixels that are PS at least once and not always masked,
    and (if ``phase_jumps`` is given) with at most ``max_jumps`` jumps. The
    candidate closest to the candidates' center of mass is returned, so the
    pick no longer changes from run to run (bug 7). Prefer
    ``rereference(ds, "auto")``.

    Returns
    -------
    tuple[int, int]
        ``(row, col)`` of the selected pixel.

    Raises
    ------
    ValueError
        If there is no candidate.

    """
    valid_pixels = np.ma.masked_where((pct_ps == 0) | (pct_mask == 100), pct_ps)
    threshold = np.nanpercentile(valid_pixels.filled(np.nan), percentile)
    valid = (valid_pixels >= threshold).filled(False)
    if phase_jumps is not None:
        valid &= phase_jumps <= max_jumps
    if not valid.any():
        raise ValueError(
            "No valid reference point found. Adjust threshold or check input data."
        )
    center = ndimage.center_of_mass(valid)
    rows, cols = np.nonzero(valid)
    idx = np.argmin((rows - center[0]) ** 2 + (cols - center[1]) ** 2)
    logger.info(f"Selected reference pixel (y/x): {(rows[idx], cols[idx])}")
    return int(rows[idx]), int(cols[idx])
