"""Coordinate and grid helpers for DISP frames.

The static-layer download helpers live in :mod:`disp_xr.geometry.static`, which
needs the ``aws`` extra.
"""

import logging
import re
import warnings
from pathlib import Path
from typing import NamedTuple, Optional, Tuple, Union

import h5py
import numpy as np
import rioxarray  # noqa: F401  (registers the .rio accessor)
import xarray as xr
import yaml  # type: ignore
from affine import Affine
from pyproj import CRS, Transformer
from rasterio.warp import Resampling

logger = logging.getLogger(__name__)

__all__ = [
    "Grid",
    "OutOfBoundsError",
    "get_burst_ids",
    "get_extent",
    "grid",
    "latlon_to_utm",
    "lonlat_to_rowcol",
    "match_grid",
    "utm_to_latlon",
]


class Grid(NamedTuple):
    """CRS, affine transform and ``(y, x)`` shape of a raster grid.

    Examples
    --------
    >>> g = grid(ds)  # doctest: +SKIP
    >>> g.crs, g.transform, g.shape  # doctest: +SKIP

    """

    crs: Optional[CRS]
    transform: Affine
    shape: tuple[int, int]


def grid(obj: Union[xr.Dataset, xr.DataArray]) -> Grid:
    """Return the grid of a DISP dataset.

    All grid and CRS access in disp-xr goes through this function, so a raster
    index (for example rasterix) can replace rioxarray in one place.

    Examples
    --------
    >>> g = grid(ds)  # doctest: +SKIP

    """
    crs = obj.rio.crs
    return Grid(
        crs=CRS.from_user_input(crs) if crs is not None else None,
        transform=obj.rio.transform(),
        shape=(obj.sizes["y"], obj.sizes["x"]),
    )


def match_grid(
    da: xr.DataArray,
    like: Union[xr.Dataset, xr.DataArray],
    resampling: Resampling = Resampling.bilinear,
) -> xr.DataArray:
    """Put ``da`` on the ``(y, x)`` grid of ``like``.

    When both share a CRS and the pixel centers of ``like`` are pixel centers
    of ``da`` (the same frame, or a subset of it), the pixels are selected,
    which stays lazy and exact. Otherwise ``da`` is resampled with
    ``rio.reproject_match``.

    Examples
    --------
    >>> tropo_on_disp = match_grid(tropo, ds)  # doctest: +SKIP

    """
    if da.rio.crs == like.rio.crs:
        dx = abs(float(like.x[1] - like.x[0])) if like.sizes["x"] > 1 else 1.0
        try:
            out = da.sel(x=like.x, y=like.y, method="nearest", tolerance=dx / 100)
            return out.assign_coords(x=like.x, y=like.y)
        except KeyError:
            pass
    return da.rio.reproject_match(like, resampling=resampling)


class OutOfBoundsError(ValueError):
    """A point lies outside the raster grid."""


def lonlat_to_rowcol(
    obj: Union[xr.Dataset, xr.DataArray], lon: float, lat: float
) -> Tuple[int, int]:
    """Return the ``(row, col)`` of the pixel that contains a lon/lat point.

    Raises
    ------
    OutOfBoundsError
        If the point is outside the grid.

    Examples
    --------
    >>> row, col = lonlat_to_rowcol(ds, -120.43, 35.90)  # doctest: +SKIP

    """
    g = grid(obj)
    if g.crs is None:
        raise ValueError("The dataset has no CRS; cannot convert lon/lat")
    x, y = Transformer.from_crs("EPSG:4326", g.crs, always_xy=True).transform(lon, lat)
    col, row = ~g.transform * (x, y)
    # floor, not round: the transform maps pixel corners, so (0.7, 0.7) is
    # inside pixel (0, 0).
    row, col = int(np.floor(row)), int(np.floor(col))
    if not (0 <= row < g.shape[0] and 0 <= col < g.shape[1]):
        raise OutOfBoundsError(
            f"Point (lon={lon}, lat={lat}) is outside the {g.shape} grid"
        )
    return row, col


def latlon_to_utm(lat: float, lon: float, utm_epsg: int) -> Tuple[float, float]:
    """Convert latitude and longitude coordinates to UTM coordinates.

    Parameters
    ----------
    lat : float
        Latitude coordinate in decimal degrees.
    lon : float
        Longitude coordinate in decimal degrees.
    utm_epsg : int
        The EPSG code for the UTM zone to which the coordinates should be transformed.

    Returns
    -------
    Tuple[float, float]
        A tuple containing the UTM easting and northing coordinates.

    Examples
    --------
    >>> easting, northing = latlon_to_utm(35.90, -120.43, 32610)
    >>> round(easting / 1e3)
    732

    """
    transformer = Transformer.from_crs("epsg:4326", f"epsg:{utm_epsg}", always_xy=True)
    return transformer.transform(lon, lat)


def utm_to_latlon(
    easting: float, northing: float, utm_epsg: int
) -> Tuple[float, float]:
    """Convert UTM coordinates to latitude and longitude.

    Parameters
    ----------
    easting : float
        UTM easting (X) coordinate in meters.
    northing : float
        UTM northing (Y) coordinate in meters.
    utm_epsg : int
        The EPSG code of the UTM projection (e.g. 32611 for UTM zone 11N).

    Returns
    -------
    Tuple[float, float]
        (latitude, longitude) in decimal degrees.

    Examples
    --------
    >>> lat, lon = utm_to_latlon(500000.0, 4000000.0, 32610)
    >>> round(lon, 2)
    -123.0

    """
    transformer = Transformer.from_crs(f"epsg:{utm_epsg}", "epsg:4326", always_xy=True)
    lon, lat = transformer.transform(easting, northing)
    return lat, lon


def get_extent(ds: xr.Dataset) -> Tuple[float, float, float, float]:
    """Calculate the DISP spatial extent in UTM.

    Parameters
    ----------
    ds : xr.Dataset
        The xarray dataset, which must have a spatial reference
        with a 'GeoTransform' attribute and 'x' and 'y' dimensions.

    Returns
    -------
    Tuple[float, float, float, float]
        A tuple representing the extent (x0, x1, y1, y0) where:
        - x0 is the starting x-coordinate,
        - x1 is the ending x-coordinate,
        - y1 is the starting y-coordinate,
        - y0 is the ending y-coordinate.

    .. deprecated::
        Use ``ds.rio.bounds()`` (rioxarray), which returns
        ``(left, bottom, right, top)``.

    Examples
    --------
    >>> west, south, east, north = get_extent(ds)  # doctest: +SKIP

    """
    warnings.warn(
        "get_extent is deprecated; use ds.rio.bounds()",
        DeprecationWarning,
        stacklevel=2,
    )

    # Extracting GeoTransform values
    transform = ds.spatial_ref.attrs["GeoTransform"].split(" ")
    x0, dx, _, y0, _, dy = map(float, transform)

    # Calculating the extent
    x1 = x0 + dx * ds.sizes["x"]
    y1 = y0 + dy * ds.sizes["y"]

    return (x0, x1, y1, y0)


def get_burst_ids(disp_path: Union[str, Path]) -> list[str]:
    """Extract burst ID and track ID from a DISP file path.

    Parameters
    ----------
    disp_path : str | Path
        Path to the DISP file.

    Returns
    -------
    list[str]
        Burst IDs in ASF format, ``T<track>_<burst>_IW<n>``.

    Examples
    --------
    >>> get_burst_ids("OPERA_L3_DISP-S1_IW_F11116_....nc")  # doctest: +SKIP

    """

    def _get_id(path: str) -> Optional[str]:
        """Extract the burst ID from the file name."""
        match = re.search(r"\d{6}-IW\d", path.split("/")[-1])
        if match:  # Check if a match was found
            return match.group(0)
        return None  # Return None if no match is found

    def _get_track(path: str) -> Optional[str]:
        """Extract the track ID from the file name."""
        match = re.search(r"T\d{3}", path.split("/")[-1])
        if match:  # Check if a match was found
            return match.group(0)
        return None  # Return None if no match is found

    # Get static layers from metadata
    with h5py.File(disp_path, "r") as f:
        runconfig = f["metadata"]["pge_runconfig"][()].decode()
        runconfig = yaml.safe_load(runconfig)
        static_lyrs = runconfig["dynamic_ancillary_file_group"]
        static_lyrs = static_lyrs["static_layers_files"]

    # Get burst and track ids from static layers
    burst_ids = np.unique(list(map(_get_id, static_lyrs)))
    track_id = np.unique(list(map(_get_track, static_lyrs)))[0]

    # Reformat to asf_vertex format
    burst_ids = [track_id + "_" + burst for burst in burst_ids]
    burst_ids = [re.sub(r"-", "_", burst) for burst in burst_ids]
    return burst_ids
