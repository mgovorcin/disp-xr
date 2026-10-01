"""Coordinate and grid helpers for DISP frames.

The static-layer download helpers live in :mod:`disp_xr.geometry.static`, which
needs the ``aws`` extra.
"""

import logging
import re
from pathlib import Path
from typing import Optional, Tuple, Union

import h5py
import numpy as np
import xarray as xr
import yaml  # type: ignore
from pyproj import Transformer

logger = logging.getLogger(__name__)

__all__ = ["get_burst_ids", "get_extent", "latlon_to_utm", "utm_to_latlon"]


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

    """
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
