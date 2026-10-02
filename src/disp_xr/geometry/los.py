"""Line-of-sight geometry: unit vectors, angles and projections.

Sign convention (DISP products, dolphin): the LOS unit vector points from the
ground to the satellite, and positive LOS displacement is motion toward the
satellite. A pure uplift ``u`` therefore appears as ``u * los_up > 0``.

Every function works on numbers, NumPy arrays or (lazy) ``xr.DataArray``.
"""

from __future__ import annotations

from typing import TypeVar

import numpy as np
import xarray as xr

__all__ = [
    "enu_to_los",
    "incidence_azimuth_to_los",
    "los_to_incidence_azimuth",
    "los_up",
    "slant_range",
]

ArrayT = TypeVar("ArrayT", float, np.ndarray, xr.DataArray)

# Mean Earth radius (IUGG), as used by opera-utils ``get_slant_range``.
EARTH_RADIUS_M = 6371.0088e3


def los_up(east: ArrayT, north: ArrayT) -> ArrayT:
    """Return the up component of a unit LOS vector from east and north."""
    return np.sqrt(1 - east**2 - north**2)  # type: ignore[return-value]


def los_to_incidence_azimuth(east: ArrayT, north: ArrayT) -> tuple[ArrayT, ArrayT]:
    """Return incidence and azimuth angles, in degrees, of a ground-to-satellite LOS.

    Incidence is measured from the vertical. Azimuth is the direction from the
    ground to the satellite, clockwise from north, in ``[0, 360)``.

    Examples
    --------
    >>> inc, az = los_to_incidence_azimuth(0.5, 0.0)
    >>> round(float(inc), 6), round(float(az), 6)
    (30.0, 90.0)

    """
    incidence = np.degrees(np.arccos(los_up(east, north)))
    azimuth = np.degrees(np.arctan2(east, north)) % 360
    return incidence, azimuth  # type: ignore[return-value]


def incidence_azimuth_to_los(
    incidence: ArrayT, azimuth: ArrayT
) -> tuple[ArrayT, ArrayT, ArrayT]:
    """Return the ``(east, north, up)`` LOS unit vector from incidence and azimuth.

    Inverse of :func:`los_to_incidence_azimuth`.
    """
    inc, az = np.radians(incidence), np.radians(azimuth)
    horizontal = np.sin(inc)
    return horizontal * np.sin(az), horizontal * np.cos(az), np.cos(inc)  # type: ignore[return-value]


def enu_to_los(
    east: ArrayT,
    north: ArrayT,
    up: ArrayT,
    los: xr.Dataset | tuple[ArrayT, ArrayT, ArrayT],
) -> ArrayT:
    """Project east, north and up motion onto the line of sight.

    Parameters
    ----------
    east, north, up : float, array or xr.DataArray
        Motion components, in any unit; the result has the same unit.
    los : xr.Dataset or tuple
        A dataset with ``los_east``, ``los_north`` and ``los_up`` (as returned
        by :func:`disp_xr.open_static`) or the three components.

    Returns
    -------
    Same type as the inputs
        LOS motion, positive toward the satellite.

    Examples
    --------
    >>> float(enu_to_los(0.0, 0.0, 1.0, (0.5, 0.0, 0.75**0.5)))  # pure uplift
    0.8660254037844386

    """
    if isinstance(los, xr.Dataset):
        le, ln, lu = los["los_east"], los["los_north"], los["los_up"]
    else:
        le, ln, lu = los
    return east * le + north * ln + up * lu  # type: ignore[return-value]


def slant_range(incidence: ArrayT, orbit_height_m: float) -> ArrayT:
    """Approximate slant range, in meters, from incidence angle and orbit height.

    Spherical Earth of radius ``EARTH_RADIUS_M`` with the satellite at radius
    ``R_s = EARTH_RADIUS_M + orbit_height_m``. Solving the triangle ground point,
    Earth center, satellite for the side opposite the Earth center gives::

        range = -R_e cos(inc) + sqrt(R_s**2 - R_e**2 sin(inc)**2)

    the same triangle opera-utils ``geometry.get_slant_range`` solves with the
    law of sines, written so it stays exact at zero incidence.

    Examples
    --------
    >>> round(float(slant_range(0.0, 693_000.0)))
    693000

    """
    orbit_radius = EARTH_RADIUS_M + orbit_height_m
    inc = np.radians(incidence)
    return -EARTH_RADIUS_M * np.cos(inc) + np.sqrt(  # type: ignore[return-value]
        orbit_radius**2 - (EARTH_RADIUS_M * np.sin(inc)) ** 2
    )
