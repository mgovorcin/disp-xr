"""Apply displacement corrections: product layers, external layers, plate motion.

DISP correction layers (``/corrections``) are relative to each granule's
reference date, like ``displacement``, and the product says to subtract them.
They are therefore applied before :func:`disp_xr.chain` (the PRD
processing order). Plate motion depends only on time differences, so it can be
removed before or after chaining.

The plate model is ITRF2014-PMM: Altamimi, Z., Métivier, L., Rebischung, P.,
Rouby, H., & Collilieux, X. (2017). ITRF2014 plate motion model. Geophysical
Journal International, 209(3), 1906-1912. https://doi.org/10.1093/gji/ggx136.
The rigid rotation ``v = omega x r`` is implemented here from the paper (Table
1 poles); no code is taken from MintPy or opera-utils.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from typing import Literal, Optional, Union

import numpy as np
import pandas as pd
import xarray as xr
from pyproj import Transformer

from ._history import recorded
from .geometry import match_grid
from .product import PRODUCTS

logger = logging.getLogger(__name__)

__all__ = [
    "ITRF2014_PMM",
    "CorrectionError",
    "correct",
    "geodetic_to_ecef",
    "plate_motion_los",
    "plate_velocity_enu",
    "remove_plate_motion",
]

ALIASES = {
    "ionosphere": "ionospheric_delay",
    "iono": "ionospheric_delay",
    "set": "solid_earth_tide",
    "tides": "solid_earth_tide",
}
DEFAULT_CORRECTIONS = ("ionospheric_delay", "solid_earth_tide")


class CorrectionError(ValueError):
    """A correction cannot be applied (missing, unknown or already applied)."""


def _applied(ds: xr.Dataset) -> list[str]:
    return list(json.loads(ds.attrs.get("corrections_applied", "[]")))


def _check_new(ds: xr.Dataset, name: str, applied: Sequence[str]) -> None:
    if name in applied:
        raise CorrectionError(f"{name!r} is already applied to this dataset")
    spec = PRODUCTS.get(ds.attrs.get("product", ""))
    if spec is not None and name in spec.pre_applied_corrections:
        raise CorrectionError(
            f"{spec.name} displacement already has {name!r} applied by the "
            "processor; applying it again would double it"
        )


def _external_on_ds(
    da: xr.DataArray, ds: xr.Dataset, name: str, kind: str
) -> xr.DataArray:
    if da.rio.crs is not None and ("y", "x") != tuple(da.dims[-2:]):
        raise CorrectionError(f"External {name!r} must have (y, x) as last dims")
    if da.rio.crs is not None and ds.rio.crs is not None:
        on_grid = match_grid(da, ds)
    else:
        on_grid = da.sel(x=ds.x, y=ds.y)
    if "time" not in on_grid.dims:
        if kind == "absolute":
            raise CorrectionError(f"An absolute {name!r} needs a time dimension")
        return on_grid

    times = pd.DatetimeIndex(ds["time"].values)
    needed = times
    if kind == "absolute":
        needed = needed.union(pd.DatetimeIndex(ds["reference_time"].values))
    missing = needed.difference(pd.DatetimeIndex(on_grid["time"].values))
    if len(missing):
        raise CorrectionError(
            f"External {name!r} has no values at "
            f"{[str(t.date()) for t in missing[:5]]}"
            f"{' ...' if len(missing) > 5 else ''}; resample it to the DISP dates"
        )
    at_secondary = on_grid.sel(time=ds["time"])
    if kind == "differential":
        return at_secondary
    at_reference = on_grid.sel(time=ds["reference_time"].values).assign_coords(
        time=ds["time"]
    )
    return at_secondary - at_reference


@recorded
def correct(
    ds: xr.Dataset,
    which: Union[str, Sequence[str], None] = "product",
    *,
    external: Optional[Mapping[str, xr.DataArray]] = None,
    external_kind: Literal["differential", "absolute"] = "differential",
) -> xr.Dataset:
    """Subtract correction layers from ``displacement``.

    Parameters
    ----------
    ds : xr.Dataset
        Unchained DISP data opened with ``open_disp(..., corrections=True)``.
    which : "product", str or sequence of str, optional
        Product layers to subtract: ``"ionospheric_delay"`` (alias
        ``"ionosphere"``) and ``"solid_earth_tide"`` (alias ``"set"``).
        ``"product"`` (default) subtracts every layer the product carries
        that its processor has not applied already: both for DISP-S1, the
        solid-earth tide only for DISP-NISAR. ``None`` or empty applies only
        ``external``.
    external : mapping of str to xr.DataArray, optional
        Extra corrections in meters of LOS displacement, positive toward the
        satellite like ``displacement``, for example a troposphere layer:
        ``(time, y, x)`` or static ``(y, x)``. Put on the DISP grid with
        :func:`disp_xr.geometry.match_grid`; dates must match exactly.
    external_kind : {"differential", "absolute"}
        ``"differential"``: each ``time`` value is already relative to that
        epoch's ``reference_time``. ``"absolute"``: one value per acquisition
        date; the correction is ``value(time) - value(reference_time)``, so
        the reference dates must be present too.

    Returns
    -------
    xr.Dataset
        ``displacement`` corrected; ``short_wavelength_displacement`` and the
        correction layers are left as they are. ``attrs["corrections_applied"]``
        lists every correction applied so far (JSON).

    Raises
    ------
    CorrectionError
        If ``ds`` is chained, a layer is missing or unknown, a correction is
        already applied, or the processor already applied it (DISP-NISAR
        ionosphere).

    Examples
    --------
    >>> raw = dxr.open_disp("F11116/", corrections=True)  # doctest: +SKIP
    >>> ds = dxr.correct(raw)  # doctest: +SKIP
    >>> ds = dxr.correct(raw, ["solid_earth_tide"])  # doctest: +SKIP
    >>> ds = dxr.correct(raw, None, external={"troposphere": tropo})  # doctest: +SKIP

    """
    if "reference_date" in ds.attrs:
        raise CorrectionError(
            "Correction layers are relative to each granule's reference date; "
            "apply them before chaining (disp_xr.chain)"
        )
    applied = _applied(ds)
    out = ds.copy()
    disp = ds["displacement"]
    spec = PRODUCTS.get(ds.attrs.get("product", ""))
    applicable = spec.applicable_corrections if spec else DEFAULT_CORRECTIONS
    if which == "product":
        pre_applied = spec.pre_applied_corrections if spec else frozenset()
        which = [c for c in applicable if c not in pre_applied]
    elif isinstance(which, str):
        which = [which]

    for requested in which or ():
        name = ALIASES.get(requested, requested)
        _check_new(ds, name, applied)
        if name not in applicable:
            raise CorrectionError(
                f"Unknown correction {requested!r}; product layers are {applicable}"
            )
        if name not in ds:
            raise CorrectionError(
                f"Layer {name!r} is not loaded; open with "
                "open_disp(..., corrections=True)"
            )
        disp = disp - ds[name]
        applied.append(name)

    for name, layer in (external or {}).items():
        _check_new(ds, name, applied)
        if name in applicable:
            raise CorrectionError(f"{name!r} is a product layer; pass it in `which`")
        disp = disp - _external_on_ds(layer, ds, name, external_kind).reset_coords(
            drop=True
        )
        applied.append(name)

    out["displacement"] = disp.astype(ds["displacement"].dtype)
    out["displacement"].attrs = ds["displacement"].attrs
    out.attrs["corrections_applied"] = json.dumps(applied)
    logger.info(f"Corrections applied: {applied}")
    return out


# --- plate motion (ITRF2014-PMM) -----------------------------------------------

# Altamimi et al. (2017), Table 1: plate rotation poles (omega_x, omega_y,
# omega_z) in milliarcseconds per year, in ITRF2014.
ITRF2014_PMM: dict[str, tuple[float, float, float]] = {
    "ANTA": (-0.248, -0.324, 0.675),
    "ARAB": (1.154, -0.136, 1.444),
    "AUST": (1.510, 1.182, 1.215),
    "EURA": (-0.085, -0.531, 0.770),
    "INDI": (1.154, -0.005, 1.454),
    "NAZC": (-0.333, -1.544, 1.623),
    "NOAM": (0.024, -0.694, -0.063),
    "NUBI": (0.099, -0.614, 0.733),
    "PCFC": (-0.409, 1.047, -2.169),
    "SOAM": (-0.270, -0.301, -0.140),
    "SOMA": (-0.121, -0.794, 0.884),
}
PLATE_NAMES = {
    "antarctica": "ANTA",
    "arabia": "ARAB",
    "australia": "AUST",
    "eurasia": "EURA",
    "india": "INDI",
    "nazca": "NAZC",
    "north_america": "NOAM",
    "nubia": "NUBI",
    "pacific": "PCFC",
    "south_america": "SOAM",
    "somalia": "SOMA",
}
MAS_TO_RAD = np.pi / (180 * 3_600_000)
WGS84_A = 6378137.0
WGS84_F = 1 / 298.257223563
YEAR_DAYS = 365.25


def _pole(plate: str) -> np.ndarray:
    code = PLATE_NAMES.get(plate.lower().replace(" ", "_"), plate.upper())
    if code not in ITRF2014_PMM:
        raise ValueError(
            f"Unknown plate {plate!r}; use one of {sorted(ITRF2014_PMM)} "
            f"or {sorted(PLATE_NAMES)}"
        )
    return np.array(ITRF2014_PMM[code]) * MAS_TO_RAD  # rad / yr


def geodetic_to_ecef(
    lat: np.ndarray, lon: np.ndarray, height: Union[float, np.ndarray] = 0.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return WGS84 ECEF ``(x, y, z)`` in meters from geodetic degrees and height.

    Examples
    --------
    >>> x, y, z = geodetic_to_ecef(0.0, 0.0)
    >>> round(float(x))
    6378137

    """
    lat_r, lon_r = np.radians(lat), np.radians(lon)
    e2 = WGS84_F * (2 - WGS84_F)
    n = WGS84_A / np.sqrt(1 - e2 * np.sin(lat_r) ** 2)
    x = (n + height) * np.cos(lat_r) * np.cos(lon_r)
    y = (n + height) * np.cos(lat_r) * np.sin(lon_r)
    z = (n * (1 - e2) + height) * np.sin(lat_r)
    return x, y, z


def plate_velocity_enu(
    plate: str, lat: np.ndarray, lon: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the rigid plate velocity ``(east, north, up)`` in meters per year.

    ``v = omega x r`` with ``r`` the WGS84 ECEF position on the ellipsoid and
    ``omega`` the plate's ITRF2014-PMM pole, rotated to local east/north/up.
    The origin rate bias of the model is not added. ``up`` is not exactly zero
    (below 0.1 mm/yr): ``omega x r`` is normal to the geocentric radius, not
    to the ellipsoid normal.

    Examples
    --------
    >>> east, north, up = plate_velocity_enu("NOAM", 35.9, -120.4)
    >>> round(float(east) * 1e3)  # mm/yr
    -12

    """
    wx, wy, wz = _pole(plate)
    x, y, z = geodetic_to_ecef(lat, lon)
    vx, vy, vz = wy * z - wz * y, wz * x - wx * z, wx * y - wy * x
    lat_r, lon_r = np.radians(lat), np.radians(lon)
    sin_lat, cos_lat = np.sin(lat_r), np.cos(lat_r)
    sin_lon, cos_lon = np.sin(lon_r), np.cos(lon_r)
    east = -sin_lon * vx + cos_lon * vy
    north = -sin_lat * cos_lon * vx - sin_lat * sin_lon * vy + cos_lat * vz
    up = cos_lat * cos_lon * vx + cos_lat * sin_lon * vy + sin_lat * vz
    return east, north, up


def plate_motion_los(static: xr.Dataset, plate: str) -> xr.DataArray:
    """Return the rigid plate velocity projected on the LOS, in meters per year.

    Parameters
    ----------
    static : xr.Dataset
        Output of :func:`disp_xr.open_static` (``los_east``, ``los_north``,
        ``los_up`` with a CRS).
    plate : str
        ITRF2014-PMM plate code (``"NOAM"``) or name (``"north_america"``).

    Returns
    -------
    xr.DataArray
        Lazy ``(y, x)`` LOS velocity, positive toward the satellite.

    Examples
    --------
    >>> v_los = plate_motion_los(dxr.open_static(ds), "NOAM")  # doctest: +SKIP

    """
    _pole(plate)
    to_lonlat = Transformer.from_crs(static.rio.crs, "EPSG:4326", always_xy=True)

    def _los_velocity(x, y, le, ln, lu):
        lon, lat = to_lonlat.transform(x, y)
        ve, vn, vu = plate_velocity_enu(plate, lat, lon)
        return (ve * le + vn * ln + vu * lu).astype(np.float32)

    xx, yy = xr.broadcast(static["x"], static["y"])
    template = static["los_east"]
    if template.chunks is not None:
        chunks = dict(zip(template.dims, template.chunks))
        xx, yy = xx.chunk(chunks), yy.chunk(chunks)
    velocity = xr.apply_ufunc(
        _los_velocity,
        xx.transpose(*template.dims),
        yy.transpose(*template.dims),
        static["los_east"],
        static["los_north"],
        static["los_up"],
        dask="parallelized",
        output_dtypes=[np.float32],
    )
    return velocity.rename("plate_motion_los").assign_attrs(
        units="meters/year", plate=plate, model="ITRF2014-PMM"
    )


@recorded
def remove_plate_motion(ds: xr.Dataset, static: xr.Dataset, plate: str) -> xr.Dataset:
    """Subtract rigid plate motion from ``displacement``.

    Each epoch loses ``v_los * (time - reference)``, where the reference is the
    granule's ``reference_time`` before :func:`disp_xr.chain` and
    ``attrs["reference_date"]`` after it, so it works on either. The result is in a
    plate-fixed frame, comparable with plate-fixed GNSS.

    Raises
    ------
    CorrectionError
        If plate motion was already removed.

    Examples
    --------
    >>> static = dxr.open_static(ds)  # doctest: +SKIP
    >>> fixed = dxr.remove_plate_motion(ds, static, "NOAM")  # doctest: +SKIP

    """
    name = "plate_motion"
    applied = _applied(ds)
    _check_new(ds, name, applied)
    velocity = match_grid(plate_motion_los(static, plate), ds).reset_coords(drop=True)
    # After chain every epoch is relative to attrs["reference_date"]; the
    # reference_time coordinate still holds each granule's own reference.
    if "reference_date" in ds.attrs:
        reference = np.datetime64(pd.Timestamp(ds.attrs["reference_date"]), "ns")
    else:
        reference = ds["reference_time"]
    years = (ds["time"] - reference) / np.timedelta64(1, "D") / YEAR_DAYS
    out = ds.copy()
    out["displacement"] = (ds["displacement"] - velocity * years).astype(
        ds["displacement"].dtype
    )
    out["displacement"].attrs = ds["displacement"].attrs
    out.attrs["corrections_applied"] = json.dumps([*applied, name])
    out.attrs["plate"] = plate
    return out
