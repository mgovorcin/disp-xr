"""Synthetic OPERA DISP granules with a known truth, for examples and tests.

``make_disp_stack(directory)`` writes a small frame of granules that open like
real ones, and returns the displacement they were built from, so a result can
be checked against the truth. The gallery and the test suite use it.

The files copy the layout of real granules (see ``tests/data/disp_s1_layout.md``):
one 2D ``(y, x)`` layer per variable, chunked and gzip-compressed with shuffle, a
length-1 ``time`` coordinate, a ``reference_time`` variable, a ``spatial_ref``
grid mapping, and the ``/corrections``, ``/identification`` and ``/metadata``
groups. Each ministack's reference date is the last secondary date of the
previous ministack, as in real frames.

Request a stack by variant name through :func:`make_disp_stack`.

``make_gnss_source(ds, pixels, cache_dir)`` serves synthetic GNSS stations
that follow the DISP displacement at given pixels, through the geepers source
interface, so :func:`disp_xr.gnss.load` and :func:`disp_xr.residuals.compare`
run without a network (``gnss`` extra). ``vertical_los(ds)`` gives matching
static layers.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import NamedTuple

import h5py
import numpy as np
import pandas as pd
import xarray as xr
from pyproj import CRS

__all__ = [
    "VARIANTS",
    "DispStack",
    "correlated_noise",
    "disp_filename",
    "make_disp_stack",
    "make_gnss_source",
    "vertical_los",
]

SPACING = 30.0
WAVELENGTH = 0.05546576
# Microseconds as in real granules: a float time since 2010 (the 2026 style)
# then decodes a few nanoseconds off, as real ones do.
FIRST_DATE = datetime(2020, 1, 1, 0, 1, 53, 435763)
REVISIT_DAYS = 12
PRODUCTION_2025 = "20250412T164531Z"
PRODUCTION_2026 = "20260920T055848Z"
EPOCH_2026 = "2010-01-01 00:00:00.000000"

# Fixed quality features, in pixel indices, so tests can assert exact pixels.
LOW_COHERENCE = (slice(0, 8), slice(0, 8))
UNWRAP_FAILED = (slice(20, 24), slice(20, 24))
WATER_COLUMNS = 4
PS_PIXELS = ((30, 30), (31, 40), (45, 12))
# pixel -> number of 2 pi cycles in the residual (real granules cluster at 2 pi)
CYCLE_PIXELS = {(10, 50): 1, (11, 51): 2}


class Frame(NamedTuple):
    """Grid of one synthetic frame: id, UTM EPSG and the upper-left corner."""

    frame_id: int
    epsg: int
    x0: float
    y0: float


FRAME_10N = Frame(11116, 32610, 499800.0, 4114500.0)
FRAME_11N = Frame(11117, 32611, 170000.0, 4114500.0)
FRAME_NISAR = Frame(26410, 32610, 499800.0, 4114500.0)
NISAR_TITLE = "OPERA_L3_DISP-NISAR Product"
NISAR_VERSION = "0.4"


@dataclass
class DispStack:
    """Paths of a written synthetic stack and the truth behind them.

    Examples
    --------
    >>> truth = make_disp_stack(folder)  # doctest: +SKIP
    >>> truth.cumulative.shape, truth.velocity  # doctest: +SKIP

    """

    paths: list[Path]
    times: pd.DatetimeIndex
    """All epochs, starting with the first reference date."""
    reference_times: pd.DatetimeIndex
    """Reference date of each secondary epoch (``times[1:]``)."""
    cumulative: np.ndarray
    """Displacement relative to ``times[0]``, shape ``(len(times), ny, nx)``."""
    velocity: np.ndarray
    """Linear velocity in m/yr, shape ``(ny, nx)``."""
    annual_amplitude: float
    step: float
    step_date: pd.Timestamp
    x: np.ndarray
    y: np.ndarray
    epsg: int
    extra: dict = field(default_factory=dict)


def disp_filename(
    reference: datetime,
    secondary: datetime,
    *,
    frame_id: int = 11116,
    version: str = "1.0",
    production: str = PRODUCTION_2025,
) -> str:
    """Return a granule name in the OPERA DISP-S1 pattern.

    Examples
    --------
    >>> from datetime import datetime
    >>> disp_filename(datetime(2020, 1, 1), datetime(2020, 1, 13))
    'OPERA_L3_DISP-S1_IW_F11116_VV_20200101T000000Z_20200113T000000Z_v1.0_20250412T164531Z.nc'

    """
    fmt = "%Y%m%dT%H%M%SZ"
    return (
        f"OPERA_L3_DISP-S1_IW_F{frame_id:05d}_VV_{reference.strftime(fmt)}_"
        f"{secondary.strftime(fmt)}_v{version}_{production}.nc"
    )


def correlated_noise(
    shape: tuple[int, int],
    *,
    correlation_range: float,
    sill: float = 1.0,
    nugget: float = 0.0,
    spacing: float = SPACING,
    seed: int = 0,
) -> np.ndarray:
    """Return a Gaussian random field with a known exponential covariance.

    The covariance is ``sill * exp(-3 h / correlation_range)`` (practical
    range ``correlation_range``, the convention of
    :class:`disp_xr.residuals.VariogramModel`) plus ``nugget`` of white
    noise, so variogram and cell-sigma estimates can be checked against the
    truth. Exact, by circulant embedding on a grid twice the size.

    Parameters
    ----------
    shape : (int, int)
        ``(ny, nx)`` in pixels.
    correlation_range : float
        Practical range, in the unit of ``spacing``.
    sill : float
        Variance of the correlated part.
    nugget : float
        Variance of the added white noise.
    spacing : float
        Pixel spacing.
    seed : int
        Random seed.

    Examples
    --------
    >>> field = correlated_noise((64, 64), correlation_range=300.0, seed=1)
    >>> field.shape
    (64, 64)

    """
    ny, nx = shape
    my, mx = 2 * ny, 2 * nx
    ly = np.minimum(np.arange(my), my - np.arange(my)) * spacing
    lx = np.minimum(np.arange(mx), mx - np.arange(mx)) * spacing
    cov = sill * np.exp(-3 * np.hypot(ly[:, None], lx[None, :]) / correlation_range)
    eigen = np.clip(np.fft.fft2(cov).real, 0, None)
    rng = np.random.default_rng(seed)
    white = rng.standard_normal((my, mx))
    field = np.fft.ifft2(np.sqrt(eigen) * np.fft.fft2(white)).real[:ny, :nx]
    if nugget:
        field = field + rng.normal(0, np.sqrt(nugget), shape)
    return field


def _dates(n_ministacks: int, epochs_per_ministack: int) -> tuple[list, list]:
    """Return the secondary dates and their reference dates."""
    secondaries, references = [], []
    reference = FIRST_DATE
    epoch = FIRST_DATE
    for _ in range(n_ministacks):
        for _ in range(epochs_per_ministack):
            epoch = epoch + timedelta(days=REVISIT_DAYS)
            secondaries.append(epoch)
            references.append(reference)
        reference = epoch
    return secondaries, references


def _signal(
    times: pd.DatetimeIndex, ny: int, nx: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, float, float, pd.Timestamp]:
    years = np.asarray((times - times[0]).days) / 365.25
    velocity = np.linspace(-0.02, 0.02, nx)[None, :] * np.ones((ny, 1))
    amplitude = 0.005
    step = 0.01
    step_date = times[len(times) // 2]
    trend = years[:, None, None] * velocity[None]
    seasonal = amplitude * np.sin(2 * np.pi * years)[:, None, None]
    jump = step * np.asarray(times >= step_date)[:, None, None]
    noise = rng.normal(0, 0.001, (len(times), ny, nx))
    cumulative = trend + seasonal + jump + noise
    cumulative -= cumulative[0]
    return cumulative, velocity, amplitude, step, step_date


def _quality_layers(ny: int, nx: int) -> dict[str, np.ndarray]:
    coherence = np.full((ny, nx), 0.9, dtype=np.float32)
    similarity = np.full((ny, nx), 0.8, dtype=np.float32)
    coherence[LOW_COHERENCE] = 0.3
    similarity[LOW_COHERENCE] = 0.2
    water = np.ones((ny, nx), dtype=np.uint8)
    water[:, -WATER_COLUMNS:] = 0
    labels = np.ones((ny, nx), dtype=np.uint16)
    labels[UNWRAP_FAILED] = 0
    ps = np.zeros((ny, nx), dtype=np.uint8)
    for iy, ix in PS_PIXELS:
        ps[iy, ix] = 1
    low_quality = (coherence < 0.6) & (similarity < 0.3)
    recommended = (~((water == 0) | (labels == 0) | low_quality)).astype(np.uint8)
    return {
        "temporal_coherence": coherence,
        "phase_similarity": similarity,
        "water_mask": water,
        "connected_component_labels": labels,
        "persistent_scatterer_mask": ps,
        "recommended_mask": recommended,
    }


def _encoding(da: xr.DataArray, fill) -> dict:
    ny, nx = da.shape
    return {
        "chunksizes": (min(32, ny), min(32, nx)),
        "zlib": True,
        "complevel": 4,
        "shuffle": True,
        "_FillValue": fill,
    }


def _spatial_ref(epsg: int, x: np.ndarray, y: np.ndarray) -> xr.DataArray:
    attrs = CRS.from_epsg(epsg).to_cf()
    attrs["GeoTransform"] = (
        f"{x[0] - SPACING / 2} {SPACING} 0.0 {y[0] + SPACING / 2} 0.0 {-SPACING}"
    )
    return xr.DataArray(np.int64(0), attrs=attrs)


def _write_granule(
    path: Path,
    *,
    reference: datetime,
    secondary: datetime,
    displacement: np.ndarray,
    quality: dict[str, np.ndarray],
    residuals: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    epsg: int,
    frame_id: int,
    version: str,
    style_2026: bool,
    title: str = "OPERA_L3_DISP-S1 Product",
) -> None:
    ny, nx = displacement.shape
    dims = ("y", "x")
    disp_dtype = np.float64 if style_2026 else np.float32
    coords = {
        "x": ("x", x, {"units": "m", "standard_name": "projection_x_coordinate"}),
        "y": ("y", y, {"units": "m", "standard_name": "projection_y_coordinate"}),
        "time": ("time", [np.datetime64(secondary, "ns")]),
    }
    data_vars = {
        "displacement": (dims, displacement.astype(disp_dtype), {"units": "meters"}),
        "short_wavelength_displacement": (
            dims,
            np.zeros((ny, nx), np.float32),
            {"units": "meters"},
        ),
        "estimated_phase_quality": (
            dims,
            np.full((ny, nx), 0.8, np.float32),
            {"units": "unitless"},
        ),
        "shp_counts": (dims, np.full((ny, nx), 50, np.int16), {"units": "unitless"}),
        "timeseries_inversion_residuals": (dims, residuals, {"units": "radians"}),
        "reference_time": ("time", [np.datetime64(reference, "ns")]),
        "spatial_ref": _spatial_ref(epsg, x, y),
    }
    for name, values in quality.items():
        data_vars[name] = (dims, values, {"units": "unitless"})
    ds = xr.Dataset(data_vars, coords=coords)
    ds["recommended_mask"].attrs.update(
        temporal_coherence_threshold="0.6", similarity_threshold="0.3"
    )
    fills = {
        np.dtype("float32"): np.nan,
        np.dtype("float64"): np.nan,
        np.dtype("uint8"): 255,
        np.dtype("uint16"): 65535,
        np.dtype("int16"): 0,
    }
    encoding = {}
    for name, da in ds.data_vars.items():
        if da.dims == dims:
            da.attrs["grid_mapping"] = "spatial_ref"
            encoding[name] = _encoding(da, fills[da.dtype])
    if style_2026:
        time_units = f"seconds since {EPOCH_2026}"
        ref_units = time_units
    else:
        time_units = f"seconds since {secondary:%Y-%m-%d %H:%M:%S.%f}"
        ref_units = f"seconds since {reference:%Y-%m-%d %H:%M:%S.%f}"
    time_enc = {"calendar": "standard", "dtype": "float64"}
    encoding["time"] = {"units": time_units, **time_enc}
    encoding["reference_time"] = {"units": ref_units, **time_enc}
    encoding["x"] = encoding["y"] = {"_FillValue": None}
    ds.attrs = {"Conventions": "CF-1.8", "title": title}
    ds.to_netcdf(path, engine="h5netcdf", encoding=encoding)

    corrections = xr.Dataset(
        {
            "ionospheric_delay": (dims, np.full((ny, nx), 0.002, np.float32)),
            "solid_earth_tide": (dims, np.full((ny, nx), -0.001, np.float32)),
            "perpendicular_baseline": (dims, np.full((ny, nx), 50.0, np.float32)),
            "reference_point": ((), np.int64(0)),
            "spatial_ref": _spatial_ref(epsg, x, y),
        },
        coords={k: coords[k] for k in ("x", "y", "time")},
    )
    corr_encoding = {
        name: _encoding(corrections[name], np.nan)
        for name in ("ionospheric_delay", "solid_earth_tide", "perpendicular_baseline")
    }
    corr_encoding["time"] = {"units": time_units, **time_enc}
    for name in tuple(corr_encoding):
        if name != "time":
            corrections[name].attrs.update(units="meters", grid_mapping="spatial_ref")
    corr_encoding["x"] = corr_encoding["y"] = {"_FillValue": None}
    corrections.to_netcdf(
        path, mode="a", group="corrections", engine="h5netcdf", encoding=corr_encoding
    )

    runconfig = (
        "dynamic_ancillary_file_group:\n"
        "  static_layers_files:\n"
        "  - OPERA_L2_CSLC-S1-STATIC_T042-088905-IW1_20140403_S1A_v1.0.h5\n"
        "  - OPERA_L2_CSLC-S1-STATIC_T042-088906-IW1_20140403_S1A_v1.0.h5\n"
    )
    with h5py.File(path, "a") as f:
        ident = f.create_group("identification")
        for key, value in {
            "frame_id": np.int64(frame_id),
            "product_version": np.bytes_(version),
            "reference_datetime": np.bytes_(f"{reference:%Y-%m-%d %H:%M:%S}.000000"),
            "secondary_datetime": np.bytes_(f"{secondary:%Y-%m-%d %H:%M:%S}.000000"),
            "radar_wavelength": np.float64(WAVELENGTH),
            "acquisition_mode": np.bytes_("IW"),
        }.items():
            ident.create_dataset(key, data=value)
        meta = f.create_group("metadata")
        meta.create_dataset("pge_runconfig", data=np.bytes_(runconfig))
        meta.create_dataset(
            "product_pixel_coordinate_convention", data=np.bytes_("pixel center")
        )
        meta.create_dataset(
            "dolphin_workflow_config", data=np.bytes_("# synthetic\n" * 10)
        )


def _residuals(ny: int, nx: int, epoch_index: int) -> np.ndarray:
    residuals = np.full((ny, nx), 0.1, np.float32)
    if epoch_index == 1:
        for (iy, ix), cycles in CYCLE_PIXELS.items():
            residuals[iy, ix] = cycles * 2 * np.pi
    return residuals


def _write_stack(
    root: Path,
    *,
    n_ministacks: int,
    epochs_per_ministack: int,
    shape: tuple[int, int],
    seed: int,
    frame: Frame,
    drop: tuple[int, ...] = (),
    style_2026_ministacks: tuple[int, ...] = (),
    nisar: bool = False,
) -> DispStack:
    ny, nx = shape
    frame_id, epsg, x0, y0 = frame
    rng = np.random.default_rng(seed)
    secondaries, references = _dates(n_ministacks, epochs_per_ministack)
    times = pd.DatetimeIndex([FIRST_DATE, *secondaries])
    cumulative, velocity, amplitude, step, step_date = _signal(times, ny, nx, rng)
    quality = _quality_layers(ny, nx)
    x = x0 + SPACING / 2 + SPACING * np.arange(nx)
    y = y0 - SPACING / 2 - SPACING * np.arange(ny)
    root.mkdir(parents=True, exist_ok=True)
    position = {t: i for i, t in enumerate(times)}
    paths = []
    for k, (secondary, reference) in enumerate(zip(secondaries, references)):
        if k in drop:
            continue
        ministack = k // epochs_per_ministack
        displacement = (
            cumulative[position[pd.Timestamp(secondary)]]
            - cumulative[position[pd.Timestamp(reference)]]
        )
        if nisar:
            # Pre-release DISP-NISAR SAS output is named <ref>_<sec>.nc.
            path = root / f"{reference:%Y%m%d}_{secondary:%Y%m%d}.nc"
        else:
            path = root / disp_filename(
                reference,
                secondary,
                frame_id=frame_id,
                production=(
                    PRODUCTION_2026
                    if ministack in style_2026_ministacks
                    else PRODUCTION_2025
                ),
            )
        _write_granule(
            path,
            reference=reference,
            secondary=secondary,
            displacement=displacement,
            quality=quality,
            residuals=_residuals(ny, nx, k % epochs_per_ministack),
            x=x,
            y=y,
            epsg=epsg,
            frame_id=frame_id,
            version=NISAR_VERSION if nisar else "1.0",
            style_2026=ministack in style_2026_ministacks,
            title=NISAR_TITLE if nisar else "OPERA_L3_DISP-S1 Product",
        )
        paths.append(path)
    return DispStack(
        paths=paths,
        # disp-xr reads times to the second, as granule names give them.
        times=times.floor("s"),
        reference_times=pd.DatetimeIndex(references).floor("s"),
        cumulative=cumulative,
        velocity=velocity,
        annual_amplitude=amplitude,
        step=step,
        step_date=step_date,
        x=x,
        y=y,
        epsg=epsg,
        extra={"quality": quality, "dropped": drop},
    )


def _add_duplicate(stack: DispStack, root: Path) -> None:
    """Write an older-version, older-production copy of the second granule."""
    original = stack.paths[1]
    older = root / original.name.replace("_v1.0_", "_v0.9_").replace(
        PRODUCTION_2025, "20240101T000000Z"
    )
    older.write_bytes(original.read_bytes())
    stack.extra["duplicate"] = older
    stack.extra["duplicate_of"] = original


VARIANTS = (
    "two_ministacks",
    "three_ministacks",
    "with_gaps",
    "duplicate",
    "mixed_dtype",
    "nisar",
    "two_utm_zones",
)


def make_disp_stack(
    root: Path,
    variant: str = "two_ministacks",
    *,
    epochs_per_ministack: int = 5,
    shape: tuple[int, int] = (64, 64),
    seed: int = 0,
) -> DispStack:
    """Write a synthetic DISP-S1 stack under ``root`` and return its truth.

    Parameters
    ----------
    root : Path
        Directory to write into (created if missing).
    variant : str
        One of :data:`VARIANTS`:

        - ``two_ministacks``: 2 ministacks, one reference change.
        - ``three_ministacks``: 3 ministacks, two reference changes.
        - ``with_gaps``: 2 ministacks with two secondaries of the first removed.
        - ``duplicate``: ``two_ministacks`` plus an older-version copy of one
          granule (path in ``extra["duplicate"]``).
        - ``mixed_dtype``: the second ministack is written in the 2026 production
          style (float64 displacement, fixed time epoch).
        - ``nisar``: ``two_ministacks`` as pre-release DISP-NISAR SAS output
          (``<ref>_<sec>.nc`` names, NISAR title, version 0.4).
        - ``two_utm_zones``: two adjacent frames, UTM 10N and 11N, in
          ``root/F11116`` and ``root/F11117``; the second stack is in
          ``extra["second_frame"]``.
    epochs_per_ministack : int
        Secondaries per ministack.
    shape : tuple[int, int]
        ``(ny, nx)`` of every layer.
    seed : int
        Seed for the noise.

    Returns
    -------
    DispStack
        Paths and truth.

    Examples
    --------
    >>> import tempfile
    >>> from pathlib import Path
    >>> truth = make_disp_stack(Path(tempfile.mkdtemp()), "two_ministacks")
    >>> len(truth.paths)
    10

    """

    def write(
        path: Path,
        n: int,
        frame: Frame = FRAME_10N,
        drop: tuple[int, ...] = (),
        style_2026_ministacks: tuple[int, ...] = (),
        nisar: bool = False,
    ) -> DispStack:
        return _write_stack(
            path,
            n_ministacks=n,
            epochs_per_ministack=epochs_per_ministack,
            shape=shape,
            seed=seed,
            frame=frame,
            drop=drop,
            style_2026_ministacks=style_2026_ministacks,
            nisar=nisar,
        )

    if variant == "two_ministacks":
        return write(root, 2)
    if variant == "three_ministacks":
        return write(root, 3)
    if variant == "with_gaps":
        return write(root, 2, drop=(1, 2))
    if variant == "duplicate":
        stack = write(root, 2)
        _add_duplicate(stack, root)
        return stack
    if variant == "mixed_dtype":
        return write(root, 2, style_2026_ministacks=(1,))
    if variant == "nisar":
        return write(root, 2, FRAME_NISAR, nisar=True)
    if variant == "two_utm_zones":
        first = write(root / "F11116", 2)
        first.extra["second_frame"] = write(root / "F11117", 2, FRAME_11N)
        return first
    raise ValueError(f"Unknown variant {variant!r}; choose one of {VARIANTS}")


# --- Synthetic GNSS -------------------------------------------------------------


@functools.cache
def synthetic_source_class() -> type:
    """Return a geepers source class serving given stations and series.

    Built on first use so that this module does not need geepers. Instances
    take ``(table, series, cache_dir)``: a station table (``id``, ``lon``,
    ``lat``, ``alt``) and a dict of daily ENU DataFrames by station id.
    """
    import geopandas as gpd
    from geepers.gps_sources.base import BaseGpsSource
    from shapely.geometry import Point

    class SyntheticGnssSource(BaseGpsSource):
        """A geepers source serving given stations and series."""

        def __init__(self, table, series, cache_dir):
            super().__init__(cache_dir=cache_dir)
            self.table = table
            self.series = series

        def _read_station_data(self):
            points = [Point(lo, la) for lo, la in zip(self.table.lon, self.table.lat)]
            return gpd.GeoDataFrame(self.table, geometry=points, crs="EPSG:4326")

        def timeseries(
            self,
            station_id,
            /,
            frame="ENU",  # noqa: ARG002
            start_date=None,
            end_date=None,
            zero_by="mean",  # noqa: ARG002
            download_if_missing=True,  # noqa: ARG002
            **kwargs,  # noqa: ARG002
        ):
            df = self.series[station_id].copy()
            if start_date:
                df = df[df.date >= pd.Timestamp(start_date)]
            if end_date:
                df = df[df.date <= pd.Timestamp(end_date)]
            return df.reset_index(drop=True)

    return SyntheticGnssSource


def make_gnss_source(
    ds: xr.Dataset,
    pixels: list[tuple[int, int]],
    cache_dir: str | Path,
    *,
    noise: float = 0.001,
    extra_rate: dict[int, float] | None = None,
    seed: int = 0,
):
    """Return a geepers GNSS source whose stations follow DISP at ``pixels``.

    Each station sits at a ``(row, col)`` pixel; its daily up motion is that
    pixel's displacement (interpolated to days) plus white noise, with east
    and north zero, so it matches DISP under a vertical line of sight
    (:func:`vertical_los`). Needs the ``gnss`` extra (geepers).

    Parameters
    ----------
    ds : xr.Dataset
        Chained, re-referenced DISP with a CRS.
    pixels : list of (int, int)
        Station pixels; stations are named ``S000``, ``S001``...
    cache_dir : str or Path
        geepers cache directory (nothing is downloaded).
    noise : float
        Standard deviation of the daily noise and the station sigmas (m).
    extra_rate : dict, optional
        Station index to an added up rate (m/yr), to make an outlier.
    seed : int
        Seed of the noise.

    Returns
    -------
    geepers source
        Pass it as ``source=`` to :func:`disp_xr.gnss.load`.

    Examples
    --------
    >>> src = make_gnss_source(ds, [(4, 4), (30, 30)], "cache")  # doctest: +SKIP
    >>> gps = dxr.gnss.load(ds, source=src)  # doctest: +SKIP

    """
    from pyproj import Transformer

    rng = np.random.default_rng(seed)
    to_ll = Transformer.from_crs(ds.rio.crs, "EPSG:4326", always_xy=True)
    days = pd.date_range(
        pd.Timestamp(ds.time.values[0]).normalize() - pd.Timedelta("5D"),
        pd.Timestamp(ds.time.values[-1]).normalize() + pd.Timedelta("5D"),
        freq="D",
    )
    rows, series = [], {}
    for k, (r, c) in enumerate(pixels):
        sid = f"S{k:03d}"
        lon, lat = to_ll.transform(float(ds.x[c]), float(ds.y[r]))
        rows.append({"id": sid, "lon": lon, "lat": lat, "alt": 0.0})
        pixel = ds.displacement.isel(y=r, x=c).to_series()
        pixel.index = pd.DatetimeIndex(pixel.index).normalize()
        up = pixel.reindex(days).interpolate(limit_direction="both").to_numpy()
        years = np.asarray((days - days[0]).days) / 365.25
        up = (
            up
            + (extra_rate or {}).get(k, 0.0) * years
            + noise * rng.standard_normal(days.size)
        )
        series[sid] = pd.DataFrame(
            {
                "date": days,
                "east": 0.0,
                "north": 0.0,
                "up": up,
                "sigma_east": noise,
                "sigma_north": noise,
                "sigma_up": noise,
                "corr_en": 0.0,
                "corr_eu": 0.0,
                "corr_nu": 0.0,
            }
        )
    return synthetic_source_class()(pd.DataFrame(rows), series, cache_dir)


def vertical_los(ds: xr.Dataset) -> xr.Dataset:
    """Return static layers with a vertical line of sight on the grid of ``ds``.

    Examples
    --------
    >>> import numpy as np, xarray as xr
    >>> ds = xr.Dataset(coords={"y": np.arange(3.0), "x": np.arange(4.0)})
    >>> vertical_los(ds.rio.write_crs("EPSG:32611")).los_up.shape
    (3, 4)

    """
    shape = (ds.sizes["y"], ds.sizes["x"])
    zeros = np.zeros(shape)
    return xr.Dataset(
        {
            "los_east": (("y", "x"), zeros),
            "los_north": (("y", "x"), zeros),
            "los_up": (("y", "x"), np.ones(shape)),
        },
        coords={"y": ds.y, "x": ds.x},
    ).rio.write_crs(ds.rio.crs)
