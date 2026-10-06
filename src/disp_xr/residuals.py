"""Residuals: spatial variograms (T27.3b) and DISP against GNSS (T32).

:func:`compare` compares DISP with GNSS at the stations, with geepers
computing the metrics (see its docstring). The rest of this docstring
covers the variograms.

The sigma of a cell mean computed as ``spread / sqrt(n)`` assumes independent
pixels. InSAR errors (troposphere, unwrapping, orbits) are correlated over
kilometres, so at 1 to 10 km cells that sigma is far too small. Following
xDEM (Hugonnet et al. 2022), this module

1. estimates an empirical variogram of a residual field (a field that should
   be zero: stable ground, or data minus a model) from a random sample of
   pixel pairs, with distances drawn log-uniformly so short lags are as well
   sampled as long ones (:func:`empirical_variogram`);
2. fits a sum of spherical, exponential or Gaussian models and a nugget
   (:func:`fit_variogram`);
3. turns the model into the effective number of independent samples of a
   block mean (:func:`mean_block_correlation`, :func:`n_eff_correlated`),
   which :func:`disp_xr.downsample` uses for ``<layer>_sigma_corr``.

The residual semivariogram of plan task T32.1 reuses this code.

References
----------
Hugonnet, R., et al. (2022). Uncertainty analysis of digital elevation models
by spatial inference from stable terrain. IEEE JSTARS 15, 6456-6472.
https://doi.org/10.1109/JSTARS.2022.3188922

Rolstad, C., Haug, T. and Denby, B. (2009). Spatially integrated geodetic
glacier mass balance and its uncertainty based on geostatistical analysis.
Journal of Glaciology 55(192), 666-680. https://doi.org/10.3189/002214309789470950

"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, Optional, Union

import numpy as np
import xarray as xr

from .units import _SCALE, _canonical

__all__ = [
    "DISP_S1_DISPLACEMENT",
    "DISP_S1_VELOCITY",
    "Comparison",
    "Requirement",
    "VariogramModel",
    "common_mode",
    "compare",
    "empirical_variogram",
    "fit_variogram",
    "mean_block_correlation",
    "mean_mask_correlation",
    "n_eff_correlated",
    "residual_spectra",
]

ModelKind = Literal["spherical", "exponential", "gaussian"]


def _component(kind: str, h: np.ndarray, sill: float, rng: float) -> np.ndarray:
    """Semivariance of one model; ``rng`` is the (practical, 95%) range."""
    r = h / rng
    if kind == "spherical":
        return sill * np.where(r < 1, 1.5 * r - 0.5 * r**3, 1.0)
    if kind == "exponential":
        return sill * (1 - np.exp(-3 * r))
    if kind == "gaussian":
        return sill * (1 - np.exp(-3 * r**2))
    raise ValueError(f"unknown variogram model {kind!r}")


@dataclass(frozen=True)
class VariogramModel:
    """A nugget plus a sum of bounded variogram models.

    ``gamma(h) = nugget + sum(sill_k * f_k(h / range_k))`` for ``h > 0`` and
    0 at ``h = 0``. Ranges are practical ranges (the semivariance reaches
    95% of the partial sill for exponential and Gaussian models, all of it
    for spherical ones), in the CRS unit of the field.

    Examples
    --------
    >>> model = dxr.fit_variogram(empirical)  # doctest: +SKIP
    >>> model.sill, model.ranges  # doctest: +SKIP

    """

    nugget: float
    components: tuple[tuple[str, float, float], ...]
    """``(kind, partial sill, range)`` per model."""

    @property
    def sill(self) -> float:
        """Total variance of a pixel: nugget plus partial sills."""
        return self.nugget + sum(c[1] for c in self.components)

    def gamma(self, h: Union[float, np.ndarray]) -> np.ndarray:
        """Return the semivariance at distance ``h``."""
        h = np.asarray(h, dtype=np.float64)
        out = np.full(h.shape, self.nugget)
        for kind, sill, rng in self.components:
            out = out + _component(kind, h, sill, rng)
        return np.where(h > 0, out, 0.0)

    def correlation(self, h: Union[float, np.ndarray]) -> np.ndarray:
        """Return the correlation of two pixels at distance ``h``."""
        return 1.0 - self.gamma(h) / self.sill

    def to_json(self) -> str:
        """Return the model as a JSON string (for attributes)."""
        return json.dumps(
            {"nugget": self.nugget, "components": [list(c) for c in self.components]}
        )


def _spacing(field: xr.DataArray) -> tuple[float, float]:
    return (
        abs(float(field["y"][1] - field["y"][0])),
        abs(float(field["x"][1] - field["x"][0])),
    )


def empirical_variogram(
    field: xr.DataArray,
    *,
    n_pairs: int = 200_000,
    n_bins: int = 30,
    max_distance: Optional[float] = None,
    seed: int = 0,
) -> xr.Dataset:
    """Estimate the empirical semivariogram of a 2D field from random pairs.

    Each pair starts at a random valid pixel; its partner lies at a distance
    drawn log-uniformly between one pixel and ``max_distance``, in a random
    direction. Every logarithmic distance bin then gets about the same number
    of pairs, unlike all pairs of a random sample, which leave short lags
    (where the cell sigma is decided) nearly empty.

    Parameters
    ----------
    field : xr.DataArray
        A ``(y, x)`` residual field, ideally zero-mean with no deformation
        signal (stable ground, or data minus a fitted model). NaN pixels are
        skipped. It is loaded into memory.
    n_pairs : int
        Number of pairs drawn (pairs whose partner is NaN or off the grid are
        dropped).
    n_bins : int
        Number of logarithmic distance bins.
    max_distance : float, optional
        Largest distance, in the CRS unit; default half the shorter side of
        the field.
    seed : int
        Random seed.

    Returns
    -------
    xr.Dataset
        Dimension ``lag``, with ``distance`` (mean pair distance in the bin,
        a coordinate), ``gamma`` (semivariance, half the mean squared
        difference) and ``count`` (pairs per bin). Empty bins are dropped.

    Examples
    --------
    >>> residual = ds.displacement.isel(time=-1) - trend  # doctest: +SKIP
    >>> empirical = dxr.empirical_variogram(
    ...     residual, max_distance=20e3
    ... )  # doctest: +SKIP

    """
    if set(field.dims) != {"y", "x"}:
        raise ValueError(f"field must have dimensions (y, x), not {field.dims}")
    dy, dx = _spacing(field)
    z = np.asarray(field.transpose("y", "x").values, dtype=np.float64)
    ny, nx = z.shape
    if max_distance is None:
        max_distance = 0.5 * min(ny * dy, nx * dx)
    min_distance = min(dy, dx)
    if max_distance <= min_distance:
        raise ValueError("max_distance must be larger than the pixel spacing")
    iy, ix = np.nonzero(np.isfinite(z))
    if iy.size < 2:
        raise ValueError("field has fewer than two valid pixels")
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, iy.size, n_pairs)
    y0, x0 = iy[pick], ix[pick]
    dist = np.exp(rng.uniform(np.log(min_distance), np.log(max_distance), n_pairs))
    angle = rng.uniform(0, np.pi, n_pairs)
    y1 = y0 + np.rint(dist * np.sin(angle) / dy).astype(np.int64)
    x1 = x0 + np.rint(dist * np.cos(angle) / dx).astype(np.int64)
    inside = (y1 >= 0) & (y1 < ny) & (x1 >= 0) & (x1 < nx) & ((y1 != y0) | (x1 != x0))
    y0, x0, y1, x1 = y0[inside], x0[inside], y1[inside], x1[inside]
    diff = z[y1, x1] - z[y0, x0]
    ok = np.isfinite(diff)
    h = np.hypot((y1 - y0)[ok] * dy, (x1 - x0)[ok] * dx)
    semi = 0.5 * diff[ok] ** 2
    edges = np.geomspace(min_distance, max_distance, n_bins + 1)
    edges[0] = 0.0
    edges[-1] = np.inf
    which = np.digitize(h, edges) - 1
    count = np.bincount(which, minlength=n_bins)[:n_bins]
    with np.errstate(invalid="ignore", divide="ignore"):
        gamma = np.bincount(which, semi, minlength=n_bins)[:n_bins] / count
        distance = np.bincount(which, h, minlength=n_bins)[:n_bins] / count
    keep = count > 0
    units = field["x"].attrs.get("units", "")
    return xr.Dataset(
        {
            "gamma": ("lag", gamma[keep], {"long_name": "semivariance"}),
            "count": ("lag", count[keep], {"long_name": "pairs in the bin"}),
        },
        coords={"distance": ("lag", distance[keep], {"units": units})},
        attrs={"field": str(field.name), "n_pairs": int(ok.sum())},
    )


def fit_variogram(
    empirical: xr.Dataset,
    models: Sequence[ModelKind] = ("spherical", "spherical"),
    *,
    nugget: bool = True,
) -> VariogramModel:
    """Fit a nugget and a sum of models to an empirical variogram.

    Weighted least squares with the approximate standard error of each bin,
    ``gamma * sqrt(2 / count)``, so every bin counts by its relative misfit
    and short lags are not swamped by the large semivariances of long ones.
    Two spherical models (a short and a long range) are xDEM's default for
    elevation residuals and fit InSAR residuals (turbulent troposphere plus
    long-wavelength errors) well.

    Parameters
    ----------
    empirical : xr.Dataset
        Output of :func:`empirical_variogram`.
    models : sequence of {"spherical", "exponential", "gaussian"}
        One entry per component, from short to long range.
    nugget : bool
        Fit a nugget (uncorrelated pixel noise); otherwise it is 0.

    Returns
    -------
    VariogramModel

    Examples
    --------
    >>> model = dxr.fit_variogram(empirical, ("exponential",))  # doctest: +SKIP
    >>> coarse = dxr.downsample(
    ...     ds, factor=33, variogram={"displacement": model}
    ... )  # doctest: +SKIP

    """
    from scipy.optimize import curve_fit

    h = empirical["distance"].values
    g = empirical["gamma"].values
    n = empirical["count"].values
    ok = np.isfinite(g) & (n > 0)
    h, g, n = h[ok], g[ok], n[ok]
    k = len(models)
    if k == 0:
        raise ValueError("give at least one model")
    if h.size < 2 * k + int(nugget):
        raise ValueError("too few lag bins for the number of parameters")
    top = float(g.max())
    first = float(h[np.argmax(g >= 0.63 * top)])
    ranges0 = np.clip(
        first * np.geomspace(1 / 3, 3, k) if k > 1 else [first], h.min(), h.max()
    )

    def model(hh: np.ndarray, *p: float) -> np.ndarray:
        out = np.full(hh.shape, p[0] if nugget else 0.0)
        for i, kind in enumerate(models):
            sill, rng = p[int(nugget) + 2 * i : int(nugget) + 2 * i + 2]
            out = out + _component(kind, hh, sill, rng)
        return out

    p0 = ([0.25 * float(g[0])] if nugget else []) + [
        v for r in ranges0 for v in (top / k, float(r))
    ]
    lower = ([0.0] if nugget else []) + [v for _ in models for v in (0.0, h.min())]
    upper = ([top] if nugget else []) + [
        v for _ in models for v in (2 * top, 2 * h.max())
    ]
    sigma = np.maximum(g, 1e-12 * top) * np.sqrt(2.0 / n)
    params, _ = curve_fit(
        model, h, g, p0=p0, sigma=sigma, bounds=(lower, upper), maxfev=20_000
    )
    offset = int(nugget)
    components = tuple(
        (str(kind), float(params[offset + 2 * i]), float(params[offset + 2 * i + 1]))
        for i, kind in enumerate(models)
    )
    order = sorted(range(k), key=lambda i: components[i][2])
    return VariogramModel(
        nugget=float(params[0]) if nugget else 0.0,
        components=tuple(components[i] for i in order),
    )


def mean_block_correlation(
    model: VariogramModel, factor: int, spacing: Union[float, tuple[float, float]]
) -> float:
    """Return the mean correlation between two distinct pixels of a block.

    Exact for a full ``factor x factor`` block: the double sum over pixel
    pairs is a sum over the ``(2 factor - 1)²`` pixel offsets, each counted
    ``(factor - |dy|) (factor - |dx|)`` times.

    Parameters
    ----------
    model : VariogramModel
        The fitted variogram.
    factor : int
        Pixels per block side.
    spacing : float or (float, float)
        Pixel spacing, or ``(dy, dx)``, in the unit of the model ranges.

    Examples
    --------
    >>> rho = mean_block_correlation(model, factor=33, spacing=30.0)  # doctest: +SKIP

    """
    if factor < 2:
        return 0.0
    dy, dx = (spacing, spacing) if np.isscalar(spacing) else spacing  # type: ignore[misc]
    lags = np.arange(-(factor - 1), factor)
    weight = factor - np.abs(lags)
    dist = np.hypot(lags[:, None] * dy, lags[None, :] * dx)
    total = float(np.sum(weight[:, None] * weight[None, :] * model.correlation(dist)))
    n = factor * factor
    return (total - n) / (n * (n - 1))


def mean_mask_correlation(
    model: VariogramModel,
    mask: np.ndarray,
    spacing: Union[float, tuple[float, float]],
) -> float:
    """Return the mean correlation between two distinct pixels of a cell shape.

    ``mask`` marks the pixels of one full cell (a hexagon, for example). The
    number of pixel pairs at each offset is the autocorrelation of the mask,
    computed with an FFT, so the double sum over pairs costs one transform.
    :func:`mean_block_correlation` is the square-block case.

    Examples
    --------
    >>> rho = mean_mask_correlation(model, cell_mask, spacing=30.0)  # doctest: +SKIP

    """
    mask = np.asarray(mask, dtype=np.float64)
    n = mask.sum()
    if n < 2:
        return 0.0
    dy, dx = (spacing, spacing) if np.isscalar(spacing) else spacing  # type: ignore[misc]
    ny, nx = mask.shape
    shape = (2 * ny - 1, 2 * nx - 1)
    spectrum = np.fft.rfft2(mask, shape)
    pairs = np.fft.irfft2(spectrum * np.conj(spectrum), shape)
    pairs = np.rint(np.fft.fftshift(pairs))
    ly = np.arange(-(ny - 1), ny)
    lx = np.arange(-(nx - 1), nx)
    dist = np.hypot(ly[:, None] * dy, lx[None, :] * dx)
    total = float(np.sum(pairs * model.correlation(dist)))
    return (total - n) / (n * (n - 1))


def n_eff_correlated(sum_w, sum_w2, rho_bar: float):
    """Return the effective sample count of a weighted mean of correlated pixels.

    The variance of a weighted mean is ``sill * Σᵢⱼ wᵢ wⱼ ρᵢⱼ / (Σw)²``.
    Replacing each off-diagonal ``ρᵢⱼ`` by the block mean ``rho_bar`` gives
    ``n_eff = (Σw)² / (Σw² + ((Σw)² - Σw²) rho_bar)``: ``(Σw)² / Σw²`` for
    independent pixels, 1 for fully correlated ones. Exact for a full block
    with equal weights; it assumes valid pixels and weights are spread
    evenly over the block, so it overestimates ``n_eff`` (underestimates the
    sigma) for blocks whose valid pixels cluster, such as edge cells.

    Examples
    --------
    >>> n_eff_correlated(sum_w=100.0, sum_w2=100.0, rho_bar=0.0)
    100.0

    """
    total2 = sum_w**2
    return total2 / (sum_w2 + (total2 - sum_w2) * rho_bar)


# --- DISP against GNSS (PRD S4, plan task T32) ---------------------------------


@dataclass(frozen=True)
class Requirement:
    """Accuracy requirement ``a + b * sqrt(L)`` against the distance ``L``.

    ``a_mm`` and ``b_mm`` are in millimeters (millimeters per year for a
    velocity requirement); calling it with distances in kilometers returns
    the threshold in meters (meters per year), the unit of DISP. The
    requirement applies over ``distance_km``; pairs outside it are not
    scored against it.

    Examples
    --------
    >>> from disp_xr.residuals import DISP_S1_DISPLACEMENT
    >>> float(DISP_S1_DISPLACEMENT(4.0))  # 3 + 3 * sqrt(4) mm
    0.009

    """

    name: str
    a_mm: float
    b_mm: float = 0.0
    distance_km: tuple[float, float] = (0.1, 50.0)
    units: str = "mm"

    def __call__(self, distance_km: Any) -> np.ndarray:
        """Return the threshold in meters (or m/yr) at ``distance_km``."""
        d = np.asarray(distance_km, dtype=np.float64)
        return (self.a_mm + self.b_mm * np.sqrt(d)) / 1000.0

    def applies(self, distance_km: Any) -> np.ndarray:
        """Return True where ``distance_km`` is inside the requirement's range."""
        d = np.asarray(distance_km, dtype=np.float64)
        return (d >= self.distance_km[0]) & (d <= self.distance_km[1])


# OPERA DISP-S1 requirements drawn by default (PRD S4 decision of 2026-10-04):
# relative displacement 3 (1 + sqrt(L)) mm and relative velocity 2 mm/yr, for
# 0.1 km < L < 50 km. Values to be confirmed against the OPERA requirements
# document before v0.3.
DISP_S1_DISPLACEMENT = Requirement("OPERA DISP-S1 displacement", 3.0, 3.0)
DISP_S1_VELOCITY = Requirement("OPERA DISP-S1 velocity", 2.0, 0.0, units="mm/yr")


def _haversine_km(lon1, lat1, lon2, lat2) -> np.ndarray:
    lon1, lat1, lon2, lat2 = (
        np.radians(np.asarray(v, float)) for v in (lon1, lat1, lon2, lat2)
    )
    h = (
        np.sin((lat2 - lat1) / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    )
    return 2 * 6371.0088 * np.arcsin(np.sqrt(h))


def _velocity_pairs(
    stations: Any, coords: dict, requirement: Requirement, scale: float
):
    """Return station-pair differences of velocity residuals against ``requirement``."""
    import pandas as pd

    ok = stations["velocity_residual"].dropna()
    ids = [s for s in ok.index if str(s) in coords]
    rows = []
    for i, a in enumerate(ids):
        for b in ids[i + 1 :]:
            (lo1, la1), (lo2, la2) = coords[str(a)], coords[str(b)]
            rows.append((a, b, float(_haversine_km(lo1, la1, lo2, la2))))
    pairs = pd.DataFrame(rows, columns=["station_1", "station_2", "distance_km"])
    pairs["difference"] = (
        ok.reindex(pairs["station_1"]).to_numpy()
        - ok.reindex(pairs["station_2"]).to_numpy()
    )
    pairs["requirement"] = requirement(pairs["distance_km"].to_numpy()) * scale
    pairs["applies"] = requirement.applies(pairs["distance_km"].to_numpy())
    pairs["passing"] = pairs["difference"].abs() <= pairs["requirement"]
    return pairs


@dataclass
class Comparison:
    """DISP against GNSS at the stations; the metrics come from geepers.

    Attributes are filled by :func:`compare`.

    Examples
    --------
    >>> result = compare(ds, gps, static)  # doctest: +SKIP
    >>> result.summary["velocity_fraction_passing"]  # doctest: +SKIP
    >>> result.plot_velocities()  # doctest: +SKIP

    """

    series: xr.Dataset
    """``(station, time)``: ``insar``, ``gnss`` (on the DISP epochs and
    reference), ``residual`` (insar - gnss) and ``sigma_gnss``."""
    stations: Any
    """DataFrame per station: velocities of both series from the same
    estimator (geepers ``fit_velocities``) with sigmas, ``velocity_residual``,
    ``relative_residual`` (less the datum offset), ``velocity_sigma``
    (combined), ``normalized_residual`` (relative residual over its sigma),
    ``chi2`` (reduced, of the displacement residual against the GNSS and
    DISP sigmas in quadrature), ``n_dates`` and ``outlier``."""
    pairs: Any
    """geepers ``pairwise_differential_rmse``: RMSE of the relative misfit of
    every station pair against distance."""
    profile: Any
    """geepers ``binned_rmse_profile`` of ``pairs`` (with ``requirement``
    and ``fraction_passing`` when a requirement is given)."""
    epochs: Any
    """geepers ``epoch_rmse`` per DISP epoch, with ``flag`` for epochs above
    the median plus ``epoch_flag`` x MAD of the RMSE."""
    semivariogram: Any
    """geepers ``misfit_semivariogram`` of ``pairs``."""
    summary: dict
    """geepers ``comparison_stats`` of the station velocities (bias, MAD,
    RMSE, r², n), ``datum_offset`` (median velocity residual: DISP is
    relative to its reference, GNSS absolute), ``disp_sigma`` (the DISP
    noise used in ``chi2``), with ``velocity_fraction_passing`` and the
    requirement names when requirements are given."""
    velocity_pairs: Any = None
    """Per station pair: distance, difference of the velocity residuals,
    the velocity ``requirement`` there, ``applies`` (inside its distance
    range) and ``passing``; None without a velocity requirement."""

    def plot_velocities(self, **kwargs: Any) -> Any:
        """Scatter GNSS against DISP velocities (geepers ``plot_velocity_scatter``)."""
        from geepers.validation import plot_velocity_scatter

        return plot_velocity_scatter(self.stations, **kwargs)

    def plot_semivariogram(self, **kwargs: Any) -> Any:
        """Plot the misfit semivariogram (geepers ``plot_semivariogram``)."""
        from geepers.validation import plot_semivariogram

        return plot_semivariogram(self.pairs, self.semivariogram, **kwargs)


def _merged_tables(series: xr.Dataset) -> dict:
    """Per-station DataFrames indexed by date with ``los_gps``, ``los_insar``."""
    import pandas as pd

    out = {}
    for station in series["station"].values:
        one = series.sel(station=station)
        out[str(station)] = pd.DataFrame(
            {"los_gps": one["gnss"].values, "los_insar": one["insar"].values},
            index=pd.DatetimeIndex(one["time"].values, name="date"),
        )
    return out


def _cell_series(cells: Any, gnss: xr.Dataset, layer: str) -> xr.DataArray:
    """Values of the cell holding each station: ``(station, time)``."""
    from scipy.spatial import cKDTree

    if hasattr(cells, "leaf_id"):  # a Quadtree
        leaf = cells.leaf_id.sel(x=gnss["x"], y=gnss["y"], method="nearest").values
        points = cells.points
        index = np.asarray(leaf, dtype=np.int64)
    else:
        points = cells
        tree = cKDTree(np.column_stack([points["x"].values, points["y"].values]))
        _, index = tree.query(np.column_stack([gnss["x"].values, gnss["y"].values]))
    values = points[layer].isel(
        point=xr.DataArray(np.maximum(index, 0), dims="station")
    )
    values = values.where(xr.DataArray(index >= 0, dims="station"))
    values = values.drop_vars([c for c in values.coords if c not in ("time",)])
    return values.assign_coords(station=gnss["station"].values).transpose(
        "station", ...
    )


def compare(
    data: Any,
    gnss: xr.Dataset,
    static: xr.Dataset,
    *,
    buffer_m: float = 0.0,
    layer: str = "displacement",
    velocity_method: str = "lsq",
    tolerance: str = "1D",
    min_common_dates: int = 3,
    min_stations: int = 3,
    n_bins: int = 10,
    requirement: Optional[Any] = DISP_S1_DISPLACEMENT,
    velocity_requirement: Optional[Requirement] = DISP_S1_VELOCITY,
    outlier_sigma: float = 3.0,
    epoch_flag: float = 3.0,
    disp_sigma: Optional[float] = None,
) -> Comparison:
    """Compare DISP with GNSS at the stations (or the cells holding them).

    geepers computes the metrics (velocities from one estimator applied to
    both series, pairwise differential RMSE against distance, its binned
    profile and semivariogram, epoch RMSE, agreement statistics); disp-xr
    prepares the series and adds normalized velocity residuals, reduced χ²
    per station and flags.

    Parameters
    ----------
    data : xr.Dataset, point table or Quadtree
        DISP at 30 m (read at each station with :func:`disp_xr.gnss.sample`),
        or downsampled cells: a point table from :func:`disp_xr.to_points`
        or :func:`disp_xr.downsample_hex` (the nearest cell center) or a
        :class:`disp_xr.Quadtree` (the leaf holding the station).
    gnss : xr.Dataset
        :func:`disp_xr.gnss.load` output for the same frame.
    static : xr.Dataset
        LOS layers for :func:`disp_xr.gnss.to_los`.
    buffer_m : float
        Window radius for sampling 30 m data.
    layer : str
        DISP layer.
    velocity_method : {"lsq", "midas", "trend"}
        Estimator for both series (geepers ``fit_velocities``); MIDAS needs
        about a year of data.
    tolerance : str
        Largest gap between a DISP epoch and a GNSS day.
    min_common_dates, min_stations, n_bins
        Passed to geepers (pairs, epochs, profile).
    requirement : Requirement or callable, optional
        Threshold of the pairwise relative displacement RMSE against
        distance, drawn on the profile with geepers' ``fraction_passing``.
        Default :data:`DISP_S1_DISPLACEMENT`. A :class:`Requirement` follows
        the units of ``layer``; a plain ``requirement(distance_km)`` must
        return the units of the data. None for no requirement.
    velocity_requirement : Requirement, optional
        Threshold of the relative velocity: every station pair's difference
        of velocity residuals is scored against it (``velocity_pairs``,
        ``summary["velocity_fraction_passing"]`` over the pairs inside its
        distance range). Default :data:`DISP_S1_VELOCITY`; None to skip.
    outlier_sigma : float
        Stations with ``|normalized_residual|`` or ``sqrt(chi2)`` above this
        are flagged.
    disp_sigma : float, optional
        Noise of one DISP epoch at a station, in the units of ``layer``,
        added in quadrature to the GNSS sigma in ``chi2``. Default: estimated
        from all stations' residuals (1.4826 MAD, less the median GNSS
        variance), so one bad station does not raise it.
    epoch_flag : float
        Epochs whose RMSE exceeds the median plus this many MADs are flagged.

    Returns
    -------
    Comparison

    Examples
    --------
    >>> gps = dxr.gnss.load(ds)  # doctest: +SKIP
    >>> result = compare(ds, gps, dxr.open_static(ds))  # doctest: +SKIP
    >>> result.stations[["velocity_residual", "outlier"]]  # doctest: +SKIP
    >>> cells = compare(dxr.quadtree(ds, n_points=5000), gps, static)  # doctest: +SKIP

    """
    from geepers.analysis import (
        binned_rmse_profile,
        epoch_rmse,
        pairwise_differential_rmse,
    )
    from geepers.validation import (
        comparison_stats,
        fit_velocities,
        misfit_semivariogram,
    )

    from . import gnss as gnss_mod

    if isinstance(data, xr.Dataset) and {"y", "x"} <= set(data.dims):
        insar = gnss_mod.sample(data, gnss, buffer_m=buffer_m, layer=layer)
        times = data["time"]
    else:
        insar = _cell_series(data, gnss, layer)
        times = insar["time"]
    projected = gnss_mod.to_los(gnss, static)
    gnss_los = gnss_mod.align(projected["los"], times, tolerance=tolerance)
    series = xr.Dataset(
        {
            "insar": insar.drop_vars(
                [c for c in insar.coords if c not in ("station", "time")]
            ),
            "gnss": gnss_los.drop_vars(
                [c for c in gnss_los.coords if c not in ("station", "time")]
            ),
        }
    )
    if "sigma_los" in projected:
        sigma = projected["sigma_los"].reindex(
            time=series["time"].values,
            method="nearest",
            tolerance=np.timedelta64(1, "D"),
        )
        series["sigma_gnss"] = sigma.drop_vars(
            [c for c in sigma.coords if c not in ("station", "time")]
        )
    series["residual"] = series["insar"] - series["gnss"]
    series = series.assign_coords(
        {c: gnss[c] for c in ("lon", "lat", "x", "y") if c in gnss.coords}
    )

    tables = _merged_tables(series)
    coords = {
        str(s): (float(lo), float(la))
        for s, lo, la in zip(
            series["station"].values, gnss["lon"].values, gnss["lat"].values
        )
    }
    stations = fit_velocities(tables, method=velocity_method)
    stations["velocity_residual"] = (
        stations["insar_velocity"] - stations["gps_velocity"]
    )
    stations["velocity_sigma"] = np.hypot(
        stations["insar_sigma"], stations["gps_sigma"]
    )
    # DISP is relative to its reference area and GNSS is absolute, so every
    # velocity residual carries the same datum offset; the median over
    # stations estimates it without knowing which station is the reference.
    datum_offset = float(np.nanmedian(stations["velocity_residual"]))
    stations["relative_residual"] = stations["velocity_residual"] - datum_offset
    stations["normalized_residual"] = (
        stations["relative_residual"] / stations["velocity_sigma"]
    )
    resid = series["residual"] - series["residual"].mean("time")
    if "sigma_gnss" in series:
        if disp_sigma is None:
            disp_sigma = _disp_sigma(resid, series["sigma_gnss"])
        variance = series["sigma_gnss"] ** 2 + disp_sigma**2
        z2 = resid**2 / variance
        count = z2.notnull().sum("time")
        chi2 = z2.sum("time") / (count - 1).where(count > 1)
        stations["chi2"] = chi2.to_series().reindex(stations.index).to_numpy()
    else:
        stations["chi2"] = np.nan
    stations["outlier"] = (stations["normalized_residual"].abs() > outlier_sigma) | (
        np.sqrt(stations["chi2"]) > outlier_sigma
    )
    pairs = pairwise_differential_rmse(
        tables, coords, min_common_dates=min_common_dates
    )
    units = _canonical(str(series["insar"].attrs.get("units", "m"))) or "m"
    scale = 1.0 / _SCALE[units]  # requirements are in meters
    threshold = requirement
    if isinstance(requirement, Requirement):

        def threshold(d, _r=requirement):
            return _r(d) * scale

    profile = binned_rmse_profile(pairs, n_bins=n_bins, requirement=threshold)
    epochs = epoch_rmse(tables, min_stations=min_stations)
    if len(epochs):
        median = epochs["rmse"].median()
        mad = 1.4826 * (epochs["rmse"] - median).abs().median()
        epochs["flag"] = epochs["rmse"] > median + epoch_flag * mad
    semivariogram = misfit_semivariogram(pairs) if len(pairs) else pairs
    summary = comparison_stats(
        stations["gps_velocity"].to_numpy(), stations["insar_velocity"].to_numpy()
    )
    summary["datum_offset"] = datum_offset
    summary["disp_sigma"] = np.nan if disp_sigma is None else float(disp_sigma)
    velocity_pairs = None
    if isinstance(requirement, Requirement):
        summary["requirement"] = requirement.name
    if velocity_requirement is not None:
        velocity_pairs = _velocity_pairs(stations, coords, velocity_requirement, scale)
        scored = velocity_pairs[velocity_pairs["applies"]]
        summary["velocity_requirement"] = velocity_requirement.name
        summary["velocity_fraction_passing"] = (
            float(scored["passing"].mean()) if len(scored) else np.nan
        )
    return Comparison(
        series, stations, pairs, profile, epochs, semivariogram, summary, velocity_pairs
    )


def _disp_sigma(resid: xr.DataArray, sigma_gnss: xr.DataArray) -> float:
    """Per-epoch DISP noise: robust residual scatter less the GNSS part."""
    values = resid.values[np.isfinite(resid.values)]
    if values.size == 0:
        return 0.0
    scatter = 1.4826 * np.median(np.abs(values - np.median(values)))
    gnss_var = float(np.nanmedian(sigma_gnss.values**2))
    return float(np.sqrt(max(scatter**2 - gnss_var, 0.0)))


def _residual_array(source: Any) -> xr.DataArray:
    return source.series["residual"] if isinstance(source, Comparison) else source


def residual_spectra(
    source: Any,
    *,
    frequencies: Optional[np.ndarray] = None,
    detrend: bool = True,
) -> xr.DataArray:
    """Return the Lomb-Scargle power spectrum of each station's residual series.

    Lomb-Scargle handles the irregular DISP epochs (gaps, 12 and 24 day
    revisits) without resampling. Peaks at 1 and 2 cycles per year are
    seasonal signal one series has and the other lacks; a red spectrum
    (power rising to low frequencies) is correlated error.

    Parameters
    ----------
    source : Comparison or xr.DataArray
        :func:`compare` output, or a ``(station, time)`` residual.
    frequencies : array, optional
        Frequencies in cycles per year; default from one cycle over the
        record to the Nyquist frequency of the median epoch spacing.
    detrend : bool
        Remove each series' linear trend first.

    Returns
    -------
    xr.DataArray
        ``(station, frequency)`` power (normalized, 0 to 1).

    Examples
    --------
    >>> power = residual_spectra(result)  # doctest: +SKIP

    """
    from scipy.signal import lombscargle

    resid = _residual_array(source)
    years = (
        (resid["time"].values - resid["time"].values[0])
        / np.timedelta64(1, "D")
        / 365.25
    )
    if frequencies is None:
        span = years[-1] - years[0]
        step = np.median(np.diff(years))
        frequencies = np.linspace(1.0 / span, 0.5 / step, 200)
    angular = 2 * np.pi * np.asarray(frequencies, dtype=np.float64)
    power = np.full((resid.sizes["station"], angular.size), np.nan)
    values = resid.transpose("station", "time").values
    for k, series in enumerate(values):
        ok = np.isfinite(series)
        if ok.sum() < 4:
            continue
        t, y = years[ok], series[ok]
        if detrend:
            y = y - np.polyval(np.polyfit(t, y, 1), t)
        power[k] = lombscargle(t, y - y.mean(), angular, normalize=True)
    return xr.DataArray(
        power,
        dims=("station", "frequency"),
        coords={"station": resid["station"].values, "frequency": frequencies},
        name="power",
        attrs={"frequency_units": "cycles per year", "method": "Lomb-Scargle"},
    )


def common_mode(
    source: Any, n_components: int = 1, min_coverage: float = 0.5
) -> xr.Dataset:
    """Return the common-mode residual across stations (geepers ``estimate_cme``).

    A signal shared by all stations' residuals (a reference or datum shift,
    a frame-wide atmospheric or orbit error) is a product error that the
    station-by-station metrics miss.

    Parameters
    ----------
    source : Comparison or xr.DataArray
        :func:`compare` output, or a ``(station, time)`` residual.
    n_components : int
        Principal components to keep.
    min_coverage : float
        Stations observed in fewer than this share of epochs are left out.

    Returns
    -------
    xr.Dataset
        ``common_mode`` ``(time, component)``, ``spatial_response``
        ``(station, component)``, ``cleaned`` ``(station, time)`` (residual
        minus the common mode) and ``explained_variance`` ``(component)``.

    Examples
    --------
    >>> cme = common_mode(result, n_components=1)  # doctest: +SKIP

    """
    from geepers.cme import estimate_cme

    resid = _residual_array(source)
    table = resid.transpose("time", "station").to_pandas()
    result = estimate_cme(table, n_components=n_components, min_coverage=min_coverage)
    comp = np.arange(n_components)
    return (
        xr.Dataset(
            {
                "common_mode": (("time", "component"), result.common_mode.to_numpy()),
                "spatial_response": (
                    ("used_station", "component"),
                    result.spatial_response.to_numpy(),
                ),
                "cleaned": (
                    ("time", "used_station"),
                    result.cleaned.to_numpy(),
                ),
                "explained_variance": (
                    "component",
                    np.asarray(result.explained_variance),
                ),
            },
            coords={
                "time": resid["time"].values,
                "component": comp,
                "used_station": result.spatial_response.index.to_numpy(),
            },
        )
        .rename({"used_station": "station"})
        .transpose("station", "time", "component")
    )
