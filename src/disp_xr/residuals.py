"""Spatial variograms of residual fields (PRD S2 cell uncertainty, plan task T27.3b).

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
from typing import Literal, Optional, Union

import numpy as np
import xarray as xr

__all__ = [
    "VariogramModel",
    "empirical_variogram",
    "fit_variogram",
    "mean_block_correlation",
    "mean_mask_correlation",
    "n_eff_correlated",
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
    """
    total2 = sum_w**2
    return total2 / (sum_w2 + (total2 - sum_w2) * rho_bar)
