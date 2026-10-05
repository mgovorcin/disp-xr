"""Polynomial trends and per-epoch de-ramping (PRD F9, plan task T24).

:class:`Trend` fits a 2D polynomial of the coordinates by weighted least
squares, on coordinates centered and scaled for conditioning (the raw UTM
values, near 10⁶, would make the normal matrix singular in float64 from
degree 2). It is the simplest spatial model and a step of the GNSS tie.

:func:`deramp` removes a polynomial ramp from every epoch of a stack. In
the temporal-denoising study, per-epoch spatial de-ramping cut velocity
scatter by 10 to 24%, more than any temporal filter. The fit only needs
sums over pixels (the normal equations), so it stays lazy and exact; it runs
on every ``stride``-th pixel by default and the ramp is subtracted from the
full grid chunk by chunk. Ramps are long-wavelength by construction: remove
them only when long-wavelength signal (plate motion, large subsidence
bowls) is not wanted, or mask such areas out of the fit.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import xarray as xr

from .._history import recorded
from ..config import bounded_threads
from .base import BaseEstimator

__all__ = ["Trend", "deramp", "polynomial_terms"]


def polynomial_terms(degree: int) -> list[tuple[int, int]]:
    """Return the ``(i, j)`` exponents of ``e**i * n**j`` with ``i + j <= degree``.

    Examples
    --------
    >>> polynomial_terms(1)
    [(0, 0), (1, 0), (0, 1)]

    """
    if degree < 0:
        raise ValueError(f"degree must be >= 0, not {degree}")
    return [(i, d - i) for d in range(degree + 1) for i in range(d, -1, -1)]


def _term_name(i: int, j: int) -> str:
    parts = [
        f"e{'' if i == 1 else i}" if i else "",
        f"n{'' if j == 1 else j}" if j else "",
    ]
    return "".join(parts) or "1"


class Trend(BaseEstimator):
    """A 2D polynomial trend of degree ``degree``, by weighted least squares.

    Parameters
    ----------
    degree : int
        0 is a constant, 1 a plane, 2 a quadratic surface.

    Attributes
    ----------
    coef_ : numpy.ndarray
        Coefficients of the terms in :func:`polynomial_terms` order, for the
        normalized coordinates ``(e - center) / scale``.
    center_, scale_ : tuple of float
        The normalization.

    Examples
    --------
    >>> plane = Trend(1).fit(coords, data)  # doctest: +SKIP
    >>> ramp = plane.grid(like=ds)  # doctest: +SKIP

    """

    def __init__(self, degree: int = 1) -> None:
        """Store the degree."""
        self.degree = degree

    def _design(self, easting: np.ndarray, northing: np.ndarray) -> np.ndarray:
        e = (easting - self.center_[0]) / self.scale_
        n = (northing - self.center_[1]) / self.scale_
        return np.column_stack([e**i * n**j for i, j in polynomial_terms(self.degree)])

    def _fit(self, easting, northing, data, weights) -> None:
        self.center_ = (float(easting.mean()), float(northing.mean()))
        span = max(np.ptp(easting), np.ptp(northing))
        self.scale_ = float(span / 2) if span > 0 else 1.0
        design = self._design(easting, northing)
        if design.shape[0] < design.shape[1]:
            raise ValueError(
                f"{design.shape[0]} points for {design.shape[1]} terms "
                f"of degree {self.degree}"
            )
        sw = np.sqrt(weights)
        self.coef_ = np.linalg.lstsq(design * sw[:, None], data * sw, rcond=None)[0]

    def _predict(self, easting, northing) -> np.ndarray:
        return self._design(easting, northing) @ self.coef_


@bounded_threads
@recorded
def deramp(
    ds: xr.Dataset,
    degree: int = 1,
    *,
    layer: str = "displacement",
    mask: Optional[xr.DataArray] = None,
    weights: Optional[xr.DataArray] = None,
    stride: int = 4,
) -> xr.Dataset:
    """Remove a polynomial ramp from every epoch of ``layer``, lazily.

    Parameters
    ----------
    ds : xr.Dataset
        Data with ``y``, ``x`` (and ``time``).
    degree : int
        Degree of the ramp: 1 a plane (default), 2 a quadratic surface.
    layer : str
        Layer to de-ramp.
    mask : xr.DataArray, optional
        Boolean ``(y, x)`` or ``(time, y, x)``: pixels to fit the ramp on
        (for example ``disp_xr.valid_mask(ds)``, with deforming areas
        removed). The ramp is subtracted everywhere.
    weights : xr.DataArray, optional
        Non-negative weights for the fit, for example quality weights.
    stride : int
        Fit on every ``stride``-th pixel in ``y`` and ``x`` (1 uses all).

    Returns
    -------
    xr.Dataset
        ``ds`` with ``layer`` de-ramped and ``<layer>_ramp_coefficients``
        ``(time, term)``: the coefficients of each epoch for coordinates
        normalized as ``(e - center) / scale`` (``ramp_center``,
        ``ramp_scale`` attributes).

    Examples
    --------
    >>> flat = dxr.models.deramp(ds, degree=1)  # doctest: +SKIP

    """
    terms = polynomial_terms(degree)
    data = ds[layer]
    x = ds["x"].values.astype(np.float64)
    y = ds["y"].values.astype(np.float64)
    center = (float(x.mean()), float(y.mean()))
    scale = float(max(np.ptp(x), np.ptp(y)) / 2) or 1.0
    e = xr.DataArray((x - center[0]) / scale, dims="x", coords={"x": ds["x"]})
    n = xr.DataArray((y - center[1]) / scale, dims="y", coords={"y": ds["y"]})
    monomials = [e**i * n**j for i, j in terms]

    sample = {"y": slice(None, None, stride), "x": slice(None, None, stride)}
    d = data.isel(sample).astype(np.float64)
    w = xr.ones_like(d) if weights is None else weights.isel(sample).astype(np.float64)
    if mask is not None:
        w = w.where(mask.isel(sample), 0.0)
    w = w.where(d.notnull(), 0.0).fillna(0.0)
    d = d.fillna(0.0)
    mono = [m.isel(sample) for m in monomials]
    spatial = ("y", "x")
    k = len(terms)
    normal = xr.concat(
        [
            xr.concat(
                [(w * mono[a] * mono[b]).sum(spatial) for b in range(k)], "term_b"
            )
            for a in range(k)
        ],
        "term",
    )
    rhs = xr.concat([(w * mono[a] * d).sum(spatial) for a in range(k)], "term")

    def solve(g: np.ndarray, b: np.ndarray) -> np.ndarray:
        out = np.full(b.shape, np.nan)
        for idx in np.ndindex(b.shape[:-1]):
            if np.linalg.matrix_rank(g[idx]) == k:
                out[idx] = np.linalg.solve(g[idx], b[idx])
        return out

    if normal.chunks is not None:
        normal = normal.chunk({"term": -1, "term_b": -1})
        rhs = rhs.chunk({"term": -1})
    coef = xr.apply_ufunc(
        solve,
        normal,
        rhs,
        input_core_dims=[["term", "term_b"], ["term"]],
        output_core_dims=[["term"]],
        dask="parallelized",
        output_dtypes=[np.float64],
    )
    ramp = sum(coef.isel(term=t, drop=True) * monomials[t] for t in range(k))
    out = ds.copy()
    flat = (data - ramp.astype(data.dtype)).transpose(*data.dims)
    out[layer] = flat.assign_attrs(data.attrs, deramped=f"polynomial degree {degree}")
    out[f"{layer}_ramp_coefficients"] = coef.assign_coords(
        term=[_term_name(i, j) for i, j in terms]
    ).assign_attrs(
        long_name=f"ramp removed from {layer}, per epoch",
        units=data.attrs.get("units", ""),
        ramp_center=list(center),
        ramp_scale=scale,
    )
    return out
