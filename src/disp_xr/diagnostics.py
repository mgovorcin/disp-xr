"""Per-pixel diagnostics of a fitted trajectory model (PRD S4, plan T32.6b-c).

:func:`model_diagnostics` describes the residuals of every pixel's fit: mean,
standard deviation, skewness, excess kurtosis, autocorrelation at chosen lags
and the Anderson-Darling statistic for normality. White-noise sigmas (the
``sigma`` of :func:`disp_xr.fit_trajectory`) hold only when the residuals are
uncorrelated and Gaussian; autocorrelation above about 0.3 or an
Anderson-Darling statistic above about 0.75 (the 5% critical value) says
they are optimistic, and :func:`disp_xr.fit_velocity` with a colored-noise
method is the better tool. The model is evaluated on the fly from the
coefficients (:func:`disp_xr.predict_trajectory`), never stored, and every
statistic comes from one pass per block of pixels over ``time``-contiguous
chunks.

:func:`significance` turns coefficients and sigmas into z-scores, two-sided
p-values and confidence intervals.

The statistics follow the Brisbane notebook (branch ``_stats_analysis.py``),
with Anderson-Darling vectorized over pixels in NumPy instead of a
per-pixel ``apply_ufunc(vectorize=True)``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Union

import numpy as np
import xarray as xr

from ._chunks import data_chunks, time_contiguous
from .timeseries.trajectory import predict_trajectory

__all__ = ["anderson_darling", "model_diagnostics", "significance"]

# Small-sample correction of the Anderson-Darling statistic with estimated
# mean and variance (Stephens 1974, https://doi.org/10.1080/01621459.1974.10480196);
# its 5% critical value is 0.752.
AD_CRITICAL_5 = 0.752


def significance(fit: xr.Dataset, alpha: float = 0.05) -> xr.Dataset:
    """Return z-scores, p-values and confidence intervals of fitted coefficients.

    Parameters
    ----------
    fit : xr.Dataset
        :func:`disp_xr.fit_trajectory` output (``coefficients`` and
        ``sigma``, ``(coefficient, y, x)``).
    alpha : float
        Confidence intervals cover ``1 - alpha``.

    Returns
    -------
    xr.Dataset
        ``z`` (coefficient over sigma), ``p_value`` (two-sided, Gaussian),
        ``ci_low`` and ``ci_high``, and ``significant`` (``p_value < alpha``).

    Examples
    --------
    >>> sig = significance(dxr.fit_trajectory(ds))  # doctest: +SKIP
    >>> sig.sel(coefficient="linear_trend").significant.mean()  # doctest: +SKIP

    """
    from scipy.special import ndtr, ndtri

    coef, sigma = fit["coefficients"], fit["sigma"]
    z = coef / sigma.where(sigma > 0)
    p = xr.apply_ufunc(
        lambda v: 2.0 * ndtr(-np.abs(v)), z, dask="parallelized", output_dtypes=[float]
    )
    half = float(ndtri(1.0 - alpha / 2.0)) * sigma
    out = xr.Dataset(
        {
            "z": z.assign_attrs(long_name="coefficient / sigma"),
            "p_value": p.assign_attrs(long_name="two-sided p-value (Gaussian)"),
            "ci_low": (coef - half).assign_attrs(coef.attrs),
            "ci_high": (coef + half).assign_attrs(coef.attrs),
            "significant": p < alpha,
        }
    )
    out.attrs = {"alpha": alpha}
    return out


def anderson_darling(values: np.ndarray, axis: int = -1) -> np.ndarray:
    """Return the Anderson-Darling normality statistic along ``axis``, ignoring NaN.

    Mean and variance are estimated from the data, with Stephens'
    small-sample correction ``A² (1 + 0.75 / n + 2.25 / n²)``. Vectorized: one
    sort for all series. NaN where fewer than 8 values are finite.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> bool(anderson_darling(rng.standard_normal(500)) < AD_CRITICAL_5)
    True

    """
    from scipy.special import log_ndtr

    x = np.moveaxis(np.asarray(values, dtype=np.float64), axis, -1)
    n = np.isfinite(x).sum(axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.nanmean(np.where(n[..., None] > 0, x, 0.0), axis=-1)
        std = np.sqrt(
            np.nansum((x - mean[..., None]) ** 2, axis=-1) / np.maximum(n - 1, 1)
        )
        z = np.sort((x - mean[..., None]) / std[..., None], axis=-1)  # NaN last
    m = x.shape[-1]
    i = np.arange(1, m + 1)
    valid = i <= n[..., None]
    # z_(n + 1 - i): mirror inside the valid part of each series.
    mirror = np.clip(n[..., None] - i, 0, m - 1)
    z_rev = np.take_along_axis(z, mirror, axis=-1)
    terms = (2 * i - 1) * (log_ndtr(z) + log_ndtr(-z_rev))
    with np.errstate(invalid="ignore", divide="ignore"):
        a2 = -n - np.where(valid, terms, 0.0).sum(axis=-1) / n
        a2 = a2 * (1.0 + 0.75 / n + 2.25 / n**2)
    return np.where((n >= 8) & (std > 0), a2, np.nan)


def _stats(residual: np.ndarray, lags: Sequence[int]) -> np.ndarray:
    """Per-pixel statistics over the last axis: one stacked array."""
    r = residual.astype(np.float64)
    ok = np.isfinite(r)
    n = ok.sum(axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(ok, r, 0.0).sum(axis=-1) / n
        c = np.where(ok, r - mean[..., None], 0.0)
        m2 = (c**2).sum(axis=-1) / n
        m3 = (c**3).sum(axis=-1) / n
        m4 = (c**4).sum(axis=-1) / n
        std = np.sqrt(m2 * n / np.maximum(n - 1, 1))
        skew = m3 / m2**1.5
        kurt = m4 / m2**2 - 3.0
        autos = []
        for lag in lags:
            both = ok[..., lag:] & ok[..., :-lag]
            num = np.where(both, c[..., lag:] * c[..., :-lag], 0.0).sum(axis=-1)
            pairs = both.sum(axis=-1)
            autos.append((num / pairs) / m2)
    ad = anderson_darling(r)
    return np.stack([n.astype(float), mean, std, skew, kurt, *autos, ad], axis=-1)


def model_diagnostics(
    data: Union[xr.Dataset, xr.DataArray],
    fit: xr.Dataset,
    *,
    layer: str = "displacement",
    lags: Sequence[int] = (1,),
) -> xr.Dataset:
    """Describe the residuals of a trajectory fit at every pixel, lazily.

    Parameters
    ----------
    data : xr.Dataset or xr.DataArray
        The data that were fitted, ``(time, y, x)``.
    fit : xr.Dataset
        :func:`disp_xr.fit_trajectory` output for ``data``.
    layer : str
        Layer of a Dataset.
    lags : sequence of int
        Lags, in epochs, of the residual autocorrelation.

    Returns
    -------
    xr.Dataset
        ``(y, x)``: ``n`` (residuals used), ``residual_mean``,
        ``residual_std``, ``skewness``,
        ``kurtosis`` (excess), ``autocorrelation_lag<k>`` and
        ``anderson_darling`` (normality; above ``AD_CRITICAL_5`` rejects at
        5%).

    Examples
    --------
    >>> fit = dxr.fit_trajectory(ds)  # doctest: +SKIP
    >>> diag = model_diagnostics(ds, fit, lags=(1, 2))  # doctest: +SKIP
    >>> (diag.autocorrelation_lag1 > 0.3).mean()  # doctest: +SKIP

    """
    da = data[layer] if isinstance(data, xr.Dataset) else data
    da = time_contiguous(da)
    model = predict_trajectory(fit, da["time"].values, dtype="float64")
    if da.chunks is not None:
        model = model.chunk({d: data_chunks(da)[d] for d in model.dims if d in da.dims})
    residual = (da.astype(np.float64) - model).transpose(..., "time")
    names = [
        "n",
        "residual_mean",
        "residual_std",
        "skewness",
        "kurtosis",
        *[f"autocorrelation_lag{lag}" for lag in lags],
        "anderson_darling",
    ]
    stacked = xr.apply_ufunc(
        _stats,
        residual,
        input_core_dims=[["time"]],
        output_core_dims=[["statistic"]],
        kwargs={"lags": tuple(int(k) for k in lags)},
        dask="parallelized",
        output_dtypes=[np.float64],
        dask_gufunc_kwargs={"output_sizes": {"statistic": len(names)}},
    )
    out = xr.Dataset(
        {name: stacked.isel(statistic=k, drop=True) for k, name in enumerate(names)}
    )
    units = da.attrs.get("units", "")
    out["residual_mean"].attrs["units"] = units
    out["residual_std"].attrs["units"] = units
    out["anderson_darling"].attrs["critical_5_percent"] = AD_CRITICAL_5
    out = out.drop_vars([c for c in out.coords if c not in ("y", "x", "spatial_ref")])
    out.attrs = {"model": fit.attrs.get("terms", "")}
    return out
