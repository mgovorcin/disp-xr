# SPDX-FileCopyrightText: 2025-2026 California Institute of Technology ("Caltech")
# SPDX-FileCopyrightText: 2015 Geoff Blewitt (original MIDAS code, Nevada Geodetic Laboratory, UNR)
# SPDX-License-Identifier: Apache-2.0
# The MIDAS pair selection (`_select_pairs`) and trimming are adapted from
# geepers (https://github.com/opera-adt/geepers, commit d542178,
# src/geepers/midas.py), Copyright (c) 2025-2026 California Institute of
# Technology, Apache License 2.0, itself a conversion of Geoff Blewitt's MIDAS
# code (https://geodesy.unr.edu/MIDAS_release.tar). The Whittle likelihood
# follows geepers trend.py. See THIRD_PARTY_NOTICES.md. Changes: vectorized
# over pixels; pairs are selected once from the stack dates and masked per
# pixel; Whittle uses a grid search with local refinement.
"""Velocity for every pixel: OLS, Whittle colored noise, or MIDAS.

``fit_velocity(ds, method=...)`` runs per pixel on a full frame:

``"ols"``
    The linear term of :func:`~disp_xr.timeseries.trajectory.fit_trajectory`,
    with white-noise sigmas (too small for real InSAR series).
``"whittle"``
    The same OLS velocity, with noise estimated per pixel from the
    periodogram of the OLS residuals (Whittle likelihood, power law plus
    white): ``kappa``, the white fraction and amplitudes, and the
    colored-noise sigma of the velocity. Reliable from about 200 epochs; on
    short records it underestimates ``|kappa|`` and the sigma
    (``benchmarks/whittle_bias.py``). The sigma uses the covariance of the
    complete record, so it is approximate for pixels with gaps, and the
    amplitudes come from the gap-filled grid, which interpolation smooths
    (about 10% low with 20% gaps). The exact grouped GLS is plan task T23.
``"midas"``
    The median of slopes between epoch pairs about one year apart, trimmed at
    two scaled MADs (Blewitt et al., 2016): robust to outliers, seasonal
    signals and steps, with its own sigma. A step biases it when the pairs
    that span it are a large share, about one year over the record length,
    so prefer records of several years. Pairs are chosen from the stack
    dates; a pair with a NaN end is dropped, where per-series MIDAS would
    choose a replacement.

Blewitt, G., Kreemer, C., Hammond, W. C., & Gazeaux, J. (2016). MIDAS robust
trend estimator for accurate GPS station velocities without step detection.
JGR Solid Earth, 121, 2054-2068. https://doi.org/10.1002/2015JB012552
"""

from __future__ import annotations

import logging
from typing import Literal, Optional, Union

import numpy as np
import xarray as xr

from .._chunks import time_contiguous
from .._grid import with_grid_mapping
from .._history import recorded
from ._datetime import _decimal_year
from .noise import _powerlaw_covariance, _sampling_grid
from .trajectory import Trajectory, _solve, check_not_denoised, fit_trajectory

logger = logging.getLogger(__name__)

__all__ = ["fit_velocity"]

# Fewer epochs than this and Whittle is biased (benchmarks/whittle_bias.py).
WHITTLE_MIN_EPOCHS = 200
# Coarse Whittle grid: kappa in [-3, 0], logit of the white fraction.
_KAPPA = np.round(np.arange(-3.0, 0.0001, 0.1), 4)
_LOGIT = np.arange(-8.0, 8.0001, 1.0)


# --- MIDAS ---------------------------------------------------------------------


def _select_pairs(times: np.ndarray, tol: float = 0.001) -> np.ndarray:
    """Return ``(n, 2)`` index pairs about one year apart (forward in time).

    For each epoch, the first later epoch within ``tol`` of one year, or else
    the next epoch after one year (Blewitt et al., 2016, without steps).
    """
    m = len(times)
    pairs = []
    k = 0
    for i in range(m):
        if times[i] > times[-1] + tol - 1.0:
            break
        for j in range(i + 1, m):
            k = max(k, j)
            fdt = times[j] - times[i] - 1.0
            if fdt < -tol:
                continue
            if fdt < tol:
                pairs.append((i, j))
            else:
                pairs.append((i, k))
                if k == m - 1:
                    k = 0
                k += 1
            break
    return np.asarray(pairs, dtype=int).reshape(-1, 2)


def _midas_pairs(times: np.ndarray) -> np.ndarray:
    """Forward and backward pairs, as MIDAS combines them."""
    forward = _select_pairs(times)
    backward = _select_pairs(-times[::-1])
    backward = len(times) - 1 - backward[:, ::-1]
    return np.vstack([forward, backward])


def _midas(data: np.ndarray, pairs: np.ndarray, years: np.ndarray) -> tuple:
    """MIDAS per pixel of ``data`` (..., time); NaN pairs are skipped."""
    shape = data.shape[:-1]
    y = data.reshape(-1, data.shape[-1]).astype(np.float64)
    dt = years[pairs[:, 1]] - years[pairs[:, 0]]
    slopes = (y[:, pairs[:, 1]] - y[:, pairs[:, 0]]) / dt  # NaN if either end is
    n_pairs = np.isfinite(slopes).sum(axis=1)
    # numbagg's NaN median is about 250x faster than np.nanmedian here.
    from numbagg import nanmedian

    with np.errstate(all="ignore"), _Quiet():
        v50 = nanmedian(slopes, axis=1)
        sd = 1.4826 * nanmedian(np.abs(slopes - v50[:, None]), axis=1)
        keep = np.abs(slopes - v50[:, None]) < 2.0 * sd[:, None]
        trimmed = np.where(keep, slopes, np.nan)
        n_kept = keep.sum(axis=1)
        v50 = nanmedian(trimmed, axis=1)
        d50 = nanmedian(np.abs(trimmed - v50[:, None]), axis=1)
        sigma = 3.0 * 1.2533 * 1.4826 * d50 / np.sqrt(n_kept / 4.0)
        outliers = (n_pairs - n_kept) / n_pairs
    empty = n_kept == 0
    v50[empty] = sigma[empty] = np.nan
    return (
        v50.reshape(shape),
        sigma.reshape(shape),
        outliers.reshape(shape),
        sd.reshape(shape),
    )


class _Quiet:
    """Silence 'All-NaN slice' warnings from fully masked pixels."""

    def __enter__(self):
        import warnings

        self._ctx = warnings.catch_warnings()
        self._ctx.__enter__()
        warnings.filterwarnings("ignore", message="All-NaN slice")
        warnings.filterwarnings("ignore", message="Mean of empty slice")
        return self

    def __exit__(self, *exc):
        return self._ctx.__exit__(*exc)


# --- Whittle -------------------------------------------------------------------


def _fill_gaps(grid: np.ndarray) -> np.ndarray:
    """Linearly interpolate NaNs along the last axis, holding the ends (np.interp)."""
    n_px, n = grid.shape
    valid = np.isfinite(grid)
    idx = np.arange(n)
    prev = np.maximum.accumulate(np.where(valid, idx, -1), axis=1)
    nxt = np.minimum.accumulate(np.where(valid, idx, n)[:, ::-1], axis=1)[:, ::-1]
    first = np.where(valid.any(axis=1), valid.argmax(axis=1), 0)
    last = n - 1 - valid[:, ::-1].argmax(axis=1)
    prev = np.where(prev < 0, first[:, None], prev)
    nxt = np.where(nxt >= n, last[:, None], nxt)
    rows = np.arange(n_px)[:, None]
    lo, hi = grid[rows, prev], grid[rows, nxt]
    span = np.where(nxt > prev, nxt - prev, 1)
    return lo + (hi - lo) * (idx - prev) / span


def _whittle_spectra(kappa: np.ndarray, logit: np.ndarray, log_two_sin: np.ndarray):
    phi = 1.0 / (1.0 + np.exp(-logit))
    return (1.0 - phi)[..., None] * np.exp(kappa[..., None] * log_two_sin) + phi[
        ..., None
    ]


def _whittle(resid_grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return per-row ``(kappa, phi, sigma2)`` maximizing the Whittle likelihood.

    The per-pixel cost of a candidate is a dot product of the periodogram
    with 1 / spectrum, so pixels that share candidates are scored with one
    matrix product. A coarse grid shared by all pixels comes first; then the
    pixels in each coarse cell share a finer grid around it, twice.
    """
    n = resid_grid.shape[1]
    centred = resid_grid - resid_grid.mean(axis=1, keepdims=True)
    pgram = np.abs(np.fft.rfft(centred, axis=1)) ** 2 / n
    freqs = np.fft.rfftfreq(n)
    # Skip f = 0, and the lowest bin, which detrending biases low.
    start = 2 if len(freqs) > 20 else 1
    pgram = pgram[:, start:]
    log_two_sin = np.log(2.0 * np.sin(np.pi * freqs[start:]))
    n_f = len(log_two_sin)

    def terms(kk: np.ndarray, zz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        spectra = _whittle_spectra(kk, zz, log_two_sin)  # (candidates, n_f)
        return np.log(spectra).sum(axis=1), (1.0 / spectra).T

    def score(pg: np.ndarray, log_sum: np.ndarray, inverse: np.ndarray) -> np.ndarray:
        return log_sum[None, :] + n_f * np.log(pg @ inverse / n_f)

    kk, zz = (a.ravel() for a in np.meshgrid(_KAPPA, _LOGIT, indexing="ij"))
    best = np.argmin(score(pgram, *terms(kk, zz)), axis=1)
    kappa, logit = kk[best], zz[best]
    for k_step, z_step, half in ((0.01, 0.1, 10), (0.001, 0.01, 5)):
        # Candidates live on an integer lattice of this level's steps, so the
        # grids of neighbouring cells share their spectra: each lattice point
        # is evaluated once.
        offsets = np.arange(-half, half + 1)
        dk, dz = (a.ravel() for a in np.meshgrid(offsets, offsets, indexing="ij"))
        cells, inverse = np.unique(
            np.stack(
                [np.round(kappa / k_step), np.round(logit / z_step)], axis=1
            ).astype(int),
            axis=0,
            return_inverse=True,
        )
        inverse = inverse.ravel()
        lattice = np.stack([cells[:, None, 0] + dk, cells[:, None, 1] + dz], axis=-1)
        points, where = np.unique(lattice.reshape(-1, 2), axis=0, return_inverse=True)
        where = where.reshape(len(cells), -1)
        pk = np.clip(points[:, 0] * k_step, -3.0, 0.01)
        pz = points[:, 1] * z_step
        log_sum, inv_spectra = terms(pk, pz)
        order = np.argsort(inverse, kind="stable")
        bounds = np.searchsorted(inverse[order], np.arange(len(cells) + 1))
        for c in range(len(cells)):
            members = order[bounds[c] : bounds[c + 1]]
            cols = where[c]
            pick = cols[
                np.argmin(
                    score(pgram[members], log_sum[cols], inv_spectra[:, cols]), axis=1
                )
            ]
            kappa[members], logit[members] = pk[pick], pz[pick]
    spectra = _whittle_spectra(kappa[:, None], logit[:, None], log_two_sin)[:, 0]
    sigma2 = (pgram / spectra).mean(axis=1)
    return np.clip(kappa, -3.0, 0.0), 1.0 / (1.0 + np.exp(-logit)), sigma2


def _velocity_variance_terms(design: np.ndarray, obs_idx: np.ndarray, column: int):
    """Velocity-variance factors of the OLS estimator for each grid kappa.

    var(v) = sigma2 [(1 - phi) q(kappa) + phi q_white], with q(kappa) =
    bᵀ C(kappa) b and b the velocity row of (AᵀA)⁻¹ Aᵀ, for the complete record.
    """
    b = np.linalg.pinv(design)[column]
    n_epochs = int(obs_idx[-1]) + 1
    kappas = np.round(np.arange(-3.0, 0.0001, 0.05), 4)
    q = np.array(
        [
            b @ _powerlaw_covariance(k, n_epochs)[np.ix_(obs_idx, obs_idx)] @ b
            for k in kappas
        ]
    )
    return kappas, q, float(b @ b)


def _whittle_block(
    data: np.ndarray,
    design: np.ndarray,
    obs_idx: np.ndarray,
    column: int,
    kappas: np.ndarray,
    q: np.ndarray,
    q_white: float,
    min_valid: int,
) -> tuple:
    shape = data.shape[:-1]
    y = data.reshape(-1, data.shape[-1]).astype(np.float64)
    coef, _, _, count = _solve(y, np.ones_like(y), design, min_valid)
    ok = np.isfinite(coef[:, column])
    out = [np.full(y.shape[0], np.nan) for _ in range(5)]
    if ok.any():
        resid = y[ok] - coef[ok] @ design.T
        grid = np.full((ok.sum(), int(obs_idx[-1]) + 1), np.nan)
        grid[:, obs_idx] = resid
        kappa, phi, sigma2 = _whittle(_fill_gaps(grid))
        q_k = np.interp(kappa, kappas, q)
        variance = sigma2 * ((1.0 - phi) * q_k + phi * q_white)
        out[0][ok] = coef[ok, column]
        out[1][ok] = np.sqrt(variance)
        out[2][ok] = kappa
        out[3][ok] = np.sqrt(sigma2 * (1.0 - phi))
        out[4][ok] = np.sqrt(sigma2 * phi)
    return tuple(o.reshape(shape) for o in out) + (count.reshape(shape),)


# --- public ----------------------------------------------------------------------


@recorded
def fit_velocity(
    obj: Union[xr.Dataset, xr.DataArray],
    method: Literal["ols", "whittle", "midas", "gls"] = "whittle",
    *,
    terms: Optional[Trajectory] = None,
    layer: str = "displacement",
    min_valid: int = 0,
    noise: str = "PLWN",
    groups: Union[str, xr.DataArray] = "coherence",
    n_groups: int = 5,
    n_samples: int = 16,
    allow_denoised: bool = False,
) -> xr.Dataset:
    """Estimate a velocity and its sigma for every pixel.

    Parameters
    ----------
    obj : xr.Dataset or xr.DataArray
        Chained displacement ``(time, y, x)``, or a Dataset holding it.
    method : {"whittle", "gls", "ols", "midas"}
        ``"whittle"`` (default): colored-noise sigmas, reliable from about
        200 epochs (6.5 years at 12 days); a warning is logged below.
        ``"gls"``: exact generalized least squares, with the noise shape
        estimated per group of pixels by exact maximum likelihood and the
        amplitude per pixel; works on short records too, costs more.
        ``"midas"``: robust to unmodelled steps and outliers, for records of
        several years. ``"ols"``: fast, but its sigma ignores correlated
        noise and is several times too small. For shorter records use
        :func:`disp_xr.fit_noise` on selected pixels. See the guide
        "Choosing a velocity method" in the documentation.
    terms : Trajectory, optional
        Deterministic model for ``"ols"`` and ``"whittle"`` (default offset,
        velocity and annual cycle). MIDAS needs none.
    layer : str
        Layer of a Dataset.
    min_valid : int
        Fewest valid epochs to fit a pixel (``"ols"``, ``"whittle"``, ``"gls"``).
    noise : {"PLWN", "FNWN", "RWFNWN", "WN", "auto"}
        Noise model of ``"gls"``; ``"auto"`` picks FNWN, PLWN or RWFNWN per
        group by the median BIC of its sampled pixels.
    groups : "coherence" or xr.DataArray
        ``"gls"`` groups: ``n_groups`` quantile bins of time-mean
        ``temporal_coherence`` (needs the Dataset), or an integer label map
        ``(y, x)``; ``-1`` leaves a pixel out.
    n_groups, n_samples : int
        Coherence bins, and pixels per group fitted to estimate its noise.
    allow_denoised : bool
        Fit data marked ``attrs["denoised"]``. Off by default: velocities and
        their sigmas come from the raw series (PRD S1).

    Returns
    -------
    xr.Dataset
        ``velocity`` and ``velocity_sigma`` (unit per year) for every method;
        ``kappa``, ``sigma_powerlaw``, ``sigma_white`` and ``n_valid`` for
        Whittle; ``outlier_fraction`` and ``velocity_scatter`` for MIDAS;
        ``noise_amplitude``, ``noise_group``, ``kappa``, ``white_fraction``
        and ``n_valid`` for GLS, with the group table in
        ``attrs["noise_groups"]`` (JSON). Lazy if the input is, except the
        sampled pixels GLS reads to estimate the noise.

    Examples
    --------
    >>> vel = dxr.fit_velocity(ds, "midas")  # doctest: +SKIP

    """
    da = obj[layer] if isinstance(obj, xr.Dataset) else obj
    if "time" not in da.dims:
        raise ValueError(f"fit_velocity needs a time dimension, not {da.dims}")
    check_not_denoised(obj, allow_denoised)
    units = da.attrs.get("units", "")
    rate = {"units": f"{units}/year"}
    terms = terms or Trajectory()
    if method == "ols":
        fit = fit_trajectory(da, terms, min_valid=min_valid)
        out = fit[["velocity", "velocity_sigma", "n_valid"]]
    elif method == "midas":
        da = time_contiguous(da)
        times = da["time"].values
        years = np.array([_decimal_year(t) for t in times])
        pairs = _midas_pairs(years)
        results = xr.apply_ufunc(
            _midas,
            da,
            input_core_dims=[["time"]],
            output_core_dims=[[], [], [], []],
            kwargs={"pairs": pairs, "years": years},
            dask="parallelized",
            output_dtypes=[np.float64] * 4,
        )
        midas_names = (
            "velocity",
            "velocity_sigma",
            "outlier_fraction",
            "velocity_scatter",
        )
        out = xr.Dataset(dict(zip(midas_names, results)))
        out["velocity_scatter"].attrs = rate
    elif method == "whittle":
        if da.sizes["time"] < WHITTLE_MIN_EPOCHS:
            logger.warning(
                f"Whittle on {da.sizes['time']} epochs: below about "
                f"{WHITTLE_MIN_EPOCHS} it underestimates |kappa| and the velocity "
                "sigma (benchmarks/whittle_bias.py)"
            )
        if "linear_trend" not in terms.names:
            raise ValueError(
                "The trajectory terms need a linear term (polynomial >= 1)"
            )
        da = time_contiguous(da)
        times = da["time"].values
        design = terms.design(times)
        obs_idx, _ = _sampling_grid(times, None)
        column = terms.names.index("linear_trend")
        kappas, q, q_white = _velocity_variance_terms(design, obs_idx, column)
        results = xr.apply_ufunc(
            _whittle_block,
            da,
            input_core_dims=[["time"]],
            output_core_dims=[[]] * 6,
            kwargs={
                "design": design,
                "obs_idx": obs_idx,
                "column": column,
                "kappas": kappas,
                "q": q,
                "q_white": q_white,
                "min_valid": min_valid,
            },
            dask="parallelized",
            output_dtypes=[np.float64] * 5 + [np.int64],
        )
        whittle_names = (
            "velocity",
            "velocity_sigma",
            "kappa",
            "sigma_powerlaw",
            "sigma_white",
            "n_valid",
        )
        out = xr.Dataset(dict(zip(whittle_names, results)))
        for name in ("sigma_powerlaw", "sigma_white"):
            out[name].attrs = {"units": units}
    elif method == "gls":
        from .gls import gls_velocity

        out = gls_velocity(
            obj,
            terms=terms,
            noise=noise,
            groups=groups,
            n_groups=n_groups,
            n_samples=n_samples,
            layer=layer,
            min_valid=min_valid,
        )
    else:
        raise ValueError(
            f"method must be 'ols', 'whittle', 'midas' or 'gls', not {method!r}"
        )
    out["velocity"].attrs = {**rate, "long_name": "velocity"}
    out["velocity_sigma"].attrs = rate
    out = out.drop_vars([c for c in out.coords if "time" in out[c].dims])
    if "spatial_ref" in da.coords:
        out = out.assign_coords(spatial_ref=da.coords["spatial_ref"])
    out.attrs = {**out.attrs, "method": method}
    if method != "midas":
        out.attrs["terms"] = repr(terms)
    return with_grid_mapping(out)
