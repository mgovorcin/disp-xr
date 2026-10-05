# SPDX-FileCopyrightText: 2025-2026 California Institute of Technology ("Caltech")
# SPDX-License-Identifier: Apache-2.0
# Adapted from geepers (https://github.com/opera-adt/geepers, commit d542178,
# src/geepers/trend.py), Copyright (c) 2025-2026 California Institute of
# Technology, Apache License 2.0. See LICENSES/geepers-Apache-2.0.txt,
# LICENSES/geepers-NOTICE.txt and THIRD_PARTY_NOTICES.md. Changes: the design
# matrix comes from disp_xr.Trajectory (or the caller); the sampling interval
# is inferred from the dates; results use disp-xr coefficient names.
"""Colored-noise trend estimation for one time series (maximum likelihood).

The deterministic model (a :class:`~disp_xr.timeseries.trajectory.Trajectory`
or any design matrix) is fitted by generalized least squares inside the
likelihood of a noise model, and the noise parameters are estimated by
(restricted) maximum likelihood. The resulting sigmas account for temporal
correlation, which ordinary least squares ignores: for GNSS and InSAR series
OLS velocity sigmas are often 5 to 10 times too small.

Noise models:

``"WN"``
    White noise only (equivalent to OLS).
``"FNWN"``
    Flicker (spectral index -1) plus white.
``"PLWN"``
    Power law with a free spectral index ``kappa`` plus white.
``"RWFNWN"``
    Random walk (-2), flicker and white.

Each likelihood evaluation factorizes an ``n x n`` covariance (O(n³)), so this
is for one series or a group of pixels; full frames use the vectorized
estimators of plan task T22. ``method="whittle"`` estimates the noise from the
periodogram of the OLS residuals (O(n log n)) and makes one exact GLS solve.

The method follows Hector / HectorP (Bos et al.), written from the papers:

- Bos, M. S., Fernandes, R. M. S., Williams, S. D. P., & Bastos, L. (2013).
  Fast error analysis of continuous GNSS observations with missing data.
  Journal of Geodesy, 87(4), 351-360. https://doi.org/10.1007/s00190-012-0605-0
- Williams, S. D. P. (2003). The effect of coloured noise on the uncertainties
  of rates estimated from geodetic time series. Journal of Geodesy, 76(9-10),
  483-494. https://doi.org/10.1007/s00190-002-0283-4
- Hosking, J. R. M. (1981). Fractional differencing. Biometrika, 68(1),
  165-176. https://doi.org/10.1093/biomet/68.1.165
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np
import pandas as pd
from scipy import linalg, optimize

from .trajectory import Trajectory

__all__ = ["NOISE_MODELS", "NoiseResult", "fit_noise"]

NoiseModel = Literal["WN", "FNWN", "PLWN", "RWFNWN"]
NOISE_MODELS = ("WN", "FNWN", "PLWN", "RWFNWN")

# Logit of the white-noise fraction that switches the white component off.
_LOGIT_NO_WHITE = -40.0
# Fixed spectral indices of the mixture components; "powerlaw" (PLWN) is the
# only component with a free kappa.
_FIXED_KAPPA = {"white": 0.0, "flicker": -1.0, "randomwalk": -2.0}
_MIXTURE_COMPONENTS = {
    "FNWN": ("flicker", "white"),
    "RWFNWN": ("randomwalk", "flicker", "white"),
}
_N_NOISE_PARAMS = {"PLWN": 3, "WN": 1, "FNWN": 2, "RWFNWN": 3}
_FAILED = 1e12


@dataclass
class NoiseResult:
    """A trajectory fitted under a colored-noise model.

    Amplitudes are in the unit of the series; rates per year.

    Examples
    --------
    >>> result = dxr.fit_noise(series, noise="PLWN")  # doctest: +SKIP
    >>> result.velocity, result.velocity_sigma, result.kappa  # doctest: +SKIP

    """

    model: str
    names: list[str]
    coefficients: np.ndarray
    sigma: np.ndarray
    kappa: float
    sigma_powerlaw: float = np.nan
    sigma_white: float = np.nan
    sigma_flicker: float = np.nan
    sigma_randomwalk: float = np.nan
    log_likelihood: float = np.nan
    aic: float = np.nan
    bic: float = np.nan
    sampling_days: float = np.nan
    residuals: np.ndarray = field(default_factory=lambda: np.array([]))
    fitted: np.ndarray = field(default_factory=lambda: np.array([]))

    @property
    def parameters(self) -> dict[str, tuple[float, float]]:
        """``name -> (value, sigma)`` for every coefficient."""
        return {
            n: (float(c), float(s))
            for n, c, s in zip(self.names, self.coefficients, self.sigma)
        }

    @property
    def velocity(self) -> float:
        """Linear rate per year (NaN without a ``linear_trend`` term)."""
        return self.parameters.get("linear_trend", (np.nan, np.nan))[0]

    @property
    def velocity_sigma(self) -> float:
        """Colored-noise sigma of the velocity."""
        return self.parameters.get("linear_trend", (np.nan, np.nan))[1]


def _fractional_diff_coeffs(d: float, n: int) -> np.ndarray:
    """Return the coefficients psi_i of (1 - B)^-d (Hosking, 1981).

    psi_0 = 1 and psi_i = psi_{i-1} (d + i - 1) / i.
    """
    psi = np.empty(n)
    psi[0] = 1.0
    if n > 1:
        i = np.arange(1, n)
        psi[1:] = np.cumprod((d + i - 1.0) / i)
    return psi


def _powerlaw_covariance(kappa: float, n: int) -> np.ndarray:
    """Return the unit-amplitude covariance of power-law noise on a regular grid.

    Non-stationary form C = U Uᵀ, with U the lower-triangular Toeplitz matrix
    of fractional-difference coefficients (Bos et al., 2013, section 2), built
    in O(n²) with per-diagonal cumulative sums:
    C[j + l, j] = sum_{m=0..j} psi_{m+l} psi_m.
    """
    psi = _fractional_diff_coeffs(-kappa / 2.0, n)
    cov = np.empty((n, n))
    for lag in range(n):
        diag = np.cumsum(psi[lag:] * psi[: n - lag])
        idx = np.arange(n - lag)
        cov[idx + lag, idx] = diag
        cov[idx, idx + lag] = diag
    return cov


def _softmax_weights(z: np.ndarray) -> np.ndarray:
    """Map K-1 free logits to K positive weights summing to 1 (last logit 0)."""
    logits = np.append(np.asarray(z, dtype=float), 0.0)
    logits -= logits.max()
    w = np.exp(logits)
    return w / w.sum()


def _nll_core(cov: np.ndarray, a: np.ndarray, y: np.ndarray, rmle: bool) -> float:
    """Return the profiled negative log-likelihood of GLS under ``cov``.

    The overall variance is profiled out (Bos et al., 2013, eq. 2 to 5).
    Returns a large value on numerical failure, so optimizers move away.
    """
    n = len(y)
    try:
        cho = linalg.cho_factor(cov, lower=True, check_finite=False)
    except linalg.LinAlgError:
        return _FAILED
    logdet = 2.0 * np.sum(np.log(np.diag(cho[0])))
    ci_a = linalg.cho_solve(cho, a, check_finite=False)
    ci_y = linalg.cho_solve(cho, y, check_finite=False)
    normal = a.T @ ci_a
    try:
        beta = linalg.solve(normal, a.T @ ci_y, assume_a="pos")
    except linalg.LinAlgError:
        return _FAILED
    r = y - a @ beta
    rcr = float(r @ linalg.cho_solve(cho, r, check_finite=False))
    if rcr <= 0:
        return _FAILED
    n_p = a.shape[1] if rmle else 0
    sigma2 = rcr / (n - n_p)
    nll = 0.5 * (logdet + (n - n_p) * np.log(sigma2) + rcr / sigma2)
    if rmle:
        sign, logdet_n = np.linalg.slogdet(normal)
        if sign <= 0:
            return _FAILED
        nll += 0.5 * logdet_n
    return nll


class _PowerLaw:
    """Power-law plus white covariance at the observed epochs, cached by kappa."""

    def __init__(self, obs_idx: np.ndarray, n_epochs: int):
        self.obs_idx, self.n_epochs = obs_idx, n_epochs
        self._kappa: Optional[float] = None
        self._cov: Optional[np.ndarray] = None

    def unit(self, kappa: float) -> np.ndarray:
        key = round(float(kappa), 10)
        if key != self._kappa:
            full = _powerlaw_covariance(kappa, self.n_epochs)
            self._cov = full[np.ix_(self.obs_idx, self.obs_idx)]
            self._kappa = key
        assert self._cov is not None
        return self._cov

    def covariance(self, kappa: float, phi: float) -> np.ndarray:
        return (1.0 - phi) * self.unit(kappa) + phi * np.eye(len(self.obs_idx))

    def nll(
        self, params: np.ndarray, a: np.ndarray, y: np.ndarray, rmle: bool
    ) -> float:
        kappa, z = params
        if not (-3.0 <= kappa <= 0.01):
            return _FAILED
        phi = 1.0 / (1.0 + np.exp(-z))
        return _nll_core(self.covariance(kappa, phi), a, y, rmle)


def _mixture_nll(
    z: np.ndarray, covs: list[np.ndarray], a: np.ndarray, y: np.ndarray, rmle: bool
) -> float:
    weights = _softmax_weights(z)
    cov = sum(w * c for w, c in zip(weights, covs))
    return _nll_core(cov, a, y, rmle)


def _minimize(fun, starts: list, args: tuple) -> optimize.OptimizeResult:
    """Polish the two best grid starts with Nelder-Mead; return the best."""
    ranked = sorted(starts, key=lambda x0: fun(np.asarray(x0, dtype=float), *args))
    best = None
    for x0 in ranked[:2]:
        res = optimize.minimize(
            fun,
            x0=np.asarray(x0, dtype=float),
            args=args,
            method="Nelder-Mead",
            options={"xatol": 1e-4, "fatol": 1e-6, "maxiter": 300},
        )
        if best is None or res.fun < best.fun:
            best = res
    assert best is not None
    return best


def _whittle_powerlaw(resid_grid: np.ndarray) -> tuple[float, float]:
    """Estimate ``(kappa, phi)`` from the periodogram of gridded residuals.

    Whittle's frequency-domain approximation to the Gaussian likelihood,
    O(n) per evaluation on the FFT periodogram.
    """
    n = len(resid_grid)
    pgram = np.abs(np.fft.rfft(resid_grid - resid_grid.mean())) ** 2 / n
    freqs = np.fft.rfftfreq(n)
    # Skip f = 0, and the lowest bin, which detrending biases low.
    start = 2 if len(freqs) > 20 else 1
    pgram = pgram[start:]
    log_two_sin = np.log(2.0 * np.sin(np.pi * freqs[start:]))
    n_freqs = len(pgram)

    def nll(params: np.ndarray) -> float:
        kappa, z = params
        if not (-3.0 <= kappa <= 0.01):
            return _FAILED
        phi = 1.0 / (1.0 + np.exp(-z))
        g = (1.0 - phi) * np.exp(kappa * log_two_sin) + phi
        sigma2 = float(np.mean(pgram / g))
        if not np.isfinite(sigma2) or sigma2 <= 0:
            return _FAILED
        return float(np.sum(np.log(g))) + n_freqs * np.log(sigma2)

    best = None
    for k0 in (-0.4, -1.0, -1.6):
        for z0 in (-2.0, 0.0, 2.0):
            res = optimize.minimize(
                nll,
                x0=np.array([k0, z0]),
                method="Nelder-Mead",
                options={"xatol": 1e-3, "fatol": 1e-8, "maxiter": 500},
            )
            if best is None or res.fun < best.fun:
                best = res
    assert best is not None
    return float(np.clip(best.x[0], -3.0, 0.0)), float(1.0 / (1.0 + np.exp(-best.x[1])))


def _sampling_grid(
    times: np.ndarray, sampling_days: Optional[float]
) -> tuple[np.ndarray, float]:
    days = (times - times[0]) / np.timedelta64(1, "D")
    if sampling_days is None:
        steps = np.diff(days)
        sampling_days = float(np.round(steps[steps > 0].min()))
    obs_idx = np.round(days / sampling_days).astype(int)
    if len(np.unique(obs_idx)) != len(obs_idx):
        raise ValueError(
            f"Two epochs fall in one sampling interval of {sampling_days:g} days; "
            "pass a smaller sampling_days"
        )
    return obs_idx, sampling_days


def fit_noise(
    times: Sequence,
    y: Sequence[float],
    design: Optional[np.ndarray] = None,
    *,
    terms: Optional[Trajectory] = None,
    names: Optional[Sequence[str]] = None,
    model: NoiseModel = "PLWN",
    rmle: bool = True,
    method: Literal["exact", "whittle"] = "exact",
    sampling_days: Optional[float] = None,
) -> NoiseResult:
    """Fit a trajectory to one series under a colored-noise model.

    Parameters
    ----------
    times : sequence of datetime-like
        Epochs; need not be regular. NaN values in ``y`` are dropped.
    y : sequence of float
        The series (any unit; outputs use the same unit).
    design : np.ndarray, optional
        ``(time, coefficient)`` design matrix. By default built from
        ``terms``.
    terms : Trajectory, optional
        Model terms when ``design`` is not given; default offset, velocity and
        an annual cycle.
    names : sequence of str, optional
        Coefficient names for a given ``design``.
    model : {"PLWN", "FNWN", "RWFNWN", "WN"}
        Noise model; see the module docstring.
    rmle : bool
        Restricted maximum likelihood, which corrects the noise estimate for
        the degrees of freedom used by the deterministic model.
    method : {"exact", "whittle"}
        Time-domain (R)MLE (O(n³) per evaluation), or the noise from the
        Whittle likelihood of the OLS residuals (O(n log n)) followed by one
        exact GLS solve. Whittle supports ``PLWN`` and ``WN``.
    sampling_days : float, optional
        Nominal sampling interval of the regular grid the noise lives on.
        Default: the smallest spacing between epochs, in whole days.

    Returns
    -------
    NoiseResult

    Raises
    ------
    ValueError
        If fewer than 10 epochs are valid, the model is rank deficient, or two
        epochs fall in one sampling interval.

    Examples
    --------
    >>> series = ds.displacement[:, 10, 20]  # doctest: +SKIP
    >>> res = fit_noise(series.time.values, series.values)  # doctest: +SKIP
    >>> res.velocity, res.velocity_sigma, res.kappa  # doctest: +SKIP

    """
    if model not in NOISE_MODELS:
        raise ValueError(f"model must be one of {NOISE_MODELS}, not {model!r}")
    if method == "whittle" and model not in ("PLWN", "WN"):
        raise ValueError("method='whittle' supports the PLWN and WN models")
    times_all = np.asarray(pd.to_datetime(np.asarray(times)), dtype="datetime64[ns]")
    y_all = np.asarray(y, dtype=float)
    if design is None:
        terms = terms or Trajectory()
        design = terms.design(times_all)
        names = terms.names
    design_all = np.asarray(design, dtype=float)
    names = (
        list(names)
        if names is not None
        else [f"c{i}" for i in range(design_all.shape[1])]
    )
    good = np.isfinite(y_all)
    t, y_obs, a = times_all[good], y_all[good], design_all[good]
    if len(y_obs) < 10:
        raise ValueError(f"Need at least 10 valid epochs, got {len(y_obs)}")
    if np.linalg.matrix_rank(a) < a.shape[1]:
        raise ValueError(
            "The trajectory model is rank deficient at the valid epochs: a step or "
            "postseismic onset outside the record gives an empty column"
        )
    obs_idx, sampling_days = _sampling_grid(t, sampling_days)
    n_epochs = int(obs_idx[-1]) + 1
    # Remove the mean to keep the likelihood well conditioned.
    constant = names.index("constant") if "constant" in names else None
    y_mean = float(np.mean(y_obs)) if constant is not None else 0.0
    y0 = y_obs - y_mean
    n = len(y0)
    powerlaw = _PowerLaw(obs_idx, n_epochs)
    weights: Optional[np.ndarray] = None

    if model in _MIXTURE_COMPONENTS:
        components = _MIXTURE_COMPONENTS[model]
        covs = [
            np.eye(n) if c == "white" else powerlaw.unit(_FIXED_KAPPA[c]).copy()
            for c in components
        ]
        n_free = len(components) - 1
        grid = (
            np.stack(np.meshgrid(*([[-2.0, 0.0, 2.0]] * n_free))).reshape(n_free, -1).T
        )
        best = _minimize(_mixture_nll, list(grid), (covs, a, y0, rmle))
        weights = _softmax_weights(best.x)
        nll = float(best.fun)
        kappa, phi = (-1.0 if model == "FNWN" else np.nan), float(weights[-1])
        cov = sum(w * c for w, c in zip(weights, covs))
    elif model == "WN":
        kappa, phi = 0.0, 1.0 - 1e-12
        nll = powerlaw.nll(np.array([0.0, 30.0]), a, y0, rmle)
        cov = np.eye(n)
    elif method == "whittle":
        beta_ols, *_ = np.linalg.lstsq(a, y0, rcond=None)
        grid_resid = np.interp(np.arange(n_epochs), obs_idx, y0 - a @ beta_ols)
        kappa, phi = _whittle_powerlaw(grid_resid)
        phi_c = float(np.clip(phi, 1e-12, 1.0 - 1e-12))
        nll = powerlaw.nll(np.array([kappa, np.log(phi_c / (1 - phi_c))]), a, y0, rmle)
        cov = powerlaw.covariance(kappa, phi)
    else:
        starts = [[k0, z0] for k0 in (-0.4, -1.0, -1.6) for z0 in (-2.0, 0.0, 2.0)]
        best = _minimize(powerlaw.nll, starts, (a, y0, rmle))
        kappa = float(best.x[0])
        phi = float(1.0 / (1.0 + np.exp(-best.x[1])))
        nll = float(best.fun)
        cov = powerlaw.covariance(kappa, phi)

    # Final GLS solve at the optimum.
    cho = linalg.cho_factor(cov, lower=True, check_finite=False)
    normal = a.T @ linalg.cho_solve(cho, a, check_finite=False)
    beta = linalg.solve(normal, a.T @ linalg.cho_solve(cho, y0), assume_a="pos")
    r = y0 - a @ beta
    n_p = a.shape[1] if rmle else 0
    sigma2 = float(r @ linalg.cho_solve(cho, r, check_finite=False)) / (n - n_p)
    beta_sigma = np.sqrt(np.diag(sigma2 * linalg.inv(normal)))
    if constant is not None:
        beta[constant] += y_mean

    amplitudes = {"sigma_powerlaw": np.nan, "sigma_white": np.nan}
    if weights is not None:
        for c, w in zip(_MIXTURE_COMPONENTS[model], weights):
            amplitudes[f"sigma_{c}"] = float(np.sqrt(sigma2 * w))
    else:
        amplitudes["sigma_powerlaw"] = float(np.sqrt(sigma2 * (1.0 - phi)))
        amplitudes["sigma_white"] = float(np.sqrt(sigma2 * phi))
    k_total = a.shape[1] + _N_NOISE_PARAMS[model]
    log_l = -nll - 0.5 * n * np.log(2.0 * np.pi)
    fitted = np.full(len(y_all), np.nan)
    fitted[good] = a @ beta  # the constant already carries the mean back
    residuals = np.full(len(y_all), np.nan)
    residuals[good] = y_obs - fitted[good]
    return NoiseResult(
        model=model,
        names=names,
        coefficients=beta,
        sigma=beta_sigma,
        kappa=float(kappa),
        log_likelihood=float(log_l),
        aic=float(2 * k_total - 2 * log_l),
        bic=float(k_total * np.log(n) - 2 * log_l),
        sampling_days=sampling_days,
        residuals=residuals,
        fitted=fitted,
        **amplitudes,
    )
