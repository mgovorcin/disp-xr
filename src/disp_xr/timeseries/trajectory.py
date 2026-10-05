"""Trajectory models fitted to every pixel by ordinary least squares.

A trajectory is a sum of terms in time: a polynomial (offset, velocity,
acceleration), annual and semiannual sine/cosine pairs, Heaviside steps, and
postseismic ``log1p(dt / tau)`` and ``1 - exp(-dt / tau)`` terms with a fixed
``tau``. The terms and their names match geepers ``trend._design_matrix``, and
the coefficient names encode the model, so :func:`get_predicted` can rebuild
it from a fit.

Every pixel is fitted with its own valid epochs. Pixels that share a
valid-epoch pattern (most of a frame: all epochs valid) are solved together,
one factorization for many right-hand sides; the rest, and weighted fits, use
per-pixel normal equations ``N = Aᵀ W A`` solved in one batched call. See
``benchmarks/README.md``. Sigmas assume white noise:
the diagonal of ``N⁻¹`` scaled by the residual variance (plan tasks T21 to
T23 replace this with colored-noise estimates).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Optional, Union

import numpy as np
import pandas as pd
import xarray as xr

from .._chunks import time_contiguous
from .._history import recorded
from ._design_matrix import create_design_matrix, get_coefficient_names

__all__ = ["Trajectory", "fit_trajectory", "predict_trajectory"]


@dataclass(frozen=True)
class Trajectory:
    """Terms of a trajectory model.

    Parameters
    ----------
    polynomial : int
        Degree: 0 offset only, 1 adds velocity, 2 adds acceleration (the
        quadratic coefficient is ``a / 2``-normalized, so it is the
        acceleration).
    periods : tuple of float
        Periods in years of sine/cosine pairs: ``1.0`` annual, ``0.5``
        semiannual.
    steps : tuple of str
        Dates of Heaviside steps (any pandas date string).
    postseismic_log, postseismic_exp : tuple of (str, float)
        ``(onset date, tau in days)`` of ``log1p(dt / tau)`` and
        ``1 - exp(-dt / tau)`` terms.

    Examples
    --------
    >>> Trajectory(steps=("2023-02-06",)).names
    ['constant', 'linear_trend', 'annual_sin', 'annual_cos', 'step_2023-02-06']

    """

    polynomial: int = 1
    periods: tuple[float, ...] = (1.0,)
    steps: tuple[str, ...] = ()
    postseismic_log: tuple[tuple[str, float], ...] = field(default=())
    postseismic_exp: tuple[tuple[str, float], ...] = field(default=())

    def _kwargs(self) -> dict:
        def by_onset(terms: tuple[tuple[str, float], ...]) -> dict[str, list]:
            out: dict[str, list] = {}
            for onset, tau in terms:
                out.setdefault(onset, []).append(int(tau))
            return out

        return {
            "polynomial_degree": self.polynomial,
            "periods": list(self.periods),
            "step_dates": list(self.steps),
            "exponential_terms": by_onset(self.postseismic_exp),
            "logarithmic_terms": by_onset(self.postseismic_log),
        }

    @property
    def names(self) -> list[str]:
        """Coefficient names, in column order of the design matrix."""
        return get_coefficient_names(**self._kwargs())

    def design(self, times: np.ndarray) -> np.ndarray:
        """Return the ``(time, coefficient)`` design matrix, time 0 at ``times[0]``."""
        return create_design_matrix(np.asarray(times), **self._kwargs())


def _solve(
    data: np.ndarray, weights: np.ndarray, design: np.ndarray, min_valid: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit ``data`` (..., time) per pixel; return coefficients, sigmas, rms, count."""
    from threadpoolctl import threadpool_limits

    from ..config import config

    # Dask runs one task per core already; BLAS threads inside each task would
    # oversubscribe the machine.
    with threadpool_limits(limits=config.get("threads") or 1):
        return _solve_block(data, weights, design, min_valid)


# A NaN pattern shared by at least this many pixels is solved as one
# least-squares problem with many right-hand sides; rarer patterns go to the
# batched per-pixel normal equations.
GROUP_MIN_PIXELS = 64


def _solve_block(
    data: np.ndarray, weights: np.ndarray, design: np.ndarray, min_valid: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    shape = data.shape[:-1]
    n_time, n_par = design.shape
    y = data.reshape(-1, n_time).astype(np.float64)
    w = weights.reshape(-1, n_time).astype(np.float64) * np.isfinite(y)
    y = np.where(w > 0, y, 0.0)
    count = (w > 0).sum(axis=1)
    coef = np.full((y.shape[0], n_par), np.nan)
    sigma = np.full((y.shape[0], n_par), np.nan)
    rms = np.full(y.shape[0], np.nan)
    enough = count >= max(min_valid, n_par + 1)

    # Uniform weights (the common case) let pixels that share a valid-epoch
    # pattern share one factorization of the design matrix.
    uniform = np.all((w == 0) | (w == w.max(axis=1, keepdims=True)), axis=1)
    keys, inverse, sizes = np.unique(
        np.packbits(w > 0, axis=1), axis=0, return_inverse=True, return_counts=True
    )
    inverse = inverse.ravel()
    grouped = np.zeros(y.shape[0], dtype=bool)
    # Pixels sorted by pattern, so each group is a slice: scanning all pixels
    # once per pattern was quadratic (thousands of patterns per block).
    order = np.argsort(inverse, kind="stable")
    ends = np.cumsum(sizes)
    eligible = enough & uniform
    for k in np.flatnonzero(sizes >= GROUP_MIN_PIXELS):
        members = order[ends[k] - sizes[k] : ends[k]]
        members = members[eligible[members]]
        if members.size < GROUP_MIN_PIXELS:
            continue
        rows = w[members[0]] > 0
        a = design[rows]
        normal = a.T @ a
        eig = np.linalg.eigvalsh(normal)
        grouped[members] = True
        if eig[0] <= 1e-10 * eig[-1]:
            continue
        inv = np.linalg.inv(normal)
        b = y[members][:, rows]
        c = b @ a @ inv  # (pixels, coefficients)
        residual = b - c @ a.T
        rss = np.einsum("nt,nt->n", residual, residual)
        dof = rows.sum() - n_par
        coef[members] = c
        sigma[members] = np.sqrt(np.diag(inv)[None, :] * (rss / dof)[:, None])
        rms[members] = np.sqrt(rss / rows.sum())

    rest = enough & ~grouped
    if rest.any():
        c, s_, r = _batched(y[rest], w[rest], count[rest], design)
        coef[rest], sigma[rest], rms[rest] = c, s_, r
    return (
        coef.reshape(*shape, n_par),
        sigma.reshape(*shape, n_par),
        rms.reshape(shape),
        count.reshape(shape),
    )


def _batched(
    y: np.ndarray, w: np.ndarray, count: np.ndarray, design: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve per-pixel normal equations ``N = Aᵀ W A`` in one batched call."""
    n_time, n_par = design.shape
    outer = np.einsum("tp,tq->tpq", design, design).reshape(n_time, n_par * n_par)
    normal = (w @ outer).reshape(-1, n_par, n_par)
    rhs = (w * y) @ design
    # N is symmetric positive semi-definite; a tiny smallest eigenvalue means
    # the valid epochs cannot separate the terms (a step with no epochs after
    # it, for example).
    eigenvalues = np.linalg.eigvalsh(normal)
    ok = eigenvalues[:, 0] > 1e-10 * eigenvalues[:, -1]
    coef = np.full((y.shape[0], n_par), np.nan)
    sigma = np.full((y.shape[0], n_par), np.nan)
    rms = np.full(y.shape[0], np.nan)
    if ok.any():
        inverse = np.linalg.inv(normal[ok])
        coef[ok] = np.einsum("npq,nq->np", inverse, rhs[ok])
        residual = (y[ok] - coef[ok] @ design.T) * (w[ok] > 0)
        rss = np.einsum("nt,nt,nt->n", w[ok], residual, residual)
        variance = rss / (count[ok] - n_par)
        sigma[ok] = np.sqrt(np.diagonal(inverse, axis1=1, axis2=2) * variance[:, None])
        rms[ok] = np.sqrt(rss / w[ok].sum(axis=1))
    return coef, sigma, rms


def check_not_denoised(obj: Union[xr.Dataset, xr.DataArray], allow: bool) -> None:
    """Refuse data marked ``attrs["denoised"]`` unless ``allow`` (PRD S1)."""
    denoised = obj.attrs.get("denoised")
    if denoised and not allow:
        raise ValueError(
            f"The data are denoised ({denoised}); velocities and sigmas must come "
            "from the raw series. Pass allow_denoised=True to fit anyway."
        )


@recorded
def fit_trajectory(
    obj: Union[xr.Dataset, xr.DataArray],
    terms: Optional[Trajectory] = None,
    *,
    layer: str = "displacement",
    weights: Optional[xr.DataArray] = None,
    min_valid: int = 0,
    allow_denoised: bool = False,
) -> xr.Dataset:
    """Fit a trajectory model to every pixel by ordinary least squares.

    Parameters
    ----------
    obj : xr.Dataset or xr.DataArray
        Chained displacement ``(time, y, x)`` (see :func:`disp_xr.stack_disp`),
        or a Dataset holding it as ``layer``. NaN epochs are skipped per
        pixel.
    terms : Trajectory, optional
        Model terms; default offset, velocity and an annual cycle.
    layer : str
        Layer of a Dataset to fit.
    weights : xr.DataArray, optional
        Per-epoch weights ``(time, y, x)`` or ``(time,)``, proportional to
        ``1 / variance`` (for example from quality layers). Zero drops an
        epoch.
    min_valid : int
        Fewest valid epochs to fit a pixel; at least one more than the number
        of coefficients is always required.
    allow_denoised : bool
        Fit data marked ``attrs["denoised"]``. Off by default: fits use the
        raw series (PRD S1).

    Returns
    -------
    xr.Dataset
        ``coefficients`` and ``sigma`` ``(coefficient, y, x)`` in the input
        unit (per year for rates), ``velocity`` and ``velocity_sigma`` when the
        model has a linear term, ``<period>_amplitude`` and
        ``<period>_amplitude_sigma`` for each sine/cosine pair, ``rms`` of the
        residuals and ``n_valid`` epochs. Lazy if the input is.

    Examples
    --------
    >>> terms = Trajectory(steps=("2023-02-06",))
    >>> fit = dxr.fit_trajectory(ds, terms)  # doctest: +SKIP
    >>> fit.velocity.disp.plot_map()  # doctest: +SKIP

    """
    terms = terms or Trajectory()
    check_not_denoised(obj, allow_denoised)
    da = obj[layer] if isinstance(obj, xr.Dataset) else obj
    if "time" not in da.dims:
        raise ValueError(f"fit_trajectory needs a time dimension, not {da.dims}")
    da = time_contiguous(da)
    design = terms.design(da["time"].values)
    names = terms.names
    if weights is None:
        weights = xr.ones_like(da["time"], dtype=np.float32)
    weights = weights.broadcast_like(da)
    if da.chunks is not None:
        weights = weights.chunk(da.chunksizes)
    coef, sigma, rms, count = xr.apply_ufunc(
        _solve,
        da,
        weights,
        input_core_dims=[["time"], ["time"]],
        output_core_dims=[["coefficient"], ["coefficient"], [], []],
        kwargs={"design": design, "min_valid": min_valid},
        dask="parallelized",
        output_dtypes=[np.float64, np.float64, np.float64, np.int64],
        dask_gufunc_kwargs={"output_sizes": {"coefficient": len(names)}},
    )
    units = da.attrs.get("units", "")
    out = xr.Dataset(
        {
            "coefficients": coef.transpose("coefficient", ...),
            "sigma": sigma.transpose("coefficient", ...),
            "rms": rms.assign_attrs(units=units, long_name="residual RMS"),
            "n_valid": count.assign_attrs(long_name="epochs fitted"),
        },
        coords={"coefficient": names},
    )
    out = out.drop_vars([c for c in out.coords if "time" in out[c].dims])
    if "linear_trend" in names:
        out["velocity"] = out.coefficients.sel(coefficient="linear_trend", drop=True)
        out["velocity_sigma"] = out.sigma.sel(coefficient="linear_trend", drop=True)
        out["velocity"].attrs = {"units": f"{units}/year", "long_name": "velocity"}
        out["velocity_sigma"].attrs = {"units": f"{units}/year"}
    for name in names:
        if name.endswith("_sin") and f"{name[:-4]}_cos" in names:
            label = name[:-4]
            s = out.coefficients.sel(coefficient=name, drop=True)
            c = out.coefficients.sel(coefficient=f"{label}_cos", drop=True)
            ss = out.sigma.sel(coefficient=name, drop=True)
            cs = out.sigma.sel(coefficient=f"{label}_cos", drop=True)
            amplitude = np.hypot(s, c)
            out[f"{label}_amplitude"] = amplitude.assign_attrs(units=units)
            # First-order propagation of the sin and cos sigmas.
            out[f"{label}_amplitude_sigma"] = (
                np.sqrt((s * ss) ** 2 + (c * cs) ** 2) / amplitude
            ).assign_attrs(units=units)
    if "spatial_ref" in da.coords:
        out = out.assign_coords(spatial_ref=da.coords["spatial_ref"])
    out.attrs = {
        "model": "trajectory, white-noise OLS",
        "reference_time": str(pd.Timestamp(da["time"].values[0])),
        "terms": repr(terms),
    }
    return out


def predict_trajectory(
    fit: xr.Dataset, times: Union[np.ndarray, Sequence], dtype: str = "float32"
) -> xr.DataArray:
    """Evaluate a fitted trajectory at ``times`` (lazily if the fit is).

    The model is rebuilt from the coefficient names and the reference time in
    ``fit.attrs``, so a fit read back from a store can be evaluated. Computed
    in ``dtype`` and never stored unless you write it.
    """
    from ._design_matrix import infer_parameters_from_names

    times = np.asarray(times, dtype="datetime64[ns]")
    reference = np.datetime64(pd.Timestamp(fit.attrs["reference_time"]), "ns")
    all_times = np.concatenate([[reference], times])
    params = infer_parameters_from_names(list(fit["coefficient"].values))
    design = create_design_matrix(all_times, **params)[1:]
    columns = xr.DataArray(
        design.astype(dtype),
        dims=("time", "coefficient"),
        coords={"time": times, "coefficient": fit["coefficient"].values},
    )
    return xr.dot(columns, fit.coefficients.astype(dtype), dim="coefficient")
