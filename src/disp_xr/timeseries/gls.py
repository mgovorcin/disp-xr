"""Full-frame velocity by generalized least squares with colored noise (T23).

The noise covariance of a pixel depends on the epochs, which of them are
valid, and the noise parameters, not on the pixel's values. So the
covariance *shape* (the spectral index and the mix of noise components) is
estimated per group of pixels, by exact maximum likelihood
(:func:`~disp_xr.timeseries.noise.fit_noise`) on a sample of each group's
pixels, and every pixel is then fitted by exact GLS under its group's shape,
with its own noise amplitude profiled from its residuals. Each pixel gets a
colored-noise velocity sigma without an n x n factorization per pixel.

Groups are quantile bins of time-averaged ``temporal_coherence`` by default,
or any integer label map (for example quadtree cells, plan task T28).

Missing epochs. Each group's covariance over the complete record is inverted
once, ``Q = C⁻¹``. For a pixel missing the epochs ``m``, the inverse
covariance of its valid epochs is the Schur complement
``P = Q - Q[:, m] Q[m, m]⁻¹ Q[m, :]``, which is zero on the missing rows and
columns. The normal equations and residual quadratic forms follow from
``Q``, ``Q A`` and a ``k x k`` inverse per pixel (``k = len(m)``), batched
over pixels with the same ``k``.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import pandas as pd
import xarray as xr

from .._chunks import time_contiguous
from .noise import NOISE_MODELS, _powerlaw_covariance, _sampling_grid, fit_noise
from .trajectory import Trajectory

logger = logging.getLogger(__name__)

__all__ = ["NoiseShape", "estimate_group_noise", "gls_velocity"]

# Spectral index of each named noise component.
_COMPONENT_KAPPA = {"white": 0.0, "flicker": -1.0, "randomwalk": -2.0}
_SELECTABLE = ("FNWN", "PLWN", "RWFNWN")
# Pixels solved together in _gls_block.
_BATCH = 8192


@dataclass(frozen=True)
class NoiseShape:
    """Covariance shape of a group: components ``(kappa, weight)``, weights sum to 1.

    The covariance of a pixel is ``sigma2 * sum(w * C(kappa))`` over the
    components, with ``C(0)`` the identity and ``sigma2`` the pixel's own
    amplitude.
    """

    model: str
    components: tuple[tuple[float, float], ...]
    n_samples: int = 0
    bic: float = np.nan

    @property
    def kappa(self) -> float:
        """Spectral index of the power-law component (NaN for mixtures)."""
        if self.model == "PLWN":
            return self.components[0][0]
        if self.model == "FNWN":
            return -1.0
        return 0.0 if self.model == "WN" else np.nan

    @property
    def white_fraction(self) -> float:
        """Share of the variance in the white component."""
        return float(sum(w for k, w in self.components if k == 0.0))

    def covariance(self, obs_idx: np.ndarray) -> np.ndarray:
        """Return the unit-amplitude covariance at the grid indices ``obs_idx``."""
        n_grid = int(obs_idx[-1]) + 1
        cov = np.zeros((len(obs_idx), len(obs_idx)))
        for kappa, weight in self.components:
            if kappa == 0.0:
                cov += weight * np.eye(len(obs_idx))
            else:
                full = _powerlaw_covariance(kappa, n_grid)
                cov += weight * full[np.ix_(obs_idx, obs_idx)]
        return cov


def _shape_of(res, model: str) -> tuple[tuple[float, float], ...]:
    """Covariance components of a fit_noise result (weights from amplitudes)."""
    if model == "WN":
        return ((0.0, 1.0),)
    if model == "PLWN":
        var = np.array([res.sigma_powerlaw, res.sigma_white]) ** 2
        return ((res.kappa, var[0] / var.sum()), (0.0, var[1] / var.sum()))
    names = (
        ("flicker", "white") if model == "FNWN" else ("randomwalk", "flicker", "white")
    )
    var = np.array([getattr(res, f"sigma_{n}") for n in names]) ** 2
    return tuple((_COMPONENT_KAPPA[n], v / var.sum()) for n, v in zip(names, var))


def _fit_sample(args: tuple) -> tuple:
    """Fit one sampled pixel with every candidate model (runs in a worker)."""
    from threadpoolctl import threadpool_limits

    times, values, terms, models = args
    out: list[tuple] = []
    # Many fits run side by side in threads; BLAS threads inside each would
    # oversubscribe the machine.
    with threadpool_limits(limits=1):
        for model in models:
            try:
                res = fit_noise(times, values, terms=terms, model=model)
                out.append((model, _shape_of(res, model), res.bic))
            except (ValueError, np.linalg.LinAlgError):
                out.append((model, None, np.nan))
    return tuple(out)


def _median_shape(shapes: list) -> tuple[tuple[float, float], ...]:
    """Median of the components across samples, weights renormalized."""
    arr = np.array([[c for pair in s for c in pair] for s in shapes])  # (samples, 2K)
    med = np.median(arr, axis=0).reshape(-1, 2)
    weights = med[:, 1] / med[:, 1].sum()
    return tuple((float(k), float(w)) for (k, _), w in zip(med, weights))


def estimate_group_noise(
    da: xr.DataArray,
    labels: xr.DataArray,
    *,
    terms: Trajectory,
    noise: str = "PLWN",
    n_samples: int = 16,
    seed: int = 0,
    max_workers: Optional[int] = None,
) -> dict[int, NoiseShape]:
    """Estimate a noise shape per group from a random sample of its pixels.

    Each sampled pixel is fitted with :func:`fit_noise` (exact RMLE); the
    group shape is the median of the samples' shapes. With ``noise="auto"``,
    FNWN, PLWN and RWFNWN are fitted and the model with the lowest median BIC
    is used for the group.
    """
    models = _SELECTABLE if noise == "auto" else (noise,)
    rng = np.random.default_rng(seed)
    label_values = np.asarray(labels.values)
    need = len(terms.names) + 10
    tasks, owners = [], []
    times = da["time"].values
    for group in np.unique(label_values[label_values >= 0]):
        rows, cols = np.nonzero(label_values == group)
        # Read only candidate pixels (three times the sample size), not the
        # whole frame, and keep those with enough valid epochs.
        pick = rng.choice(len(rows), size=min(3 * n_samples, len(rows)), replace=False)
        series = da.isel(
            y=xr.DataArray(rows[pick], dims="s"), x=xr.DataArray(cols[pick], dims="s")
        ).values
        usable = [c for c in series.T if np.isfinite(c).sum() >= need][:n_samples]
        for column in usable:
            tasks.append((times, column, terms, models))
            owners.append(int(group))
    # Threads, not processes: a spawned process re-imports the caller's
    # script, which fails for scripts without a __main__ guard. The Cholesky
    # factorizations that dominate each fit release the GIL.
    from ..config import config

    workers = max_workers or min(len(tasks), int(config.get("num_workers") or 1))
    if workers > 1 and len(tasks) > 1:
        with ThreadPoolExecutor(workers) as pool:
            results = list(pool.map(_fit_sample, tasks))
    else:
        results = [_fit_sample(t) for t in tasks]

    shapes: dict[int, NoiseShape] = {}
    for group in sorted(set(owners)):
        fits = [r for r, g in zip(results, owners) if g == group]
        best = None
        for i, model in enumerate(models):
            good = [f[i] for f in fits if f[i][1] is not None]
            if not good:
                continue
            bic = float(np.median([f[2] for f in good]))
            if best is None or bic < best.bic:
                best = NoiseShape(
                    model, _median_shape([f[1] for f in good]), len(good), bic
                )
        if best is not None:
            shapes[group] = best
    return shapes


def _gls_block(
    data: np.ndarray,
    labels: np.ndarray,
    design: np.ndarray,
    inverses: np.ndarray,
    groups: np.ndarray,
    column: int,
    min_valid: int,
) -> tuple:
    """Exact GLS per pixel under its group's covariance shape."""
    shape = data.shape[:-1]
    n_t, n_p = design.shape
    y_all = data.reshape(-1, n_t).astype(np.float64)
    lab = labels.reshape(-1)
    valid_all = np.isfinite(y_all)
    count_all = valid_all.sum(axis=1)
    velocity = np.full(y_all.shape[0], np.nan)
    sigma = np.full(y_all.shape[0], np.nan)
    amplitude = np.full(y_all.shape[0], np.nan)
    enough = count_all >= max(min_valid, n_p + 1)
    for g_index, group in enumerate(groups):
        q = inverses[g_index]
        b = q @ design  # (n_t, n_p)
        m_full = design.T @ b
        members_all = np.flatnonzero((lab == group) & enough)
        # Batches bound the temporaries (several pixels x epochs arrays and
        # the k x k Schur blocks) to a few hundred MB per task.
        for start in range(0, members_all.size, _BATCH):
            members = members_all[start : start + _BATCH]
            y = np.where(valid_all[members], y_all[members], 0.0)
            valid = valid_all[members]
            missing = n_t - valid.sum(axis=1)
            qy = y @ q  # (pixels, n_t); y is zero at missing epochs
            rhs = y @ b  # Aᵀ Q y per pixel
            normal = np.broadcast_to(m_full, (len(members), n_p, n_p)).copy()
            for k in np.unique(missing[missing > 0]):
                sel = np.flatnonzero(missing == k)
                idx = np.nonzero(~valid[sel])[1].reshape(len(sel), k)  # missing epochs
                s_inv = np.linalg.inv(q[idx[:, :, None], idx[:, None, :]])  # (s, k, k)
                b_m = b[idx]  # (s, k, n_p)
                z_m = np.take_along_axis(qy[sel], idx, axis=1)  # (s, k)
                corr = np.einsum("skp,skl->spl", b_m, s_inv)  # B_mᵀ S
                normal[sel] -= np.einsum("spl,slq->spq", corr, b_m)
                rhs[sel] -= np.einsum("spl,sl->sp", corr, z_m)
            try:
                inv_normal = np.linalg.inv(normal)
            except np.linalg.LinAlgError:
                inv_normal = np.linalg.pinv(normal)
            beta = np.einsum("spq,sq->sp", inv_normal, rhs)
            r = np.where(valid, y - beta @ design.T, 0.0)
            qr = r @ q
            quad = np.einsum("st,st->s", r, qr)
            for k in np.unique(missing[missing > 0]):
                sel = np.flatnonzero(missing == k)
                idx = np.nonzero(~valid[sel])[1].reshape(len(sel), k)
                s_inv = np.linalg.inv(q[idx[:, :, None], idx[:, None, :]])
                w_m = np.take_along_axis(qr[sel], idx, axis=1)
                quad[sel] -= np.einsum("sk,skl,sl->s", w_m, s_inv, w_m)
            dof = valid.sum(axis=1) - n_p
            sigma2 = quad / dof
            velocity[members] = beta[:, column]
            sigma[members] = np.sqrt(sigma2 * inv_normal[:, column, column])
            amplitude[members] = np.sqrt(sigma2)
    return (
        velocity.reshape(shape),
        sigma.reshape(shape),
        amplitude.reshape(shape),
        count_all.reshape(shape),
    )


def coherence_groups(ds: xr.Dataset, n_groups: int = 5) -> xr.DataArray:
    """Label pixels by quantile bin of their (time-mean) ``temporal_coherence``."""
    if "temporal_coherence" not in ds:
        raise ValueError("groups='coherence' needs the temporal_coherence layer")
    coherence = ds["temporal_coherence"]
    if "time" in coherence.dims:
        coherence = coherence.mean("time")
    coherence = coherence.compute()
    edges = np.nanquantile(coherence.values, np.linspace(0, 1, n_groups + 1))
    labels = np.digitize(coherence.values, edges[1:-1])
    labels = np.where(np.isfinite(coherence.values), labels, -1)
    return xr.DataArray(labels, dims=("y", "x"), coords=coherence.coords)


def gls_velocity(
    obj: Union[xr.Dataset, xr.DataArray],
    *,
    terms: Trajectory,
    noise: str = "PLWN",
    groups: Union[str, xr.DataArray] = "coherence",
    n_groups: int = 5,
    n_samples: int = 16,
    layer: str = "displacement",
    min_valid: int = 0,
    seed: int = 0,
    shapes: Optional[dict[int, NoiseShape]] = None,
) -> xr.Dataset:
    """Velocity and colored-noise sigma for every pixel by grouped exact GLS.

    See :func:`disp_xr.fit_velocity` with ``method="gls"``.
    """
    if noise != "auto" and noise not in NOISE_MODELS:
        raise ValueError(
            f"noise must be 'auto' or one of {NOISE_MODELS}, not {noise!r}"
        )
    if "linear_trend" not in terms.names:
        raise ValueError("The trajectory terms need a linear term (polynomial >= 1)")
    da = obj[layer] if isinstance(obj, xr.Dataset) else obj
    if isinstance(groups, str):
        if groups == "quadtree":
            raise NotImplementedError(
                "Quadtree groups come with plan task T28; pass groups='coherence' or "
                "an integer label map"
            )
        if groups != "coherence":
            raise ValueError(
                f"groups must be 'coherence' or a label map, not {groups!r}"
            )
        if not isinstance(obj, xr.Dataset):
            raise ValueError(
                "groups='coherence' needs the Dataset with temporal_coherence"
            )
        labels = coherence_groups(obj, n_groups)
    else:
        labels = groups.astype(int)
    da = time_contiguous(da)
    times = da["time"].values
    if shapes is None:
        shapes = estimate_group_noise(
            da, labels, terms=terms, noise=noise, n_samples=n_samples, seed=seed
        )
    if not shapes:
        raise ValueError("No group has a pixel with enough valid epochs to fit noise")
    obs_idx, _ = _sampling_grid(times, None)
    design = terms.design(times)
    group_ids = np.array(sorted(shapes))
    inverses = np.stack(
        [np.linalg.inv(shapes[g].covariance(obs_idx)) for g in group_ids]
    )
    column = terms.names.index("linear_trend")
    labels = labels.where(labels.isin(group_ids), -1)
    if da.chunks is not None:
        labels = labels.chunk({d: da.chunksizes[d] for d in labels.dims})
    velocity, sigma, amplitude, count = xr.apply_ufunc(
        _gls_block,
        da,
        labels,
        input_core_dims=[["time"], []],
        output_core_dims=[[], [], [], []],
        kwargs={
            "design": design,
            "inverses": inverses,
            "groups": group_ids,
            "column": column,
            "min_valid": min_valid,
        },
        dask="parallelized",
        output_dtypes=[np.float64, np.float64, np.float64, np.int64],
    )
    units = da.attrs.get("units", "")
    kappa = xr.full_like(labels, np.nan, dtype=np.float64)
    white = xr.full_like(labels, np.nan, dtype=np.float64)
    for g, s in shapes.items():
        kappa = kappa.where(labels != g, s.kappa)
        white = white.where(labels != g, s.white_fraction)
    out = xr.Dataset(
        {
            "velocity": velocity,
            "velocity_sigma": sigma,
            "noise_amplitude": amplitude.assign_attrs(
                units=units, long_name="total noise amplitude (per pixel)"
            ),
            "noise_group": labels.assign_attrs(long_name="noise group (-1: none)"),
            "kappa": kappa.assign_attrs(long_name="power-law index of the group"),
            "white_fraction": white,
            "n_valid": count,
        }
    )
    table = pd.DataFrame(
        [
            {
                "group": g,
                "model": s.model,
                "kappa": s.kappa,
                "white_fraction": s.white_fraction,
                "n_samples": s.n_samples,
                "median_bic": s.bic,
                "components": s.components,
            }
            for g, s in shapes.items()
        ]
    )
    out.attrs["noise_groups"] = table.to_json(orient="records")
    return out
