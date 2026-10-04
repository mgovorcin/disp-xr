"""Temporal denoising: gated Kalman smoother and robust LOWESS (PRD S3, T31).

Both methods share one interface, :func:`denoise`, and evaluate on the
acquisitions or on a regular grid (:func:`disp_xr.timeseries.regular_time_grid`)
with refused gaps. They are for display, resampling, linearity checks and
transient hunting: the temporal-denoising study (Kiritimati DISP-S1, 2026-09)
found that no filter improves a velocity, because InSAR residuals at 12-day
sampling are white and ordinary least squares is already optimal. Fit rates on
the raw series; :func:`disp_xr.fit_trajectory` refuses data marked as denoised.

**Kalman** (``method="kalman"``): a local-linear-trend model (displacement and
velocity, an integrated random walk) with a Rauch-Tung-Striebel smoother, or a
fixed-lag smoother for products that publish as data arrive. The only knobs
are the timescale ``tau_days`` (20 to 30 d), which sets the process noise
``q = 3 R / tau³`` against the measurement variance ``R`` (estimated per pixel
from the data), and the χ² gate: an update whose normalized innovation squared
exceeds ``gate`` (9) is skipped. A one-epoch outlier makes one huge innovation
and is rejected; a real transient builds over several epochs that stay inside
the gate. In the study it left 0.2 to 0.4% of an injected spike and kept 93% of
a 60-day transient. Declared breaks inflate the covariance at the first epoch
after them (``break_mode``: offset, rate or both), and the gate is suspended
there, so a real step is kept; the inflated covariance also stops the backward
pass at the break.

Unlike the study, each pixel keeps its own 2 x 2 covariance: DISP pixels have
their own missing epochs, and the study's production notes ask for a
per-pixel gate (unwrapping jumps) rather than its scene-wide one.

**LOWESS** (``method="lowess"``): robust local-linear regression (tricube
kernel of ``bandwidth_days``, 60 d, and ``iterations`` bisquare reweightings,
2, per pixel), defined at any time. Its outlier rejection depends on sampling
density (0.1% of a spike survives at 12-day sampling, about 30% at 24-day).
Breaks are handled by fitting each segment separately.

Known limitation: both methods were evaluated on one flat, aseismic site
(Kiritimati). A deforming site with topography-correlated troposphere is
part of the T29 trade-study frames.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Any, Literal, Optional, Union

import numpy as np
import pandas as pd
import xarray as xr

from .._chunks import time_contiguous
from .._history import recorded
from ..config import bounded_threads
from .temporal import break_epochs, gap_mask, regular_time_grid, split_at_breaks

__all__ = ["denoise", "kalman_smooth", "lowess", "q_from_timescale"]

YEAR_DAYS = 365.25


def q_from_timescale(tau_days: float, r: Union[float, np.ndarray]) -> Any:
    """Return the process noise giving a smoothing timescale of ``tau_days``.

    For an integrated random walk the timescale where process noise balances
    measurement noise ``r`` is ``tau = (3 r / q)^(1/3)`` (in years). Setting
    ``q`` directly is how a filter silently returns a straight line (``q`` too
    small for ``r``).
    """
    tau = tau_days / YEAR_DAYS
    return 3.0 * np.asarray(r) / tau**3


def _measurement_variance(y: np.ndarray) -> np.ndarray:
    """Robust white-noise variance per pixel from first differences.

    ``var(y[i+1] - y[i]) = 2 R`` for white noise plus a slowly varying signal;
    the median difference (the trend) is removed and the MAD scaled.
    """
    n = y.shape[0]
    if n < 3:
        return np.full(y.shape[1:], np.nan)
    diff = np.diff(y, axis=0)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="All-NaN slice")
        centered = diff - np.nanmedian(diff, axis=0)
        mad = 1.4826 * np.nanmedian(np.abs(centered), axis=0)
    return mad**2 / 2.0


def kalman_smooth(
    t_years: np.ndarray,
    y: np.ndarray,
    *,
    tau_days: float = 20.0,
    gate: Optional[float] = 9.0,
    r: Optional[np.ndarray] = None,
    p0_velocity: float = 1.0,
    breaks: Optional[np.ndarray] = None,
    break_mode: Literal["offset", "rate", "both"] = "offset",
    break_var: float = 1.0e2,
    lag: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Gated local-linear-trend Kalman filter and smoother on ``(time, npix)``.

    Parameters
    ----------
    t_years : numpy.ndarray
        Increasing times in years; epochs where ``y`` is NaN are predicted
        without an update (use them for grid times).
    y : numpy.ndarray
        ``(time, npix)`` series.
    tau_days, gate : float
        Smoothing timescale and χ² gate (None turns the gate off).
    r : numpy.ndarray, optional
        Measurement variance per pixel; default estimated from ``y``.
    p0_velocity : float
        Prior velocity variance; must stay diffuse (a tight prior shrinks
        every rate toward zero), in data units squared per year squared.
    breaks : numpy.ndarray, optional
        Boolean per epoch: the covariance is inflated there.
    break_mode : {"offset", "rate", "both"}
        What may jump at a break.
    break_var : float
        Variance added at a break, in data units squared (and per year
        squared for the rate).
    lag : int, optional
        Fixed-lag smoothing: epoch ``j`` uses data through ``j + lag``
        (about 3 tau converges). None is the full RTS smoother.

    Returns
    -------
    (numpy.ndarray, numpy.ndarray)
        Smoothed displacement and velocity, ``(time, npix)``.

    """
    t = np.asarray(t_years, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n, npix = y.shape
    if r is None:
        r = _measurement_variance(y)
    fallback = float(np.nanmedian(r)) if np.isfinite(r).any() else 1.0
    r = np.where(np.isfinite(r) & (r > 0), r, fallback)
    q = q_from_timescale(tau_days, r)
    brk = np.zeros(n, bool) if breaks is None else np.asarray(breaks, bool)
    # Diffuse start: the first observation sets the level, later ones the rate.
    x0 = np.zeros(npix)
    x1 = np.zeros(npix)
    p00 = np.full(npix, 1.0e6 * float(np.nanmax(r)))
    p01 = np.zeros(npix)
    p11 = np.full(npix, p0_velocity)
    xs = np.empty((n, 2, npix))
    xp = np.empty((n, 2, npix))
    ps = np.empty((n, 3, npix))
    pp = np.empty((n, 3, npix))
    for i in range(n):
        dt = t[i] - t[i - 1] if i else 0.0
        if dt > 0:
            x0 = x0 + dt * x1
            p00 = p00 + 2 * dt * p01 + dt * dt * p11 + q * dt**3 / 3
            p01 = p01 + dt * p11 + q * dt**2 / 2
            p11 = p11 + q * dt
        if brk[i]:
            if break_mode in ("offset", "both"):
                p00 = p00 + break_var
            if break_mode in ("rate", "both"):
                p11 = p11 + break_var
        xp[i, 0], xp[i, 1] = x0, x1
        pp[i, 0], pp[i, 1], pp[i, 2] = p00, p01, p11
        s = p00 + r
        innov = y[i] - x0
        update = np.isfinite(innov)
        if gate is not None and not brk[i]:
            with np.errstate(invalid="ignore"):
                update &= innov**2 / s <= gate
        k0 = np.where(update, p00 / s, 0.0)
        k1 = np.where(update, p01 / s, 0.0)
        innov = np.where(update, innov, 0.0)
        x0 = x0 + k0 * innov
        x1 = x1 + k1 * innov
        p00, p01, p11 = p00 - k0 * p00, p01 - k0 * p01, p11 - k1 * p01
        xs[i, 0], xs[i, 1] = x0, x1
        ps[i, 0], ps[i, 1], ps[i, 2] = p00, p01, p11

    def gain(i: int) -> tuple[np.ndarray, ...]:
        """RTS gain C = Ps[i] F' inv(Pp[i+1]) for each pixel (2 x 2)."""
        dt = t[i + 1] - t[i]
        a00 = ps[i, 0] + dt * ps[i, 1]
        a01 = ps[i, 1]
        a10 = ps[i, 1] + dt * ps[i, 2]
        a11 = ps[i, 2]
        b00, b01, b11 = pp[i + 1, 0], pp[i + 1, 1], pp[i + 1, 2]
        det = b00 * b11 - b01 * b01
        i00, i01, i11 = b11 / det, -b01 / det, b00 / det
        return (
            a00 * i00 + a01 * i01,
            a00 * i01 + a01 * i11,
            a10 * i00 + a11 * i01,
            a10 * i01 + a11 * i11,
        )

    def back(
        i: int, nxt0: np.ndarray, nxt1: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        c00, c01, c10, c11 = gain(i)
        d0 = nxt0 - xp[i + 1, 0]
        d1 = nxt1 - xp[i + 1, 1]
        return xs[i, 0] + c00 * d0 + c01 * d1, xs[i, 1] + c10 * d0 + c11 * d1

    disp = np.empty((n, npix))
    rate = np.empty((n, npix))
    if lag is None:
        s0, s1 = xs[n - 1, 0], xs[n - 1, 1]
        disp[n - 1], rate[n - 1] = s0, s1
        for i in range(n - 2, -1, -1):
            s0, s1 = back(i, s0, s1)
            disp[i], rate[i] = s0, s1
    else:
        for j in range(n):
            stop = min(j + int(lag), n - 1)
            s0, s1 = xs[stop, 0], xs[stop, 1]
            for i in range(stop - 1, j - 1, -1):
                s0, s1 = back(i, s0, s1)
            disp[j], rate[j] = s0, s1
    return disp, rate


def _tricube(u: np.ndarray) -> np.ndarray:
    return (1.0 - np.clip(np.abs(u), 0.0, 1.0) ** 3) ** 3


def lowess(
    t_years: np.ndarray,
    y: np.ndarray,
    t_out: np.ndarray,
    *,
    bandwidth_days: float = 60.0,
    iterations: int = 2,
    min_points: int = 3,
) -> np.ndarray:
    """Robust local-linear regression of ``(time, npix)`` series at ``t_out``.

    The window at each output time is ``bandwidth_days`` wide on each side,
    widened to the ``min_points`` nearest acquisitions where sampling is
    sparse; pixels with fewer than ``min_points`` valid values in it are NaN.
    Each of ``iterations`` passes reweights points by the bisquare of their
    residual over six times the pixel's median absolute residual (Cleveland
    1979, https://doi.org/10.1080/01621459.1979.10481038).
    """
    t = np.asarray(t_years, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    out = np.full((len(t_out), y.shape[1]), np.nan)
    bw = bandwidth_days / YEAR_DAYS
    valid_all = np.isfinite(y)
    for o, t0 in enumerate(np.asarray(t_out, dtype=np.float64)):
        d = np.abs(t - t0)
        h = max(bw, float(np.sort(d)[min(min_points, d.size) - 1]))
        m = d <= h
        tw = t[m] - t0
        yw = y[m]
        valid = valid_all[m]
        kernel = _tricube(tw / max(h, 1e-12))[:, None] * valid
        robust = np.ones_like(yw)
        yz = np.where(valid, yw, 0.0)
        enough = valid.sum(axis=0) >= min_points
        for it in range(iterations + 1):
            w = kernel * robust
            sw = w.sum(axis=0)
            swt = (w * tw[:, None]).sum(axis=0)
            swtt = (w * tw[:, None] ** 2).sum(axis=0)
            swy = (w * yz).sum(axis=0)
            swty = (w * tw[:, None] * yz).sum(axis=0)
            with np.errstate(invalid="ignore", divide="ignore"):
                det = sw * swtt - swt**2
                slope = np.where(det > 0, (sw * swty - swt * swy) / det, 0.0)
                level = (swy - slope * swt) / sw
            if it == iterations:
                break
            resid = np.where(valid, yw - (level + slope * tw[:, None]), np.nan)
            with np.errstate(invalid="ignore"), warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="All-NaN slice")
                scale = np.nanmedian(np.abs(resid), axis=0)
                scale = np.where(scale > 0, scale, np.inf)
                u = np.nan_to_num(resid / (6.0 * scale), nan=1.0)
            robust = np.where(np.abs(u) < 1, (1 - u**2) ** 2, 0.0)
        out[o] = np.where(enough & (sw > 0), level, np.nan)
    return out


def _years(times: np.ndarray, origin: np.datetime64) -> np.ndarray:
    return (
        (np.asarray(times, "datetime64[ns]") - origin)
        / np.timedelta64(1, "D")
        / YEAR_DAYS
    )


def _kernel(
    y: np.ndarray,
    *,
    method: str,
    t_in: np.ndarray,
    t_out: np.ndarray,
    out_from_merged: Optional[np.ndarray],
    breaks_merged: Optional[np.ndarray],
    segments_in: list,
    segments_out: list,
    params: dict,
) -> np.ndarray:
    """Denoise ``(..., time)`` blocks: returns ``(..., out_time)``."""
    lead = y.shape[:-1]
    flat = y.reshape(-1, y.shape[-1]).T  # (time, npix)
    if method == "kalman":
        merged = np.full((params["t_merged"].size, flat.shape[1]), np.nan)
        merged[params["obs_in_merged"]] = flat
        disp, _ = kalman_smooth(
            params["t_merged"],
            merged,
            r=_measurement_variance(flat),
            breaks=breaks_merged,
            **params["kalman"],
        )
        result = disp[out_from_merged]
    else:
        result = np.full((t_out.size, flat.shape[1]), np.nan)
        for idx_in, idx_out in zip(segments_in, segments_out):
            if idx_out.size == 0:
                continue
            if idx_in.size < params["lowess"]["min_points"]:
                continue
            result[idx_out] = lowess(
                t_in[idx_in], flat[idx_in], t_out[idx_out], **params["lowess"]
            )
    return result.T.reshape(*lead, t_out.size)


@bounded_threads
@recorded
def denoise(
    obj: Union[xr.Dataset, xr.DataArray],
    method: Literal["kalman", "lowess"] = "kalman",
    *,
    grid: Union[None, str, Sequence, pd.DatetimeIndex] = None,
    layer: str = "displacement",
    tau_days: float = 20.0,
    gate: Optional[float] = 9.0,
    lag: Optional[int] = None,
    bandwidth_days: float = 60.0,
    iterations: int = 2,
    breaks: Optional[Sequence[Any]] = None,
    break_mode: Literal["offset", "rate", "both"] = "offset",
    max_gap_days: Optional[float] = None,
) -> Any:
    """Denoise displacement series in time, on the acquisitions or a grid.

    Parameters
    ----------
    obj : xr.Dataset or xr.DataArray
        ``(time, y, x)`` (or any dims with ``time``) series.
    method : {"kalman", "lowess"}
        See the module docstring.
    grid : None, "auto" or dates
        Output times: the acquisitions (None), a regular grid
        (``"auto"``: :func:`disp_xr.timeseries.regular_time_grid`), or given
        dates. Grid dates inside gaps longer than ``max_gap_days`` are NaN.
    layer : str
        Layer of a Dataset.
    tau_days, gate, lag
        Kalman timescale, χ² gate (None: off) and fixed lag (None: RTS).
    bandwidth_days, iterations
        LOWESS half-window and robust passes.
    breaks : sequence of date-like, optional
        Declared discontinuities.
    break_mode : {"offset", "rate", "both"}
        Kalman: what may jump at a break.
    max_gap_days : float, optional
        Longest hole a grid may bridge (default from
        :func:`disp_xr.timeseries.gap_mask`).

    Returns
    -------
    Same type as ``obj``
        Denoised ``layer`` (a Dataset keeps only that layer) with ``time`` the
        output times; ``attrs["denoised"]`` names the method, so fits refuse
        it. Lazy if the input is.

    """
    if method not in ("kalman", "lowess"):
        raise ValueError(f"method must be 'kalman' or 'lowess', not {method!r}")
    da = obj[layer] if isinstance(obj, xr.Dataset) else obj
    da = time_contiguous(da)
    times = pd.DatetimeIndex(da["time"].values)
    if grid is None:
        out_times = times
    elif isinstance(grid, str):
        if grid != "auto":
            raise ValueError(f"grid must be None, 'auto' or dates, not {grid!r}")
        out_times = regular_time_grid(times)
    else:
        out_times = pd.DatetimeIndex(pd.to_datetime(np.asarray(grid)))
    origin = np.datetime64(times[0], "ns")
    t_in = _years(times.values, origin)
    t_out = _years(out_times.values, origin)
    params: dict[str, Any] = {
        "kalman": {
            "tau_days": tau_days,
            "gate": gate,
            "lag": lag,
            "break_mode": break_mode,
        },
        "lowess": {
            "bandwidth_days": bandwidth_days,
            "iterations": iterations,
            "min_points": 3,
        },
    }
    out_from_merged = breaks_merged = None
    segments_in = segments_out = []
    if method == "kalman":
        merged = np.union1d(
            times.values.astype("datetime64[ns]"),
            out_times.values.astype("datetime64[ns]"),
        )
        params["t_merged"] = _years(merged, origin)
        params["obs_in_merged"] = np.searchsorted(
            merged, times.values.astype("datetime64[ns]")
        )
        out_from_merged = np.searchsorted(
            merged, out_times.values.astype("datetime64[ns]")
        )
        breaks_merged = break_epochs(pd.DatetimeIndex(merged), breaks)
    else:
        segments_in = split_at_breaks(times, breaks)
        edges = [times[s[0]] for s in segments_in[1:]]
        bounds = [pd.Timestamp.min, *edges, pd.Timestamp.max]
        segments_out = [
            np.flatnonzero((out_times >= lo) & (out_times < hi))
            for lo, hi in zip(bounds[:-1], bounds[1:])
        ]
    result = xr.apply_ufunc(
        _kernel,
        da,
        input_core_dims=[["time"]],
        output_core_dims=[["out_time"]],
        kwargs={
            "method": method,
            "t_in": t_in,
            "t_out": t_out,
            "out_from_merged": out_from_merged,
            "breaks_merged": breaks_merged,
            "segments_in": segments_in,
            "segments_out": segments_out,
            "params": params,
        },
        dask="parallelized",
        output_dtypes=[np.float64],
        dask_gufunc_kwargs={"output_sizes": {"out_time": out_times.size}},
    )
    result = result.rename(out_time="time").assign_coords(time=out_times.values)
    if grid is not None:
        refused = gap_mask(times, out_times, max_gap_days)
        result = result.where(
            xr.DataArray(~refused, dims="time", coords={"time": out_times.values})
        )
    result = result.transpose(*[d for d in da.dims if d in result.dims])
    result = result.astype(da.dtype).assign_attrs({**da.attrs, "denoised": method})
    if isinstance(obj, xr.Dataset):
        out = result.to_dataset(name=layer)
        out.attrs = {**obj.attrs, "denoised": method}
        return out
    return result.rename(da.name)
