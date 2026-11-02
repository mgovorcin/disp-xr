"""Per-series models from geepers, as maps: smooth rates and a linearity test.

:func:`fit_series` runs a geepers model on every multilooked cell of a chained,
re-referenced cube: the median of ``window x window`` pixels, in median bins of
``bin_days``, one fit per cell. These models factorize their covariance many
times per series, so a fit costs 0.1 to 0.3 s on a 10-year record, before the read:
about 5 core-hours for a frame at 1 km, 200 at 150 m, and too much at 30 m, where
:func:`disp_xr.fit_velocity` is the tool.

Series that are not re-referenced carry what the whole frame shares (atmosphere,
the reference), which these models report as change: on a 128 x 128 block of
F11116 the rate was -13 mm/yr before removing the block's median and 0.6 after.

Requires the ``gnss`` extra, with a geepers that has ``geepers.smooth``.

``"linearity"`` is a **draft**: not validated on DISP (needs V&V) and to be
improved, so its output may change. ``"smooth"`` is provisional, with the checks
against GNSS described in the geepers documentation (``docs/analysis-modules.md``).
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Any, Literal, Optional, Union

import numpy as np
import pandas as pd
import xarray as xr

from .._chunks import time_contiguous
from .._grid import with_grid_mapping
from .._history import recorded
from .trajectory import check_not_denoised

__all__ = ["fit_series"]

SHAPES = ("linear", "piecewise", "smooth")
LINEARITY_MODELS = ("linear", "quadratic", "piecewise")
HORIZON_DRIVERS = ("rate uncertainty", "non-linearity", "capped")
DRAFT = "draft: needs V&V and improvement"
_TO_MM = {
    "m": 1000.0,
    "meter": 1000.0,
    "meters": 1000.0,
    "metre": 1000.0,
    "metres": 1000.0,
    "mm": 1.0,
    "millimeter": 1.0,
    "millimeters": 1.0,
}
_NOT_SET = -1  # code of a cell that was not fitted


def _models() -> tuple[Any, Any, Any]:
    try:
        from geepers.linearity import linearity_test, validity_horizon
        from geepers.smooth import fit_smooth_trend
    except ImportError as err:
        raise ImportError(
            "fit_series needs a geepers with geepers.smooth: "
            "pip install 'disp-xr[gnss]'"
        ) from err
    return fit_smooth_trend, linearity_test, validity_horizon


def _median_windows(block: np.ndarray, window: int) -> np.ndarray:
    """Return the NaN-aware median of ``window x window`` cells of (time, y, x)."""
    # numbagg's NaN median is about 250x faster than np.nanmedian (velocity.py).
    from numbagg import nanmedian

    nt, ny, nx = block.shape
    cells = block.reshape(nt, ny // window, window, nx // window, window)
    return nanmedian(cells, axis=(2, 4))


def _multilook(da: xr.DataArray, window: int) -> xr.DataArray:
    """Median of ``window x window`` pixels, lazily if ``da`` is; edges are trimmed."""
    da = da.transpose("time", "y", "x")
    if window == 1:
        return da
    ny = da.sizes["y"] // window * window
    nx = da.sizes["x"] // window * window
    if ny == 0 or nx == 0:
        raise ValueError(f"window={window} is larger than the frame {da.shape[1:]}")
    da = da.isel(y=slice(ny), x=slice(nx))
    if da.chunks is None:
        data = _median_windows(da.values, window)
    else:
        # Whole windows in every chunk, so each one reduces on its own.
        sizes = {
            dim: max(1, max(da.chunksizes[dim]) // window) * window
            for dim in ("y", "x")
        }
        aligned = da.chunk(sizes).data
        data = aligned.map_blocks(
            _median_windows,
            window,
            chunks=(
                aligned.chunks[0],
                tuple(c // window for c in aligned.chunks[1]),
                tuple(c // window for c in aligned.chunks[2]),
            ),
            dtype=da.dtype,
        )
    coords = {
        dim: da[dim].values.reshape(-1, window).mean(axis=1) for dim in ("y", "x")
    }
    return xr.DataArray(
        data,
        dims=("time", "y", "x"),
        coords={"time": da["time"].values, **coords},
        attrs=da.attrs,
    )


def _bins(times: np.ndarray, bin_days: float) -> tuple[np.ndarray, np.ndarray]:
    """Return each epoch's bin and the start time of every bin."""
    times = np.asarray(times, dtype="datetime64[s]")
    days = (times - times[0]) / np.timedelta64(1, "D")
    index = np.floor(days / bin_days).astype(int)
    seconds = np.arange(index.max() + 1) * bin_days * 86400.0
    return index, times[0] + seconds.astype("timedelta64[s]")


def _bin_medians(y: np.ndarray, index: np.ndarray, n_bins: int) -> np.ndarray:
    """Return the median per bin of ``y`` (cells, time)."""
    from numbagg import nanmedian

    out = np.full((y.shape[0], n_bins), np.nan)
    for k in np.unique(index):
        out[:, k] = nanmedian(y[:, index == k], axis=1)
    return out


def _smooth_block(
    data: np.ndarray,
    index: np.ndarray,
    dates: np.ndarray,
    to_mm: float,
    min_valid: int,
    fit_options: dict,
) -> tuple[np.ndarray, ...]:
    from threadpoolctl import threadpool_limits

    fit_smooth_trend, _, _ = _models()
    shape = data.shape[:-1]
    y = _bin_medians(data.reshape(-1, data.shape[-1]) * to_mm, index, dates.size)
    n = y.shape[0]
    code = np.full(n, _NOT_SET, dtype=np.int8)
    evidence, rate, rate_sigma = (np.full(n, np.nan) for _ in range(3))
    rates, rates_sigma = (np.full(y.shape, np.nan) for _ in range(2))
    n_steps, n_changes = (np.zeros(n, dtype=np.int16) for _ in range(2))
    # Dask runs one task per core already: BLAS threads would oversubscribe it.
    with threadpool_limits(limits=1):
        for i in np.flatnonzero(np.isfinite(y).sum(axis=1) >= min_valid):
            finite = np.isfinite(y[i])
            try:
                fit = fit_smooth_trend(dates, y[i], units="mm", **fit_options)
            except (ValueError, np.linalg.LinAlgError):
                continue  # too few usable epochs for the model: left as NaN
            code[i] = SHAPES.index(fit.shape)
            evidence[i] = fit.evidence
            rate[i], rate_sigma[i] = fit.rate / to_mm, fit.rate_sigma / to_mm
            rates[i, finite] = np.asarray(fit.rates) / to_mm
            rates_sigma[i, finite] = np.asarray(fit.rates_sigma) / to_mm
            n_steps[i], n_changes[i] = len(fit.step_dates), len(fit.rate_change_dates)
    return (
        code.reshape(shape),
        evidence.reshape(shape),
        rate.reshape(shape),
        rate_sigma.reshape(shape),
        n_steps.reshape(shape),
        n_changes.reshape(shape),
        rates.reshape(*shape, -1),
        rates_sigma.reshape(*shape, -1),
    )


def _linearity_block(
    data: np.ndarray,
    index: np.ndarray,
    dates: np.ndarray,
    to_mm: float,
    min_valid: int,
    tolerance_mm: float,
    fit_options: dict,
) -> tuple[np.ndarray, ...]:
    from threadpoolctl import threadpool_limits

    _, linearity_test, validity_horizon = _models()
    shape = data.shape[:-1]
    y = _bin_medians(data.reshape(-1, data.shape[-1]) * to_mm, index, dates.size)
    n = y.shape[0]
    code = np.full(n, _NOT_SET, dtype=np.int8)
    driver = np.full(n, _NOT_SET, dtype=np.int8)
    breakpoints = np.full(n, np.datetime64("NaT"), dtype="datetime64[ns]")
    names = (
        "velocity",
        "velocity_sigma",
        "p_quadratic",
        "p_piecewise",
        "acceleration",
        "acceleration_sigma",
        "rate_before",
        "rate_after",
        "rate_change_sigma",
        "departure",
        "validity_years",
    )
    values = {name: np.full(n, np.nan) for name in names}
    with threadpool_limits(limits=1):
        for i in np.flatnonzero(np.isfinite(y).sum(axis=1) >= min_valid):
            try:
                result = linearity_test(dates, y[i], **fit_options)
            except (ValueError, np.linalg.LinAlgError):
                continue  # too few usable epochs for the model: left as NaN
            horizon = validity_horizon(result, tolerance=tolerance_mm)
            code[i] = LINEARITY_MODELS.index(result.model)
            driver[i] = HORIZON_DRIVERS.index(horizon.driver)
            if result.breakpoint is not None:
                breakpoints[i] = np.datetime64(result.breakpoint, "ns")
            values["velocity"][i] = result.trend.velocity / to_mm
            values["velocity_sigma"][i] = result.trend.velocity_uncertainty / to_mm
            values["p_quadratic"][i] = result.p_values.get("quadratic", np.nan)
            values["p_piecewise"][i] = result.p_values.get("piecewise", np.nan)
            values["acceleration"][i] = result.acceleration / to_mm
            values["acceleration_sigma"][i] = result.acceleration_sigma / to_mm
            values["rate_before"][i] = result.rate_before / to_mm
            values["rate_after"][i] = result.rate_after / to_mm
            values["rate_change_sigma"][i] = result.rate_change_sigma / to_mm
            values["departure"][i] = result.departure / to_mm
            values["validity_years"][i] = horizon.years
    return (
        code.reshape(shape),
        driver.reshape(shape),
        breakpoints.reshape(shape),
        *(values[name].reshape(shape) for name in names),
    )


def _flags(names: Sequence[str], long_name: str) -> dict:
    return {
        "long_name": long_name,
        "flag_values": list(range(len(names))),
        "flag_meanings": " ".join(n.replace(" ", "_") for n in names),
        "_FillValue": _NOT_SET,
    }


def _check_input(obj: Union[xr.Dataset, xr.DataArray], da: xr.DataArray) -> None:
    if "time" not in da.dims:
        raise ValueError(f"fit_series needs a time dimension, not {da.dims}")
    if not isinstance(obj, xr.Dataset):
        return  # a layer does not carry the Dataset's chain and reference records
    if "reference_date" not in obj.attrs and "reference_time" in obj.coords:
        if np.unique(obj["reference_time"].values).size > 1:
            raise ValueError(
                "fit_series needs one continuous series: run disp_xr.chain first"
            )
    if "reference" not in obj.attrs:
        warnings.warn(
            "fit_series on data that was not re-referenced: what the whole frame "
            "shares is reported as change. Run .disp.rereference first",
            UserWarning,
            stacklevel=3,
        )


@recorded
def fit_series(
    obj: Union[xr.Dataset, xr.DataArray],
    model: Literal["smooth", "linearity"] = "smooth",
    *,
    layer: str = "displacement",
    window: int = 5,
    bin_days: float = 30.0,
    periods: tuple[float, ...] = (1.0, 0.5),
    steps: Sequence[str] = (),
    min_valid: int = 24,
    max_random_walk_mm: Optional[float] = 5.0,
    min_evidence: float = 6.0,
    knots_per_year: float = 1.0,
    detect_changes: bool = True,
    noise: str = "FNWN",
    tolerance_mm: float = 10.0,
    n_simulations: int = 2000,
    allow_denoised: bool = False,
) -> xr.Dataset:
    """Run a geepers per-series model on every multilooked cell.

    Parameters
    ----------
    obj : xr.Dataset or xr.DataArray
        Chained, re-referenced displacement ``(time, y, x)``. A Dataset that is
        not chained raises, and one that is not re-referenced warns: a DataArray
        carries neither record, so check it yourself.
    model : {"smooth", "linearity"}
        ``"smooth"``: a time-varying rate without an assumed shape,
        ``geepers.smooth.fit_smooth_trend``. Provisional. ``"linearity"``: is a
        single rate enough, and for how long, ``geepers.linearity``. **Draft:
        needs V&V and improvement**, so it warns, and its output may change.
    layer : str
        Layer of a Dataset.
    window : int
        Pixels per side of the median cell; the edges are trimmed. 1 fits every
        pixel, which is rarely affordable (see the module docstring).
    bin_days : float
        Median bins in time. The cost per fit grows with the cube of the bins.
    periods : tuple of float
        Periods in years of the sine/cosine pairs in every model.
    steps : sequence of str
        Dates of known offsets, estimated in every model.
    min_valid : int
        Fewest valid bins to fit a cell; others are NaN.
    max_random_walk_mm : float or None
        ``"smooth"``: largest random walk, in mm/sqrt(yr), the straight line may
        be given before a smooth trend is preferred. The default of 5 is the
        geepers recommendation for DISP-S1 with 5 x 5 medians in 30-day bins,
        from station pairs on two frames (F11116, F38502). It depends on the
        product and the multilook: check it when you change either.
    min_evidence, knots_per_year, detect_changes
        ``"smooth"``: as in ``fit_smooth_trend``.
    noise : {"FNWN", "PLWN", "PL", "WN", "RWFNWN"}
        ``"linearity"``: noise model of ``linearity_test``.
    tolerance_mm : float
        ``"linearity"``: extrapolation error that sets ``validity_years``.
    n_simulations : int
        ``"linearity"``: noise realizations for the piecewise p-value.
    allow_denoised : bool
        Fit data marked ``attrs["denoised"]``. Off by default (PRD S1).

    Returns
    -------
    xr.Dataset
        On the multilooked grid, rates in the input unit per year, lazy if the
        input is. Cells that were not fitted are NaN (code -1).

        ``"smooth"``: ``trend_shape`` (0 linear, 1 piecewise, 2 smooth), ``evidence``,
        ``rate`` and ``rate_sigma`` (at the last epoch), ``n_steps``,
        ``n_rate_changes``, and ``rates``, ``rates_sigma`` ``(time, y, x)`` at
        the start of each bin.

        ``"linearity"``: ``linearity_model`` (0 linear, 1 quadratic, 2
        piecewise), ``velocity``, ``velocity_sigma``, ``p_quadratic``,
        ``p_piecewise``, ``acceleration``, ``acceleration_sigma``,
        ``breakpoint``, ``rate_before``, ``rate_after``, ``rate_change_sigma``,
        ``departure``, ``validity_years`` and ``validity_driver``.

    Raises
    ------
    ImportError
        Without a geepers that has these models.
    ValueError
        For an unknown model, unknown units or a Dataset that is not chained.

    Examples
    --------
    >>> cube = dxr.chain(ds).disp.rereference("median")  # doctest: +SKIP
    >>> maps = dxr.fit_series(cube.sel(x=xs, y=ys))  # doctest: +SKIP
    >>> maps.rate.where(maps.trend_shape == 2).plot()  # doctest: +SKIP

    """
    if model not in ("smooth", "linearity"):
        raise ValueError(f"model must be 'smooth' or 'linearity', not {model!r}")
    if window < 1:
        raise ValueError(f"window must be at least 1, not {window}")
    _models()
    if model == "linearity":
        warnings.warn(
            "fit_series(model='linearity') is a draft: it needs V&V on DISP and "
            "improvement, and its output may change",
            UserWarning,
            stacklevel=3,
        )
    check_not_denoised(obj, allow_denoised)
    da = obj[layer] if isinstance(obj, xr.Dataset) else obj
    _check_input(obj, da)
    units = str(da.attrs.get("units", "")).lower()
    if units not in _TO_MM:
        raise ValueError(
            f"fit_series needs displacement in meters or millimeters, not "
            f"{units!r}: the limits of these models are physical (set attrs['units'])"
        )
    cells = time_contiguous(_multilook(time_contiguous(da), window))
    index, starts = _bins(cells["time"].values, bin_days)
    dates = starts.astype("datetime64[ns]")
    common = {
        "sampling_days": bin_days,
        "periods_years": tuple(periods),
        "step_dates": np.array(list(steps), dtype="datetime64[ns]") if steps else None,
    }
    kwargs = {
        "index": index,
        "dates": dates,
        "to_mm": _TO_MM[units],
        "min_valid": min_valid,
    }
    rate_units = {"units": f"{da.attrs['units']}/year"}
    if model == "smooth":
        kwargs["fit_options"] = {
            **common,
            "max_random_walk_mm": max_random_walk_mm,
            "min_evidence": min_evidence,
            "knots_per_year": knots_per_year,
            "detect_changes": detect_changes,
        }
        results = xr.apply_ufunc(
            _smooth_block,
            cells,
            input_core_dims=[["time"]],
            output_core_dims=[[]] * 6 + [["bin_time"]] * 2,
            kwargs=kwargs,
            dask="parallelized",
            output_dtypes=[np.int8, float, float, float, np.int16, np.int16]
            + [float, float],
            dask_gufunc_kwargs={"output_sizes": {"bin_time": dates.size}},
        )
        code, evidence, rate, rate_sigma, n_steps, n_changes, rates, sigmas = results
        out = xr.Dataset(
            {
                "trend_shape": code.assign_attrs(_flags(SHAPES, "trend used")),
                "evidence": evidence.assign_attrs(
                    long_name="restricted log-likelihood gain over a line, x 2"
                ),
                "rate": rate.assign_attrs(rate_units, long_name="rate, last epoch"),
                "rate_sigma": rate_sigma.assign_attrs(rate_units),
                "n_steps": n_steps.assign_attrs(long_name="steps in the model"),
                "n_rate_changes": n_changes.assign_attrs(
                    long_name="rate changes in the model"
                ),
                "rates": rates.transpose("bin_time", ...).assign_attrs(rate_units),
                "rates_sigma": (
                    sigmas.transpose("bin_time", ...).assign_attrs(rate_units)
                ),
            },
            coords={"bin_time": dates},
        ).rename(bin_time="time")
    else:
        kwargs["tolerance_mm"] = tolerance_mm
        kwargs["fit_options"] = {
            **common,
            "noise_model": noise,
            "n_simulations": n_simulations,
        }
        results = xr.apply_ufunc(
            _linearity_block,
            cells,
            input_core_dims=[["time"]],
            output_core_dims=[[]] * 14,
            kwargs=kwargs,
            dask="parallelized",
            output_dtypes=[np.int8, np.int8, "datetime64[ns]"] + [float] * 11,
        )
        out = xr.Dataset(
            {
                "linearity_model": (
                    results[0].assign_attrs(
                        _flags(LINEARITY_MODELS, "model with the lowest BIC")
                    )
                ),
                "validity_driver": (
                    results[1].assign_attrs(
                        _flags(HORIZON_DRIVERS, "what sets validity_years")
                    )
                ),
                "breakpoint": results[2],
                "velocity": results[3].assign_attrs(rate_units),
                "velocity_sigma": results[4].assign_attrs(rate_units),
                "p_quadratic": results[5],
                "p_piecewise": results[6],
                "acceleration": (
                    results[7].assign_attrs(units=f"{da.attrs['units']}/year^2")
                ),
                "acceleration_sigma": (
                    results[8].assign_attrs(units=f"{da.attrs['units']}/year^2")
                ),
                "rate_before": results[9].assign_attrs(rate_units),
                "rate_after": results[10].assign_attrs(rate_units),
                "rate_change_sigma": results[11].assign_attrs(rate_units),
                "departure": results[12].assign_attrs(units=da.attrs["units"]),
                "validity_years": (
                    results[13].assign_attrs(units="years", tolerance_mm=tolerance_mm)
                ),
            }
        )
    if "spatial_ref" in da.coords:
        out = out.assign_coords(spatial_ref=da.coords["spatial_ref"])
    out.attrs = {
        "model": f"geepers {model}, {window} x {window} medians, {bin_days:g}-day bins",
        "status": DRAFT if model == "linearity" else "provisional",
        "reference_time": str(pd.Timestamp(cells["time"].values[0])),
    }
    return with_grid_mapping(out)
