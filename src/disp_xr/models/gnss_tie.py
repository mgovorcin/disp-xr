"""GNSS tie: remove long-wavelength DISP error with GNSS (PRD F9, plan task T33).

The residual DISP minus GNSS at the stations is fitted with a smooth surface
(a polynomial trend, optionally plus a damped spline) and the predicted field
is subtracted from DISP, either for the velocity or for every epoch.

**What the tie field may hold.** Only long-wavelength error: orbits,
long-wavelength troposphere and ionosphere, reference-frame and plate-motion
differences. Local deformation (groundwater subsidence, coseismic steps) must
stay in the calibrated product, so stations in deforming areas are left out of
the fit (``exclude``) and their signal is restored by construction: the
smooth field does not reach it. A plane (``degree=1``) is the default; add the
spline only with damping strong enough that its scale stays well above the
station spacing. Leave-one-station-out scores improve as the field gets more
local, partly by construction (a field that bends to each station predicts
its neighbors well), so a better score alone does not show that the signal
was preserved.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Optional, Union

import numpy as np
import pandas as pd
import xarray as xr

from .._history import recorded
from .base import BaseEstimator, Chain, clone
from .gridding import Spline
from .trend import Trend, polynomial_terms

__all__ = ["GNSSTie", "TieResult", "apply_tie", "cross_validate_tie", "fit_tie"]

Mode = Literal["velocity", "epoch"]


class GNSSTie(BaseEstimator):
    """A polynomial trend, optionally plus a damped spline of what it leaves.

    Parameters
    ----------
    degree : int
        Degree of the trend (1, a plane, by default).
    damping : float, optional
        Damping of a biharmonic spline fitted to the trend's residual; None
        (default) fits the trend alone. Keep it large: the tie field should
        hold long-wavelength error only.

    """

    def __init__(self, degree: int = 1, damping: Optional[float] = None) -> None:
        """Store the parameters."""
        self.degree = degree
        self.damping = damping

    def _fit(self, easting, northing, data, weights) -> None:
        steps: list[tuple[str, BaseEstimator]] = [("trend", Trend(self.degree))]
        if self.damping is not None:
            steps.append(("spline", Spline(self.damping)))
        self.chain_ = Chain(steps)
        self.chain_._fit(easting, northing, data, weights)

    def _predict(self, easting, northing) -> np.ndarray:
        return self.chain_._predict(easting, northing)


@dataclass
class TieResult:
    """A fitted GNSS tie: one estimator (velocity) or one per epoch."""

    mode: Mode
    estimators: Any
    """The fitted tie (velocity mode) or ``{time: tie}`` (epoch mode)."""
    stations: pd.DataFrame
    """Stations used and left out, with coordinates and residuals."""

    def correction(self, like: Union[xr.Dataset, xr.DataArray]) -> xr.DataArray:
        """Return the correction field on the grid of ``like`` (lazy in time)."""
        if self.mode == "velocity":
            return self.estimators.grid(like=like, name="tie_correction")
        times = pd.DatetimeIndex(like["time"].values)
        x, y = like["x"].values, like["y"].values
        trend_only = all(est.damping is None for est in self.estimators.values())
        if trend_only:
            return _polynomial_correction(self.estimators, times, like)
        import dask
        import dask.array as da

        def one(t: pd.Timestamp) -> np.ndarray:
            est = self.estimators.get(t)
            if est is None:
                return np.zeros((y.size, x.size), np.float32)
            ee, nn = np.meshgrid(x, y)
            return est.predict((ee, nn)).astype(np.float32)

        blocks = [
            da.from_delayed(dask.delayed(one)(t), (y.size, x.size), dtype=np.float32)
            for t in times
        ]
        return xr.DataArray(
            da.stack(blocks),
            dims=("time", "y", "x"),
            coords={"time": times.values, "y": y, "x": x},
            name="tie_correction",
        )


def _polynomial_correction(
    estimators: dict, times: pd.DatetimeIndex, like: Union[xr.Dataset, xr.DataArray]
) -> xr.DataArray:
    """Per-epoch polynomial corrections as coefficients times monomials (lazy)."""
    first = next(iter(estimators.values()))
    terms = polynomial_terms(first.degree)
    coef = np.zeros((times.size, len(terms)))
    centers = np.zeros((times.size, 2))
    scales = np.ones(times.size)
    for k, t in enumerate(times):
        est = estimators.get(t)
        if est is None:
            continue
        trend = est.chain_.named_steps_["trend"]
        coef[k] = trend.coef_
        centers[k] = trend.center_
        scales[k] = trend.scale_
    x = xr.DataArray(
        like["x"].values.astype(np.float64), dims="x", coords={"x": like["x"]}
    )
    y = xr.DataArray(
        like["y"].values.astype(np.float64), dims="y", coords={"y": like["y"]}
    )
    tdim = {"dims": "time", "coords": {"time": times.values}}
    e = (x - xr.DataArray(centers[:, 0], **tdim)) / xr.DataArray(scales, **tdim)
    n = (y - xr.DataArray(centers[:, 1], **tdim)) / xr.DataArray(scales, **tdim)
    field = sum(
        xr.DataArray(coef[:, k], **tdim) * e**i * n**j for k, (i, j) in enumerate(terms)
    )
    return field.transpose("time", "y", "x").rename("tie_correction")


def _excluded(stations: pd.DataFrame, exclude: Any) -> np.ndarray:
    """Boolean per station: left out of the fit."""
    if exclude is None:
        return np.zeros(len(stations), bool)
    if isinstance(exclude, (list, tuple, set, np.ndarray, pd.Index)):
        return stations.index.isin(list(exclude))
    from shapely import contains_xy

    geometry = exclude.union_all() if hasattr(exclude, "union_all") else exclude
    return contains_xy(geometry, stations["lon"].to_numpy(), stations["lat"].to_numpy())


def _station_table(comparison: Any) -> pd.DataFrame:
    series = comparison.series
    table = pd.DataFrame(
        {name: series[name].values for name in ("x", "y", "lon", "lat")},
        index=pd.Index(series["station"].values.astype(str), name="station"),
    )
    return table


def fit_tie(
    comparison: Any,
    *,
    mode: Mode = "velocity",
    degree: int = 1,
    damping: Optional[float] = None,
    exclude: Any = None,
) -> TieResult:
    """Fit a GNSS tie to the DISP-minus-GNSS residual at the stations.

    Parameters
    ----------
    comparison : disp_xr.residuals.Comparison
        :func:`disp_xr.residuals.compare` output.
    mode : {"velocity", "epoch"}
        Fit the velocity residual (``insar_velocity - gps_velocity``,
        weights ``1 / velocity_sigma²``) or the displacement residual of
        every epoch (weights ``1 / sigma_gnss²``).
    degree, damping
        :class:`GNSSTie` parameters.
    exclude : list of station ids, or a lon/lat polygon, optional
        Stations left out of the fit, for example those in deforming areas
        (subsidence, coseismic): their signal must stay in the product.

    Returns
    -------
    TieResult

    """
    if mode not in ("velocity", "epoch"):
        raise ValueError(f"mode must be 'velocity' or 'epoch', not {mode!r}")
    table = _station_table(comparison)
    table["excluded"] = _excluded(table, exclude)
    used = ~table["excluded"].to_numpy()
    coords = (table["x"].to_numpy()[used], table["y"].to_numpy()[used])
    template = GNSSTie(degree, damping)
    if mode == "velocity":
        stations = comparison.stations.reindex(table.index)
        table["residual"] = stations["velocity_residual"].to_numpy()
        sigma = stations["velocity_sigma"].to_numpy()
        weights = 1.0 / np.where(sigma > 0, sigma, np.nan) ** 2
        est = clone(template).fit(
            coords, table["residual"].to_numpy()[used], weights[used]
        )
        table["fitted"] = est.predict((table["x"].to_numpy(), table["y"].to_numpy()))
        return TieResult("velocity", est, table)
    series = comparison.series
    resid = series["residual"].transpose("time", "station").values
    if "sigma_gnss" in series:
        sig = series["sigma_gnss"].transpose("time", "station").values
        weights_all = 1.0 / np.where(sig > 0, sig, np.nan) ** 2
    else:
        weights_all = np.ones_like(resid)
    estimators = {}
    n_terms = len(polynomial_terms(degree))
    for k, t in enumerate(pd.DatetimeIndex(series["time"].values)):
        data = resid[k][used]
        w = weights_all[k][used]
        ok = np.isfinite(data) & np.isfinite(w) & (w > 0)
        if ok.sum() <= n_terms:
            continue
        estimators[t] = clone(template).fit(coords, data, w)
    table["n_epochs"] = len(estimators)
    return TieResult("epoch", estimators, table)


@recorded
def apply_tie(
    ds: xr.Dataset, tie: TieResult, *, layer: str = "displacement"
) -> xr.Dataset:
    """Subtract a GNSS tie from ``layer``.

    Epoch mode subtracts each epoch's correction (zero where no tie was
    fitted). Velocity mode subtracts the correction from a velocity layer, or
    from displacement as the corrected rate times the time since the first
    epoch.
    """
    out = ds.copy()
    data = ds[layer]
    correction = tie.correction(ds)
    if tie.mode == "velocity" and "time" in data.dims:
        years = (data["time"] - data["time"][0]) / np.timedelta64(1, "D") / 365.25
        correction = correction * years
    out[layer] = (data - correction.astype(data.dtype)).assign_attrs(
        data.attrs, gnss_tie=f"{tie.mode}, degree {next(iter(_estimators(tie))).degree}"
    )
    return out


def _estimators(tie: TieResult) -> list:
    return [tie.estimators] if tie.mode == "velocity" else list(tie.estimators.values())


def cross_validate_tie(
    comparison: Any,
    *,
    mode: Mode = "velocity",
    degree: int = 1,
    damping: Optional[float] = None,
    exclude: Any = None,
) -> pd.DataFrame:
    """Leave-one-station-out: residual RMSE at each station before and after.

    Each station used in the fit is left out in turn, the tie is fitted on
    the others and predicted at it. Excluded stations are scored with the
    tie of all used stations (they never enter the fit).

    Returns
    -------
    pandas.DataFrame
        Per station: ``rmse_before`` (of its residual) and ``rmse_after``
        (after subtracting the prediction), ``excluded``; ``attrs`` holds the
        network RMSE before and after.

    """
    table = _station_table(comparison)
    excluded = _excluded(table, exclude)
    names = list(table.index)
    rows = []
    for k, name in enumerate(names):
        ids = [n for j, n in enumerate(names) if j != k and not excluded[j]]
        tie = fit_tie(
            comparison,
            mode=mode,
            degree=degree,
            damping=damping,
            exclude=[n for n in names if n not in ids],
        )
        xy = (table["x"].to_numpy()[[k]], table["y"].to_numpy()[[k]])
        if mode == "velocity":
            before = comparison.stations.loc[name, "velocity_residual"]
            after = before - float(tie.estimators.predict(xy)[0])
            rows.append((name, abs(before), abs(after), bool(excluded[k])))
        else:
            resid = comparison.series["residual"].sel(station=name)
            pred = np.array(
                [
                    (
                        float(tie.estimators[t].predict(xy)[0])
                        if t in tie.estimators
                        else np.nan
                    )
                    for t in pd.DatetimeIndex(resid["time"].values)
                ]
            )
            r = resid.values
            ok = np.isfinite(r) & np.isfinite(pred)
            rows.append(
                (
                    name,
                    float(np.sqrt(np.mean(r[ok] ** 2))),
                    float(np.sqrt(np.mean((r[ok] - pred[ok]) ** 2))),
                    bool(excluded[k]),
                )
            )
    out = pd.DataFrame(
        rows, columns=["station", "rmse_before", "rmse_after", "excluded"]
    )
    out = out.set_index("station")
    used = ~out["excluded"]
    out.attrs = {
        "rmse_before": float(np.sqrt(np.mean(out.loc[used, "rmse_before"] ** 2))),
        "rmse_after": float(np.sqrt(np.mean(out.loc[used, "rmse_after"] ** 2))),
        "mode": mode,
        "degree": degree,
        "damping": damping,
    }
    return out
