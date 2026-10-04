"""Estimator base for spatial models (PRD F9, plan task T19).

Spatial models (trends, GNSS ties, gridders) share one shape, after verde
(the design reference only, not a dependency; PRD decision):

- ``fit(coords, data, weights=None)`` estimates the parameters and returns
  the estimator; fitted attributes end in ``_``;
- ``predict(coords)`` evaluates the model at any coordinates;
- ``score(coords, data, weights=None)`` is the weighted R²;
- ``grid(...)`` predicts on a regular grid as an ``xr.DataArray`` with a CRS;
- :class:`Chain` fits steps in sequence, each on the residual of the last.

Coordinates are a tuple ``(easting, northing)`` of arrays of one shape, in
the CRS of the data (projected, in meters for DISP frames).
:func:`coordinates` builds them from a raster, a point table or a
DataFrame. :func:`cross_val_score` scores an estimator with spatial block
k-fold cross-validation, which keeps neighbors out of each other's folds
(Roberts et al. 2017, https://doi.org/10.1111/ecog.02881).
"""

from __future__ import annotations

import inspect
from collections.abc import Sequence
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
import xarray as xr

__all__ = [
    "BaseEstimator",
    "Chain",
    "Mean",
    "NotFittedError",
    "clone",
    "coordinates",
    "cross_val_score",
]

Coords = tuple[np.ndarray, np.ndarray]


class NotFittedError(RuntimeError):
    """Raised when ``predict`` runs before ``fit``."""


def coordinates(obj: Union[xr.Dataset, xr.DataArray, pd.DataFrame]) -> Coords:
    """Return ``(easting, northing)`` of a raster, a point table or a DataFrame.

    A raster with ``x`` and ``y`` dimensions gives 2D arrays (``(y, x)``
    order); a point table (``x`` and ``y`` coordinates along one dimension,
    as from :func:`disp_xr.to_points`) or a DataFrame with ``x`` and ``y``
    columns gives 1D arrays.
    """
    if isinstance(obj, pd.DataFrame):
        return obj["x"].to_numpy(np.float64), obj["y"].to_numpy(np.float64)
    if {"x", "y"} <= set(obj.dims):
        easting, northing = np.meshgrid(
            obj["x"].values.astype(np.float64), obj["y"].values.astype(np.float64)
        )
        return easting, northing
    if "x" in obj.coords and "y" in obj.coords and obj["x"].dims == obj["y"].dims:
        return obj["x"].values.astype(np.float64), obj["y"].values.astype(np.float64)
    raise ValueError(
        "need x and y dimensions, or x and y coordinates along one dimension"
    )


def _flat(coords: Coords) -> tuple[np.ndarray, np.ndarray, tuple[int, ...]]:
    easting, northing = (np.asarray(c, dtype=np.float64) for c in coords)
    if easting.shape != northing.shape:
        raise ValueError(
            f"easting {easting.shape} and northing {northing.shape} differ"
        )
    return easting.ravel(), northing.ravel(), easting.shape


class BaseEstimator:
    """Base class of disp-xr spatial estimators.

    Subclasses take their parameters as keyword arguments of ``__init__``
    (stored under the same names) and implement ``_fit(easting, northing,
    data, weights)`` and ``_predict(easting, northing)`` on 1D arrays
    without NaN; the base class checks inputs, drops NaN and reshapes.
    """

    def _fit(
        self,
        easting: np.ndarray,
        northing: np.ndarray,
        data: np.ndarray,
        weights: np.ndarray,
    ) -> None:
        raise NotImplementedError

    def _predict(self, easting: np.ndarray, northing: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def get_params(self) -> dict[str, Any]:
        """Return the constructor parameters."""
        names = [
            p for p in inspect.signature(type(self).__init__).parameters if p != "self"
        ]
        return {name: getattr(self, name) for name in names}

    def set_params(self, **params: Any) -> BaseEstimator:
        """Set constructor parameters; returns the estimator."""
        valid = self.get_params()
        for name, value in params.items():
            if name not in valid:
                raise ValueError(f"{type(self).__name__} has no parameter {name!r}")
            setattr(self, name, value)
        return self

    def __repr__(self) -> str:
        params = ", ".join(f"{k}={v!r}" for k, v in self.get_params().items())
        return f"{type(self).__name__}({params})"

    def _is_fitted(self) -> bool:
        return any(k.endswith("_") and not k.startswith("_") for k in vars(self))

    def fit(
        self,
        coords: Coords,
        data: Any,
        weights: Optional[Any] = None,
    ) -> BaseEstimator:
        """Fit the model; returns the estimator.

        Parameters
        ----------
        coords : (easting, northing)
            Arrays of one shape.
        data : array-like
            Values at ``coords`` (same shape); NaN points are skipped.
        weights : array-like, optional
            Non-negative weights (same shape), for example ``1 / sigma**2``;
            points with a zero or NaN weight are skipped.

        """
        easting, northing, shape = _flat(coords)
        values = np.asarray(data, dtype=np.float64)
        if values.shape != shape:
            raise ValueError(f"data {values.shape} and coordinates {shape} differ")
        values = values.ravel()
        if weights is None:
            w = np.ones_like(values)
        else:
            w = np.asarray(weights, dtype=np.float64)
            if w.shape != shape:
                raise ValueError(f"weights {w.shape} and coordinates {shape} differ")
            w = w.ravel()
            if np.any(w < 0):
                raise ValueError("weights must be non-negative")
        ok = np.isfinite(easting) & np.isfinite(northing) & np.isfinite(values)
        # Zero-weight points carry no information; dropping them keeps
        # estimators that place something at each point (splines) consistent.
        ok &= np.isfinite(w) & (w > 0)
        if not ok.any():
            raise ValueError("no point with data and a positive weight")
        self.region_ = (
            float(easting[ok].min()),
            float(easting[ok].max()),
            float(northing[ok].min()),
            float(northing[ok].max()),
        )
        self._fit(easting[ok], northing[ok], values[ok], w[ok])
        return self

    def predict(self, coords: Coords) -> np.ndarray:
        """Return the model at ``coords`` (same shape; NaN where they are NaN)."""
        if not self._is_fitted():
            raise NotFittedError(f"{type(self).__name__} is not fitted; call fit first")
        easting, northing, shape = _flat(coords)
        out = np.full(easting.shape, np.nan)
        ok = np.isfinite(easting) & np.isfinite(northing)
        out[ok] = self._predict(easting[ok], northing[ok])
        return out.reshape(shape)

    def score(self, coords: Coords, data: Any, weights: Optional[Any] = None) -> float:
        """Return the weighted R² of the predictions at ``coords``."""
        predicted = self.predict(coords).ravel()
        values = np.asarray(data, dtype=np.float64).ravel()
        w = (
            np.ones_like(values)
            if weights is None
            else np.asarray(weights, float).ravel()
        )
        ok = np.isfinite(values) & np.isfinite(predicted) & np.isfinite(w)
        values, predicted, w = values[ok], predicted[ok], w[ok]
        mean = np.sum(w * values) / w.sum()
        total = np.sum(w * (values - mean) ** 2)
        return float(1.0 - np.sum(w * (values - predicted) ** 2) / total)

    def grid(
        self,
        *,
        region: Optional[Sequence[float]] = None,
        spacing: Optional[float] = None,
        like: Optional[Union[xr.Dataset, xr.DataArray]] = None,
        crs: Optional[Any] = None,
        name: str = "prediction",
    ) -> xr.DataArray:
        """Predict on a regular grid.

        Parameters
        ----------
        region : (west, east, south, north), optional
            Bounds of the pixel centers; default the fitted data's bounds.
        spacing : float, optional
            Pixel size, needed unless ``like`` is given.
        like : xr.Dataset or xr.DataArray, optional
            Predict on this grid (its ``x``, ``y`` and CRS).
        crs : optional
            CRS to attach (default: the CRS of ``like``).
        name : str
            Name of the returned DataArray.

        Returns
        -------
        xr.DataArray
            ``(y, x)``, north up, with the CRS when known.

        """
        if like is not None:
            x = like["x"].values.astype(np.float64)
            y = like["y"].values.astype(np.float64)
            crs = crs if crs is not None else like.rio.crs
        else:
            if spacing is None:
                raise ValueError("give spacing (or like)")
            if region is None:
                if not self._is_fitted():
                    raise NotFittedError("fit first, or give region")
                region = self.region_
            west, east, south, north = region
            x = np.arange(west, east + spacing / 2, spacing)
            y = np.arange(north, south - spacing / 2, -spacing)
        easting, northing = np.meshgrid(x, y)
        out = xr.DataArray(
            self.predict((easting, northing)),
            dims=("y", "x"),
            coords={"y": y, "x": x},
            name=name,
        )
        if crs is not None:
            out = out.rio.write_crs(crs)
        return out


class Mean(BaseEstimator):
    """The weighted mean of the data: the simplest model (a baseline)."""

    def __init__(self) -> None:
        """Take no parameters."""

    def _fit(
        self,
        easting,  # noqa: ARG002 (estimator interface)
        northing,  # noqa: ARG002
        data,
        weights,
    ) -> None:
        self.mean_ = float(np.sum(weights * data) / np.sum(weights))

    def _predict(self, easting, northing) -> np.ndarray:  # noqa: ARG002
        return np.full(easting.shape, self.mean_)


class Chain(BaseEstimator):
    """Fit estimators in sequence, each on the residual of the previous ones.

    ``predict`` sums the steps. For example, a planar trend then a gridder of
    what the trend leaves.

    Parameters
    ----------
    steps : sequence of (name, estimator)
        The estimators, in order.

    """

    def __init__(self, steps: Sequence[tuple[str, BaseEstimator]]) -> None:
        """Store the steps."""
        self.steps = list(steps)

    def _fit(self, easting, northing, data, weights) -> None:
        residual = data.copy()
        for _, step in self.steps:
            step.fit((easting, northing), residual, weights)
            residual = residual - step.predict((easting, northing))
        self.named_steps_ = dict(self.steps)

    def _predict(self, easting, northing) -> np.ndarray:
        return sum(step.predict((easting, northing)) for _, step in self.steps)


def clone(estimator: BaseEstimator) -> BaseEstimator:
    """Return an unfitted copy of ``estimator`` with the same parameters."""
    params = {}
    for name, value in estimator.get_params().items():
        if isinstance(value, BaseEstimator):
            value = clone(value)
        elif name == "steps":
            value = [(n, clone(s)) for n, s in value]
        params[name] = value
    return type(estimator)(**params)


def cross_val_score(
    estimator: BaseEstimator,
    coords: Coords,
    data: Any,
    *,
    spacing: float,
    weights: Optional[Any] = None,
    n_splits: int = 5,
    seed: int = 0,
) -> np.ndarray:
    """Score an estimator with spatial block k-fold cross-validation.

    Points are grouped into square blocks of side ``spacing``; the blocks are
    shuffled and split into ``n_splits`` folds. Each fold is predicted by a
    copy fitted on the other folds, and scored with the weighted R².

    Returns
    -------
    numpy.ndarray
        One score per fold.

    """
    easting, northing, shape = _flat(coords)
    values = np.asarray(data, dtype=np.float64).ravel()
    w = np.ones_like(values) if weights is None else np.asarray(weights, float).ravel()
    ok = np.isfinite(easting) & np.isfinite(northing) & np.isfinite(values)
    easting, northing, values, w = easting[ok], northing[ok], values[ok], w[ok]
    block = np.floor(easting / spacing).astype(np.int64) * 1_000_003 + np.floor(
        northing / spacing
    ).astype(np.int64)
    unique, inverse = np.unique(block, return_inverse=True)
    if unique.size < n_splits:
        raise ValueError(f"{unique.size} blocks for {n_splits} folds; reduce spacing")
    fold_of_block = np.random.default_rng(seed).permutation(unique.size) % n_splits
    fold = fold_of_block[inverse]
    scores = []
    for k in range(n_splits):
        test = fold == k
        model = clone(estimator).fit(
            (easting[~test], northing[~test]), values[~test], w[~test]
        )
        scores.append(
            model.score((easting[test], northing[test]), values[test], w[test])
        )
    return np.array(scores)
