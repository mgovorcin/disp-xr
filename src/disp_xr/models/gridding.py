"""Gridders for point data (PRD F9, plan task T24.4).

:class:`Spline` is the biharmonic spline of Sandwell (1987,
https://doi.org/10.1029/GL014i002p00139): the field is a sum of Green's
functions ``g(r) = r² (ln r - 1)`` centered on the data points, with forces
from weighted least squares and optional damping (verde's ``Spline`` is the
design reference, not a dependency). It suits smooth fields from scattered
points (GNSS velocities, downsampled cells). The system is dense, ``n x n``
for ``n`` points, so keep ``n`` to a few thousand (downsample first).

:class:`Nearest` copies the value of the nearest point with positive
weight: a fallback with no smoothing.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .base import BaseEstimator

__all__ = ["Nearest", "Spline"]

# Points per block when evaluating Green's functions (memory: block x n).
BLOCK = 4096


def _green(r: np.ndarray) -> np.ndarray:
    """Biharmonic Green's function in 2D; 0 at ``r = 0`` (its limit)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        out = r**2 * (np.log(r) - 1.0)
    return np.where(r > 0, out, 0.0)


class Spline(BaseEstimator):
    """Biharmonic spline gridder (Sandwell 1987) with optional damping.

    Parameters
    ----------
    damping : float, optional
        Regularization of the forces (``damping * ||f||²``); larger values
        give a smoother surface that no longer passes through the data.
        None or 0 interpolates.

    Attributes
    ----------
    force_coords_ : tuple of numpy.ndarray
        The data points, where the forces sit.
    forces_ : numpy.ndarray
        The force at each point.

    """

    def __init__(self, damping: Optional[float] = None) -> None:
        """Store the damping."""
        self.damping = damping

    def _normalize(self, easting, northing):
        return (easting - self.center_[0]) / self.scale_, (
            northing - self.center_[1]
        ) / self.scale_

    def _jacobian(self, easting, northing) -> np.ndarray:
        e, n = self._normalize(easting, northing)
        fe, fn = self._normalize(*self.force_coords_)
        return _green(np.hypot(e[:, None] - fe[None, :], n[:, None] - fn[None, :]))

    def _fit(self, easting, northing, data, weights) -> None:
        from scipy.linalg import lstsq

        self.center_ = (float(easting.mean()), float(northing.mean()))
        span = max(np.ptp(easting), np.ptp(northing))
        self.scale_ = float(span) if span > 0 else 1.0
        self.force_coords_ = (easting.copy(), northing.copy())
        jac = self._jacobian(easting, northing)
        sw = np.sqrt(weights)
        a, b = jac * sw[:, None], data * sw
        if self.damping:
            a = np.vstack([a, np.sqrt(self.damping) * np.eye(easting.size)])
            b = np.concatenate([b, np.zeros(easting.size)])
        self.forces_ = lstsq(a, b, lapack_driver="gelsd")[0]

    def _predict(self, easting, northing) -> np.ndarray:
        out = np.empty(easting.size)
        for start in range(0, easting.size, BLOCK):
            stop = start + BLOCK
            out[start:stop] = (
                self._jacobian(easting[start:stop], northing[start:stop]) @ self.forces_
            )
        return out


class Nearest(BaseEstimator):
    """Nearest-neighbor gridder: the value of the closest data point."""

    def __init__(self) -> None:
        """Take no parameters."""

    def _fit(self, easting, northing, data, weights) -> None:  # noqa: ARG002
        from scipy.spatial import cKDTree

        self.tree_ = cKDTree(np.column_stack([easting, northing]))
        self.values_ = data.copy()

    def _predict(self, easting, northing) -> np.ndarray:
        _, index = self.tree_.query(np.column_stack([easting, northing]))
        return self.values_[index]
