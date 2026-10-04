"""Spatial models with a fit / predict interface (PRD F9).

See :mod:`disp_xr.models.base` for the interface; estimators (trends, the
GNSS tie, gridders) build on :class:`BaseEstimator`.
"""

from .base import (
    BaseEstimator,
    Chain,
    Mean,
    NotFittedError,
    clone,
    coordinates,
    cross_val_score,
)

__all__ = [
    "BaseEstimator",
    "Chain",
    "Mean",
    "NotFittedError",
    "clone",
    "coordinates",
    "cross_val_score",
]
