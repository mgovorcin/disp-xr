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
from .gnss_tie import GNSSTie, TieResult, apply_tie, cross_validate_tie, fit_tie
from .gridding import Nearest, Spline
from .trend import Trend, deramp, polynomial_terms

__all__ = [
    "BaseEstimator",
    "Chain",
    "GNSSTie",
    "Mean",
    "Nearest",
    "NotFittedError",
    "Spline",
    "TieResult",
    "Trend",
    "apply_tie",
    "clone",
    "coordinates",
    "cross_val_score",
    "cross_validate_tie",
    "deramp",
    "fit_tie",
    "polynomial_terms",
]
