"""Regular time grids, refused gaps and declared breaks (PRD S3, plan task T30).

Rules from the temporal-denoising study (Kiritimati DISP-S1, 2026-09):

- **Grid step.** 15 d where sampling is genuinely 12-day, 30 d for 24-day
  sampling. A 15-day grid is finer than 24-day sampling supports. The step is
  chosen from the 75th percentile of the acquisition intervals rather than
  the median: the study's ``pre`` segment has a 12-day *median* (18 of its
  34 intervals) but 14 intervals of 24 days, and the study recommends 30 d
  for such frames. The step is 15 d per 12 d of that percentile, rounded.
- **Gaps are refused, not interpolated.** A grid point farther than
  ``max_gap_days / 2`` from every acquisition lies inside a hole longer than
  ``max_gap_days`` and is marked; a 15-day grid across ``pre``'s 288-day hole
  would otherwise invent about 19 samples.
- **Breaks are declared** (earthquake catalogs, user input); no automatic
  detection before 1.0. A break at ``tb`` lands on the first epoch at or after
  it: the first observation that already contains the jump. Filters that know
  nothing about discontinuities respect them when run per segment
  (:func:`apply_segmented`).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
import xarray as xr

__all__ = [
    "apply_segmented",
    "break_epochs",
    "gap_mask",
    "grid_step_days",
    "regular_time_grid",
    "split_at_breaks",
]

TimesLike = Union[Sequence, np.ndarray, pd.DatetimeIndex, xr.DataArray]


def _times(times: TimesLike) -> pd.DatetimeIndex:
    values = times.values if isinstance(times, xr.DataArray) else times
    out = pd.DatetimeIndex(pd.to_datetime(np.asarray(values)))
    if not out.is_monotonic_increasing:
        raise ValueError("times must be increasing")
    return out


def grid_step_days(times: TimesLike) -> int:
    """Return the grid step in days for these acquisitions (15 per 12 d of sampling).

    Uses the 75th percentile of the intervals, so a frame with a mix of 12-
    and 24-day intervals gets 30 d (see the module docstring).

    Examples
    --------
    >>> import pandas as pd
    >>> grid_step_days(pd.date_range("2020-01-01", periods=30, freq="12D"))
    15

    """
    t = _times(times)
    if t.size < 2:
        raise ValueError("need at least two acquisitions")
    intervals = np.diff(t.values).astype("timedelta64[s]").astype(np.float64) / 86400.0
    typical = float(np.percentile(intervals, 75))
    return int(15 * max(1, round(typical / 12.0)))


def regular_time_grid(
    times: TimesLike,
    step: Optional[int] = None,
    *,
    start: Optional[Any] = None,
    end: Optional[Any] = None,
) -> pd.DatetimeIndex:
    """Return a regular grid of dates covering the acquisitions.

    Parameters
    ----------
    times : array of datetimes
        Acquisition times, increasing.
    step : int, optional
        Grid step in days; default :func:`grid_step_days`.
    start, end : date-like, optional
        Grid bounds; default the first and last acquisition days.

    Returns
    -------
    pandas.DatetimeIndex
        Midnight dates from ``start`` every ``step`` days, not beyond ``end``.

    Examples
    --------
    >>> import pandas as pd
    >>> times = pd.date_range("2020-01-01", periods=30, freq="12D")
    >>> grid = regular_time_grid(times)
    >>> int((grid[1] - grid[0]).days)
    15

    """
    t = _times(times)
    step = grid_step_days(t) if step is None else int(step)
    if step <= 0:
        raise ValueError(f"step must be positive, not {step}")
    first = pd.Timestamp(start if start is not None else t[0]).normalize()
    last = pd.Timestamp(end if end is not None else t[-1])
    return pd.date_range(first, last, freq=f"{step}D")


def gap_mask(
    times: TimesLike,
    grid: TimesLike,
    max_gap_days: Optional[float] = None,
) -> np.ndarray:
    """Return True for grid dates inside a refused gap.

    A grid date farther than ``max_gap_days / 2`` from every acquisition lies
    in a hole longer than ``max_gap_days``; resampled values there must be
    NaN, never interpolated.

    Parameters
    ----------
    times : array of datetimes
        Acquisition times.
    grid : array of datetimes
        Grid dates (for example :func:`regular_time_grid`).
    max_gap_days : float, optional
        Longest hole to bridge; default four times the typical (75th
        percentile) interval, so a missed acquisition or two is bridged and a
        season-long hole is not.

    Returns
    -------
    numpy.ndarray
        Boolean, one per grid date.

    Examples
    --------
    >>> import pandas as pd
    >>> times = pd.to_datetime(["2020-01-01", "2020-01-13", "2020-09-01"])
    >>> grid = pd.date_range("2020-01-01", "2020-09-01", freq="30D")
    >>> gap_mask(times, grid, max_gap_days=60).sum()  # doctest: +SKIP

    """
    t = _times(times)
    g = _times(grid)
    if max_gap_days is None:
        intervals = np.diff(t.values).astype("timedelta64[s]").astype(float) / 86400.0
        max_gap_days = 4.0 * float(np.percentile(intervals, 75))
    tn = t.values.astype("datetime64[s]").astype(np.int64)
    gn = g.values.astype("datetime64[s]").astype(np.int64)
    after = np.clip(np.searchsorted(tn, gn), 0, tn.size - 1)
    before = np.clip(after - 1, 0, tn.size - 1)
    nearest = np.minimum(np.abs(tn[after] - gn), np.abs(gn - tn[before])) / 86400.0
    return nearest > max_gap_days / 2.0


def break_epochs(times: TimesLike, breaks: Optional[Sequence[Any]]) -> np.ndarray:
    """Return True at the first epoch at or after each break (inside the series)."""
    t = _times(times)
    mask = np.zeros(t.size, dtype=bool)
    if breaks is None:
        return mask
    listed = breaks if pd.api.types.is_list_like(breaks) else [breaks]
    for tb in pd.to_datetime(list(listed)).values:
        idx = int(np.searchsorted(t.values, tb, side="left"))
        if 0 < idx < t.size:
            mask[idx] = True
    return mask


def split_at_breaks(
    times: TimesLike, breaks: Optional[Sequence[Any]]
) -> list[np.ndarray]:
    """Return the epoch indices of each segment between breaks.

    Examples
    --------
    >>> import pandas as pd
    >>> times = pd.date_range("2020-01-01", periods=6, freq="12D")
    >>> [len(s) for s in split_at_breaks(times, ["2020-02-01"])]
    [3, 3]

    """
    t = _times(times)
    edges = [0, *np.flatnonzero(break_epochs(t, breaks)).tolist(), t.size]
    return [np.arange(a, b) for a, b in zip(edges, edges[1:]) if b > a]


def apply_segmented(
    fn: Callable,
    data: Union[xr.DataArray, xr.Dataset, np.ndarray],
    breaks: Optional[Sequence[Any]],
    *,
    times: Optional[TimesLike] = None,
    min_len: int = 3,
) -> Any:
    """Run ``fn`` on each segment between breaks independently.

    A filter that knows nothing about discontinuities then never smooths
    across one, at the cost of one-sided support at each segment edge.

    Parameters
    ----------
    fn : callable
        For xarray input, ``fn(segment) -> segment`` on the time slice (same
        shape). For NumPy input, ``fn(t_segment, y_segment)`` with
        ``y`` of shape ``(time, ...)``.
    data : xr.DataArray, xr.Dataset or numpy.ndarray
        Series with ``time`` first (NumPy) or a ``time`` dimension (xarray).
    breaks : sequence of date-like, optional
        Break dates.
    times : array of datetimes, optional
        Acquisition times for NumPy input (xarray uses its ``time``).
    min_len : int
        Segments shorter than this are passed through unchanged: one or two
        epochs carry nothing a smoother could use.

    Returns
    -------
    Same type as ``data``.

    Examples
    --------
    >>> breaks = ["2023-02-06"]
    >>> smooth = apply_segmented(my_filter, ds.displacement, breaks)  # doctest: +SKIP

    """
    if isinstance(data, (xr.DataArray, xr.Dataset)):
        pieces = []
        for idx in split_at_breaks(data["time"], breaks):
            segment = data.isel(time=idx)
            pieces.append(segment if idx.size < min_len else fn(segment))
        return xr.concat(pieces, dim="time")
    if times is None:
        raise ValueError("NumPy input needs times=")
    t = _times(times)
    y = np.asarray(data, dtype=np.float64)
    out = np.empty_like(y)
    for idx in split_at_breaks(t, breaks):
        out[idx] = y[idx] if idx.size < min_len else fn(t[idx], y[idx])
    return out
