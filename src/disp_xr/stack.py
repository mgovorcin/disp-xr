"""Chain DISP ministacks into one displacement series from the first date.

Each DISP granule's displacement is relative to its ministack's reference date.
For an epoch ``t`` in ministack ``k`` with reference date ``r_k``::

    cumulative(t) = displacement(t) + cumulative(r_k),    cumulative(r_0) = 0

``cumulative(r_k)`` is read from the epoch whose secondary date is ``r_k`` (the
"crossover" granule of the previous ministack), so every reference date after
the first must also be an epoch of the stack.

``NaNPolicy`` and the chaining idea are adapted from opera-utils
(src/opera_utils/disp/_rebase.py, commit dff4b45), Copyright (c) 2022 California
Institute of Technology, BSD-3-Clause. See THIRD_PARTY_NOTICES.md. Changes: the
offset comes from the crossover epoch by date (opera-utils assumes it is the
previous epoch), a missing crossover raises, and the input dtype is kept.
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Sequence
from datetime import datetime
from enum import Enum
from typing import Optional, Union

import numpy as np
import pandas as pd
import xarray as xr

from ._history import recorded
from .log.logging import log_runtime

logger = logging.getLogger(__name__)

__all__ = ["MissingCrossoverError", "NaNPolicy", "stack_disp"]

DISPLACEMENT_LAYERS = ("displacement", "short_wavelength_displacement")


class NaNPolicy(str, Enum):
    """How a NaN at a crossover epoch affects later ministacks.

    ``propagate`` (default): later epochs of that pixel become NaN, since their
    offset is unknown. ``omit``: the missing offset is taken as zero, which
    assumes no motion over the missing interval.
    """

    propagate = "propagate"
    omit = "omit"

    def __str__(self) -> str:
        return self.value


class MissingCrossoverError(ValueError):
    """A reference date has no epoch to take its offset from."""


def _check_crossovers(times: np.ndarray, refs: np.ndarray) -> None:
    first = refs.min()
    missing = sorted({r for r in refs if r != first and r not in set(times)})
    if missing:
        raise MissingCrossoverError(
            "No epoch at reference dates "
            f"{[str(pd.Timestamp(m).date()) for m in missing]}: the granule that "
            "links each ministack to the previous one is missing"
        )


def _chain(
    raw: np.ndarray, times: np.ndarray, refs: np.ndarray, nan_policy: str
) -> np.ndarray:
    """Chain a block with time on the last axis (the apply_ufunc layout)."""
    out = np.empty_like(raw)
    later_refs = set(refs) - {refs.min()}
    offset = {refs.min(): np.zeros(raw.shape[:-1], dtype=raw.dtype)}
    for i in np.argsort(times, kind="stable"):
        out[..., i] = raw[..., i] + offset[refs[i]]
        if times[i] in later_refs:
            value = out[..., i]
            offset[times[i]] = (
                np.nan_to_num(value) if nan_policy == NaNPolicy.omit else value
            )
    return out


@log_runtime
@recorded
def stack_disp(
    ds: xr.Dataset,
    *,
    start: Optional[Union[str, datetime, pd.Timestamp]] = None,
    end: Optional[Union[str, datetime, pd.Timestamp]] = None,
    nan_policy: Union[str, NaNPolicy] = NaNPolicy.propagate,
    zero_epoch: bool = True,
) -> xr.Dataset:
    """Chain ministacks so displacement is relative to one reference date.

    Parameters
    ----------
    ds : xr.Dataset
        Output of :func:`disp_xr.open_disp`: ``time`` and ``reference_time``
        coordinates, displacement relative to each granule's reference date.
    start : date-like, optional
        Chain from this ministack reference date; earlier ministacks are
        dropped. Must be one of the reference dates.
    end : date-like, optional
        Drop epochs after this date.
    nan_policy : {"propagate", "omit"}
        See :class:`NaNPolicy`.
    zero_epoch : bool
        Prepend the reference date as an epoch with zero displacement. Other
        layers get their no-data value there (NaN for floats, ``_FillValue``
        for integers), so their dtype is kept.

    Returns
    -------
    xr.Dataset
        Same variables; ``displacement`` and ``short_wavelength_displacement``
        relative to the first reference date, with the whole time axis in one
        chunk. ``attrs["reference_date"]`` records that date.

    Raises
    ------
    MissingCrossoverError
        If a reference date after the first is not an epoch of ``ds``.
    ValueError
        If ``start`` is not a reference date.

    """
    nan_policy = NaNPolicy(nan_policy)
    refs_all = np.unique(ds["reference_time"].values)
    if start is not None:
        start = np.datetime64(pd.Timestamp(start), "ns")
        if start not in refs_all:
            raise ValueError(
                f"start must be a ministack reference date; choose one of "
                f"{[str(pd.Timestamp(r).date()) for r in refs_all]}"
            )
        ds = ds.sel(time=ds["reference_time"].values >= start)
    if end is not None:
        ds = ds.sel(time=ds["time"].values <= np.datetime64(pd.Timestamp(end), "ns"))

    times = ds["time"].values
    refs = ds["reference_time"].values
    _check_crossovers(times, refs)
    first = refs.min()

    chained = ds.copy()
    for name in DISPLACEMENT_LAYERS:
        if name not in ds:
            continue
        da = ds[name].chunk({"time": -1})
        chained[name] = xr.apply_ufunc(
            _chain,
            da,
            input_core_dims=[["time"]],
            output_core_dims=[["time"]],
            kwargs={"times": times, "refs": refs, "nan_policy": nan_policy},
            dask="parallelized",
            output_dtypes=[da.dtype],
        ).transpose(*da.dims)
        chained[name].attrs = ds[name].attrs

    if zero_epoch:
        chained = _prepend_zero_epoch(chained, first)
    chained.attrs["reference_date"] = str(pd.Timestamp(first))
    return chained


def _prepend_zero_epoch(ds: xr.Dataset, first: np.datetime64) -> xr.Dataset:
    fill = {}
    for name, var in ds.data_vars.items():
        if np.issubdtype(var.dtype, np.integer):
            fill[name] = var.attrs.get("_FillValue", var.encoding.get("_FillValue", 0))
        else:
            fill[name] = np.nan
    new_times = np.concatenate([[first], ds["time"].values])
    out = ds.reindex(time=new_times, fill_value=fill)
    refs = out["reference_time"].values.copy()
    refs[0] = first
    out = out.assign_coords(reference_time=("time", refs))
    for name in DISPLACEMENT_LAYERS:
        if name in out:
            zero = xr.zeros_like(out[name].isel(time=0))
            out[name] = xr.concat(
                [zero.expand_dims(time=[first]), out[name].isel(time=slice(1, None))],
                dim="time",
            ).chunk({"time": -1})
            out[name].attrs = ds[name].attrs
    return out


# --- deprecated entry points (removed in plan task T38) -----------------------


def combine_disp_product(
    disp_df: pd.DataFrame,
    chunks: Optional[dict] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> xr.Dataset:
    """Open and chain the granules listed by :func:`disp_xr.product.get_disp_info`.

    .. deprecated::
        Use ``stack_disp(open_disp(paths))``.

    ``start_date`` keeps ministacks whose reference date is on or after it,
    ``end_date`` those whose reference date is on or before it.
    """
    warnings.warn(
        "combine_disp_product is deprecated; use stack_disp(open_disp(paths))",
        DeprecationWarning,
        stacklevel=2,
    )
    from .open import open_disp

    ds = open_disp(list(disp_df["path"]), chunks=chunks or "timeseries")
    refs = pd.DatetimeIndex(np.unique(ds["reference_time"].values))
    start = refs[refs >= pd.Timestamp(start_date)].min() if start_date else None
    if end_date:
        keep = ds["reference_time"].values <= np.datetime64(pd.Timestamp(end_date))
        ds = ds.sel(time=keep)
    return stack_disp(ds, start=start)


def rebase_disp(
    disp_df: pd.DataFrame,
    chunks: Optional[dict] = None,
    add_reference_time: bool = False,
) -> xr.Dataset:
    """Open and chain granules (``timeseries_fitting`` branch API).

    .. deprecated::
        Use ``stack_disp(open_disp(paths), zero_epoch=...)``.
    """
    warnings.warn(
        "rebase_disp is deprecated; use stack_disp(open_disp(paths))",
        DeprecationWarning,
        stacklevel=2,
    )
    from .open import open_disp

    ds = open_disp(list(disp_df["path"]), chunks=chunks or "timeseries")
    return stack_disp(ds, zero_epoch=add_reference_time)


def create_rebased_displacement(
    da_displacement: xr.DataArray,
    reference_datetimes: Sequence[Union[datetime, pd.Timestamp]],
    process_chunk_size: tuple[int, int] = (512, 512),  # noqa: ARG001
    add_reference_time: bool = False,
    nan_policy: Union[str, NaNPolicy] = NaNPolicy.propagate,
) -> xr.DataArray:
    """Chain one displacement DataArray.

    .. deprecated::
        Use :func:`stack_disp` on a dataset with a ``reference_time`` coordinate.
        ``process_chunk_size`` is ignored.
    """
    warnings.warn(
        "create_rebased_displacement is deprecated; use stack_disp",
        DeprecationWarning,
        stacklevel=2,
    )
    refs = np.asarray(pd.DatetimeIndex(reference_datetimes).tz_localize(None))
    ds = da_displacement.to_dataset(name="displacement").assign_coords(
        reference_time=("time", refs.astype("datetime64[ns]"))
    )
    out = stack_disp(ds, nan_policy=nan_policy, zero_epoch=add_reference_time)
    return out["displacement"].rename(da_displacement.name)
