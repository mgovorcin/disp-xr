"""Quality-layer statistics and masks for DISP stacks."""

import json
import logging
import warnings
from collections.abc import Sequence
from functools import reduce
from typing import Any, Callable, Optional

import numpy as np
import xarray as xr

from ._history import recorded

logger = logging.getLogger(__name__)


def func_stat(mode: str) -> Callable:
    """Return a statistical operation based on the specified mode.

    Parameters
    ----------
    mode : str
        The statistical operation to perform. Example values include 'mean',
        'median', 'max', 'min', 'std', 'var', and 'sum'.

    Returns
    -------
    callable
        A lambda function that performs the corresponding statistical operation
        on an xarray object.

    """
    kwargs = {"skipna": True, "keepdims": True}
    # Mapping mode to corresponding statistical operation
    operations = {
        "mean": lambda x: x.mean(dim="time", **kwargs),
        "median": lambda x: x.median(dim="time", **kwargs),
        "max": lambda x: x.max(dim="time", **kwargs),
        "min": lambda x: x.min(dim="time", **kwargs),
        "std": lambda x: x.std(dim="time", **kwargs),
        "var": lambda x: x.var(dim="time", **kwargs),
        "sum": lambda x: x.sum(dim="time", **kwargs),
    }

    # Check if the mode is valid and return the corresponding function
    try:
        return operations[mode]
    except KeyError:
        raise ValueError(
            (
                f"Unsupported mode: '{mode}'."
                " Valid options are 'mean', 'median', 'max',"
                " 'min', 'std', 'var', 'sum'"
            )
        )


def get_stack_stat(stack_xr: xr.DataArray, mode: str = "mean") -> xr.DataArray:
    """Compute a statistical operation (mean, median, etc.).

    Parameters
    ----------
    stack_xr : xr.DataArray
        The xarray DataArray on which the statistical operation..
    mode : str, optional
        The statistical operation to perform. Default is 'mean'.
        Options: 'median', 'max', 'min', 'std', 'var', and 'sum'.

    Returns
    -------
    xr.DataArray
        The result of the statistical operation, with the operation applied along the
        'time' dimension and any NaN values handled.

    """
    if not isinstance(stack_xr, xr.DataArray):
        raise TypeError("Input must be an xarray Dataset.")

    logger.info(f"Get {mode} of {stack_xr.name}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        warnings.filterwarnings("ignore", message="All-NaN slice encountered")
        data = func_stat(mode)(stack_xr)
    return data.squeeze()


# Map_block template


def get_value_percentage(
    stack_xr: xr.DataArray, value: int, reverse: bool = False
) -> xr.DataArray:
    """Return, per pixel, the percentage of epochs equal to ``value``.

    Parameters
    ----------
    stack_xr : xr.DataArray
        A ``(time, y, x)`` layer.
    value : int
        The value to count.
    reverse : bool, optional
        Count epochs *not* equal to ``value`` instead.

    Returns
    -------
    xr.DataArray
        Lazy ``(y, x)`` percentage of all epochs.

    """
    if not isinstance(stack_xr, xr.DataArray):
        raise TypeError("Input must be an xarray DataArray.")
    logger.info(f"Get percentage of {stack_xr.name}")
    hits = (stack_xr != value) if reverse else (stack_xr == value)
    return hits.sum("time") / stack_xr.sizes["time"] * 100


# --- masks and summaries (plan task T10) ---------------------------------------

DISPLACEMENT_LAYERS = ("displacement", "short_wavelength_displacement")

# Value of a good pixel in the product's 0/1 layers.
_GOOD = {"recommended_mask": 1, "water_mask": 1}


def combine_quality_masks(
    quality_datasets: Sequence[xr.DataArray],
    thresholds: Sequence[float],
    reduction_func: Callable = np.logical_or,
) -> xr.DataArray:
    """Combine thresholded quality layers into one boolean mask.

    Each layer is compared with ``> threshold``, then the results are reduced
    with ``reduction_func`` (``np.logical_or``: any layer passes;
    ``np.logical_and``: all pass).

    Ported from opera-utils (src/opera_utils/disp/_reformat.py, commit
    97f77d9), Copyright (c) 2022 California Institute of Technology,
    BSD-3-Clause. See THIRD_PARTY_NOTICES.md.

    Examples
    --------
    >>> import xarray as xr
    >>> a = xr.DataArray([0.3, 0.7], dims="x")
    >>> b = xr.DataArray([0.6, 0.4], dims="x")
    >>> combine_quality_masks([a, b], [0.5, 0.5]).values.tolist()
    [True, True]

    """
    return reduce(
        reduction_func,
        [qd > t for qd, t in zip(quality_datasets, thresholds, strict=True)],
    )


def _nodata(da: xr.DataArray) -> Any:
    return da.attrs.get("_FillValue", da.encoding.get("_FillValue"))


def _has_data(da: xr.DataArray) -> xr.DataArray:
    if np.issubdtype(da.dtype, np.floating):
        return da.notnull()
    fill = _nodata(da)
    return da != fill if fill is not None else xr.ones_like(da, dtype=bool)


def _criteria_for(ds: xr.Dataset, preset: Optional[str]) -> dict[str, Any]:
    if preset is None:
        return {}
    if preset == "recommended":
        return {"recommended": True}
    if preset == "strict":
        attrs = ds["recommended_mask"].attrs if "recommended_mask" in ds else {}
        return {
            "recommended": True,
            "connected_components": True,
            "water": True,
            "temporal_coherence": float(attrs.get("temporal_coherence_threshold", 0.6)),
        }
    raise ValueError(f"Unknown preset {preset!r}; use 'recommended' or 'strict'")


def _require(ds: xr.Dataset, name: str) -> xr.DataArray:
    if name not in ds:
        raise ValueError(f"Criterion needs the {name!r} layer, which is not loaded")
    return ds[name]


def valid_mask(
    ds: xr.Dataset,
    preset: Optional[str] = "recommended",
    *,
    recommended: Optional[bool] = None,
    temporal_coherence: Optional[float] = None,
    phase_similarity: Optional[float] = None,
    connected_components: Optional[bool] = None,
    water: Optional[bool] = None,
) -> xr.DataArray:
    """Return True where a pixel passes every chosen quality criterion.

    Presets: ``"recommended"`` uses the product's ``recommended_mask``;
    ``"strict"`` adds connected components, water and ``temporal_coherence >=``
    the threshold stored on ``recommended_mask`` (0.6 if absent). Keyword
    arguments add to or override the preset; ``False`` turns a criterion off.
    A pixel at an integer layer's no-data value fails.

    Raises
    ------
    ValueError
        If no criterion is chosen, or a needed layer is not in ``ds``.

    Examples
    --------
    >>> good = dxr.valid_mask(ds, "recommended")  # doctest: +SKIP
    >>> good = dxr.valid_mask(ds, None, temporal_coherence=0.7)  # doctest: +SKIP

    """
    chosen = _criteria_for(ds, preset)
    for key, value in {
        "recommended": recommended,
        "temporal_coherence": temporal_coherence,
        "phase_similarity": phase_similarity,
        "connected_components": connected_components,
        "water": water,
    }.items():
        if value is not None:
            chosen[key] = value
    chosen = {k: v for k, v in chosen.items() if v is not False}
    if not chosen:
        raise ValueError("No quality criterion chosen")

    passes = []
    if chosen.get("recommended"):
        passes.append(_require(ds, "recommended_mask") == _GOOD["recommended_mask"])
    if chosen.get("water"):
        passes.append(_require(ds, "water_mask") == _GOOD["water_mask"])
    if chosen.get("connected_components"):
        labels = _require(ds, "connected_component_labels")
        passes.append((labels != 0) & _has_data(labels))
    for name in ("temporal_coherence", "phase_similarity"):
        if name in chosen:
            passes.append(_require(ds, name) >= chosen[name])
    out = reduce(np.logical_and, passes)
    out.attrs = {"criteria": json.dumps(chosen, sort_keys=True)}
    return out.rename("valid")


@recorded
def mask(
    ds: xr.Dataset,
    preset: Optional[str] = "recommended",
    **criteria: Any,
) -> xr.Dataset:
    """Set displacement to NaN where a pixel fails the quality criteria.

    Only ``displacement`` and ``short_wavelength_displacement`` are masked;
    quality layers are kept as they are. See :func:`valid_mask` for presets and
    criteria. Every layer is per epoch, so each epoch is masked by its own
    granule's quality. Works before or after chaining; the zero epoch that
    :func:`disp_xr.stack_disp` prepends has no quality layers and stays 0.

    Examples
    --------
    >>> masked = dxr.mask(ds, "strict", temporal_coherence=0.7)  # doctest: +SKIP

    """
    valid = valid_mask(ds, preset, **criteria)
    keep = valid | (ds["time"] == ds["reference_time"])
    out = ds.copy()
    for name in DISPLACEMENT_LAYERS:
        if name in ds:
            out[name] = ds[name].where(keep)
            out[name].attrs = ds[name].attrs
    out.attrs["mask_criteria"] = valid.attrs["criteria"]
    return out


@recorded
def quality_summary(
    ds: xr.Dataset,
    *,
    reliability_threshold: float = 0.9,
    by_ministack: bool = False,
) -> xr.Dataset:
    """Summarize quality per pixel over time (lazy).

    Percentages divide by the epochs where the layer has data, so a stacked
    zero epoch or missing granules do not count as bad.

    Returns
    -------
    xr.Dataset
        ``pct_masked`` (% of epochs with ``recommended_mask == 0``),
        ``pct_ps`` (% of epochs flagged persistent scatterer),
        ``temporal_coherence_hmean`` (harmonic mean, which down-weights low
        coherence periods), ``reliability`` (fraction of epochs with
        ``recommended_mask == 1``) and ``reliable``
        (``reliability >= reliability_threshold``). With ``by_ministack``,
        each is computed per ``reference_time`` group (flox groupby).

    Examples
    --------
    >>> summary = dxr.quality_summary(ds).compute()  # doctest: +SKIP
    >>> summary.reliable.plot.imshow()  # doctest: +SKIP
    >>> per_ministack = dxr.quality_summary(ds, by_ministack=True)  # doctest: +SKIP

    """

    def total(da: xr.DataArray) -> xr.DataArray:
        if by_ministack:
            return da.groupby("reference_time").sum()
        return da.sum("time")

    out = {}
    if "recommended_mask" in ds:
        rm = ds["recommended_mask"]
        n = total(_has_data(rm))
        good = total(rm == _GOOD["recommended_mask"])
        out["pct_masked"] = (n - good) / n * 100
        out["reliability"] = good / n
        out["reliable"] = out["reliability"] >= reliability_threshold
    if "persistent_scatterer_mask" in ds:
        ps = ds["persistent_scatterer_mask"]
        out["pct_ps"] = total(ps == 1) / total(_has_data(ps)) * 100
    if "temporal_coherence" in ds:
        coh = ds["temporal_coherence"].where(ds["temporal_coherence"] > 0)
        out["temporal_coherence_hmean"] = total(coh.notnull()) / total(1.0 / coh)
    if not out:
        raise ValueError("No quality layers in the dataset")
    return xr.Dataset(out)
