"""Quality weights from the DISP quality layers (PRD S2, plan task T26).

Each layer maps to a weight between 0 and 1, and the weights combine as a
weighted geometric mean, ``prod(w_i ** (a_i / sum(a)))``:

``phase_similarity``, ``temporal_coherence`` (good near 1)
    Linear stretch ``clip((v - floor) / (1 - floor), 0, 1)``: values at or
    below the floor get weight 0 and exclude the pixel at that epoch.
``timeseries_inversion_residuals`` (good near 0)
    The per-date sum of absolute residuals, in radians, of dolphin's L1
    time-series inversion over every interferogram that includes the date.
    An unwrapping error of one cycle in one interferogram adds about 2 pi to
    both of its dates, so on real granules the residuals cluster near
    multiples of 2 pi (``residual_cycle_histogram``). The residual splits
    into ``cycles = round(r / 2 pi)`` and the distance to that lattice
    ``frac = |r / 2 pi - cycles|`` (0 to 0.5). The weight is
    ``penalty ** cycles * exp(-frac / f0)``, clamped to ``min_residual`` so a
    cycle jump down-weights a pixel but never removes it (PRD decision). A
    ``frac`` near 0.5 is ambiguous between two cycle counts and is flagged.

The per-epoch weight ``(time, y, x)`` combines the per-epoch layer weights;
the static weight ``(y, x)`` combines their time means, with the residual
term replaced by one minus the share of dates that have a cycle jump.

Defaults for the floors, ``penalty``, ``f0`` and ``min_residual`` are
provisional until the trade study (plan task T29).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import xarray as xr

from ._history import recorded

__all__ = [
    "network_degree",
    "quality_weights",
    "residual_cycle_histogram",
    "residual_cycles",
]

TWO_PI = 2.0 * np.pi
STRETCH_LAYERS = ("phase_similarity", "temporal_coherence")
RESIDUAL_LAYER = "timeseries_inversion_residuals"
DEFAULT_LAYERS = (*STRETCH_LAYERS, RESIDUAL_LAYER)
# Provisional (T29). The coherence and similarity floors follow the
# processor's recommended-mask thresholds when the granules record them.
DEFAULT_FLOORS = {"phase_similarity": 0.3, "temporal_coherence": 0.6}
# Half-cycle ambiguity: frac within this of 0.5.
AMBIGUOUS_WIDTH = 0.1


def residual_cycles(residuals: xr.DataArray) -> xr.Dataset:
    """Split inversion residuals into 2 pi cycles and the distance to them.

    Returns
    -------
    xr.Dataset
        ``cycles`` (integer count of 2 pi), ``frac`` (0 to 0.5, distance to
        the nearest whole cycle) and ``ambiguous`` (``frac`` within
        ``AMBIGUOUS_WIDTH`` of 0.5: the cycle count could be either).

    Examples
    --------
    >>> r = xr.DataArray([0.1, 2 * np.pi + 0.2, 3 * np.pi], dims="time")
    >>> residual_cycles(r).cycles.values.tolist()
    [0, 1, 2]

    """
    turns = residuals / TWO_PI
    # The method, not np.round: numpy's round passes `out=`, which xarray does
    # not take, and falls back to np.asarray, loading a dask array whole.
    cycles = turns.round()
    frac = abs(turns - cycles)
    return xr.Dataset(
        {
            "cycles": cycles.fillna(0).astype(np.int16),
            "frac": frac,
            "ambiguous": abs(frac - 0.5) < AMBIGUOUS_WIDTH,
        }
    )


def max_bandwidth_from_granule(path: Union[str, Path]) -> Optional[int]:
    """Read ``interferogram_network.max_bandwidth`` from a granule's dolphin config.

    Returns None if the config is missing or the network is not
    bandwidth-limited.
    """
    import yaml

    from ._remote import open_h5

    with open_h5(path) as f:
        if "metadata/dolphin_workflow_config" not in f:
            return None
        raw = f["metadata/dolphin_workflow_config"][()]
    config = yaml.safe_load(raw.decode() if isinstance(raw, bytes) else raw)
    network = (config or {}).get("interferogram_network") or {}
    value = network.get("max_bandwidth")
    return int(value) if value is not None else None


def network_degree(ds: xr.Dataset, max_bandwidth: int) -> xr.DataArray:
    """Return the interferogram count of each epoch's date in its ministack.

    In a bandwidth-limited network each date pairs with up to
    ``max_bandwidth`` neighbours on each side within its ministack, whose
    first date is the (compressed) reference. A secondary at position ``p``
    of ``N`` dates has ``min(p, b) + min(N - 1 - p, b)`` interferograms.
    """
    refs = pd.Series(ds["reference_time"].values)
    degree = np.zeros(len(refs), dtype=np.int16)
    for _, idx in refs.groupby(refs).groups.items():
        n = len(idx) + 1  # the reference date plus the secondaries
        for p, i in enumerate(sorted(idx, key=lambda j: ds["time"].values[j]), start=1):
            degree[i] = min(p, max_bandwidth) + min(n - 1 - p, max_bandwidth)
    return xr.DataArray(
        degree, dims="time", coords={"time": ds["time"]}, name="network_degree"
    )


def _stretch(layer: xr.DataArray, floor: float) -> xr.DataArray:
    return ((layer - floor) / (1.0 - floor)).clip(0.0, 1.0)


def _floor_for(ds: xr.Dataset, name: str, floors: Mapping[str, float]) -> float:
    if name in floors:
        return float(floors[name])
    attrs = ds["recommended_mask"].attrs if "recommended_mask" in ds else {}
    recorded_threshold = {
        "temporal_coherence": attrs.get("temporal_coherence_threshold"),
        "phase_similarity": attrs.get("similarity_threshold"),
    }.get(name)
    if recorded_threshold is not None:
        return float(recorded_threshold)
    return DEFAULT_FLOORS[name]


def _geometric_mean(weights: dict[str, xr.DataArray], exponents: Mapping[str, float]):
    total = sum(exponents[name] for name in weights)
    out = None
    for name, w in weights.items():
        term = w ** (exponents[name] / total)
        out = term if out is None else out * term
    return out


@recorded
def quality_weights(
    ds: xr.Dataset,
    layers: Sequence[str] = DEFAULT_LAYERS,
    *,
    exponents: Optional[Mapping[str, float]] = None,
    floors: Optional[Mapping[str, float]] = None,
    penalty: float = 0.5,
    f0: float = 0.25,
    min_residual: float = 0.05,
) -> xr.Dataset:
    """Combine quality layers into pixel weights between 0 and 1.

    Parameters
    ----------
    ds : xr.Dataset
        DISP data with the chosen layers (as from :func:`disp_xr.open_disp`),
        chained or not.
    layers : sequence of str
        Any of ``phase_similarity``, ``temporal_coherence``,
        ``timeseries_inversion_residuals``.
    exponents : mapping, optional
        Relative weight of each layer in the geometric mean (default 1 each).
    floors : mapping, optional
        Stretch floor per similarity or coherence layer. Default: the
        processor's thresholds recorded on ``recommended_mask``, else 0.3 for
        similarity and 0.6 for coherence.
    penalty : float
        Residual weight factor per 2 pi cycle jump.
    f0 : float
        Scale of the residual weight's decay with the distance to the nearest
        whole cycle.
    min_residual : float
        Lowest residual weight: cycle jumps down-weight, never exclude.

    Returns
    -------
    xr.Dataset
        ``weight`` (time, y, x) and ``static_weight`` (y, x), one
        ``weight_<layer>`` per layer, and for the residuals ``cycles``,
        ``ambiguous`` and ``jump_fraction`` (share of dates with a jump).
        Lazy if the input is.

    Examples
    --------
    >>> w = dxr.quality_weights(ds)  # doctest: +SKIP
    >>> fit = dxr.fit_trajectory(ds, weights=w.weight)  # doctest: +SKIP

    """
    unknown = set(layers) - set(DEFAULT_LAYERS)
    if unknown:
        raise ValueError(
            f"Unknown weight layers {sorted(unknown)}; use {DEFAULT_LAYERS}"
        )
    missing = [name for name in layers if name not in ds]
    if missing:
        raise ValueError(f"The dataset has no {missing} layer")
    if not 0 < min_residual <= 1:
        raise ValueError("min_residual must be in (0, 1]: cycle jumps never exclude")
    exponents = dict.fromkeys(layers, 1.0) | dict(exponents or {})
    floors = dict(floors or {})
    epoch: dict[str, xr.DataArray] = {}
    static: dict[str, xr.DataArray] = {}
    extra: dict[str, xr.DataArray] = {}
    for name in layers:
        if name in STRETCH_LAYERS:
            w = _stretch(ds[name], _floor_for(ds, name, floors))
            epoch[name] = w
            static[name] = w.mean("time") if "time" in w.dims else w
        else:
            parts = residual_cycles(ds[name])
            w = (penalty**parts.cycles) * np.exp(-parts.frac / f0)
            w = w.clip(min_residual, 1.0).where(ds[name].notnull())
            epoch[name] = w
            jumps = (parts.cycles > 0).where(ds[name].notnull())
            jump_fraction = jumps.mean("time")
            static[name] = (1.0 - jump_fraction).clip(min_residual, 1.0)
            extra["cycles"] = parts.cycles
            extra["ambiguous"] = parts.ambiguous
            extra["jump_fraction"] = jump_fraction
    out = xr.Dataset(
        {
            "weight": _geometric_mean(epoch, exponents),
            "static_weight": _geometric_mean(static, exponents),
            **{f"weight_{name}": w for name, w in epoch.items()},
            **extra,
        }
    )
    out = out.drop_vars(
        [c for c in out.coords if c not in ("time", "y", "x", "spatial_ref")]
    )
    out.attrs = {
        "layers": list(layers),
        "exponents": str(exponents),
        "penalty": penalty,
        "f0": f0,
        "min_residual": min_residual,
    }
    return out


def residual_cycle_histogram(
    ds: xr.Dataset, bins: Optional[np.ndarray] = None, min_residual: float = 0.05
) -> pd.Series:
    """Histogram of the inversion residuals in units of 2 pi.

    On real DISP-S1 granules the counts peak near whole numbers (one 2 pi
    unwrapping error per interferogram), which the cycle decomposition of
    :func:`quality_weights` assumes. Residuals below ``min_residual`` radians
    (clean pixels) are left out so the peaks show.

    Returns
    -------
    pd.Series
        Counts indexed by the lower bin edge, in cycles of 2 pi.

    """
    bins = np.arange(0.0, 4.01, 0.25) if bins is None else np.asarray(bins)
    turns = ds[RESIDUAL_LAYER] / TWO_PI
    values = turns.where(ds[RESIDUAL_LAYER] > min_residual)
    counts, edges = np.histogram(np.asarray(values).ravel(), bins=bins)
    return pd.Series(counts, index=edges[:-1], name="count")
