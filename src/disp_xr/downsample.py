"""Quality-weighted downsampling to a regular coarse grid (PRD S2, plan task T27).

Each output cell aggregates a ``factor x factor`` block of 30 m pixels:

- float layers: the weighted mean (``stat="wmean"``) or weighted median
  (``stat="wmedian"``), with the weighted spread, the effective sample count
  ``n_eff = (Σw)² / Σw²`` and ``sigma = spread / sqrt(n_eff)``;
- integer mask and label layers: the most common value (``"mode"``) or
  whether any pixel is set (``"any"``), never a mean.

Every cell also gets ``n_valid``, ``valid_fraction`` (of the full block, so
edge cells padded beyond the frame show how much they hold) and ``quality``
(mean static weight). Weighted-mean levels keep ``sum_w`` and ``sum_w2`` so a
coarser mean can be rebuilt from them; a weighted median cannot, so every
median level is computed from 30 m. The ``sigma`` treats pixels as
independent, which is far too small at 1 to 10 km; pass a variogram model
(:mod:`disp_xr.residuals`) to also get ``<layer>_sigma_corr``, which accounts
for spatially correlated errors.

Odd factors (33, 67, 333 for 990 m, 2,010 m, 9,990 m) put each cell centre on
a 30 m pixel centre, so the coarse transform is exact. Run after chaining and
re-referencing: averaging series with different reference dates is wrong.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from typing import Literal, Optional, Union

import numpy as np
import pandas as pd
import xarray as xr

from ._chunks import data_chunks
from ._history import recorded
from .config import bounded_threads
from .residuals import VariogramModel, mean_block_correlation, n_eff_correlated

__all__ = ["downsample", "to_points"]

Stat = Literal["wmean", "wmedian"]
Categorical = Literal["mode", "any"]
# Mask and label layers and their default reduction.
CATEGORICAL_DEFAULTS: dict[str, Categorical] = {
    "recommended_mask": "mode",
    "water_mask": "mode",
    "persistent_scatterer_mask": "any",
    "connected_component_labels": "mode",
    "layover_shadow_mask": "any",
}


def _factor(obj: xr.Dataset, factor: Optional[int], spacing: Optional[float]) -> int:
    if (factor is None) == (spacing is None):
        raise ValueError("Give exactly one of factor or spacing")
    if factor is None:
        res = abs(float(obj["x"][1] - obj["x"][0]))
        factor = int(round(spacing / res))  # type: ignore[operator]
        if factor % 2 == 0:
            # An odd factor keeps cell centres on pixel centres.
            factor += 1 if spacing / res >= factor else -1  # type: ignore[operator]
    if factor < 1:
        raise ValueError(f"factor must be a positive integer, not {factor}")
    return int(factor)


def _centres(coord: np.ndarray, factor: int) -> np.ndarray:
    """Block-centre coordinates on the pixel grid (edge blocks extend past it)."""
    step = float(coord[1] - coord[0])
    n_cells = math.ceil(len(coord) / factor)
    return coord[0] + (np.arange(n_cells) * factor + (factor - 1) / 2) * step


def _coarse_coords(obj: xr.Dataset, factor: int) -> dict[str, xr.DataArray]:
    out = {
        d: xr.DataArray(_centres(obj[d].values, factor), dims=d, attrs=obj[d].attrs)
        for d in ("y", "x")
    }
    return out


def _pad(da: xr.DataArray, factor: int, fill) -> xr.DataArray:
    pad = {d: (0, -da.sizes[d] % factor) for d in ("y", "x") if d in da.dims}
    if not any(p[1] for p in pad.values()):
        return da
    return da.pad(pad, mode="constant", constant_values=fill)


def _aligned(da: xr.DataArray, factor: int) -> xr.DataArray:
    """Pad to whole blocks and align dask chunks to them (coarsen needs both)."""
    if da.chunks is None:
        return da
    sizes = dict(zip(da.dims, da.data.chunks))
    chunks = {
        d: max(factor, (sizes[d][0] // factor) * factor)
        for d in ("y", "x")
        if d in da.dims
    }
    return da.chunk(chunks)


def _block_sum(da: xr.DataArray, factor: int) -> xr.DataArray:
    return _aligned(_pad(da, factor, 0), factor).coarsen(y=factor, x=factor).sum()


def _block_reduce(arrays: list, factor: int, func, dtype, fill=np.nan) -> xr.DataArray:
    """Apply ``func`` to (..., cells_y, cells_x, factor * factor) blocks, lazily.

    ``func`` receives one block array per input in ``arrays``. Dask chunks
    are aligned to whole blocks, so each chunk is reduced on its own.
    """
    padded = [_aligned(_pad(a, factor, fill), factor) for a in arrays]
    template = padded[0]
    dims = template.dims

    def blocks(*arrs: np.ndarray) -> np.ndarray:
        out = []
        for arr in arrs:
            *lead, by, bx = arr.shape
            b = arr.reshape(*lead, by // factor, factor, bx // factor, factor)
            b = np.moveaxis(b, -3, -2).reshape(
                *lead, by // factor, bx // factor, factor * factor
            )
            out.append(b)
        return func(*out).astype(dtype)

    order = [d for d in dims if d not in ("y", "x")] + ["y", "x"]
    data = [p.transpose(*order).data for p in padded]
    if isinstance(data[0], np.ndarray):
        result = blocks(*data)
    else:
        import dask.array as da

        # The arrays' own chunks (coordinates may be chunked differently).
        data = [da.asarray(d).rechunk(data[0].chunks) for d in data]
        out_chunks = data[0].chunks[:-2] + tuple(
            tuple(c // factor for c in ch) for ch in data[0].chunks[-2:]
        )
        result = da.map_blocks(blocks, *data, chunks=out_chunks, dtype=dtype)
    coords = {d: template[d] for d in order[:-2] if d in template.coords}
    return xr.DataArray(result, dims=order, coords=coords)


def _weighted_median_blocks(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Return the lower weighted median over the last axis.

    The smallest value whose cumulative weight reaches half the total.
    """
    ok = np.isfinite(values) & (weights > 0)
    values = np.where(ok, values, np.inf)
    weights = np.where(ok, weights, 0.0)
    order = np.argsort(values, axis=-1)
    v = np.take_along_axis(values, order, axis=-1)
    cw = np.cumsum(np.take_along_axis(weights, order, axis=-1), axis=-1)
    total = cw[..., -1:]
    idx = np.argmax(cw >= 0.5 * total, axis=-1)
    out = np.take_along_axis(v, idx[..., None], axis=-1)[..., 0]
    return np.where(total[..., 0] > 0, out, np.nan)


def _mode_blocks(block: np.ndarray, fill) -> np.ndarray:
    """Most common value in each block, ignoring ``fill``; ties take the smallest."""
    *lead, n = block.shape
    flat = block.reshape(-1, n)
    out = np.full(flat.shape[0], fill, dtype=block.dtype)
    for i, row in enumerate(flat):
        row = row[row != fill]
        if row.size:
            vals, counts = np.unique(row, return_counts=True)
            out[i] = vals[np.argmax(counts)]
    return out.reshape(lead)


def _categorical(da: xr.DataArray, factor: int, how: Categorical) -> xr.DataArray:
    fill = da.attrs.get("_FillValue", da.encoding.get("_FillValue"))
    fill = da.dtype.type(fill if fill is not None else 0 if how == "any" else -1)
    if how == "any":
        reduced = _block_reduce(
            [da],
            factor,
            lambda b: ((b != fill) & (b != 0)).any(axis=-1).astype(np.uint8),
            np.uint8,
            fill=fill,
        )
    else:
        reduced = _block_reduce(
            [da], factor, lambda b: _mode_blocks(b, fill), da.dtype, fill=fill
        )
    return reduced.assign_attrs(da.attrs, reduction=how)


def _resolve_weights(
    ds: xr.Dataset, weights: Union[str, xr.DataArray, None]
) -> tuple[xr.DataArray, xr.DataArray]:
    """Return per-element weights and static (y, x) weights."""
    if isinstance(weights, str):
        if weights != "quality":
            raise ValueError(
                f"weights must be 'quality', a DataArray or None, not {weights!r}"
            )
        from .weights import quality_weights

        present = [
            n
            for n in (
                "phase_similarity",
                "temporal_coherence",
                "timeseries_inversion_residuals",
            )
            if n in ds
        ]
        if not present:
            raise ValueError(
                "weights='quality' needs quality layers; pass weights=None "
                "or a DataArray"
            )
        qw = quality_weights(ds, present)
        static = qw.static_weight.fillna(0.0)
        # Epochs without quality layers (the zero epoch of a chained stack)
        # take the static weight.
        return qw.weight.fillna(static), static
    if weights is None:
        template = ds[
            next(n for n, v in ds.data_vars.items() if {"y", "x"} <= set(v.dims))
        ]
        ones = xr.ones_like(
            template.isel(time=0, drop=True) if "time" in template.dims else template,
            dtype=np.float64,
        )
        return ones, ones
    w = weights.fillna(0.0).clip(min=0.0)
    static = w.mean("time") if "time" in w.dims else w
    return w, static


SUMS = ("sum_w", "sum_w2", "sum_wv", "sum_wvv", "n_valid")


def _pixel_sums(v: np.ndarray, w: np.ndarray, reduce: Callable) -> np.ndarray:
    """Return Σw, Σw², Σwv, Σwv² and the valid count per cell, in one pass.

    ``reduce`` sums a pixel array into cells. Stacked on a new first axis in
    the order of ``SUMS``. One pass over ``v`` and ``w`` instead of five
    full-size temporaries is what makes downsampling memory-bound rather
    than memory-starved on a busy node.
    """
    ok = np.isfinite(v)
    w0 = np.where(ok, w, 0.0)
    vz = np.where(ok, v, 0.0)
    wv = w0 * vz
    return np.stack(
        [reduce(w0), reduce(w0 * w0), reduce(wv), reduce(wv * vz), reduce(ok * 1.0)]
    )


def _block_cell_sums(
    v: xr.DataArray, w: xr.DataArray, factor: int
) -> tuple[xr.DataArray, ...]:
    """Per-block Σw, Σw², Σwv, Σwv² and valid count, lazily (see ``SUMS``)."""
    order = [d for d in v.dims if d not in ("y", "x")] + ["y", "x"]
    vp = _aligned(_pad(v.transpose(*order), factor, np.nan), factor)
    wp = _pad(w.transpose(*order), factor, 0.0)

    def reduce(a: np.ndarray) -> np.ndarray:
        *lead, by, bx = a.shape
        return a.reshape(*lead, by // factor, factor, bx // factor, factor).sum(
            axis=(-3, -1)
        )

    def kernel(vb: np.ndarray, wb: np.ndarray) -> np.ndarray:
        return _pixel_sums(vb, wb, reduce)

    if vp.chunks is None:
        stacked = kernel(np.asarray(vp.data), np.asarray(wp.data))
    else:
        import dask.array as da

        data = vp.data
        weights = da.asarray(wp.data).rechunk(data.chunks)
        stacked = da.map_blocks(
            kernel,
            data,
            weights,
            new_axis=0,
            chunks=((len(SUMS),),)
            + data.chunks[:-2]
            + tuple(tuple(c // factor for c in ch) for ch in data.chunks[-2:]),
            dtype=np.float64,
        )
    coords = {d: vp[d] for d in order[:-2] if d in vp.coords}
    return tuple(
        xr.DataArray(stacked[i], dims=order, coords=coords) for i in range(len(SUMS))
    )


def _aggregate_layers(
    ds: xr.Dataset,
    w: xr.DataArray,
    *,
    layers: list[str],
    cell_sums: Callable[[xr.DataArray, xr.DataArray], tuple[xr.DataArray, ...]],
    median: Callable[[xr.DataArray, xr.DataArray], xr.DataArray],
    full: float,
    stat: Stat,
    nan_threshold: float,
    variogram: Mapping[str, VariogramModel],
    rho_bar: Callable[[VariogramModel], float],
) -> tuple[dict[str, xr.DataArray], dict[str, xr.DataArray]]:
    """Per-cell value, sigma and counts of float layers, for any cell shape.

    ``cell_sums(v, w)`` returns the per-cell sums named in ``SUMS``,
    ``median`` returns the weighted
    median per cell, ``full`` is the pixel count of a full cell, and
    ``rho_bar`` the mean pixel correlation of a full cell under a variogram.
    Returns the per-layer variables and those shared by all layers.
    """
    out: dict[str, xr.DataArray] = {}
    shared: dict[str, xr.DataArray] = {}
    for name in layers:
        v = ds[name]
        if v.dtype.kind in "iu":
            fill = v.attrs.get("_FillValue", v.encoding.get("_FillValue"))
            v = v.where(v != fill) if fill is not None else v
        v = v.astype(np.float64)
        wb = w.broadcast_like(v)
        if v.chunks is not None:
            wb = wb.chunk(data_chunks(v))
        sum_w, sum_w2, sum_wv, sum_wvv, n_valid = cell_sums(v, wb)
        n_valid = n_valid.astype(np.int32)
        mean = sum_wv / sum_w.where(sum_w > 0)
        mean_square = sum_wvv / sum_w.where(sum_w > 0)
        # E[v²] - E[v]² cancels; below this relative size it is rounding.
        variance = mean_square - mean**2
        variance = variance.where(variance > 1e-12 * mean_square, 0.0)
        spread = np.sqrt(variance)
        n_eff = sum_w**2 / sum_w2.where(sum_w2 > 0)
        if stat == "wmean":
            value = mean
        else:
            value = median(v, wb.where(v.notnull(), 0.0))
            value = value.drop_vars([c for c in value.coords if c not in value.dims])
        keep = (1.0 - n_valid / full) <= nan_threshold
        keep &= sum_w > 0
        # An integer layer's fill value means nothing once it is averaged.
        attrs = {k: v for k, v in ds[name].attrs.items() if k != "_FillValue"}
        out[name] = value.where(keep).drop_attrs().assign_attrs(attrs, aggregation=stat)
        out[f"{name}_sigma"] = (
            (spread / np.sqrt(n_eff))
            .where(keep)
            .drop_attrs()
            .assign_attrs(
                units=ds[name].attrs.get("units", ""),
                long_name=f"sigma of the cell {stat} (pixels independent)",
            )
        )
        if name in variogram:
            model = variogram[name]
            rho = rho_bar(model)
            n_eff_corr = n_eff_correlated(sum_w, sum_w2, rho).where(keep)
            sigma_corr = np.sqrt(model.sill / n_eff_corr).drop_attrs()
            out[f"{name}_sigma_corr"] = sigma_corr.assign_attrs(
                units=ds[name].attrs.get("units", ""),
                long_name=f"sigma of the cell {stat} (spatially correlated errors)",
                variogram=model.to_json(),
                mean_block_correlation=rho,
            )
            if "n_eff_corr" not in shared:
                shared["n_eff_corr"] = n_eff_corr.drop_attrs().assign_attrs(
                    long_name="effective independent pixels (variogram)"
                )
        if "n_eff" not in shared:
            shared |= {
                "n_eff": (
                    n_eff.where(keep)
                    .drop_attrs()
                    .assign_attrs(
                        long_name="effective sample count, (sum w)^2 / sum w^2"
                    )
                ),
                "n_valid": (
                    n_valid.drop_attrs().assign_attrs(
                        long_name="valid pixels in the cell"
                    )
                ),
                "valid_fraction": (
                    (n_valid / full)
                    .drop_attrs()
                    .assign_attrs(long_name="valid share of the full cell")
                ),
            }
            if stat == "wmean":
                shared["sum_w"] = sum_w.drop_attrs().assign_attrs(
                    long_name="sum of weights"
                )
                shared["sum_w2"] = sum_w2.drop_attrs().assign_attrs(
                    long_name="sum of squared weights"
                )
    return out, shared


@bounded_threads
@recorded
def downsample(
    ds: xr.Dataset,
    *,
    factor: Optional[int] = None,
    spacing: Optional[float] = None,
    stat: Stat = "wmean",
    weights: Union[str, xr.DataArray, None] = "quality",
    layers: Optional[list[str]] = None,
    categorical: Optional[Mapping[str, Categorical]] = None,
    nan_threshold: float = 1.0,
    variogram: Optional[Mapping[str, VariogramModel]] = None,
) -> xr.Dataset:
    """Aggregate a DISP dataset to coarse cells with quality weights.

    Parameters
    ----------
    ds : xr.Dataset
        Chained, re-referenced data with ``y`` and ``x`` (and optionally
        ``time``) dimensions.
    factor : int, optional
        Pixels per cell side. Odd factors keep cell centres on pixel centres
        (33, 67, 333 for about 1, 2 and 10 km at 30 m).
    spacing : float, optional
        Cell size in the CRS unit instead of ``factor``; rounded to the
        nearest odd factor.
    stat : {"wmean", "wmedian"}
        Weighted mean or weighted median of float layers.
    weights : "quality", xr.DataArray or None
        ``"quality"`` (default): :func:`disp_xr.quality_weights` from the
        quality layers present, per epoch when the data have time. A
        DataArray ``(time, y, x)`` or ``(y, x)`` of weights, or None for equal
        weights.
    layers : list of str, optional
        Float layers to aggregate; default every float layer with ``y`` and
        ``x`` except quality layers used for the weights.
    categorical : mapping, optional
        Mask and label layers and their reduction, ``"mode"`` or ``"any"``;
        defaults in ``CATEGORICAL_DEFAULTS`` (mode for masks and labels, any
        for persistent scatterers and layover/shadow). Other integer layers
        (counts) are averaged like floats, ignoring their fill value.
    nan_threshold : float
        Largest share of missing pixels (of the full block) a cell may have;
        cells above it become NaN. 1.0 keeps any cell with data. Follows
        xarray-regrid's ``nan_threshold``.
    variogram : mapping of str to VariogramModel, optional
        A fitted variogram (:func:`disp_xr.residuals.fit_variogram`) per
        layer, estimated on that layer's residual field. Adds
        ``<layer>_sigma_corr = sqrt(sill / n_eff_corr)`` and ``n_eff_corr``,
        the effective number of independent pixels of the cell mean given the
        spatial correlation (Hugonnet et al. 2022). The model is the same for
        every epoch.

    Returns
    -------
    xr.Dataset
        One variable per aggregated layer on the coarse grid, with
        ``<layer>_sigma`` and, shared by all, ``n_eff``, ``n_valid``,
        ``valid_fraction``, ``quality``, ``centroid_x``, ``centroid_y``, and
        for ``"wmean"`` ``sum_w`` and ``sum_w2``; with ``variogram``,
        ``<layer>_sigma_corr`` and ``n_eff_corr``. Integer layers are reduced
        with mode or any. Lazy if the input is.

    Examples
    --------
    >>> coarse = dxr.downsample(ds, factor=33)  # doctest: +SKIP
    >>> points = dxr.to_points(coarse)  # doctest: +SKIP

    """
    if stat not in ("wmean", "wmedian"):
        raise ValueError(f"stat must be 'wmean' or 'wmedian', not {stat!r}")
    f = _factor(ds, factor, spacing)
    w, static = _resolve_weights(ds, weights)
    weight_layers = {
        "phase_similarity",
        "temporal_coherence",
        "timeseries_inversion_residuals",
    }
    categorical = {**CATEGORICAL_DEFAULTS, **dict(categorical or {})}
    if layers is None:
        # Float layers and integer counts; masks and labels are categorical.
        layers = [
            str(n)
            for n, v in ds.data_vars.items()
            if v.dtype.kind in "fiu"
            and {"y", "x"} <= set(v.dims)
            and n not in weight_layers
            and n not in categorical
        ]
    categorical = {k: v for k, v in categorical.items() if k in ds}
    variogram = dict(variogram or {})
    if missing := set(variogram) - set(layers):
        raise ValueError(f"variogram given for layers not aggregated: {missing}")
    spacing_yx = (
        abs(float(ds["y"][1] - ds["y"][0])),
        abs(float(ds["x"][1] - ds["x"][0])),
    )
    coords = _coarse_coords(ds, f)
    out, shared = _aggregate_layers(
        ds,
        w,
        layers=layers,
        cell_sums=lambda v, w: _block_cell_sums(v, w, f),
        median=lambda v, wv: _block_reduce(
            [v, wv], f, _weighted_median_blocks, np.float64
        ),
        full=f**2,
        stat=stat,
        nan_threshold=nan_threshold,
        variogram=variogram,
        rho_bar=lambda model: mean_block_correlation(model, f, spacing_yx),
    )
    for name, how in categorical.items():
        if {"y", "x"} <= set(ds[name].dims):
            out[name] = _categorical(ds[name], f, how)
    # Static per-cell quality and weighted centroid.
    total = _block_sum(static, f)
    xx, yy = xr.broadcast(ds["x"].astype(np.float64), ds["y"].astype(np.float64))
    shared["quality"] = (
        (total / f**2)
        .drop_attrs()
        .assign_attrs(long_name="mean static weight of the block")
    )
    for name, coord in (("centroid_x", xx), ("centroid_y", yy)):
        centroid = _block_sum(static * coord, f) / total.where(total > 0)
        shared[name] = centroid.drop_attrs().assign_attrs(
            units=ds[name[-1]].attrs.get("units", ""),
            long_name="weighted centroid (static weights)",
        )
    result = xr.Dataset({**out, **shared}).unify_chunks().drop_encoding()
    result = result.drop_vars([c for c in ("y", "x") if c in result.coords])
    result = result.assign_coords(coords)
    for name in list(result.coords):
        if name not in ("y", "x", "time") and name != "spatial_ref":
            if set(result[name].dims) & {"y", "x"}:
                result = result.drop_vars(name)
    if "spatial_ref" in ds.coords:
        result = result.assign_coords(spatial_ref=ds.coords["spatial_ref"])
        result = result.rio.write_transform(result.rio.transform(recalc=True))
    result.attrs = {
        **{k: v for k, v in ds.attrs.items() if k != "history"},
        "history": ds.attrs.get("history", ""),
        "downsample_factor": f,
        "downsample_stat": stat,
        "downsample_weights": (
            weights if isinstance(weights, str) or weights is None else "user"
        ),
        "resolution": f"{abs(float(ds['x'][1] - ds['x'][0])) * f:g}",
    }
    return result


def to_points(coarse: xr.Dataset, min_valid_fraction: float = 0.0) -> xr.Dataset:
    """Return the occupied cells of a downsampled dataset as points.

    The result has dimension ``point`` (and ``time`` if present), ordered by
    cell row then column, with the weighted centroid (``centroid_x``,
    ``centroid_y``), the cell centre (``x``, ``y``), the ``cell`` id
    (``row * n_cols + col``) and every cell variable. ``.to_dataframe()``
    gives the table.
    """
    stacked = coarse.stack(point=("y", "x"))
    keep = (stacked["n_valid"] > 0) & (stacked["valid_fraction"] >= min_valid_fraction)
    if "time" in keep.dims:
        keep = keep.any("time")
    keep = keep.compute()
    n_cols = coarse.sizes["x"]
    rows = np.repeat(np.arange(coarse.sizes["y"]), n_cols)
    cols = np.tile(np.arange(n_cols), coarse.sizes["y"])
    stacked = stacked.assign_coords(cell=("point", rows * n_cols + cols))
    points = stacked.isel(point=np.flatnonzero(keep.values))
    index = pd.MultiIndex.from_arrays(
        [points["y"].values, points["x"].values], names=("y", "x")
    )
    points = points.drop_vars(["point", "y", "x"]).assign_coords(
        y=("point", index.get_level_values("y")),
        x=("point", index.get_level_values("x")),
    )
    return points
