"""Quality-weighted quadtree downsampling (PRD S2 quadtree mode, plan task T28).

Cells of level ``k`` cover ``2**k x 2**k`` pixels of the 30 m grid, anchored
at its first pixel, from ``min_level`` (finest) to ``max_level`` (coarsest).
A cell splits into its four children while its split score exceeds a
threshold and every child keeps enough reliable pixels; the cells that do
not split are the leaves. Smooth areas end up in large cells and areas with
structure (deformation, unwrapping errors) in small ones.

The per-cell sums of :func:`disp_xr.downsample` (Σw, Σw², Σwv, Σwv², valid
count) compose exactly: a level is the 2 x 2 block sum of the level below.
So the data are read twice: once for the sums of the finest level (and from
them every coarser level and every score), once for the values of the
leaves. Leaf selection works on the small score maps, level by level, as
boolean masks, with no recursion per cell.

Split metrics (``metric``):

- ``"displacement"``: RMS over epochs of the weighted spread of the pixels
  in the cell, in the unit of the data, so a threshold reads as "split
  where pixels in a cell differ by more than 5 mm";
- ``"velocity"``: weighted spread of a per-pixel velocity (pass
  ``velocity``, for example from :func:`disp_xr.fit_velocity`), in its unit;
- ``"normalized"``: the weighted spread of the pixels in the cell against
  the spread their noise alone would give, RMS over epochs: a reduced
  chi-square per cell, about 1 where the cell holds only noise, so noisy
  (low-coherence) cells no longer split for being noisy. Pixel noise is
  taken as ``sigma0**2 / w`` (``w`` the quality weight); ``sigma0`` per
  epoch is the median of that spread over the cells of ``min_level``, where
  the scatter is mostly noise. Unitless; split above about 2;
- ``"structure"``: the spread of the four children's means divided by
  their noise (the mean of their sigmas), both RMS over epochs: signal
  structure at the cell scale against noise, after the structure function
  of GPS Imaging (Hammond et al. 2016,
  https://doi.org/10.1002/2016JB013458). Unitless; split above about 2;
- or a callable ``metric(level, children) -> (y, x) DataArray`` that gets
  the level's sums as a Dataset and those of its children (None at
  ``min_level``).

Plan task T29 compares the three and sets defaults.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import Literal, NamedTuple, Optional, Union

import numpy as np
import xarray as xr

from ._chunks import data_chunks
from ._grid import coordinate_grids
from ._history import record
from .config import bounded_threads
from .downsample import SUMS, _block_cell_sums, _block_sum, _resolve_weights

__all__ = ["Quadtree", "quadtree"]

MetricName = Literal["displacement", "velocity", "normalized", "structure"]
Metric = Union[MetricName, Callable[[xr.Dataset, Optional[xr.Dataset]], xr.DataArray]]


class Quadtree(NamedTuple):
    """Leaves of a quadtree.

    Examples
    --------
    >>> tree = dxr.quadtree(ds, n_points=5000)  # doctest: +SKIP
    >>> tree.points.level.to_series().value_counts()  # doctest: +SKIP
    >>> tree.to_raster("displacement", time=-1).plot.imshow()  # doctest: +SKIP

    """

    points: xr.Dataset
    """One point per leaf (dimension ``point``, and ``time``)."""
    leaf_id: xr.DataArray
    """``(y, x)`` index into ``points`` of the leaf holding each pixel, -1
    for pixels in no leaf (no reliable pixels in the cell)."""

    def to_raster(self, layer: str = "displacement", time=-1) -> xr.DataArray:
        """Return a leaf variable painted back on the ``(y, x)`` grid.

        ``time`` is an epoch index (default the last) or a date (nearest);
        pixels in no leaf are NaN.
        """
        values = self.points[layer]
        if "time" in values.dims:
            values = (
                values.isel(time=time)
                if isinstance(time, (int, np.integer))
                else values.sel(time=time, method="nearest")
            )
        flat = np.append(np.asarray(values.values, dtype=np.float64), np.nan)
        painted = flat[self.leaf_id.values]  # -1 picks the appended NaN
        return self.leaf_id.copy(data=painted).rename(layer).assign_attrs(values.attrs)


def _coarsen2(level: xr.Dataset) -> xr.Dataset:
    """Return the next coarser level: 2 x 2 block sums of every variable."""
    return xr.Dataset({name: _block_sum(var, 2) for name, var in level.items()})


def _moments(level: xr.Dataset) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    """Return the weighted mean, variance and n_eff of each cell."""
    sum_w = level["sum_w"].where(level["sum_w"] > 0)
    mean = level["sum_wv"] / sum_w
    mean_square = level["sum_wvv"] / sum_w
    variance = mean_square - mean**2
    variance = variance.where(variance > 1e-12 * mean_square, 0.0)
    n_eff = level["sum_w"] ** 2 / level["sum_w2"].where(level["sum_w2"] > 0)
    return mean, variance, n_eff


def _nanmean(da: xr.DataArray, dim: str) -> xr.DataArray:
    """Mean ignoring NaN; NaN (without a warning) where all are NaN."""
    count = da.notnull().sum(dim)
    return da.sum(dim, skipna=True) / count.where(count > 0)


def _rms_over_time(da: xr.DataArray) -> xr.DataArray:
    return np.sqrt(_nanmean(da, "time")) if "time" in da.dims else np.sqrt(da)


def _displacement_score(
    level: xr.Dataset,
    children: Optional[xr.Dataset],  # noqa: ARG001 (metric interface)
) -> xr.DataArray:
    _, variance, _ = _moments(level)
    return _rms_over_time(variance)


def _scatter(level: xr.Dataset) -> xr.DataArray:
    """Return the weighted scatter per degree of freedom of each cell.

    ``sum(w (v - m)**2) / (n - 1)``, whose expectation is ``sigma0**2`` when
    each pixel's noise variance is ``sigma0**2 / w``; NaN for cells with
    fewer than two valid pixels.
    """
    sum_w = level["sum_w"].where(level["sum_w"] > 0)
    scatter = level["sum_wvv"] - level["sum_wv"] ** 2 / sum_w
    scatter = scatter.where(scatter > 1e-12 * level["sum_wvv"], 0.0)
    return scatter / (level["n_valid"] - 1).where(level["n_valid"] > 1)


def _noise_floor(finest: xr.DataArray) -> xr.DataArray:
    """Return ``sigma0**2``: the median scatter of the finest cells per epoch."""
    finest = finest.chunk({"y": -1, "x": -1}) if finest.chunks is not None else finest
    return xr.apply_ufunc(
        lambda a: np.nanmedian(a, axis=(-2, -1)),
        finest,
        input_core_dims=[["y", "x"]],
        dask="parallelized",
        output_dtypes=[np.float64],
    )


def _normalized_score(
    level: xr.Dataset,
    children: Optional[xr.Dataset],  # noqa: ARG001 (metric interface)
    *,
    noise: xr.DataArray,
) -> xr.DataArray:
    return _rms_over_time(_scatter(level) / noise.where(noise > 0))


def _structure_score(level: xr.Dataset, children: Optional[xr.Dataset]) -> xr.DataArray:
    if children is None:
        return xr.full_like(
            (
                level["sum_w"].isel(time=0, drop=True)
                if "time" in level.dims
                else level["sum_w"]
            ),
            np.nan,
        )
    mean, variance, n_eff = _moments(children)
    sigma2 = variance / n_eff
    ny, nx = level.sizes["y"], level.sizes["x"]

    def blocks(da: xr.DataArray) -> xr.DataArray:
        da = da.pad(
            y=(0, 2 * ny - da.sizes["y"]),
            x=(0, 2 * nx - da.sizes["x"]),
            constant_values=np.nan,
        )
        return (
            da.coarsen(y=2, x=2)
            .construct(y=("y", "cy"), x=("x", "cx"))
            .stack(child=("cy", "cx"))
        )

    child_mean = blocks(mean)
    # Population variance of the four child means; pairs differ by 4/3 of it.
    centered = child_mean - _nanmean(child_mean, "child")
    spread = _nanmean(centered**2, "child")
    noise = _nanmean(blocks(sigma2), "child")
    noise = _rms_over_time(noise)
    # No noise estimate (single-pixel children): no score, so no split.
    score = _rms_over_time(spread) / noise.where(noise > 0)
    return score.drop_vars([c for c in score.coords if c not in ("y", "x")])


def _velocity_score(
    level: xr.Dataset,
    children: Optional[xr.Dataset],  # noqa: ARG001 (metric interface)
) -> xr.DataArray:
    sw = level["vel_w"].where(level["vel_w"] > 0)
    mean = level["vel_wv"] / sw
    mean_square = level["vel_wvv"] / sw
    variance = mean_square - mean**2
    variance = variance.where(variance > 1e-12 * mean_square, 0.0)
    return np.sqrt(variance)


METRICS: dict[str, Callable] = {
    "displacement": _displacement_score,
    "velocity": _velocity_score,
    "structure": _structure_score,
    "normalized": _normalized_score,  # bound to the noise floor in quadtree
}


def _select(
    scores: dict[int, np.ndarray],
    reliable: dict[int, np.ndarray],
    threshold: float,
    min_level: int,
    max_level: int,
    min_count: dict[int, np.ndarray],
) -> dict[int, np.ndarray]:
    """Return the leaf mask of each level, top-down.

    ``min_count[k]`` is the number of reliable pixels each cell of level
    ``k`` needs to be a child of a split: a share of its pixels inside the
    frame, 0 for cells outside it (which never block a split).
    """
    active = np.ones_like(scores[max_level], dtype=bool)
    leaves: dict[int, np.ndarray] = {}
    for k in range(max_level, min_level - 1, -1):
        split = np.zeros_like(active)
        if k > min_level:
            child_ok = reliable[k - 1] >= min_count[k - 1]
            cy, cx = active.shape
            pad = np.ones((2 * cy, 2 * cx), dtype=bool)
            pad[: child_ok.shape[0], : child_ok.shape[1]] = child_ok[: 2 * cy, : 2 * cx]
            all_ok = pad.reshape(cy, 2, cx, 2).all(axis=(1, 3))
            with np.errstate(invalid="ignore"):
                split = active & (scores[k] > threshold) & all_ok
        leaves[k] = active & ~split & (reliable[k] > 0)
        if k > min_level:
            below = scores[k - 1].shape
            active = np.repeat(np.repeat(split, 2, axis=0), 2, axis=1)[
                : below[0], : below[1]
            ]
    return leaves


def _threshold_for(
    n_points: int,
    scores: dict[int, np.ndarray],
    reliable: dict[int, np.ndarray],
    min_level: int,
    max_level: int,
    min_count: dict[int, np.ndarray],
) -> float:
    """Return the threshold whose leaf count is closest to ``n_points``.

    The leaf count does not grow with the threshold, and it only changes at
    score values, so a bisection over the sorted scores finds it.
    """

    def count(threshold: float) -> int:
        leaves = _select(scores, reliable, threshold, min_level, max_level, min_count)
        return int(sum(mask.sum() for mask in leaves.values()))

    finite = [a[np.isfinite(a)] for k, a in scores.items() if k > min_level]
    candidates = np.concatenate([[-np.inf], np.unique(np.concatenate(finite))])
    lo, hi = 0, candidates.size - 1
    if count(candidates[hi]) >= n_points:
        return float(candidates[hi])
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if count(candidates[mid]) > n_points:
            lo = mid
        else:
            hi = mid
    above, below = count(candidates[lo]), count(candidates[hi])
    return float(
        candidates[lo] if above - n_points < n_points - below else candidates[hi]
    )


@bounded_threads
def quadtree(
    ds: xr.Dataset,
    *,
    threshold: Optional[float] = None,
    n_points: Optional[int] = None,
    metric: Metric = "structure",
    min_level: int = 3,
    max_level: int = 8,
    min_reliable: float = 0.5,
    layer: str = "displacement",
    weights: Union[str, xr.DataArray, None] = "quality",
    velocity: Optional[xr.DataArray] = None,
) -> Quadtree:
    """Downsample to quadtree leaves: small cells where the data vary.

    Parameters
    ----------
    ds : xr.Dataset
        Chained, re-referenced data with ``y``, ``x`` (and ``time``).
    threshold : float, optional
        Split score above which a cell splits (see the module docstring for
        the unit of each metric).
    n_points : int, optional
        Number of leaves wanted instead of a threshold: the threshold whose
        leaf count is closest is chosen (recorded as ``quadtree_threshold``).
        Counts outside what ``min_level`` and ``max_level`` allow give the
        nearest extreme. Pass ``threshold`` or ``n_points``, not both.
    metric : {"displacement", "velocity", "normalized", "structure"} or callable
        Split metric. Default ``"structure"``, the best of the three in the
        T29 trade study (``docs/studies/split_metric.md``); at the same
        number of points the regular grid of :func:`downsample` still
        reconstructs smooth signals better. ``"normalized"`` scores
        ``"displacement"`` against the pixels' expected noise (plan task
        T28.6).
    min_level, max_level : int
        Finest and coarsest levels: cells of ``2**min_level`` to
        ``2**max_level`` pixels on a side (3 and 8 are 240 m and 7.7 km at
        30 m).
    min_reliable : float
        Share of a child's pixels inside the frame that must be reliable
        (static weight above 0 and data at some epoch) in each of the four
        children for a cell to split. Children outside the frame are
        ignored, so cells on the frame edge split like the others.
    layer : str
        Layer to aggregate and score.
    weights : "quality", xr.DataArray or None
        As in :func:`disp_xr.downsample`.
    velocity : xr.DataArray, optional
        ``(y, x)`` velocity for ``metric="velocity"``.

    Returns
    -------
    Quadtree
        ``points``: one point per leaf with ``level``, ``size`` (meters),
        the cell center ``x``, ``y``, ``row``, ``col`` (at its level), the
        weighted centroid, ``score``, the weighted mean ``<layer>`` and its
        ``<layer>_sigma``, ``n_eff``, ``n_valid``, ``valid_fraction`` and
        ``quality``; lazy in the data. ``leaf_id``: the ``(y, x)`` leaf of
        each pixel.

    Examples
    --------
    >>> tree = dxr.quadtree(ds, threshold=0.005)  # doctest: +SKIP
    >>> tree.points.level.to_series().value_counts()  # doctest: +SKIP
    >>> tree = dxr.quadtree(ds, n_points=5000)  # doctest: +SKIP

    """
    import dask

    if (threshold is None) == (n_points is None):
        raise ValueError("pass threshold or n_points, not both or neither")
    if n_points is not None and n_points < 1:
        raise ValueError(f"n_points must be positive, not {n_points}")
    if not 0 <= min_level <= max_level:
        raise ValueError(
            f"need 0 <= min_level <= max_level, not {min_level}, {max_level}"
        )
    score_fn = METRICS.get(metric, metric) if isinstance(metric, str) else metric
    if not callable(score_fn):
        raise ValueError(f"unknown metric {metric!r}")
    if metric == "velocity" and velocity is None:
        raise ValueError("metric='velocity' needs velocity=")
    w, static = _resolve_weights(ds, weights)
    v = ds[layer].astype(np.float64)
    wb = w.broadcast_like(v)
    if v.chunks is not None:
        wb = wb.chunk(data_chunks(v))
    f0 = 2**min_level
    sums = _block_cell_sums(v, wb, f0)
    finest = xr.Dataset(dict(zip(SUMS, sums)))
    # Static per-cell sums: quality weight, centroid, reliable pixels.
    xx, yy = coordinate_grids(ds, static)
    any_valid = v.notnull().any("time") if "time" in v.dims else v.notnull()
    reliable_px = ((static > 0) & any_valid).astype(np.float64)
    finest["static_w"] = _block_sum(static, f0)
    finest["static_wx"] = _block_sum(static * xx, f0)
    finest["static_wy"] = _block_sum(static * yy, f0)
    finest["reliable"] = _block_sum(reliable_px, f0)
    finest["inside"] = _block_sum(xr.ones_like(static, dtype=np.float64), f0)
    if velocity is not None:
        vel = velocity.astype(np.float64)
        ok = vel.notnull() & (static > 0)
        sw = static.where(ok, 0.0)
        finest["vel_w"] = _block_sum(sw, f0)
        finest["vel_wv"] = _block_sum(sw * vel.fillna(0.0), f0)
        finest["vel_wvv"] = _block_sum(sw * vel.fillna(0.0) ** 2, f0)
    finest = finest.drop_vars([c for c in finest.coords if c not in ("time",)])
    levels = {min_level: finest}
    for k in range(min_level + 1, max_level + 1):
        levels[k] = _coarsen2(levels[k - 1])
    if score_fn is _normalized_score:
        score_fn = partial(_normalized_score, noise=_noise_floor(_scatter(finest)))

    lazy_scores = {
        k: score_fn(levels[k], levels.get(k - 1) if k > min_level else None)
        for k in levels
    }
    computed = dask.compute(
        {k: s.data for k, s in lazy_scores.items()},
        {k: lv["reliable"].data for k, lv in levels.items()},
        {k: lv["inside"].data for k, lv in levels.items()},
    )
    scores = {k: np.asarray(a, dtype=np.float64) for k, a in computed[0].items()}
    reliable = {k: np.asarray(a) for k, a in computed[1].items()}
    # A share of the pixels inside the frame: edge cells can split too.
    min_count = {k: min_reliable * np.asarray(a) for k, a in computed[2].items()}
    if threshold is None:
        assert n_points is not None
        threshold = _threshold_for(
            n_points, scores, reliable, min_level, max_level, min_count
        )
    leaves = _select(scores, reliable, threshold, min_level, max_level, min_count)

    dy = float(ds["y"][1] - ds["y"][0])
    dx = float(ds["x"][1] - ds["x"][0])
    x0, y0 = float(ds["x"][0]), float(ds["y"][0])
    parts = []
    ny, nx = ds.sizes["y"], ds.sizes["x"]
    leaf_id = np.full((ny, nx), -1, dtype=np.int64)
    start = 0
    for k in range(max_level, min_level - 1, -1):
        rows, cols = np.nonzero(leaves[k])
        if rows.size == 0:
            continue
        size = 2**k
        pick = {
            "y": xr.DataArray(rows, dims="point"),
            "x": xr.DataArray(cols, dims="point"),
        }
        level = levels[k].isel(pick)
        mean, variance, n_eff = _moments(level)
        part = xr.Dataset(
            {
                layer: mean.assign_attrs(ds[layer].attrs, aggregation="wmean"),
                f"{layer}_sigma": (
                    np.sqrt(variance / n_eff).assign_attrs(
                        units=ds[layer].attrs.get("units", ""),
                        long_name="sigma of the cell wmean (pixels independent)",
                    )
                ),
                "n_eff": n_eff,
                "n_valid": level["n_valid"].astype(np.int32),
                "valid_fraction": level["n_valid"] / size**2,
                "quality": level["static_w"] / size**2,
                "centroid_x": (
                    level["static_wx"] / level["static_w"].where(level["static_w"] > 0)
                ),
                "centroid_y": (
                    level["static_wy"] / level["static_w"].where(level["static_w"] > 0)
                ),
                "score": ("point", scores[k][rows, cols]),
            },
            coords={
                "level": ("point", np.full(rows.size, k)),
                "size": ("point", np.full(rows.size, size * abs(dx))),
                "row": ("point", rows),
                "col": ("point", cols),
                "x": ("point", x0 + (cols * size + (size - 1) / 2) * dx),
                "y": ("point", y0 + (rows * size + (size - 1) / 2) * dy),
            },
        )
        parts.append(part)
        ids = np.full(leaves[k].shape, -1, dtype=np.int64)
        ids[rows, cols] = start + np.arange(rows.size)
        up = np.repeat(np.repeat(ids, size, axis=0), size, axis=1)[:ny, :nx]
        leaf_id = np.where(up >= 0, up, leaf_id)
        start += rows.size
    if not parts:
        raise ValueError("no leaf holds reliable pixels")
    points = xr.concat(parts, dim="point", coords="minimal", compat="override")
    points = points.assign_coords(leaf=("point", np.arange(points.sizes["point"])))
    if "spatial_ref" in ds.coords:
        points = points.assign_coords(spatial_ref=ds.coords["spatial_ref"])
    points.attrs = {
        **{k: v for k, v in ds.attrs.items() if k != "history"},
        "history": ds.attrs.get("history", ""),
        "downsample_grid": "quadtree",
        "quadtree_metric": metric if isinstance(metric, str) else "custom",
        "quadtree_threshold": float(threshold),
        "quadtree_levels": f"{min_level}-{max_level}",
        "downsample_weights": (
            weights if isinstance(weights, str) or weights is None else "user"
        ),
    }
    leaf_da = xr.DataArray(
        leaf_id, dims=("y", "x"), coords={"y": ds["y"], "x": ds["x"]}, name="leaf_id"
    )
    if "spatial_ref" in ds.coords:
        leaf_da = leaf_da.assign_coords(spatial_ref=ds.coords["spatial_ref"])
    points = record(
        points,
        "quadtree",
        threshold=threshold,
        metric=points.attrs["quadtree_metric"],
        min_level=min_level,
        max_level=max_level,
        min_reliable=min_reliable,
        layer=layer,
    )
    return Quadtree(points, leaf_da)
