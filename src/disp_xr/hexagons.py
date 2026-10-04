"""Quality-weighted downsampling to hexagonal cells (PRD S2 hexagon mode, T27.8).

Pixels are grouped into pointy-top hexagons of a planar lattice in the
frame's CRS, anchored at the CRS origin, so overlapping frames of one UTM
zone share cell ids. ``spacing`` is the distance between neighboring cell
centers (the flat-to-flat width); a hexagon covers ``sqrt(3) / 2 * spacing²``,
so ``spacing = 1.0746 * side`` matches the area of a square of that side.

Each pixel goes to the hexagon whose center is nearest, found with axial
coordinates and cube rounding (Patel, "Hexagonal Grids", Red Blob Games).
The per-cell statistics are those of :func:`disp_xr.downsample` (weighted
mean or median, sigma, ``n_eff``, valid fraction, quality, weighted
centroid, mode or "any" for masks and labels), computed with grouped sums
(one ``np.bincount`` per dask block) instead of block sums. Hexagons are not
a raster, so the result is a ``(time, point)`` table in the
:func:`disp_xr.to_points` layout.

Why hexagons: all six neighbors are equidistant and a hexagon is more
compact than a square of equal area, so the sampling is more isotropic
(Birch et al. 2007, https://doi.org/10.1016/j.ecolmodel.2007.03.041). The
lattice is the one ``matplotlib.hexbin`` and PostGIS ``ST_HexagonGrid``
build (Carr et al. 1987, https://doi.org/10.1080/01621459.1987.10478445);
here the cells carry quality weights and sigmas, lazily.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal, Optional, Union

import numpy as np
import xarray as xr

from ._history import recorded
from .config import bounded_threads
from .downsample import (
    CATEGORICAL_DEFAULTS,
    SUMS,
    Categorical,
    Stat,
    _aggregate_layers,
    _pixel_sums,
    _resolve_weights,
)
from .residuals import VariogramModel, mean_mask_correlation

__all__ = ["downsample_hex", "hex_axial", "hex_centers", "hexagon_vertices"]

SQRT3 = float(np.sqrt(3.0))
WEIGHT_LAYERS = (
    "phase_similarity",
    "temporal_coherence",
    "timeseries_inversion_residuals",
)


def hex_axial(
    x: np.ndarray, y: np.ndarray, spacing: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return the axial coordinates ``(q, r)`` of the hexagon holding each point.

    The lattice is pointy-top, anchored at ``(0, 0)``, with centers at
    ``x = spacing * (q + r / 2)``, ``y = spacing * sqrt(3) / 2 * r``.
    """
    qf = np.asarray(x, dtype=np.float64) / spacing - np.asarray(y, dtype=np.float64) / (
        SQRT3 * spacing
    )
    rf = 2.0 * np.asarray(y, dtype=np.float64) / (SQRT3 * spacing)
    sf = -qf - rf
    q, r, s = np.rint(qf), np.rint(rf), np.rint(sf)
    dq, dr, ds = np.abs(q - qf), np.abs(r - rf), np.abs(s - sf)
    fix_q = (dq > dr) & (dq > ds)
    fix_r = ~fix_q & (dr > ds)
    q = np.where(fix_q, -r - s, q)
    r = np.where(fix_r, -q - s, r)
    return q.astype(np.int64), r.astype(np.int64)


def hex_centers(
    q: np.ndarray, r: np.ndarray, spacing: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return the center ``(x, y)`` of hexagons ``(q, r)``."""
    q = np.asarray(q, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)
    return spacing * (q + r / 2.0), spacing * SQRT3 / 2.0 * r


def hexagon_vertices(x: np.ndarray, y: np.ndarray, spacing: float) -> np.ndarray:
    """Return the six vertices of pointy-top hexagons centered at ``(x, y)``.

    Shape ``(n, 6, 2)``, for ``matplotlib.collections.PolyCollection`` or
    shapely polygons.
    """
    radius = spacing / SQRT3
    angles = np.radians(30.0 + 60.0 * np.arange(6))
    x = np.asarray(x, dtype=np.float64)[:, None]
    y = np.asarray(y, dtype=np.float64)[:, None]
    return np.stack([x + radius * np.cos(angles), y + radius * np.sin(angles)], -1)


def _id_range(ds: xr.Dataset, spacing: float) -> tuple[int, int, int, int]:
    """Return ``(q0, r0, nq, nr)`` bounding the axial coordinates of the grid."""
    xs, ys = ds["x"].values, ds["y"].values
    r_lo = int(np.floor(2 * ys.min() / (SQRT3 * spacing))) - 1
    r_hi = int(np.ceil(2 * ys.max() / (SQRT3 * spacing))) + 1
    q_lo = int(np.floor(xs.min() / spacing - r_hi / 2)) - 1
    q_hi = int(np.ceil(xs.max() / spacing - r_lo / 2)) + 1
    return q_lo, r_lo, q_hi - q_lo + 1, r_hi - r_lo + 1


def _labels(ds: xr.Dataset, spacing: float, bounds) -> np.ndarray:
    """Return the ``(y, x)`` array of cell ids (cheap, so computed eagerly)."""
    q0, r0, nq, _ = bounds
    x = ds["x"].values.astype(np.float64)[None, :]
    y = ds["y"].values.astype(np.float64)[:, None]
    q, r = hex_axial(
        np.broadcast_to(x, (y.size, x.size)),
        np.broadcast_to(y, (y.size, x.size)),
        spacing,
    )
    return (r - r0) * nq + (q - q0)


def _bincount_cells(values: np.ndarray, labels: np.ndarray, n_ids: int) -> np.ndarray:
    """Sum ``(..., y, x)`` values into cells: one bincount over all leading indices."""
    lead = values.shape[:-2]
    flat = values.reshape(-1, labels.size)
    index = np.arange(flat.shape[0])[:, None] * n_ids + labels.reshape(1, -1)
    sums = np.bincount(
        index.ravel(), weights=flat.ravel(), minlength=flat.shape[0] * n_ids
    )
    return sums.reshape(*lead, n_ids)


def _row_windows(labels: np.ndarray, ychunks) -> tuple[np.ndarray, int]:
    """Return the first cell id of each row of blocks, and the window width."""
    edges = np.concatenate([[0], np.cumsum(ychunks)])
    bands = [labels[a:b] for a, b in zip(edges[:-1], edges[1:])]
    starts = np.array([band.min() for band in bands])
    stops = np.array([band.max() + 1 for band in bands])
    return starts, int((stops - starts).max())


def _group_reduce(
    arrays: list[xr.DataArray], labels: np.ndarray, n_ids: int, kernel, k: int
) -> xr.DataArray:
    """Reduce ``(..., y, x)`` arrays into cells with ``kernel``, lazily for dask.

    ``kernel(*blocks, reduce)`` returns ``k`` stacked per-cell results, where
    ``reduce`` sums a pixel array into cells. Cell ids grow with the hexagon
    row, so the pixels of one row of dask blocks fall in a contiguous window
    of ids, known from ``labels``. Each block is reduced into its window with
    ``np.bincount``; the blocks of a row are added, and the row windows are
    overlap-added into all cells. The partial sums then hold about one copy
    of the output, not one per block (a frame at 1 km has about 84,000
    cells). Returns dims ``("stat", ..., "cell")``.
    """
    order = [d for d in arrays[0].dims if d not in ("y", "x")] + ["y", "x"]
    arrays = [a.transpose(*order) for a in arrays]
    lead_dims = order[:-2]
    data = arrays[0].data
    if isinstance(data, np.ndarray):
        result = kernel(
            *(np.asarray(a.data) for a in arrays),
            lambda a: _bincount_cells(a, labels, n_ids),
        )
    else:
        import dask.array as darr

        ychunks, xchunks = data.chunks[-2:]
        starts, width = _row_windows(labels, ychunks)
        datas = [darr.asarray(a.data).rechunk(data.chunks) for a in arrays]
        lab = darr.from_array(labels, chunks=(ychunks, xchunks))

        def block(*blocks: np.ndarray, block_info=None) -> np.ndarray:
            *values, lb = blocks
            row = block_info[0]["chunk-location"][-2]
            local = lb - starts[row]
            out = kernel(*values, lambda a: _bincount_cells(a, local, width))
            return out.reshape(*out.shape[:-1], 1, 1, width)

        partial = darr.map_blocks(
            block,
            *datas,
            lab,
            new_axis=[0, data.ndim + 1],
            chunks=((k,),)
            + data.chunks[:-2]
            + ((1,) * len(ychunks), (1,) * len(xchunks), (width,)),
            dtype=np.float64,
        )
        rows = partial.sum(axis=-2).rechunk({partial.ndim - 3: -1})

        def overlap_add(windows: np.ndarray) -> np.ndarray:
            out = np.zeros((*windows.shape[:-2], n_ids))
            for i, start in enumerate(starts):
                stop = min(start + width, n_ids)
                out[..., start:stop] += windows[..., i, : stop - start]
            return out

        result = darr.map_blocks(
            overlap_add,
            rows,
            drop_axis=rows.ndim - 2,
            chunks=rows.chunks[:-2] + ((n_ids,),),
            dtype=np.float64,
        )
    coords = {d: arrays[0][d] for d in lead_dims if d in arrays[0].coords}
    return xr.DataArray(result, dims=("stat", *lead_dims, "cell"), coords=coords)


def _group_sum(da: xr.DataArray, labels: np.ndarray, n_ids: int) -> xr.DataArray:
    """Sum a ``(..., y, x)`` DataArray into cells."""
    out = _group_reduce(
        [da], labels, n_ids, lambda v, reduce: reduce(v.astype(np.float64))[None], 1
    )
    return out.isel(stat=0, drop=True)


def _hex_cell_sums(
    v: xr.DataArray, w: xr.DataArray, labels: np.ndarray, n_ids: int
) -> tuple[xr.DataArray, ...]:
    """Per-cell Σw, Σw², Σwv, Σwv² and valid count in one pass (see ``SUMS``)."""
    stacked = _group_reduce([v, w], labels, n_ids, _pixel_sums, len(SUMS))
    return tuple(stacked.isel(stat=i, drop=True) for i in range(len(SUMS)))


def _hex_mask(spacing: float, dy: float, dx: float) -> np.ndarray:
    """Pixels of a hexagon centered on a pixel (for the block correlation)."""
    half = int(np.ceil(spacing / min(dx, dy))) + 1
    iy, ix = np.mgrid[-half : half + 1, -half : half + 1]
    q, r = hex_axial(ix * dx, -iy * dy, spacing)
    mask = (q == 0) & (r == 0)
    rows, cols = np.nonzero(mask)
    return mask[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1]


def _group_sorted(labels: np.ndarray, values: np.ndarray, weights: np.ndarray):
    """Sort valid entries by (label, value); return them and the group starts."""
    ok = np.isfinite(values) & (weights > 0) & (labels >= 0)
    lab, val, w = labels[ok], values[ok], weights[ok]
    order = np.lexsort((val, lab))
    lab, val, w = lab[order], val[order], w[order]
    groups, starts = np.unique(lab, return_index=True)
    return lab, val, w, groups, starts


def _weighted_median_cells(
    values: np.ndarray, weights: np.ndarray, labels: np.ndarray, n_ids: int
) -> np.ndarray:
    """Lower weighted median per cell id, for each leading index."""
    lead = values.shape[:-2]
    flat_v = values.reshape(-1, labels.size)
    flat_w = np.broadcast_to(weights, values.shape).reshape(-1, labels.size)
    flat_l = labels.ravel()
    out = np.full((flat_v.shape[0], n_ids), np.nan)
    for i in range(flat_v.shape[0]):
        _, val, w, groups, starts = _group_sorted(flat_l, flat_v[i], flat_w[i])
        if groups.size == 0:
            continue
        cw = np.cumsum(w)
        ends = np.append(starts[1:], cw.size)
        before = np.where(starts > 0, cw[np.maximum(starts - 1, 0)], 0.0)
        total = cw[ends - 1] - before
        idx = np.searchsorted(cw, before + 0.5 * total, side="left")
        out[i, groups] = val[np.minimum(idx, ends - 1)]
    return out.reshape(*lead, n_ids)


def _mode_cells(values: np.ndarray, labels: np.ndarray, n_ids: int, fill) -> np.ndarray:
    """Most common value per cell, ignoring ``fill``; ties take the smallest."""
    lead = values.shape[:-2]
    flat_v = values.reshape(-1, labels.size)
    flat_l = labels.ravel()
    out = np.full((flat_v.shape[0], n_ids), fill, dtype=values.dtype)
    for i in range(flat_v.shape[0]):
        keep = (flat_v[i] != fill) & (flat_l >= 0)
        lab, val = flat_l[keep], flat_v[i][keep]
        if lab.size == 0:
            continue
        order = np.lexsort((val, lab))
        lab, val = lab[order], val[order]
        new = np.ones(lab.size, bool)
        new[1:] = (lab[1:] != lab[:-1]) | (val[1:] != val[:-1])
        run_start = np.flatnonzero(new)
        counts = np.diff(np.append(run_start, lab.size))
        run_lab, run_val = lab[run_start], val[run_start]
        # Per label: the largest count, then the smallest value.
        best = np.lexsort((run_val, -counts, run_lab))
        first = np.ones(best.size, bool)
        first[1:] = run_lab[best][1:] != run_lab[best][:-1]
        pick = best[first]
        out[i, run_lab[pick]] = run_val[pick]
    return out.reshape(*lead, n_ids)


def _banded(
    func,
    arrays: list[xr.DataArray],
    labels: np.ndarray,
    n_ids: int,
    halo: int,
    dtype,
    missing,
    target_mb: float = 128.0,
    **kwargs,
) -> xr.DataArray:
    """Apply a whole-cell reduction (median, mode) by bands of rows, lazily.

    A hexagon crosses chunk borders, so its median needs all its pixels in
    one task. Each band of rows owns the cells whose center falls in it, and
    reads ``halo`` extra rows on both sides, enough for every pixel of those
    cells; ``func(*blocks, local_labels, width)`` sees the band's pixels with
    labels local to the band's window of ids, and -1 for cells another band
    owns. One task then holds a band of the frame for a few epochs instead of
    whole epochs of the frame.
    """
    order = [d for d in arrays[0].dims if d not in ("y", "x")] + ["y", "x"]
    arrays = [a.transpose(*order) for a in arrays]
    lead_dims = order[:-2]
    data = arrays[0].data
    coords = {d: arrays[0][d] for d in lead_dims if d in arrays[0].coords}
    if isinstance(data, np.ndarray):
        result = func(*(np.asarray(a.data) for a in arrays), labels, n_ids, **kwargs)
        return xr.DataArray(result, dims=(*lead_dims, "cell"), coords=coords)

    import dask.array as darr

    ny, nx = labels.shape
    band = max(data.chunks[-2][0], halo)
    itemsize = 8 * len(arrays)
    per_epoch = (band + 2 * halo) * nx * itemsize
    lead_chunk = {
        i: max(1, int(target_mb * 1e6 // per_epoch)) if i == 0 else -1
        for i in range(data.ndim - 2)
    }
    chunks = {**lead_chunk, data.ndim - 2: band, data.ndim - 1: -1}
    datas = [darr.asarray(a.data).rechunk(chunks) for a in arrays]
    ychunks = datas[0].chunks[-2]
    edges = np.concatenate([[0], np.cumsum(ychunks)])
    # Owner band of each cell: the band holding the row of its center.
    rows = np.arange(ny)
    first_row = np.full(n_ids, ny, dtype=np.int64)
    last_row = np.full(n_ids, -1, dtype=np.int64)
    flat = labels.ravel()
    row_of = np.repeat(rows, nx)
    np.minimum.at(first_row, flat, row_of)
    np.maximum.at(last_row, flat, row_of)
    center_row = (first_row + last_row) // 2
    owner = np.searchsorted(edges[1:], center_row, side="right")
    present = last_row >= 0
    lo = np.array(
        [
            np.flatnonzero(present & (owner == k)).min(initial=0)
            for k in range(len(ychunks))
        ]
    )
    hi = np.array(
        [
            np.flatnonzero(present & (owner == k)).max(initial=-1) + 1
            for k in range(len(ychunks))
        ]
    )
    hi = np.maximum(hi, lo)
    width = max(int((hi - lo).max()), 1)
    if (last_row - first_row)[present].max() > 2 * halo:
        raise AssertionError("halo smaller than a cell")
    depth = {data.ndim - 2: halo}
    padded = [darr.overlap.overlap(d, depth=depth, boundary="none") for d in datas]
    lab = darr.from_array(labels, chunks=(ychunks, (nx,)))
    lab = darr.overlap.overlap(lab, depth={0: halo}, boundary="none")

    def block(*blocks: np.ndarray, block_info=None) -> np.ndarray:
        *values, lb = blocks
        k = block_info[0]["chunk-location"][-2]
        local = np.where((lb >= lo[k]) & (lb < hi[k]), lb - lo[k], -1)
        out = func(*values, local, width, **kwargs)
        return out.reshape(*out.shape[:-1], 1, width)

    partial = darr.map_blocks(
        block,
        *padded,
        lab,
        chunks=datas[0].chunks[:-2] + ((1,) * len(ychunks), (width,)),
        dtype=dtype,
    )
    windows = partial.rechunk({partial.ndim - 2: -1})

    def assemble(win: np.ndarray) -> np.ndarray:
        out = np.full((*win.shape[:-2], n_ids), missing, dtype=dtype)
        for k in range(win.shape[-2]):
            out[..., lo[k] : hi[k]] = win[..., k, : hi[k] - lo[k]]
        return out

    result = darr.map_blocks(
        assemble,
        windows,
        drop_axis=windows.ndim - 2,
        chunks=windows.chunks[:-2] + ((n_ids,),),
        dtype=dtype,
    )
    return xr.DataArray(result, dims=(*lead_dims, "cell"), coords=coords)


@bounded_threads
@recorded
def downsample_hex(
    ds: xr.Dataset,
    *,
    spacing: float,
    grid: Literal["planar"] = "planar",
    stat: Stat = "wmean",
    weights: Union[str, xr.DataArray, None] = "quality",
    layers: Optional[list[str]] = None,
    categorical: Optional[Mapping[str, Categorical]] = None,
    nan_threshold: float = 1.0,
    variogram: Optional[Mapping[str, VariogramModel]] = None,
) -> xr.Dataset:
    """Aggregate a DISP dataset to hexagonal cells with quality weights.

    Parameters
    ----------
    ds : xr.Dataset
        Chained, re-referenced data with ``y`` and ``x`` (and optionally
        ``time``) dimensions, in a projected CRS.
    spacing : float
        Distance between neighboring cell centers, in the CRS unit (a
        hexagon covers ``0.866 * spacing²``).
    grid : {"planar"}
        Planar hexagons in the frame's CRS. H3 global cells are plan task
        T27.8, second part.
    stat, weights, layers, categorical, nan_threshold, variogram
        As in :func:`disp_xr.downsample`. ``valid_fraction`` is relative to
        the expected pixel count of a full hexagon (its area over the pixel
        area), so it can slightly exceed 1.

    Returns
    -------
    xr.Dataset
        Dimension ``point`` (and ``time``), one point per hexagon holding at
        least one pixel of the grid, ordered by cell id; coordinates ``cell``,
        ``q``, ``r`` and the center ``x``, ``y``; the variables of
        :func:`disp_xr.downsample`. Lazy if the input is.

    Examples
    --------
    >>> hexes = dxr.downsample_hex(ds, spacing=1075)  # doctest: +SKIP
    >>> dxr.plot.plot_hex(hexes)  # doctest: +SKIP

    """
    if grid != "planar":
        raise ValueError(f"grid must be 'planar' (H3 is planned), not {grid!r}")
    if stat not in ("wmean", "wmedian"):
        raise ValueError(f"stat must be 'wmean' or 'wmedian', not {stat!r}")
    if spacing <= 0:
        raise ValueError(f"spacing must be positive, not {spacing}")
    dy = abs(float(ds["y"][1] - ds["y"][0]))
    dx = abs(float(ds["x"][1] - ds["x"][0]))
    if spacing < 2 * max(dx, dy):
        raise ValueError(f"spacing {spacing} is under two pixels")
    w, static = _resolve_weights(ds, weights)
    categorical = {**CATEGORICAL_DEFAULTS, **dict(categorical or {})}
    if layers is None:
        layers = [
            str(n)
            for n, v in ds.data_vars.items()
            if v.dtype.kind in "fiu"
            and {"y", "x"} <= set(v.dims)
            and n not in WEIGHT_LAYERS
            and n not in categorical
        ]
    categorical = {k: v for k, v in categorical.items() if k in ds}
    variogram = dict(variogram or {})
    if missing := set(variogram) - set(layers):
        raise ValueError(f"variogram given for layers not aggregated: {missing}")

    bounds = _id_range(ds, spacing)
    n_ids = bounds[2] * bounds[3]
    labels = _labels(ds, spacing, bounds)
    # Rows a hexagon reaches above or below its center, plus one.
    halo = int(np.ceil(spacing / SQRT3 / dy)) + 1

    def group_sum(da: xr.DataArray) -> xr.DataArray:
        return _group_sum(da, labels, n_ids)

    full = SQRT3 / 2.0 * spacing**2 / (dx * dy)
    mask = _hex_mask(spacing, dy, dx)
    out, shared = _aggregate_layers(
        ds,
        w,
        layers=layers,
        cell_sums=lambda v, w: _hex_cell_sums(v, w, labels, n_ids),
        median=lambda v, wv: _banded(
            _weighted_median_cells, [v, wv], labels, n_ids, halo, np.float64, np.nan
        ),
        full=full,
        stat=stat,
        nan_threshold=nan_threshold,
        variogram=variogram,
        rho_bar=lambda model: mean_mask_correlation(model, mask, (dy, dx)),
    )
    for name, how in categorical.items():
        var = ds[name]
        if not {"y", "x"} <= set(var.dims):
            continue
        fill = var.attrs.get("_FillValue", var.encoding.get("_FillValue"))
        fill = var.dtype.type(fill if fill is not None else 0 if how == "any" else -1)
        if how == "any":
            hit = ((var != fill) & (var != 0)).astype(np.uint8)
            reduced = (group_sum(hit) > 0).astype(np.uint8)
        else:
            reduced = _banded(
                _mode_cells, [var], labels, n_ids, halo, var.dtype, fill, fill=fill
            )
        out[name] = reduced.drop_attrs().assign_attrs(var.attrs, reduction=how)
    total = group_sum(static)
    xx, yy = xr.broadcast(ds["x"].astype(np.float64), ds["y"].astype(np.float64))
    shared["quality"] = (
        (total / full)
        .drop_attrs()
        .assign_attrs(long_name="mean static weight of the cell (over a full hexagon)")
    )
    for name, coord in (("centroid_x", xx), ("centroid_y", yy)):
        centroid = group_sum(static * coord) / total.where(total > 0)
        shared[name] = centroid.drop_attrs().assign_attrs(
            units=ds[name[-1]].attrs.get("units", ""),
            long_name="weighted centroid (static weights)",
        )
    pixels = np.bincount(labels.ravel(), minlength=n_ids)
    occupied = np.flatnonzero(pixels > 0)
    q0, r0, nq, _ = bounds
    q, r = occupied % nq + q0, occupied // nq + r0
    cx, cy = hex_centers(q, r, spacing)
    result = xr.Dataset({**out, **shared}).isel(cell=occupied)
    result = result.drop_vars(
        [c for c in result.coords if c != "time" and c not in ("spatial_ref",)]
    )
    result = result.rename({"cell": "point"}).assign_coords(
        cell=("point", occupied),
        q=("point", q),
        r=("point", r),
        x=("point", cx, ds["x"].attrs),
        y=("point", cy, ds["y"].attrs),
    )
    if "spatial_ref" in ds.coords:
        result = result.assign_coords(spatial_ref=ds.coords["spatial_ref"])
    result["pixels"] = ("point", pixels[occupied].astype(np.int32))
    result["pixels"].attrs["long_name"] = "grid pixels in the cell"
    result.attrs = {
        **{k: v for k, v in ds.attrs.items() if k != "history"},
        "history": ds.attrs.get("history", ""),
        "downsample_grid": "hexagon_planar",
        "hex_spacing": float(spacing),
        "downsample_stat": stat,
        "downsample_weights": (
            weights if isinstance(weights, str) or weights is None else "user"
        ),
    }
    return result
