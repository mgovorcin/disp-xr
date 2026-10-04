"""Hexagonal downsampling (plan task T27.8)."""

import dask.array as darr
import numpy as np
import pytest
import xarray as xr

import disp_xr as dxr
from disp_xr.hexagons import (
    downsample_hex,
    hex_axial,
    hex_centers,
    hexagon_vertices,
)
from disp_xr.residuals import VariogramModel, mean_mask_correlation

SPACING = 97.0  # not a multiple of the 30 m pixel, so no pixel sits on an edge


def _nearest_center(x, y, spacing):
    """Brute force: the lattice center nearest each point, by distance."""
    q0, r0 = hex_axial(x, y, spacing)
    best_q, best_r = q0.copy(), r0.copy()
    best = np.full(x.shape, np.inf)
    for dq in range(-2, 3):
        for dr in range(-2, 3):
            cx, cy = hex_centers(q0 + dq, r0 + dr, spacing)
            d = np.hypot(x - cx, y - cy)
            closer = d < best
            best = np.where(closer, d, best)
            best_q = np.where(closer, q0 + dq, best_q)
            best_r = np.where(closer, r0 + dr, best_r)
    return best_q, best_r


def _dataset(nt=3, ny=20, nx=23, seed=0, chunks=None):
    rng = np.random.default_rng(seed)
    values = rng.normal(size=(nt, ny, nx))
    values[rng.random(values.shape) < 0.25] = np.nan
    weights = rng.random((nt, ny, nx))
    x = 500000.0 + 30.0 * np.arange(nx) + 15
    y = 4100000.0 - 30.0 * np.arange(ny) - 15
    mask = (rng.random((nt, ny, nx)) < 0.7).astype(np.uint8)
    ds = xr.Dataset(
        {
            "displacement": (("time", "y", "x"), values, {"units": "meters"}),
            "recommended_mask": (("time", "y", "x"), mask, {"_FillValue": 255}),
        },
        coords={"time": np.arange(nt), "y": y, "x": x},
    ).rio.write_crs("EPSG:32610")
    w = xr.DataArray(weights, dims=("time", "y", "x"), coords=ds.displacement.coords)
    if chunks:
        ds, w = ds.chunk(chunks), w.chunk(chunks)
    return ds, w


def _brute(ds, w, spacing, stat):
    xx, yy = np.meshgrid(ds.x.values, ds.y.values)
    q, r = _nearest_center(xx, yy, spacing)
    cells = sorted(set(zip(q.ravel(), r.ravel())))
    values, weights = ds.displacement.values, w.values
    out = np.full((values.shape[0], len(cells)), np.nan)
    sigma = np.full_like(out, np.nan)
    modes = np.zeros((values.shape[0], len(cells)), np.uint8)
    for k, (cq, cr) in enumerate(cells):
        inside = (q == cq) & (r == cr)
        for t in range(values.shape[0]):
            v, wt = values[t][inside], weights[t][inside]
            counts = np.bincount(ds.recommended_mask.values[t][inside], minlength=2)
            modes[t, k] = np.argmax(counts)
            ok = np.isfinite(v) & (wt > 0)
            if not ok.any():
                continue
            v, wt = v[ok], wt[ok]
            mean = np.sum(wt * v) / wt.sum()
            if stat == "wmean":
                out[t, k] = mean
            else:
                order = np.argsort(v)
                cw = np.cumsum(wt[order])
                out[t, k] = v[order][np.argmax(cw >= 0.5 * cw[-1])]
            spread = np.sqrt(np.sum(wt * (v - mean) ** 2) / wt.sum())  # two-pass
            sigma[t, k] = spread / np.sqrt(wt.sum() ** 2 / np.sum(wt**2))
    return cells, out, sigma, modes


def test_axial_round_trip_and_nearest():
    rng = np.random.default_rng(3)
    x, y = rng.uniform(-5e5, 5e5, 10_000), rng.uniform(3e6, 5e6, 10_000)
    q, r = hex_axial(x, y, SPACING)
    bq, br = _nearest_center(x, y, SPACING)
    np.testing.assert_array_equal(q, bq)
    np.testing.assert_array_equal(r, br)
    cx, cy = hex_centers(q, r, SPACING)
    assert np.hypot(x - cx, y - cy).max() <= SPACING / np.sqrt(3) + 1e-6
    # Neighbors are all one spacing apart.
    nx, ny = hex_centers(np.array([1, 0, -1]), np.array([0, 1, 1]), SPACING)
    np.testing.assert_allclose(np.hypot(nx, ny), SPACING)


def test_vertices():
    verts = hexagon_vertices(np.array([0.0]), np.array([0.0]), SPACING)
    assert verts.shape == (1, 6, 2)
    x, y = verts[0].T
    area = 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    assert area == pytest.approx(np.sqrt(3) / 2 * SPACING**2)


@pytest.mark.parametrize("stat", ["wmean", "wmedian"])
def test_matches_brute_force(stat):
    ds, w = _dataset(chunks={"y": 7, "x": 9})
    out = downsample_hex(ds, spacing=SPACING, stat=stat, weights=w)
    assert isinstance(out.displacement.data, darr.Array)
    out = out.compute()
    cells, ref, sigma, modes = _brute(ds, w, SPACING, stat)
    assert list(zip(out.q.values, out.r.values)) == sorted(
        cells, key=lambda c: (c[1], c[0])
    )
    order = [cells.index(c) for c in zip(out.q.values, out.r.values)]
    np.testing.assert_allclose(
        out.displacement, ref[:, order], rtol=1e-10, equal_nan=True
    )
    np.testing.assert_allclose(
        out.displacement_sigma, sigma[:, order], rtol=1e-8, atol=1e-14, equal_nan=True
    )
    np.testing.assert_array_equal(out.recommended_mask, modes[:, order])


def test_chunk_independent_and_layout():
    whole_ds, w = _dataset()
    split_ds, ws = _dataset(chunks={"y": 5, "x": 4})
    a = downsample_hex(whole_ds, spacing=SPACING, stat="wmedian", weights=w).compute()
    b = downsample_hex(split_ds, spacing=SPACING, stat="wmedian", weights=ws).compute()
    xr.testing.assert_allclose(a.displacement, b.displacement)
    assert a.displacement.dims == ("time", "point")
    assert np.all(np.diff(a.cell.values) > 0)
    assert int(a.pixels.sum()) == 20 * 23  # every pixel in exactly one cell
    assert a.attrs["downsample_grid"] == "hexagon_planar"
    assert a.rio.crs is not None or "spatial_ref" in a.coords
    full = np.sqrt(3) / 2 * SPACING**2 / 900
    np.testing.assert_allclose(a.valid_fraction, a.n_valid / full)


def test_overlapping_frames_share_cell_ids():
    ds, w = _dataset(ny=30, nx=30)
    left = downsample_hex(
        ds.isel(x=slice(0, 20)), spacing=SPACING, weights=w.isel(x=slice(0, 20))
    )
    right = downsample_hex(
        ds.isel(x=slice(10, 30)), spacing=SPACING, weights=w.isel(x=slice(10, 30))
    )
    shared = set(zip(left.q.values, left.r.values)) & set(
        zip(right.q.values, right.r.values)
    )
    assert shared
    for qq, rr in shared:
        a = left.where((left.q == qq) & (left.r == rr), drop=True)
        b = right.where((right.q == qq) & (right.r == rr), drop=True)
        np.testing.assert_allclose(a.x, b.x)
        np.testing.assert_allclose(a.y, b.y)


def test_variogram_sigma_on_hexagons():
    ds, w = _dataset(nt=1, ny=30, nx=30)
    model = VariogramModel(1.0, (("exponential", 4.0, 900.0),))
    out = downsample_hex(
        ds, spacing=SPACING * 3, weights=None, variogram={"displacement": model}
    ).compute()
    filled = out.n_valid > 0
    floor = np.sqrt(model.sill / out.n_valid.where(filled))
    assert (
        out.displacement_sigma_corr.where(filled) >= floor - 1e-12
    ).sum() == filled.sum()
    rho = out.displacement_sigma_corr.attrs["mean_block_correlation"]
    assert 0 < rho < 1
    square = np.ones((5, 5))
    assert mean_mask_correlation(model, square, 30.0) == pytest.approx(
        dxr.residuals.mean_block_correlation(model, 5, 30.0)
    )


def test_quality_weights_on_fixture(stack_dir):
    ds = dxr.stack_disp(dxr.open_disp(stack_dir))
    out = downsample_hex(ds, spacing=300).compute()
    assert np.isfinite(out.displacement.isel(time=-1)).any()
    assert "temporal_coherence" not in out
    assert out.persistent_scatterer_mask.attrs["reduction"] == "any"
    assert 0 <= float(out.quality.max()) <= 1.05


def test_errors():
    ds, w = _dataset()
    with pytest.raises(ValueError, match="planar"):
        downsample_hex(ds, spacing=300, grid="h3", weights=w)
    with pytest.raises(ValueError, match="two pixels"):
        downsample_hex(ds, spacing=40, weights=w)
    with pytest.raises(ValueError, match="not aggregated"):
        downsample_hex(
            ds, spacing=300, weights=w, variogram={"velocity": VariogramModel(0.0, ())}
        )


def test_plot_hex(stack_dir):
    import matplotlib

    matplotlib.use("Agg")
    ds = dxr.stack_disp(dxr.open_disp(stack_dir))
    hexes = downsample_hex(ds, spacing=300)
    ax = dxr.plot.plot_hex(hexes)
    assert len(ax.collections) == 1
    assert "300 m hexagons" in ax.get_title(loc="left")


def test_group_sum_windows_match_dense():
    from disp_xr.hexagons import _group_sum, _id_range, _labels, _row_windows

    ds, _ = _dataset(nt=4, ny=200, nx=40)
    bounds = _id_range(ds, SPACING)
    n_ids = bounds[2] * bounds[3]
    labels = _labels(ds, SPACING, bounds)
    values = ds.displacement.fillna(0.0)
    dense = _group_sum(values, labels, n_ids)  # numpy path
    lazy = _group_sum(values.chunk({"y": 13, "x": 17}), labels, n_ids)
    np.testing.assert_allclose(lazy.compute(), dense, rtol=1e-12)
    # Each block's partial covers a window of cells, not all of them.
    _, width = _row_windows(labels, values.chunk({"y": 13}).chunks[1])
    assert width < n_ids / 5


def test_banded_median_and_mode_match_whole_frame():
    from disp_xr.hexagons import (
        _banded,
        _id_range,
        _labels,
        _mode_cells,
        _weighted_median_cells,
    )

    ds, w = _dataset(nt=5, ny=150, nx=31, seed=4)
    bounds = _id_range(ds, SPACING)
    n_ids = bounds[2] * bounds[3]
    labels = _labels(ds, SPACING, bounds)
    halo = int(np.ceil(SPACING / np.sqrt(3) / 30.0)) + 1
    v = ds.displacement
    for chunks in ({"y": 10}, {"y": 37, "time": 2}):
        whole = _banded(
            _weighted_median_cells, [v, w], labels, n_ids, halo, float, np.nan
        )
        banded = _banded(
            _weighted_median_cells,
            [v.chunk(chunks), w.chunk(chunks)],
            labels,
            n_ids,
            halo,
            float,
            np.nan,
        )
        np.testing.assert_array_equal(banded.compute(), whole)
        mask = ds.recommended_mask
        mode_whole = _banded(
            _mode_cells, [mask], labels, n_ids, halo, np.uint8, 255, fill=255
        )
        mode_banded = _banded(
            _mode_cells,
            [mask.chunk(chunks)],
            labels,
            n_ids,
            halo,
            np.uint8,
            255,
            fill=255,
        )
        np.testing.assert_array_equal(mode_banded.compute(), mode_whole)
