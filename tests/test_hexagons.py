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


def _weighted_spread(v, w, value):
    """Reference within-cell spread: weighted std, RMS about value, MAD."""

    def wmedian(a):
        order = np.argsort(a)
        cw = np.cumsum(w[order])
        return a[order][np.argmax(cw >= 0.5 * cw[-1])]

    mean = np.sum(w * v) / w.sum()
    std = np.sqrt(np.sum(w * (v - mean) ** 2) / w.sum())
    rmse = np.sqrt(np.sum(w * (v - value) ** 2) / w.sum())
    return std, rmse, wmedian(np.abs(v - wmedian(v)))


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
    ds = dxr.chain(dxr.open_disp(stack_dir))
    out = downsample_hex(ds, spacing=300).compute()
    assert np.isfinite(out.displacement.isel(time=-1)).any()
    assert "temporal_coherence" not in out
    assert out.persistent_scatterer_mask.attrs["reduction"] == "any"
    assert 0 <= float(out.quality.max()) <= 1.05


def test_errors():
    ds, w = _dataset()
    with pytest.raises(ValueError, match="takes resolution"):
        downsample_hex(ds, spacing=300, grid="h3", weights=w)
    with pytest.raises(ValueError, match="takes spacing"):
        downsample_hex(ds, resolution=8, weights=w)
    with pytest.raises(ValueError, match="grid must be"):
        downsample_hex(ds, spacing=300, grid="s2", weights=w)
    with pytest.raises(ValueError, match="two pixels"):
        downsample_hex(ds, spacing=40, weights=w)
    with pytest.raises(ValueError, match="not aggregated"):
        downsample_hex(
            ds, spacing=300, weights=w, variogram={"velocity": VariogramModel(0.0, ())}
        )


def test_plot_hex(stack_dir):
    import matplotlib

    matplotlib.use("Agg")
    ds = dxr.chain(dxr.open_disp(stack_dir))
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
    v = ds.displacement
    for chunks in ({"y": 10}, {"y": 37, "time": 2}):
        whole = _banded(_weighted_median_cells, [v, w], labels, n_ids, float, np.nan)
        banded = _banded(
            _weighted_median_cells,
            [v.chunk(chunks), w.chunk(chunks)],
            labels,
            n_ids,
            float,
            np.nan,
        )
        np.testing.assert_array_equal(banded.compute(), whole)
        mask = ds.recommended_mask
        mode_whole = _banded(
            _mode_cells, [mask], labels, n_ids, np.uint8, 255, fill=255
        )
        mode_banded = _banded(
            _mode_cells,
            [mask.chunk(chunks)],
            labels,
            n_ids,
            np.uint8,
            255,
            fill=255,
        )
        np.testing.assert_array_equal(mode_banded.compute(), mode_whole)


# --- H3 cells and GeoDataFrame export (T27.8, second part) ---

h3 = pytest.importorskip("h3")


def _exact_h3(ds, resolution):
    import h3.api.numpy_int as h3n
    from pyproj import Transformer

    to_ll = Transformer.from_crs(ds.rio.crs, "EPSG:4326", always_xy=True)
    xx, yy = np.meshgrid(ds.x.values, ds.y.values)
    lon, lat = to_ll.transform(xx.ravel(), yy.ravel())
    cells = [h3n.latlng_to_cell(a, b, resolution) for a, b in zip(lat, lon)]
    return np.array(cells, dtype=np.uint64).reshape(xx.shape)


def test_h3_block_shortcut_is_exact():
    from disp_xr.hexagons import _h3_pixels

    ds, _ = _dataset(nt=1, ny=150, nx=170)
    fast = _h3_pixels(ds, 8, 30.0)  # stride 4: most pixels filled by blocks
    np.testing.assert_array_equal(fast, _exact_h3(ds, 8))


@pytest.mark.parametrize("stat", ["wmean", "wmedian"])
def test_h3_matches_brute_force(stat):
    ds, w = _dataset(nt=2, ny=90, nx=100, seed=2, chunks={"y": 23, "x": 31})
    out = downsample_hex(ds, grid="h3", resolution=8, stat=stat, weights=w).compute()
    exact = _exact_h3(ds, 8)
    assert set(out.h3_cell.values) == set(np.unique(exact))
    values, weights = ds.displacement.values, w.values
    for k, cell in enumerate(out.h3_cell.values):
        inside = exact == cell
        assert int(out.pixels[k]) == inside.sum()
        for t in range(values.shape[0]):
            v, wt = values[t][inside], weights[t][inside]
            ok = np.isfinite(v) & (wt > 0)
            if not ok.any():
                assert np.isnan(out.displacement[t, k])
                continue
            v, wt = v[ok], wt[ok]
            if stat == "wmean":
                ref = np.sum(wt * v) / wt.sum()
            else:
                order = np.argsort(v)
                cw = np.cumsum(wt[order])
                ref = v[order][np.argmax(cw >= 0.5 * cw[-1])]
            assert float(out.displacement[t, k]) == pytest.approx(ref, rel=1e-10)
    area = np.array(
        [h3.cell_area(h3.int_to_str(int(c)), unit="m^2") for c in out.h3_cell.values]
    )
    np.testing.assert_allclose(out.valid_fraction, out.n_valid / (area / 900.0))
    assert out.attrs["downsample_grid"] == "hexagon_h3"
    assert out.attrs["h3_resolution"] == 8


def test_h3_ids_shared_across_utm_zones():
    """The same ground in two UTM zones gets the same H3 cells."""
    from pyproj import Transformer

    lon0, lat0 = -120.0, 37.0  # on the 10/11 zone boundary
    frames = []
    for epsg in (32610, 32611):
        x0, y0 = Transformer.from_crs(4326, epsg, always_xy=True).transform(lon0, lat0)
        x = x0 - 1500 + 30.0 * np.arange(100) + 15
        y = y0 + 1500 - 30.0 * np.arange(100) - 15
        ds = xr.Dataset(
            {"displacement": (("y", "x"), np.ones((100, 100)))},
            coords={"y": y, "x": x},
        ).rio.write_crs(f"EPSG:{epsg}")
        frames.append(downsample_hex(ds, grid="h3", resolution=8, weights=None))
    a, b = frames
    shared = set(a.h3_cell.values) & set(b.h3_cell.values)
    assert len(shared) >= 4
    for cell in shared:
        ia = int(np.flatnonzero(a.h3_cell.values == cell)[0])
        ib = int(np.flatnonzero(b.h3_cell.values == cell)[0])
        assert float(a.lon[ia]) == float(b.lon[ib])
        assert float(a.lat[ia]) == float(b.lat[ib])


def test_h3_ids_follow_rows():
    from disp_xr.hexagons import _h3_cells, _row_windows

    ds, _ = _dataset(nt=1, ny=300, nx=60)
    cells = _h3_cells(ds, 8, 30.0, 30.0)
    _, width = _row_windows(cells.labels, (50,) * 6)
    assert width < cells.n_ids / 2  # a band holds a window of ids, not all


def test_h3_banded_median_matches_whole_frame():
    ds, w = _dataset(nt=2, ny=120, nx=80, seed=5)
    whole = downsample_hex(ds, grid="h3", resolution=8, stat="wmedian", weights=w)
    split = downsample_hex(
        ds.chunk({"y": 17, "x": 23}),
        grid="h3",
        resolution=8,
        stat="wmedian",
        weights=w.chunk({"y": 17, "x": 23}),
    ).compute()
    np.testing.assert_array_equal(split.displacement, whole.displacement)


@pytest.mark.parametrize("grid", ["planar", "h3"])
def test_to_geodataframe(grid):
    ds, w = _dataset(nt=3, ny=60, nx=70)
    kwargs = {"spacing": 300.0} if grid == "planar" else {"resolution": 8}
    hexes = downsample_hex(ds, grid=grid, weights=w, **kwargs).compute()
    gdf = dxr.to_geodataframe(hexes)
    assert len(gdf) == hexes.sizes["point"]
    assert gdf.crs.to_epsg() == 32610
    assert gdf.geometry.is_valid.all()
    np.testing.assert_allclose(gdf.geometry.centroid.x, hexes.x, atol=30)
    if grid == "planar":
        np.testing.assert_allclose(gdf.geometry.area, np.sqrt(3) / 2 * 300.0**2)
    else:
        area = [
            h3.cell_area(h3.int_to_str(int(c)), unit="m^2")
            for c in hexes.h3_cell.values
        ]
        np.testing.assert_allclose(gdf.geometry.area, area, rtol=0.01)
    long = dxr.to_geodataframe(hexes, time=None)
    assert len(long) == hexes.sizes["point"] * hexes.sizes["time"]
    assert {"time", "displacement", "geometry"} <= set(long.columns)


def test_plot_hex_h3():
    import matplotlib

    matplotlib.use("Agg")
    ds, w = _dataset(nt=1, ny=60, nx=70)
    times = np.array(["2020-01-01"], dtype="datetime64[ns]")
    ds, w = ds.assign_coords(time=times), w.assign_coords(time=times)
    hexes = downsample_hex(ds, grid="h3", resolution=8, weights=w)
    ax = dxr.plot.plot_hex(hexes)
    assert "H3 resolution 8" in ax.get_title(loc="left")


@pytest.mark.parametrize("stat", ["wmean", "wmedian"])
def test_spread_matches_brute_force(stat):
    ds, w = _dataset(chunks={"y": 7, "x": 9})
    out = downsample_hex(
        ds, spacing=SPACING, stat=stat, weights=w, spread=["std", "rmse", "mad"]
    ).compute()
    xx, yy = np.meshgrid(ds.x.values, ds.y.values)
    q, r = _nearest_center(xx, yy, SPACING)
    values, weights = ds.displacement.values, w.values
    for k, (cq, cr) in enumerate(zip(out.q.values, out.r.values)):
        inside = (q == cq) & (r == cr)
        for t in range(values.shape[0]):
            v, wt = values[t][inside], weights[t][inside]
            ok = np.isfinite(v) & (wt > 0)
            if not ok.any():
                assert np.isnan(out.displacement_mad[t, k])
                continue
            ref = _weighted_spread(v[ok], wt[ok], float(out.displacement[t, k]))
            got = [
                float(out[f"displacement_{n}"][t, k]) for n in ("std", "rmse", "mad")
            ]
            np.testing.assert_allclose(got, ref, rtol=1e-8, atol=1e-12)
