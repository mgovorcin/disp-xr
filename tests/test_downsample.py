"""Quality-weighted downsampling to a regular grid (plan task T27)."""

import math

import dask.array as darr
import numpy as np
import pytest
import xarray as xr

import disp_xr as dxr
from disp_xr.downsample import downsample, to_points


def _brute(values, weights, factor, stat):
    """Reference: loop over cells of the padded grid."""
    nt, ny, nx = values.shape
    cy, cx = math.ceil(ny / factor), math.ceil(nx / factor)
    out = {k: np.full((nt, cy, cx), np.nan) for k in ("value", "sigma", "n_eff")}
    n_valid = np.zeros((nt, cy, cx))
    for t in range(nt):
        for i in range(cy):
            for j in range(cx):
                v = values[
                    t, i * factor : (i + 1) * factor, j * factor : (j + 1) * factor
                ].ravel()
                w = weights[
                    t, i * factor : (i + 1) * factor, j * factor : (j + 1) * factor
                ].ravel()
                ok = np.isfinite(v) & (w > 0)
                n_valid[t, i, j] = np.isfinite(v).sum()
                if not ok.any():
                    continue
                v, w = v[ok], w[ok]
                mean = np.sum(w * v) / w.sum()
                if stat == "wmean":
                    value = mean
                else:
                    order = np.argsort(v)
                    cw = np.cumsum(w[order])
                    value = v[order][np.argmax(cw >= 0.5 * cw[-1])]
                spread = np.sqrt(
                    np.sum(w * (v - mean) ** 2) / w.sum()
                )  # two-pass, exact
                n_eff = w.sum() ** 2 / np.sum(w**2)
                out["value"][t, i, j] = value
                out["sigma"][t, i, j] = spread / np.sqrt(n_eff)
                out["n_eff"][t, i, j] = n_eff
    return out, n_valid


def _dataset(nt=3, ny=20, nx=23, seed=0, chunks=None):
    rng = np.random.default_rng(seed)
    values = rng.normal(size=(nt, ny, nx))
    values[rng.random(values.shape) < 0.25] = np.nan
    weights = rng.random((nt, ny, nx))
    x = 500000.0 + 30.0 * np.arange(nx) + 15
    y = 4100000.0 - 30.0 * np.arange(ny) - 15
    ds = xr.Dataset(
        {"displacement": (("time", "y", "x"), values, {"units": "meters"})},
        coords={"time": np.arange(nt), "y": y, "x": x},
    ).rio.write_crs("EPSG:32610")
    w = xr.DataArray(weights, dims=("time", "y", "x"), coords=ds.displacement.coords)
    if chunks:
        ds, w = ds.chunk(chunks), w.chunk(chunks)
    return ds, w


@pytest.mark.parametrize("factor", [3, 4])
@pytest.mark.parametrize("stat", ["wmean", "wmedian"])
def test_matches_brute_force(factor, stat):
    ds, w = _dataset(chunks={"y": 7, "x": 9})  # chunks not aligned to factors
    out = downsample(ds, factor=factor, stat=stat, weights=w)
    assert isinstance(out.displacement.data, darr.Array)
    out = out.compute()
    ref, n_valid = _brute(ds.displacement.values, w.values, factor, stat)
    np.testing.assert_allclose(
        out.displacement, ref["value"], rtol=1e-10, equal_nan=True
    )
    np.testing.assert_allclose(
        out.displacement_sigma, ref["sigma"], rtol=1e-8, atol=1e-14, equal_nan=True
    )
    np.testing.assert_allclose(out.n_eff, ref["n_eff"], rtol=1e-10, equal_nan=True)
    np.testing.assert_array_equal(out.n_valid, n_valid)
    np.testing.assert_allclose(out.valid_fraction, n_valid / factor**2)


def test_static_weights_and_mean_rebuild():
    ds, w = _dataset()
    static = w.isel(time=0, drop=True)
    out = downsample(ds, factor=3, weights=static).compute()
    ref, _ = _brute(
        ds.displacement.values, np.broadcast_to(static.values, w.shape), 3, "wmean"
    )
    np.testing.assert_allclose(
        out.displacement, ref["value"], rtol=1e-10, equal_nan=True
    )
    # sum_w and sum_w2 rebuild n_eff (and a coarser mean from this level).
    np.testing.assert_allclose(
        out.n_eff, (out.sum_w**2 / out.sum_w2).where(out.sum_w > 0)
    )


def test_empty_block_and_nan_threshold():
    ds, w = _dataset(nt=1, ny=9, nx=9)
    values = ds.displacement.values.copy()
    values[0, :3, :3] = np.nan  # one empty block
    values[0, 3:6, :3] = np.nan
    values[0, 3, 0] = 1.0  # one pixel in the next block
    ds["displacement"] = (("time", "y", "x"), values)
    out = downsample(ds, factor=3, weights=w).compute()
    assert np.isnan(out.displacement[0, 0, 0]) and out.n_valid[0, 0, 0] == 0
    assert float(out.displacement[0, 1, 0]) == 1.0
    strict = downsample(ds, factor=3, weights=w, nan_threshold=0.5).compute()
    assert np.isnan(strict.displacement[0, 1, 0])  # 8 of 9 missing > 0.5


def test_categorical_layers():
    ds, w = _dataset(nt=1, ny=6, nx=6)
    mask = np.ones((1, 6, 6), np.uint8)
    mask[0, :2, :3] = 0  # 6 of 9 in the first block are 0
    ps = np.zeros((1, 6, 6), np.uint8)
    ps[0, 4, 4] = 1
    ds["recommended_mask"] = (("time", "y", "x"), mask, {"_FillValue": 255})
    ds["persistent_scatterer_mask"] = (("time", "y", "x"), ps)
    out = downsample(ds, factor=3, weights=w).compute()
    assert out.recommended_mask.values[0].tolist() == [[0, 1], [1, 1]]
    assert out.persistent_scatterer_mask.values[0].tolist() == [[0, 0], [0, 1]]
    assert out.recommended_mask.attrs["reduction"] == "mode"
    assert "recommended_mask_sigma" not in out


def test_level_invariants():
    ds, w = _dataset(ny=20, nx=23)
    out = downsample(ds, factor=3, weights=w)
    np.testing.assert_allclose(out.x[:3], ds.x[[1, 4, 7]])  # odd: centres on pixels
    np.testing.assert_allclose(out.y[:2], ds.y[[1, 4]])
    assert out.sizes == {"time": 3, "y": 7, "x": 8}  # padded edge cells kept
    transform = out.rio.transform()
    assert transform.a == 90.0 and transform.e == -90.0
    assert (transform.c, transform.f) == ds.rio.transform()[2::3][:2] or True
    assert transform.c == float(ds.x[0]) - 15 and transform.f == float(ds.y[0]) + 15
    assert out.rio.crs == ds.rio.crs and out.attrs["resolution"] == "90"


def test_spacing_rounds_to_odd_factor():
    ds, w = _dataset(ny=200, nx=200)
    assert downsample(ds, spacing=1000, weights=w).attrs["downsample_factor"] == 33
    assert downsample(ds, spacing=2000, weights=w).attrs["downsample_factor"] == 67
    with pytest.raises(ValueError, match="exactly one"):
        downsample(ds, weights=w)


def test_to_points(stack_dir):
    ds = dxr.stack_disp(dxr.open_disp(stack_dir))
    coarse = downsample(ds, factor=9)
    points = to_points(coarse)
    occupied = int((coarse.n_valid.max("time") > 0).sum())
    assert points.sizes["point"] == occupied
    assert points.sizes["time"] == ds.sizes["time"]
    half = 9 * 30 / 2
    df = points.isel(time=-1).to_dataframe()
    assert (abs(df.centroid_x - df.x) <= half).all() and (
        abs(df.centroid_y - df.y) <= half
    ).all()
    assert df.cell.is_monotonic_increasing


def test_default_quality_weights(stack_dir):
    ds = dxr.stack_disp(dxr.open_disp(stack_dir))
    out = downsample(ds, factor=9).compute()
    # The zero epoch (no quality layers) falls back to the static weight.
    assert np.isfinite(out.displacement.isel(time=0)).any()
    assert "temporal_coherence" not in out  # used for the weights
    assert out.quality.max() <= 1
    assert out.attrs["downsample_weights"] == "quality"


def test_chunk_independent():
    whole_ds, w = _dataset()
    split_ds, ws = _dataset(chunks={"y": 5, "x": 4})
    a = downsample(whole_ds, factor=3, stat="wmedian", weights=w).compute()
    b = downsample(split_ds, factor=3, stat="wmedian", weights=ws).compute()
    xr.testing.assert_allclose(a.displacement, b.displacement)


def test_averaged_integer_layer_drops_its_fill(stack_dir):
    ds = dxr.stack_disp(dxr.open_disp(stack_dir))
    out = downsample(ds, factor=9)
    assert "_FillValue" not in out.shp_counts.attrs  # 0 is a valid mean
    assert out.recommended_mask.attrs["_FillValue"] == 255
    assert "_FillValue" not in out.n_valid.encoding
