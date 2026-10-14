"""Re-referencing (plan task T11).

The ``get_reference_values`` cases are ported from opera-utils
tests/test_disp_reference.py (commit 97f77d9), Copyright (c) 2022 California
Institute of Technology, BSD-3-Clause; modified for disp-xr.
"""

import json

import dask
import numpy as np
import pytest
import xarray as xr
from fixtures.make_disp import CYCLE_PIXELS, LOW_COHERENCE, PS_PIXELS
from pyproj import Transformer

import disp_xr as dxr
from disp_xr.reference import (
    OutOfBoundsError,
    ReferenceMethod,
    find_stable_point,
    get_reference_values,
    rereference,
)

# --- get_reference_values (ported) ---------------------------------------------


@pytest.fixture
def sample_da():
    rng = np.random.default_rng(42)
    return xr.DataArray(
        rng.standard_normal((5, 10, 15)),
        dims=["time", "y", "x"],
        coords={"time": np.arange(5), "y": np.arange(10.0), "x": np.arange(15.0)},
    )


def test_point_rowcol(sample_da):
    ref = get_reference_values(sample_da, method=ReferenceMethod.point, row=5, col=7)
    xr.testing.assert_equal(ref, sample_da[:, 5, 7])


def test_point_string_method(sample_da):
    ref = get_reference_values(sample_da, method="point", row=5, col=7)
    xr.testing.assert_equal(ref, sample_da[:, 5, 7])


def test_point_missing_params(sample_da):
    with pytest.raises(ValueError, match=r"Need \(row, col\) or \(lon, lat\)"):
        get_reference_values(sample_da, method="point")


@pytest.mark.parametrize("row, col", [(-1, 0), (0, 15), (10, 0)])
def test_point_out_of_bounds(sample_da, row, col):
    """Regression: upstream bug 5, negative indices wrapped around."""
    with pytest.raises(OutOfBoundsError):
        get_reference_values(sample_da, method="point", row=row, col=col)


def test_median(sample_da):
    ref = get_reference_values(sample_da, method="median")
    np.testing.assert_allclose(ref, np.median(sample_da.values, axis=(1, 2)))


def test_median_lazy_matches_numpy(sample_da):
    """The median over (y, x) of a chunked cube equals the NumPy median."""
    lazy = sample_da.chunk({"time": 1, "y": 3, "x": 4})
    ref = get_reference_values(lazy, method="median")
    assert ref.chunks is not None
    np.testing.assert_allclose(ref, np.median(sample_da.values, axis=(1, 2)))


def test_nanmedian_maps_matches_numpy():
    from disp_xr.reference import _nanmedian_maps

    rng = np.random.default_rng(3)
    a = rng.normal(size=(6, 21, 30)).astype("float32")
    a[a > 0.7] = np.nan
    a[2] = np.nan  # an all-NaN map
    good = rng.random((21, 30)) > 0.3
    with np.errstate(all="ignore"), pytest.warns(RuntimeWarning):
        expected = np.nanmedian(a, axis=(1, 2))
    np.testing.assert_array_equal(_nanmedian_maps(a), expected)
    masked = np.where(good, a, np.nan)
    with pytest.warns(RuntimeWarning):
        expected = np.nanmedian(masked, axis=(1, 2))
    np.testing.assert_array_equal(_nanmedian_maps(a, good), expected)
    assert _nanmedian_maps(a).dtype == np.float32


def test_median_honors_mask(sample_da):
    """Regression: upstream bug 5, the median ignored good_pixel_mask."""
    good = np.zeros((10, 15), bool)
    good[2:4, 3:6] = True
    ref = get_reference_values(sample_da, method="median", good_pixel_mask=good)
    np.testing.assert_allclose(
        ref, np.median(sample_da.values[:, 2:4, 3:6], axis=(1, 2))
    )


@pytest.mark.parametrize("n", [1, 2, 3])
def test_border(sample_da, n):
    mask = np.zeros((10, 15), bool)
    mask[:n], mask[-n:], mask[:, :n], mask[:, -n:] = True, True, True, True
    ref = get_reference_values(sample_da, method="border", border_pixels=n)
    np.testing.assert_allclose(ref, np.median(sample_da.values[:, mask], axis=1))


def test_high_coherence(sample_da):
    good = np.random.default_rng(0).random((10, 15)) > 0.5
    ref = get_reference_values(sample_da, method="high_coherence", good_pixel_mask=good)
    np.testing.assert_allclose(ref, np.median(sample_da.values[:, good], axis=1))


def test_high_coherence_requires_mask(sample_da):
    """Regression: upstream bug 5, no mask silently used every pixel."""
    with pytest.raises(ValueError, match="needs good_pixel_mask"):
        get_reference_values(sample_da, method="high_coherence")


def test_unknown_method(sample_da):
    with pytest.raises(ValueError, match="is not a valid ReferenceMethod"):
        get_reference_values(sample_da, method="unknown_method")


# --- rereference on the synthetic frame ---------------------------------------


@pytest.fixture
def ds(stack_dir):
    return dxr.chain(dxr.open_disp(stack_dir))


def _lonlat(ds, row, col, dx=0.0, dy=0.0):
    to_lonlat = Transformer.from_crs(ds.rio.crs, "EPSG:4326", always_xy=True)
    return to_lonlat.transform(float(ds.x[col]) + dx, float(ds.y[row]) + dy)


def test_point_reference_pixel_is_zero(ds):
    out = rereference(ds, "point", row=40, col=40)
    assert (out.displacement[:, 40, 40] == 0).all()
    np.testing.assert_allclose(
        out.displacement[:, 5, 50],
        ds.displacement[:, 5, 50] - ds.displacement[:, 40, 40],
    )
    assert json.loads(out.attrs["reference"]) == {
        "method": "point",
        "row": 40,
        "col": 40,
        "x": float(ds.x[40]),
        "y": float(ds.y[40]),
    }


def test_point_from_lonlat(ds):
    lon, lat = _lonlat(ds, 40, 41)
    out = rereference(ds, "point", lon=lon, lat=lat)
    assert json.loads(out.attrs["reference"])["col"] == 41


def test_lonlat_outside_frame(ds):
    lon, lat = _lonlat(ds, 0, 0)
    with pytest.raises(OutOfBoundsError, match="outside"):
        rereference(ds, "point", lon=lon - 1, lat=lat)


def test_masked_point_raises(ds):
    masked = dxr.mask(ds, "strict")
    with pytest.raises(OutOfBoundsError, match="is masked"):
        rereference(masked.isel(time=slice(1, None)), "point", row=0, col=0)


def test_area_median(ds):
    # Corners half a pixel off the centers, so the lon/lat box (slightly
    # rotated on the UTM grid) holds exactly rows 35-44 and columns 31-40.
    (west, south) = _lonlat(ds, 44, 31, dx=-15, dy=-15)
    (east, north) = _lonlat(ds, 35, 40, dx=15, dy=15)
    out = rereference(ds, "area", bbox=(west, south, east, north))
    area = ds.displacement.values[:, 35:45, 31:41]
    np.testing.assert_allclose(
        out.displacement[:, 0, 0],
        ds.displacement[:, 0, 0] - np.median(area, axis=(1, 2)),
        atol=1e-6,
    )
    assert "bbox" in json.loads(out.attrs["reference"])


def test_area_reads_only_the_box(ds):
    # Every chunk read is logged; the reference values must come from the
    # chunks under the box, not from the whole frame.
    read = []

    def logged(block, block_info=None):
        read.append(tuple(block_info[0]["chunk-location"][1:]))
        return block

    chunked = ds.chunk({"time": -1, "y": 16, "x": 16})
    disp = chunked.displacement
    chunked["displacement"] = disp.copy(
        data=disp.data.map_blocks(logged, dtype=disp.dtype)
    )
    (west, south) = _lonlat(ds, 44, 31, dx=-15, dy=-15)
    (east, north) = _lonlat(ds, 35, 40, dx=15, dy=15)
    out = rereference(chunked, "area", bbox=(west, south, east, north))
    expected = rereference(ds, "area", bbox=(west, south, east, north))
    read.clear()
    ref = (chunked.displacement - out.displacement).isel(y=0, x=0)
    np.testing.assert_allclose(
        ref.values, (ds.displacement - expected.displacement)[:, 0, 0], atol=1e-6
    )
    # Rows 35-44 and columns 31-40 lie in chunk rows 2 and columns 1-2; the
    # pixel (0, 0) read for the check is chunk (0, 0).
    assert set(read) <= {(2, 1), (2, 2), (0, 0)}


def test_area_needs_one_shape(ds):
    with pytest.raises(ValueError, match="exactly one of bbox or geometry"):
        rereference(ds, "area")


def test_auto_is_deterministic_and_avoids_bad_pixels(ds):
    row, col = find_stable_point(ds)
    assert (row, col) == find_stable_point(ds)
    assert (row, col) in PS_PIXELS
    assert (row, col) not in CYCLE_PIXELS
    out = rereference(ds, "auto")
    assert (out.displacement[:, row, col] == 0).all()
    assert json.loads(out.attrs["reference"])["method"] == "auto"


def test_auto_skips_phase_jumps(ds):
    only = xr.zeros_like(ds.temporal_coherence).where(False, 0.1)
    for y, x in CYCLE_PIXELS:
        only[:, y, x] = 0.99
    trial = ds.assign(temporal_coherence=only)
    with pytest.raises(ValueError, match="No stable pixel"):
        find_stable_point(trial)


def test_lazy_median_and_accessor_history(ds):
    def fail(*_args, **_kwargs):
        raise AssertionError("computed")

    with dask.config.set(scheduler=fail):
        out = ds.disp.rereference("median", good_pixels=dxr.valid_mask(ds, "strict"))
    assert isinstance(out.displacement.data, dask.array.Array)
    assert out.disp.history[-1]["op"] == "rereference"


@pytest.mark.parametrize("method", ["median", "high_coherence"])
def test_whole_frame_median_keeps_chunks_and_values(ds, method):
    """Per-epoch medians are subtracted one epoch at a time, chunks restored."""
    tiled = ds.chunk({"time": 1, "y": 16, "x": 24})
    good = dxr.valid_mask(ds, "strict")
    out = rereference(tiled, method, good_pixels=good)
    assert out.displacement.chunks == tiled.displacement.chunks
    assert not any("p2p" in name for name in out.displacement.data.dask.layers)
    values = ds.displacement.values
    with pytest.warns(RuntimeWarning):  # the all-NaN epochs
        ref = np.nanmedian(np.where(good.values, values, np.nan), axis=(1, 2))
    np.testing.assert_array_equal(out.displacement.values, values - ref[:, None, None])


def test_none_is_identity(ds):
    xr.testing.assert_identical(rereference(ds, "none").displacement, ds.displacement)


def test_low_coherence_patch_is_not_auto_reference(ds):
    row, col = find_stable_point(ds)
    assert not (
        LOW_COHERENCE[0].start <= row < LOW_COHERENCE[0].stop
        and LOW_COHERENCE[1].start <= col < LOW_COHERENCE[1].stop
    )
