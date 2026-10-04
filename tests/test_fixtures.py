"""The synthetic DISP stacks match the real layout and their own truth."""

import h5py
import numpy as np
import pytest
import xarray as xr
from fixtures.make_disp import (
    PRODUCTION_2026,
    VARIANTS,
    WATER_COLUMNS,
    make_disp_stack,
)

LAYERS_2D = {
    "displacement",
    "short_wavelength_displacement",
    "temporal_coherence",
    "phase_similarity",
    "estimated_phase_quality",
    "timeseries_inversion_residuals",
    "recommended_mask",
    "water_mask",
    "persistent_scatterer_mask",
    "connected_component_labels",
    "shp_counts",
}


def test_granule_layout_matches_real_products(two_ministacks):
    with h5py.File(two_ministacks.paths[0]) as f:
        assert set(f) >= LAYERS_2D | {"corrections", "identification", "metadata"}
        for name in LAYERS_2D:
            ds = f[name]
            assert ds.ndim == 2
            assert ds.compression == "gzip"
            assert ds.shuffle
            assert ds.attrs["grid_mapping"] == "spatial_ref"
        assert f["displacement"].dtype == np.float32
        assert f["recommended_mask"].dtype == np.uint8
        assert f["connected_component_labels"].dtype == np.uint16
        assert f["time"].dtype == np.float64
        assert h5py.h5ds.is_scale(f["time"].id)
        assert not h5py.h5ds.is_scale(f["reference_time"].id)
        assert "crs_wkt" in f["spatial_ref"].attrs
        assert set(f["corrections"]) >= {"ionospheric_delay", "solid_earth_tide"}


def test_granules_decode_to_their_filename_dates(two_ministacks):
    for path, secondary, reference in zip(
        two_ministacks.paths, two_ministacks.times[1:], two_ministacks.reference_times
    ):
        # Files carry microseconds like real granules; names and the truth
        # are whole seconds.
        ds = xr.open_dataset(path, engine="h5netcdf")
        assert ds.time.dt.floor("s").values[0] == np.datetime64(secondary)
        assert ds.reference_time.dt.floor("s").values[0] == np.datetime64(reference)
        assert ds.time.dt.microsecond.values[0] == 435763
        assert secondary.strftime("%Y%m%dT%H%M%S") in path.name


def test_next_reference_is_last_secondary_of_previous_ministack(disp_stacks):
    stack = disp_stacks["three_ministacks"]
    refs = stack.reference_times.unique()
    assert len(refs) == 3
    for later in refs[1:]:
        assert later in stack.times[1:]


def test_granule_displacement_is_cumulative_difference(two_ministacks):
    stack = two_ministacks
    k = len(stack.paths) - 1
    ds = xr.open_dataset(stack.paths[k], engine="h5netcdf")
    i_sec = stack.times.get_loc(stack.times[k + 1])
    i_ref = stack.times.get_loc(stack.reference_times[k])
    expected = stack.cumulative[i_sec] - stack.cumulative[i_ref]
    np.testing.assert_allclose(ds.displacement.values, expected, atol=1e-6)


def test_quality_features_sit_at_fixed_pixels(two_ministacks):
    ds = xr.open_dataset(two_ministacks.paths[0], engine="h5netcdf")
    water = ds.water_mask.values
    assert (water[:, -WATER_COLUMNS:] == 0).all()
    assert (water[:, :-WATER_COLUMNS] == 1).all()
    recommended = ds.recommended_mask.values
    assert recommended[0, 0] == 0  # low coherence and low similarity
    assert recommended[21, 21] == 0  # unwrapping failed
    assert recommended[40, 40] == 1


def test_mixed_dtype_second_ministack_uses_2026_style(disp_stacks):
    stack = disp_stacks["mixed_dtype"]
    first, last = stack.paths[0], stack.paths[-1]
    assert PRODUCTION_2026 in last.name and PRODUCTION_2026 not in first.name
    with h5py.File(first) as f1, h5py.File(last) as f2:
        assert f1["displacement"].dtype == np.float32
        assert f2["displacement"].dtype == np.float64
        assert "2010-01-01" in f2["time"].attrs["units"]
        assert "2010-01-01" not in f1["time"].attrs["units"]


def test_duplicate_has_older_version_and_production(disp_stacks):
    stack = disp_stacks["duplicate"]
    older, newer = stack.extra["duplicate"], stack.extra["duplicate_of"]
    assert "_v0.9_" in older.name and "_v1.0_" in newer.name
    assert older.name.split("_")[-1] < newer.name.split("_")[-1]


def test_two_utm_zones_frames_have_different_crs(disp_stacks):
    first = disp_stacks["two_utm_zones"]
    second = first.extra["second_frame"]
    assert (first.epsg, second.epsg) == (32610, 32611)


def test_with_gaps_drops_two_secondaries(disp_stacks):
    assert len(disp_stacks["with_gaps"].paths) == 8


def test_unknown_variant_raises(tmp_path):
    with pytest.raises(ValueError, match="Unknown variant 'nope'"):
        make_disp_stack(tmp_path, "nope")


def test_every_variant_is_listed():
    assert len(VARIANTS) == len(set(VARIANTS)) == 7


def test_nisar_granules_use_sas_names_and_nisar_title(disp_stacks):
    stack = disp_stacks["nisar"]
    assert stack.paths[0].name == "20200101_20200113.nc"
    with h5py.File(stack.paths[0]) as f:
        assert f.attrs["title"] == "OPERA_L3_DISP-NISAR Product"
        assert f["identification/product_version"][()] == b"0.4"
