"""Corrections and plate motion (plan task T13)."""

import json

import dask
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from pyproj import Transformer

import disp_xr as dxr
from disp_xr import corrections
from disp_xr.corrections import CorrectionError

IONO, SET = 0.002, -0.001  # constant fixture values


@pytest.fixture
def ds(stack_dir):
    return dxr.open_disp(stack_dir, corrections=True)


def _close(a, b):
    np.testing.assert_allclose(a.values, b.values, atol=1e-6)


def test_default_subtracts_both_layers(ds):
    out = dxr.correct(ds)
    _close(out.displacement, ds.displacement - IONO - SET)
    assert json.loads(out.attrs["corrections_applied"]) == [
        "ionospheric_delay",
        "solid_earth_tide",
    ]
    assert out.displacement.dtype == ds.displacement.dtype


def test_alias_and_short_wavelength_untouched(ds):
    out = dxr.correct(ds, ["ionosphere"])
    _close(out.displacement, ds.displacement - IONO)
    xr.testing.assert_identical(
        out.short_wavelength_displacement, ds.short_wavelength_displacement
    )


def test_never_twice(ds):
    once = dxr.correct(ds, ["ionosphere"])
    with pytest.raises(CorrectionError, match="already applied"):
        dxr.correct(once, ["ionospheric_delay"])
    twice = dxr.correct(once, ["set"])
    assert len(json.loads(twice.attrs["corrections_applied"])) == 2


def test_pre_applied_is_refused(ds):
    nisar = ds.assign_attrs(product="DISP-NISAR")
    with pytest.raises(CorrectionError, match="already has 'ionospheric_delay'"):
        dxr.correct(nisar, ["ionosphere"])
    dxr.correct(nisar, ["solid_earth_tide"])


def test_default_is_what_the_product_has_not_applied(ds):
    # DISP-NISAR carries the ionosphere already applied: the default subtracts
    # the solid-earth tide only, where it used to raise.
    nisar = ds.assign_attrs(product="DISP-NISAR")
    out = dxr.correct(nisar)
    assert json.loads(out.attrs["corrections_applied"]) == ["solid_earth_tide"]
    xr.testing.assert_allclose(
        out.displacement, (nisar.displacement - nisar.solid_earth_tide)
    )


def test_one_correction_as_a_string(ds):
    out = dxr.correct(ds, "set")
    assert json.loads(out.attrs["corrections_applied"]) == ["solid_earth_tide"]


def test_missing_and_unknown_layers(stack_dir, ds):
    with pytest.raises(CorrectionError, match=r"corrections=True"):
        dxr.correct(dxr.open_disp(stack_dir))
    with pytest.raises(
        CorrectionError, match="Unknown correction 'perpendicular_baseline'"
    ):
        dxr.correct(ds, ["perpendicular_baseline"])


def test_refuses_chained_data(ds):
    with pytest.raises(CorrectionError, match="before chain"):
        dxr.correct(dxr.chain(ds))


def test_correct_then_stack_matches_stacked_correction(ds):
    """Corrections chain like displacement: each epoch loses one per ministack."""
    out = dxr.chain(dxr.correct(ds))
    raw = dxr.chain(ds)
    n_refs = xr.DataArray(
        pd.Series(ds.reference_time.values).rank(method="dense").to_numpy(),
        dims="time",
    )
    expected = (
        raw.displacement.isel(time=slice(1, None))
        - (IONO + SET) * n_refs.values[:, None, None]
    )
    _close(out.displacement.isel(time=slice(1, None)), expected)


def _external(ds, values, times):
    data = np.asarray(values, "f4")[:, None, None] * np.ones(
        (1, ds.sizes["y"], ds.sizes["x"]), "f4"
    )
    da = xr.DataArray(
        data, dims=("time", "y", "x"), coords={"time": times, "y": ds.y, "x": ds.x}
    )
    return da.rio.write_crs(ds.rio.crs)


def test_external_differential(ds):
    tropo = _external(ds, np.arange(ds.sizes["time"]) * 0.01, ds.time.values)
    out = dxr.correct(ds, None, external={"troposphere": tropo})
    _close(out.displacement[:, 5, 5], ds.displacement[:, 5, 5] - tropo[:, 5, 5])
    assert json.loads(out.attrs["corrections_applied"]) == ["troposphere"]


def test_external_absolute_uses_reference_dates(ds):
    dates = pd.DatetimeIndex(np.union1d(ds.time.values, ds.reference_time.values))
    values = np.arange(len(dates)) * 0.01
    tropo = _external(ds, values, dates)
    out = dxr.correct(
        ds, None, external={"troposphere": tropo}, external_kind="absolute"
    )
    lookup = dict(zip(dates, values))
    expected = [
        lookup[pd.Timestamp(t)] - lookup[pd.Timestamp(r)]
        for t, r in zip(ds.time.values, ds.reference_time.values)
    ]
    np.testing.assert_allclose(
        (ds.displacement - out.displacement)[:, 5, 5], expected, atol=1e-6
    )


def test_external_missing_dates(ds):
    tropo = _external(ds, [0.0], ds.time.values[:1])
    with pytest.raises(CorrectionError, match="no values at"):
        dxr.correct(ds, None, external={"troposphere": tropo})


def test_external_resampled_from_coarser_grid(ds):
    coarse = _external(ds, [0.005], ds.time.values[:1]).isel(time=0, drop=True)
    coarse = coarse.rio.reproject(coarse.rio.crs, resolution=120.0)
    out = dxr.correct(ds, None, external={"troposphere": coarse})
    _close(out.displacement[:, 20, 20], ds.displacement[:, 20, 20] - 0.005)


def test_external_cannot_shadow_product_layer(ds):
    with pytest.raises(CorrectionError, match="is a product layer"):
        dxr.correct(ds, None, external={"ionospheric_delay": ds.ionospheric_delay})


def test_correct_is_lazy_and_recorded(ds):
    def fail(*_args, **_kwargs):
        raise AssertionError("computed")

    with dask.config.set(scheduler=fail):
        out = ds.disp.correct(["ionosphere"])
    assert isinstance(out.displacement.data, dask.array.Array)
    assert out.disp.history[-1]["op"] == "correct"


# --- plate motion ----------------------------------------------------------------


@pytest.mark.parametrize(
    "lat, lon, h", [(0, 0, 0), (36.2, -120.5, 300), (-45, 170, -20)]
)
def test_geodetic_to_ecef_matches_pyproj(lat, lon, h):
    expected = Transformer.from_crs("EPSG:4979", "EPSG:4978", always_xy=True).transform(
        lon, lat, h
    )
    np.testing.assert_allclose(
        corrections.geodetic_to_ecef(lat, lon, h), expected, atol=1e-6
    )


def test_rotation_about_z_is_eastward():
    corrections.ITRF2014_PMM["TEST"] = (0.0, 0.0, 1.0)
    try:
        e, n, u = corrections.plate_velocity_enu("TEST", 0.0, 0.0)
    finally:
        del corrections.ITRF2014_PMM["TEST"]
    # 1 mas/yr at the equator: a * 1 mas in radians.
    assert e == pytest.approx(6378137.0 * np.pi / (180 * 3_600_000))
    assert n == pytest.approx(0, abs=1e-12) and u == pytest.approx(0, abs=1e-12)


def test_published_plate_speeds():
    # Pacific plate at Hawaii (Kokee, 22.1N 159.7W): about 7 cm/yr to the
    # west-northwest in ITRF2014 (Altamimi et al. 2017, Fig. 1).
    e, n, u = corrections.plate_velocity_enu("pacific", 22.13, -159.66)
    assert 0.065 < np.hypot(e, n) < 0.075 and e < 0 and n > 0
    # Rigid rotation is horizontal up to the ellipsoid's tilt: below 0.1 mm/yr.
    assert abs(u) < 1e-4
    # North America at Boulder: about 1.5-2 cm/yr to the west-southwest.
    e, n, _ = corrections.plate_velocity_enu("NOAM", 40.0, -105.3)
    assert 0.012 < np.hypot(e, n) < 0.022 and e < 0 and n < 0


def test_unknown_plate():
    with pytest.raises(ValueError, match="Unknown plate 'atlantis'"):
        corrections.plate_velocity_enu("atlantis", 0, 0)


@pytest.fixture
def static(ds):
    los = {"los_east": 0.6, "los_north": -0.1, "los_up": float(np.sqrt(1 - 0.37))}
    out = xr.Dataset(
        {
            k: xr.full_like(ds.temporal_coherence.isel(time=0, drop=True), v)
            for k, v in los.items()
        }
    )
    return out.rio.write_crs(ds.rio.crs)


def test_plate_motion_los_matches_point_projection(ds, static):
    v = dxr.corrections.plate_motion_los(static, "NOAM").compute()
    lon, lat = Transformer.from_crs(ds.rio.crs, "EPSG:4326", always_xy=True).transform(
        float(ds.x[3]), float(ds.y[7])
    )
    e, n, u = corrections.plate_velocity_enu("NOAM", lat, lon)
    expected = dxr.enu_to_los(e, n, u, (0.6, -0.1, np.sqrt(1 - 0.37)))
    assert float(v[7, 3]) == pytest.approx(expected, rel=1e-5)
    assert v.attrs["units"] == "meters/year"


@pytest.mark.parametrize("chained", [False, True])
def test_remove_plate_motion(ds, static, chained):
    data = dxr.chain(ds) if chained else ds
    out = dxr.remove_plate_motion(data, static, "NOAM")
    v = float(dxr.corrections.plate_motion_los(static, "NOAM")[10, 10])
    reference = (
        np.datetime64(pd.Timestamp(data.attrs["reference_date"]), "ns")
        if chained
        else data.reference_time.values
    )
    years = (data.time.values - reference) / np.timedelta64(1, "D") / 365.25
    np.testing.assert_allclose(
        (data.displacement - out.displacement)[:, 10, 10],
        v * years,
        rtol=1e-4,
        atol=1e-7,
    )
    with pytest.raises(CorrectionError, match="already applied"):
        dxr.remove_plate_motion(out, static, "NOAM")
