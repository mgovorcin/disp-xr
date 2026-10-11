"""GNSS adapters over geepers (plan task T25)."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import disp_xr as dxr

geepers = pytest.importorskip("geepers")
from fixtures.fake_gnss import FakeSource  # noqa: E402

from disp_xr import gnss  # noqa: E402


@pytest.fixture
def frame(stack_dir):
    return dxr.chain(dxr.open_disp(stack_dir))


def _station_lonlat(ds, rows_cols):
    from pyproj import Transformer

    to_ll = Transformer.from_crs(ds.rio.crs, "EPSG:4326", always_xy=True)
    out = []
    for r, c in rows_cols:
        out.append(to_ll.transform(float(ds.x[c]), float(ds.y[r])))
    return out


@pytest.fixture
def fake(frame, tmp_path):
    inside = _station_lonlat(frame, [(10, 12), (40, 50)])
    lonlat = [*inside, (inside[0][0] + 1.0, inside[0][1])]  # one far east
    ids = ["AAA1", "BBB2", "FAR3"]
    table = pd.DataFrame(
        {
            "id": ids,
            "lon": [p[0] for p in lonlat],
            "lat": [p[1] for p in lonlat],
            "alt": 0.0,
        }
    )
    days = pd.date_range("2019-12-01", "2021-12-31", freq="D")
    years = np.asarray((days - days[0]).days) / 365.25
    series = {}
    for k, sid in enumerate(ids):
        series[sid] = pd.DataFrame(
            {
                "date": days,
                "east": 0.002 * years * (k + 1),
                "north": -0.001 * years,
                "up": 0.01 * years * (k + 1),
                "sigma_east": 0.001,
                "sigma_north": 0.001,
                "sigma_up": 0.003,
                "corr_en": 0.1,
                "corr_eu": 0.0,
                "corr_nu": 0.0,
            }
        )
    return FakeSource(table, series, cache_dir=tmp_path / "cache")


def _static(ds, east=-0.6, north=-0.1):
    up = np.sqrt(1 - east**2 - north**2)
    shape = (ds.sizes["y"], ds.sizes["x"])
    return xr.Dataset(
        {
            "los_east": (("y", "x"), np.full(shape, east)),
            "los_north": (("y", "x"), np.full(shape, north)),
            "los_up": (("y", "x"), np.full(shape, up)),
        },
        coords={"y": ds.y, "x": ds.x},
    ).rio.write_crs(ds.rio.crs)


def test_stations_inside_the_frame(frame, fake):
    table = gnss.stations(frame, source=fake)
    assert sorted(table.id) == ["AAA1", "BBB2"]
    assert {"x", "y"} <= set(table.columns)


def test_load_builds_a_station_time_cube(frame, fake):
    cube = gnss.load(frame, source=fake, start="2020-01-01")
    assert cube.east.dims == ("station", "time")
    assert sorted(cube.station.values) == ["AAA1", "BBB2"]
    assert str(cube.time.values[0])[:10] == "2020-01-01"
    assert {"sigma_up", "corr_en", "lon", "lat", "x", "y"} <= set(cube.variables)
    assert cube.attrs["gnss_source"] == "SyntheticGnssSource"
    by_id = gnss.load(["BBB2"], source=fake)
    assert list(by_id.station.values) == ["BBB2"]
    with pytest.raises(ValueError, match="source must be"):
        gnss.get_source("ngl")


def test_to_los_projects_and_propagates_sigma(frame, fake):
    cube = gnss.load(frame, source=fake)
    out = gnss.to_los(cube, _static(frame))
    e, n, u = -0.6, -0.1, np.sqrt(1 - 0.37)
    expected = e * cube.east + n * cube.north + u * cube.up
    xr.testing.assert_allclose(
        out.los.drop_vars(["los_east", "los_north", "los_up"]), expected
    )
    from geepers.uncertainty import get_sigma_los

    ref = get_sigma_los(np.array([e, n, u]), 0.001, 0.001, 0.003, 0.1, 0.0, 0.0)
    np.testing.assert_allclose(out.sigma_los.values, ref)
    with pytest.raises(ValueError, match="x and y"):
        gnss.to_los(cube.drop_vars(["x", "y"]), _static(frame))


def test_sample_reads_the_station_window(frame, fake):
    cube = gnss.load(frame, source=fake)
    at_pixel = gnss.sample(frame, cube)
    assert at_pixel.dims == ("station", "time")
    k = list(at_pixel.station.values).index("AAA1")
    np.testing.assert_allclose(
        at_pixel.values[k], frame.displacement.isel(y=10, x=12).values, equal_nan=True
    )
    windowed = gnss.sample(frame, cube, buffer_m=60.0)  # two pixels at 30 m
    block = frame.displacement.isel(y=slice(8, 13), x=slice(10, 15))
    expected = block.median(("y", "x"), skipna=True).values
    if (
        "buffer_meters"
        not in __import__("inspect")
        .signature(__import__("geepers.processing", fromlist=["x"]).sample_insar)
        .parameters
    ):
        np.testing.assert_allclose(windowed.values[k], expected, equal_nan=True)
    assert windowed.attrs["gnss_buffer_m"] == 60.0


def test_align_on_disp_epochs_and_reference(frame, fake):
    cube = gnss.load(frame, source=fake)
    los = gnss.to_los(cube, _static(frame))["los"]
    aligned = gnss.align(los, frame.time)
    assert aligned.sizes["time"] == frame.sizes["time"]
    np.testing.assert_allclose(aligned.isel(time=0), 0.0, atol=1e-12)
    linear = gnss.align(los, frame.time, method="linear")
    np.testing.assert_allclose(linear.values, aligned.values, atol=2e-5)
    # An epoch beyond the GNSS record stays NaN.
    late = frame.time.values.copy()
    late[-1] = np.datetime64("2030-01-01")
    assert np.isnan(gnss.align(los, late).isel(time=-1)).all()
    with pytest.raises(ValueError, match="method"):
        gnss.align(los, frame.time, method="cubic")


def test_align_takes_the_epochs_own_utc_day():
    """Regression: epochs after 12:00 UTC took the next day's daily solution."""
    days = pd.date_range("2020-01-01", periods=6, freq="D")
    daily = xr.DataArray(
        np.arange(6.0)[None, :],
        dims=("station", "time"),
        coords={"station": ["A"], "time": days},
    )
    epochs = pd.DatetimeIndex(
        ["2020-01-02 00:26", "2020-01-03 14:07", "2020-01-04 19:16", "2020-01-05"]
    )
    aligned = gnss.align(daily, epochs)
    # Day index of each epoch's own day, relative to the first epoch's day.
    # Midnight is 12 h from both neighbouring noons; it belongs to its own day.
    np.testing.assert_array_equal(aligned.values[0], [0.0, 1.0, 2.0, 3.0])
    linear = gnss.align(daily, epochs, method="linear")
    # The value is the day index at noon, so linear in days since 2020-01-01 12:00.
    days_since_noon = (epochs - pd.Timestamp("2020-01-01 12:00")) / pd.Timedelta("1D")
    expected = days_since_noon - days_since_noon[0]
    np.testing.assert_allclose(linear.values[0], expected, atol=1e-3)


def test_align_references_late_stations_to_their_first_epoch():
    """Regression: a station without GNSS on the reference date was all NaN."""
    days = pd.date_range("2020-01-01", periods=10, freq="D")
    values = np.vstack([np.arange(10.0), np.arange(10.0) * 2])
    values[1, :4] = np.nan  # station B starts on day 4
    daily = xr.DataArray(
        values, dims=("station", "time"), coords={"station": ["A", "B"], "time": days}
    )
    epochs = days[::3] + pd.Timedelta(hours=14)  # days 0, 3, 6, 9

    aligned = gnss.align(daily, epochs)

    np.testing.assert_array_equal(aligned.sel(station="A").values, [0, 3, 6, 9])
    np.testing.assert_array_equal(
        aligned.sel(station="B").values, [np.nan, np.nan, 0, 6]
    )
    assert aligned["gnss_reference"].sel(station="A").values == epochs[0]
    assert aligned["gnss_reference"].sel(station="B").values == epochs[2]


def test_drop_relocations_keeps_the_longest_segment():
    # Station B is moved about 1 km after day 3 (GINN in UNR, 2019-2020).
    days = pd.date_range("2020-01-01", periods=10, freq="D")
    east = np.tile(np.arange(10.0) * 1e-3, (2, 1))
    east[1, 4:] += -349.0
    zeros = np.zeros_like(east)
    cube = xr.Dataset(
        {
            "east": (("station", "time"), east),
            "north": (("station", "time"), zeros),
            "up": (("station", "time"), zeros),
            "sigma_east": (("station", "time"), zeros + 1e-3),
        },
        coords={"station": ["A", "B"], "time": days},
    )
    out = gnss.drop_relocations(cube)
    assert out["relocated"].values.tolist() == [False, True]
    np.testing.assert_array_equal(out["east"].sel(station="A"), east[0])
    b = out["east"].sel(station="B")
    assert np.isnan(b[:4]).all() and np.isfinite(b[4:]).all()
    assert np.isnan(out["sigma_east"].sel(station="B")[:4]).all()
    # A 5 m coseismic step stays.
    quake = cube.copy(deep=True)
    quake["east"][0, 5:] += 5.0
    assert not gnss.drop_relocations(quake)["relocated"].sel(station="A")


def test_centred_leaves_sub_daily_series():
    times = pd.date_range("2020-01-01 03:00", periods=4, freq="6h")
    series = xr.DataArray(np.arange(4.0), dims="time", coords={"time": times})
    assert gnss.centred(series)["time"].equals(series["time"])
    daily = series.assign_coords(time=pd.date_range("2020-01-01", periods=4))
    assert (gnss.centred(daily)["time"].dt.hour == 12).all()


@pytest.mark.parametrize(
    "dx, dy", [(-10.0, 10.0), (10.0, 10.0), (-10.0, -10.0), (10.0, -10.0)]
)
def test_sample_finds_the_pixel_holding_an_off_center_station(frame, dx, dy):
    """geepers rounds the pixel index; a point in the right or lower half of
    its 30 m pixel must still read that pixel."""
    from pyproj import Transformer

    to_ll = Transformer.from_crs(frame.rio.crs, "EPSG:4326", always_xy=True)
    lon, lat = to_ll.transform(float(frame.x[20]) + dx, float(frame.y[30]) + dy)
    table = pd.DataFrame({"id": ["S"], "lon": [lon], "lat": [lat]})
    got = gnss.sample(frame, table)
    np.testing.assert_allclose(
        got.values[0], frame.displacement.isel(y=30, x=20).values, equal_nan=True
    )


def test_sample_needs_units(frame, fake):
    cube = gnss.load(frame, source=fake)
    bare = frame.assign(displacement=frame.displacement.drop_attrs())
    with pytest.raises(ValueError, match="units"):
        gnss.sample(bare, cube)
