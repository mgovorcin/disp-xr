"""Static layers and LOS geometry (plan task T12)."""

from pathlib import Path

import dask
import numpy as np
import pytest
import xarray as xr

import disp_xr as dxr
from disp_xr.geometry import los, static

FRAME = 11116
STATIC_NAME = "OPERA_L3_DISP-S1-STATIC_F11116_20140403_S1A_v1.0_{}.tif"
LE, LN = 0.6, -0.1
LU = float(np.sqrt(1 - LE**2 - LN**2))


@pytest.fixture
def ds(stack_dir):
    return dxr.open_disp(stack_dir)


def _raster(ds, values, spacing=30.0, pad=8, dtype="float32", nodata=0):
    """A raster on the frame grid extended by ``pad`` pixels on every side."""
    x0, y0 = float(ds.x[0]) - 15, float(ds.y[0]) + 15
    ny, nx = (ds.sizes["y"] + 2 * pad) * 30 // int(spacing), (
        ds.sizes["x"] + 2 * pad
    ) * 30 // int(spacing)
    x = x0 - pad * 30 + spacing * (np.arange(nx) + 0.5)
    y = y0 + pad * 30 - spacing * (np.arange(ny) + 0.5)
    data = np.broadcast_to(
        np.asarray(values, dtype)[:, None, None], (len(values), ny, nx)
    )
    da = xr.DataArray(
        data.copy(),
        dims=("band", "y", "x"),
        coords={"band": np.arange(1, len(values) + 1), "y": y, "x": x},
    )
    return da.rio.write_crs(ds.rio.crs).rio.write_nodata(nodata)


@pytest.fixture
def static_dir(ds, tmp_path):
    d = tmp_path / "static"
    d.mkdir()
    los_da = _raster(ds, [LE, LN, LU])
    los_da[:, 0:10, 0:10] = 0  # nodata corner, outside the frame after the pad
    los_da[:, 8, 8] = 0  # nodata at frame pixel (0, 0)
    los_da.rio.to_raster(d / STATIC_NAME.format("line_of_sight_enu"))
    _raster(ds, [120.0], nodata=None).rio.to_raster(d / STATIC_NAME.format("dem"))
    mask = _raster(ds, [0], dtype="uint8", nodata=255)
    mask[:, 20, 20] = 2
    mask.rio.to_raster(d / STATIC_NAME.format("layover_shadow_mask"))
    return d


# --- LOS geometry --------------------------------------------------------------


def test_pure_uplift_projects_with_los_up():
    assert dxr.enu_to_los(0.0, 0.0, 1.0, (LE, LN, LU)) == pytest.approx(LU)


def test_motion_toward_satellite_is_positive():
    assert dxr.enu_to_los(LE, LN, LU, (LE, LN, LU)) == pytest.approx(1.0)
    assert dxr.enu_to_los(-LE, -LN, -LU, (LE, LN, LU)) == pytest.approx(-1.0)


@pytest.mark.parametrize("inc, az", [(30.0, 80.0), (42.0, 280.0), (5.0, 190.0)])
def test_incidence_azimuth_round_trip(inc, az):
    e, n, u = los.incidence_azimuth_to_los(inc, az)
    assert e**2 + n**2 + u**2 == pytest.approx(1)
    assert u == pytest.approx(los.los_up(e, n))
    np.testing.assert_allclose(los.los_to_incidence_azimuth(e, n), (inc, az))


def test_slant_range():
    assert los.slant_range(0.0, 693e3) == pytest.approx(693e3)
    # Flat-Earth limit for a low orbit: h / cos(inc).
    assert los.slant_range(30.0, 1e3) == pytest.approx(
        1e3 / np.cos(np.radians(30)), rel=1e-3
    )
    assert 850e3 < los.slant_range(40.0, 693e3) < 900e3  # Sentinel-1 near 40 degrees


# --- open_static (DISP-S1-STATIC) ---------------------------------------------


def test_open_static_on_disp_grid(ds, static_dir):
    st = dxr.open_static(ds, static_dir=static_dir, download=False)
    assert set(st.data_vars) == {
        "los_east",
        "los_north",
        "los_up",
        "incidence_angle",
        "azimuth_angle",
        "dem",
        "layover_shadow_mask",
    }
    np.testing.assert_array_equal(st.x, ds.x)
    np.testing.assert_array_equal(st.y, ds.y)
    assert st.rio.crs == ds.rio.crs
    np.testing.assert_allclose(st.los_east[5, 5], LE, rtol=1e-6)
    inc, az = los.los_to_incidence_azimuth(LE, LN)
    np.testing.assert_allclose(st.incidence_angle[5, 5], inc, rtol=1e-5)
    np.testing.assert_allclose(st.azimuth_angle[5, 5], az, rtol=1e-5)
    assert float(st.dem[5, 5]) == 120
    assert int(st.layover_shadow_mask[12, 12]) == 2


def test_open_static_nodata_is_nan(ds, static_dir):
    st = dxr.open_static(ds, static_dir=static_dir, download=False)
    assert np.isnan(st.los_east[0, 0]) and np.isnan(st.incidence_angle[0, 0])


def test_open_static_follows_a_subset(ds, static_dir):
    sub = ds.isel(y=slice(10, 20), x=slice(30, 34))
    st = dxr.open_static(sub, static_dir=static_dir, download=False)
    np.testing.assert_array_equal(st.x, sub.x)
    assert int(st.layover_shadow_mask[2, -1]) == 0


def test_open_static_resamples_other_grid(ds, tmp_path):
    files = {"dem": tmp_path / "dem.tif"}
    _raster(ds, [50.0], spacing=60.0, nodata=None).rio.to_raster(files["dem"])
    st = dxr.open_static(ds, files=files)
    assert st.dem.shape == (ds.sizes["y"], ds.sizes["x"])
    np.testing.assert_allclose(st.dem.values, 50)


def test_open_static_is_lazy(ds, static_dir):
    def fail(*_args, **_kwargs):
        raise AssertionError("computed")

    with dask.config.set(scheduler=fail):
        st = dxr.open_static(ds, static_dir=static_dir, download=False)
    assert isinstance(st.los_up.data, dask.array.Array)


def test_open_static_missing_without_download(ds, tmp_path):
    with pytest.raises(FileNotFoundError):
        dxr.open_static(ds, static_dir=tmp_path, download=False)


def test_project_enu_on_static(ds, static_dir):
    st = dxr.open_static(ds, static_dir=static_dir, download=False)
    up = dxr.enu_to_los(0.0, 0.0, 0.01, st)
    np.testing.assert_allclose(up[5, 5], 0.01 * LU, rtol=1e-5)


# --- download_static (CMR and HTTPS faked) -------------------------------------


class _Response:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self.payload


class _Session:
    def __init__(self, items):
        self.items, self.calls = items, []

    def get(self, _url, params=None):
        self.calls.append(params)
        return _Response({"items": self.items})


def _umm(version):
    base = (
        f"https://example.test/OPERA_L3_DISP-S1-STATIC_F11116_20140403_S1A_v{version}"
    )
    urls = [{"Type": "GET DATA", "URL": f"{base}_{n}.tif"} for n in static.STATIC_FILES]
    urls.append({"Type": "GET DATA VIA DIRECT ACCESS", "URL": "s3://bucket/x_dem.tif"})
    return {
        "umm": {
            "RelatedUrls": urls,
            "AdditionalAttributes": [{"Name": "PRODUCT_VERSION", "Values": [version]}],
        }
    }


def test_find_static_urls_picks_newest_version():
    session = _Session([_umm("1.0"), _umm("1.10"), _umm("1.9")])
    urls = static.find_static_urls(FRAME, session=session)
    assert set(urls) == set(static.STATIC_FILES)
    assert all("v1.10_" in u for u in urls.values())
    assert session.calls[0]["attribute[]"] == "int,FRAME_NUMBER,11116"


def test_find_static_urls_none():
    assert static.find_static_urls(FRAME, session=_Session([])) == {}


def test_download_static_once(monkeypatch, tmp_path):
    fetched = []

    class FakeFS:
        def get(self, url, path):
            fetched.append(url)
            Path(path).write_bytes(b"tif")

    monkeypatch.setattr(
        static,
        "find_static_urls",
        lambda _frame_id: {
            n: (
                f"https://example.test/OPERA_L3_DISP-S1-STATIC_F11116_20140403_S1A_v1.0_{n}.tif"
            )
            for n in static.STATIC_FILES
        },
    )
    monkeypatch.setattr("disp_xr._remote.get_filesystem", lambda _url, **_kw: FakeFS())
    first = static.download_static(FRAME, tmp_path)
    second = static.download_static(FRAME, tmp_path)
    assert first == second and len(fetched) == 3
    assert not list(tmp_path.glob("*.part"))


def test_download_static_missing_frame(monkeypatch, tmp_path):
    monkeypatch.setattr(static, "find_static_urls", lambda _frame_id: {})
    with pytest.raises(FileNotFoundError, match="stitch_cslc_static"):
        static.download_static(FRAME, tmp_path)


def test_cache_dir_from_config(tmp_path):
    with dxr.config.set({"cache.dir": str(tmp_path)}):
        assert static._cache_dir() == tmp_path


# --- stitch_cslc_static --------------------------------------------------------


def _cslc_burst(path, ds, cols, east):
    """A CSLC-STATIC file (5 m x 10 m) covering the frame columns ``cols``."""
    x_left = float(ds.x[cols.start]) - 15
    x_right = float(ds.x[cols.stop - 1]) + 15
    y_top, y_bottom = float(ds.y[0]) + 15, float(ds.y[-1]) - 15
    x = np.arange(x_left + 2.5, x_right, 5.0)
    y = np.arange(y_top - 5, y_bottom, -10.0)
    shape = (len(y), len(x))
    mask = np.zeros(shape, np.int8)
    mask[:, :6] = 127
    data = xr.Dataset(
        {
            "los_east": (
                ("y_coordinates", "x_coordinates"),
                np.full(shape, east, "f4"),
            ),
            "los_north": (("y_coordinates", "x_coordinates"), np.full(shape, LN, "f4")),
            "layover_shadow_mask": (("y_coordinates", "x_coordinates"), mask),
            "projection": ((), np.int32(0), {"epsg_code": ds.rio.crs.to_epsg()}),
        },
        coords={"x_coordinates": x, "y_coordinates": y},
    )
    data.to_netcdf(path, group="data", engine="h5netcdf")
    return path


def test_stitch_cslc_static(ds, tmp_path):
    a = _cslc_burst(tmp_path / "a.h5", ds, slice(0, 40), 0.6)
    b = _cslc_burst(tmp_path / "b.h5", ds, slice(30, 60), 0.5)
    st = static.stitch_cslc_static([a, b], ds).compute()
    np.testing.assert_allclose(st.los_east[:, 10], 0.6)
    np.testing.assert_allclose(st.los_east[:, 35], 0.6)  # overlap: first file wins
    np.testing.assert_allclose(st.los_east[:, 50], 0.5)
    assert np.isnan(st.los_east[:, 62]).all()  # no burst covers it
    np.testing.assert_allclose(st.los_up[:, 50], los.los_up(0.5, LN), rtol=1e-6)
    assert st.layover_shadow_mask.dtype == np.uint8
    assert (
        int(st.layover_shadow_mask[5, 0]) == 255
        and int(st.layover_shadow_mask[5, 10]) == 0
    )
    assert st.attrs["source"] == "CSLC-S1-STATIC"


def test_stitch_unknown_layer(ds):
    with pytest.raises(ValueError, match="Unknown CSLC-STATIC layers"):
        static.stitch_cslc_static([], ds, layers=["dem"])


def test_no_opera_utils_import():
    src = Path(static.__file__).parents[1]
    offenders = [
        p
        for p in src.rglob("*.py")
        if "opera_utils" in p.read_text()
        and (
            "import opera_utils" in p.read_text() or "from opera_utils" in p.read_text()
        )
    ]
    assert offenders == []


def test_legacy_helpers_are_deprecated():
    from rasterio.errors import RasterioIOError

    with pytest.warns(DeprecationWarning, match="open_static"):
        with pytest.raises(RasterioIOError):
            static.los_unit2inc_azimuth("missing_e.tif", "missing_n.tif", ".")


def test_static_source_in_spec():
    from disp_xr.product import DISP_S1

    assert DISP_S1.static_source == "DISP-S1-STATIC"
