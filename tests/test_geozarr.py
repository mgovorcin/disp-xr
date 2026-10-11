"""GeoZarr stores that bowser opens directly (plan task T27.9)."""

import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
import zarr
from click.testing import CliRunner
from pyproj import CRS

import disp_xr as dxr
from disp_xr.cli import cli_app
from disp_xr.export.geozarr import (
    bowser_attrs,
    display_factors,
    histogram_stats,
    los_attrs,
    reference_lonlat,
)
from disp_xr.export.zarr import chunk_scheme
from disp_xr.geometry.los import incidence_azimuth_to_los

BOWSER_CHECK = Path(__file__).with_name("bowser_check.py")


@pytest.fixture
def ds(stack_dir):
    stacked = dxr.chain(dxr.open_disp(stack_dir))
    return stacked.disp.rereference("point", row=30, col=30)


def _static(shape, azimuth, inc=(32.0, 44.0)):
    incidence = np.linspace(*inc, shape[1])[None].repeat(shape[0], 0)
    e, n, u = incidence_azimuth_to_los(incidence, azimuth)
    dims = ("y", "x")
    return xr.Dataset(
        {"los_east": (dims, e), "los_north": (dims, n), "los_up": (dims, u)}
    )


def test_display_factors_follow_bowser():
    assert display_factors((7000, 9000)) == [2, 4, 8, 16]
    assert display_factors((511, 4000)) == []
    assert display_factors((512, 512)) == [2]
    assert display_factors((10**6, 10**6)) == [2, 4, 8, 16, 32, 64]  # at most 6


def test_bowser_attrs(ds):
    out = bowser_attrs(ds)
    assert out.displacement.attrs["bowser_uses_spatial_ref"] is True
    # Units in meters or radians would make bowser re-reference these.
    for name in ("short_wavelength_displacement", "timeseries_inversion_residuals"):
        assert out[name].attrs["bowser_uses_spatial_ref"] is False
    assert out.displacement.dtype == np.float32
    assert out.temporal_coherence.attrs["long_name"] == "temporal coherence"
    assert out.recommended_mask.dtype == ds.recommended_mask.dtype


def test_reference_lonlat(ds, stack_dir):
    lon, lat = reference_lonlat(ds)
    assert -180 < lon < 180 and -90 < lat < 90
    median = dxr.chain(dxr.open_disp(stack_dir)).disp.rereference("median")
    assert reference_lonlat(median) is None


@pytest.mark.parametrize(
    ("azimuth", "look", "heading"),
    [(260.0, "right", 350.0), (100.0, "right", 190.0), (80.0, "left", 350.0)],
)
def test_los_attrs(azimuth, look, heading):
    attrs = los_attrs(_static((20, 50), azimuth), look_side=look)
    assert attrs["heading_angle_deg"] == pytest.approx(heading)
    inc = attrs["incidence_angle_deg"]
    assert inc["near"] < inc["center"] < inc["far"]
    assert inc["center"] == pytest.approx(38.0)
    enu = attrs["los_enu_ground_to_satellite"]["center"]
    assert np.hypot(np.hypot(enu["east"], enu["north"]), enu["up"]) == pytest.approx(1)


def test_histogram_matches_numpy():
    rng = np.random.default_rng(0)
    values = rng.normal(size=(3, 40, 50))
    values[0, :5] = np.nan
    da = xr.DataArray(values, dims=("time", "y", "x")).chunk({"y": 20})
    stats = histogram_stats(da)
    assert len(stats) == 3
    valid = values[0][np.isfinite(values[0])]
    assert stats[0]["p98"] == pytest.approx(np.percentile(valid, 98))
    assert sum(stats[0]["counts"]) == valid.size and len(stats[0]["bins"]) == 101
    empty = histogram_stats(xr.DataArray(np.full((4, 4), np.nan), dims=("y", "x")))
    assert empty[0]["bins"] == [] and empty[0]["max"] == 0.0


def test_tiles_preset():
    ds = xr.Dataset({"v": (("time", "y", "x"), np.zeros((10, 3000, 700), "f4"))})
    chunks, shards = chunk_scheme(ds, "tiles")
    assert chunks == {"time": 1, "y": 256, "x": 256}
    assert shards == {"time": 4, "y": 1024, "x": 768}


def _replay_bowser_read(store: Path) -> list[xr.Dataset]:
    """Follow bowser's reader (geozarr.py, state.py) without importing it."""
    root = xr.open_zarr(str(store), consolidated=False)
    layout = root.attrs["multiscales"]["layout"]
    assert isinstance(layout, list) and layout[0]["asset"] == "0"
    attrs = dict(zarr.open_group(store, mode="r").attrs)
    levels = []
    for entry in layout:
        level = xr.open_zarr(str(store), group=entry["asset"], consolidated=False)
        level.attrs = {**attrs, **level.attrs}
        CRS.from_string(level.attrs["proj:code"])
        assert level.y[0] > level.y[-1]
        for name, var in level.data_vars.items():
            if {"x", "y"} <= set(var.dims):
                assert var.dims[-2:] == ("y", "x")
                assert len([d for d in var.dims if d not in ("y", "x")]) <= 1
        levels.append(level)
    sizes = [float(abs(lv.x[1] - lv.x[0])) for lv in levels]
    assert sizes == sorted(sizes)  # bowser picks levels in this order
    return levels


def test_to_geozarr_bowser_profile(ds, tmp_path):
    store = tmp_path / "frame.zarr"
    static = _static((ds.sizes["y"], ds.sizes["x"]), 260.0)
    tree = dxr.to_geozarr(ds, store, static=static, factors=(33,), min_size=16)
    factors = [e["disp_xr:factor"] for e in tree.attrs["multiscales"]["layout"]]
    assert factors == [1, 2, 4, 33]  # display levels of a 64-pixel frame, then 33
    levels = _replay_bowser_read(store)
    base = levels[0]
    assert set(levels[0].data_vars) <= set(levels[-1].data_vars)
    # Attributes stay JSON objects in Zarr, as bowser expects.
    assert isinstance(base.attrs["reference_lonlat"], list)
    assert isinstance(base.attrs["incidence_angle_deg"], dict)
    assert base.attrs["heading_angle_deg"] == pytest.approx(350.0)
    hist = base.displacement.attrs["bowser_histogram"]
    assert len(hist) == ds.sizes["time"] and "p98" in hist[0]
    assert "bowser_histogram" not in levels[1].displacement.attrs
    assert levels[1].displacement_sigma.attrs["bowser_uses_spatial_ref"] is False
    # Integer masks keep their dtype and fill; xarray decodes the fill to NaN.
    assert base.recommended_mask.dtype.kind == "f"
    assert int(base.recommended_mask.isnull().sum()) >= 0
    array = zarr.open_group(store / "0", mode="r")["displacement"]
    assert array.dtype == np.float32
    assert array.chunks[0] == 1


def test_to_geozarr_flat_and_standard(ds, tmp_path):
    flat = tmp_path / "flat.zarr"
    assert dxr.to_geozarr(ds, flat, pyramid=False, histogram=False) is None
    opened = xr.open_zarr(flat, consolidated=False)
    assert opened.displacement.attrs["bowser_uses_spatial_ref"] is True
    assert isinstance(opened.attrs["reference_lonlat"], list)
    tree = dxr.to_geozarr(ds, tmp_path / "std.zarr", profile="standard", factors=(9,))
    assert "bowser_uses_spatial_ref" not in tree["0"].displacement.attrs
    with pytest.raises(ValueError, match="too small"):
        dxr.to_geozarr(ds, tmp_path / "small.zarr")
    with pytest.raises(ValueError, match="profile"):
        dxr.to_geozarr(ds, tmp_path / "x.zarr", profile="other")


def test_netcdf_attrs_stay_strings(ds, tmp_path):
    ds = ds.assign_attrs(extra=[1, 2])
    dxr.to_netcdf(ds, tmp_path / "out.nc")
    with xr.open_dataset(tmp_path / "out.nc") as opened:
        assert opened.attrs["extra"] == "[1, 2]"


def test_cli_geozarr(stack_dir, tmp_path):
    out = tmp_path / "cli.zarr"
    result = CliRunner().invoke(
        cli_app,
        ["export", str(stack_dir), "--format", "geozarr", "--out", str(out)]
        + ["--factors", "9,33"],
    )
    assert result.exit_code == 0, result.output
    layout = zarr.open_group(out, mode="r").attrs["multiscales"]["layout"]
    assert [e["disp_xr:factor"] for e in layout] == [1, 9, 33]
    bad = CliRunner().invoke(
        cli_app,
        ["export", str(stack_dir), "--format", "geozarr", "--out", str(out)],
    )
    assert bad.exit_code != 0 and "exists" in bad.output


@pytest.mark.skipif(
    not os.environ.get("DISP_XR_BOWSER_PYTHON"),
    reason="set DISP_XR_BOWSER_PYTHON to a python with bowser installed",
)
def test_bowser_opens_and_serves_the_store(ds, tmp_path):
    store = tmp_path / "frame.zarr"
    static = _static((ds.sizes["y"], ds.sizes["x"]), 260.0)
    dxr.to_geozarr(ds, store, static=static, factors=(9,), min_size=16)
    run = subprocess.run(
        [os.environ["DISP_XR_BOWSER_PYTHON"], str(BOWSER_CHECK), str(store)],
        capture_output=True,
        text=True,
        timeout=600,
    )
    lines = run.stdout.splitlines()
    report = json.loads(lines[lines.index("--- bowser_check report ---") + 1])
    assert run.returncode == 0, report["errors"]
    assert len(report["levels"]) == 4  # 30, 60, 120 and 270 m
    assert report["variables"]["displacement"]["uses_spatial_ref"] is True
    assert (
        report["variables"]["timeseries_inversion_residuals"]["uses_spatial_ref"]
        is False
    )
    assert report["reference_lonlat"] is not None
    assert report["los_metadata"]["heading_deg"] == pytest.approx(350.0)
