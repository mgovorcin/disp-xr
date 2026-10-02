"""Command-line interface (plan task T15)."""

import json

import numpy as np
import pandas as pd
import pytest
import rasterio
import xarray as xr
from click.testing import CliRunner
from test_static import STATIC_NAME, _raster

import disp_xr as dxr
from disp_xr.cli import cli_app


def run(*args, ok=True):
    result = CliRunner().invoke(cli_app, [str(a) for a in args])
    if ok:
        assert result.exit_code == 0, result.output + repr(result.exception)
    return result


# --- help ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("command", "text"),
    [
        (None, "Xarray tools for OPERA DISP products"),
        ("info", "Summarize the DISP granules"),
        ("stack", "fixed order of the PRD"),
        ("export", "GeoTIFF, Zarr, NetCDF or MintPy"),
        ("geometry", "static layers"),
        ("search", "Find DISP granules"),
        ("virtual", "granule references in Icechunk"),
    ],
)
def test_help(command, text):
    out = run(*([command] if command else []), "--help").output
    assert text in " ".join(out.split())


def test_version():
    assert dxr.__version__ in run("--version").output


# --- info ----------------------------------------------------------------------


def test_info(stack_dir):
    out = run("info", stack_dir).output
    assert "DISP-S1" in out
    assert "10 in 2 ministacks" in out
    assert "Dropped duplicates: 0" in out


def test_info_json_and_table(disp_stacks, tmp_path):
    stack = disp_stacks["duplicate"]
    table = tmp_path / "granules.csv"
    out = run("info", stack.paths[0].parent, "--json", "--table", table).stdout
    summary = json.loads(out)
    assert summary["granules"] == 10 and summary["frames"] == [11116]
    assert summary["dropped_duplicates"] == [stack.extra["duplicate"].name]
    assert len(pd.read_csv(table)) == 10


def test_user_error_is_one_line(tmp_path):
    (tmp_path / "x.nc").write_bytes(b"")
    result = run("info", tmp_path / "x.nc", ok=False)
    assert result.exit_code == 1
    assert result.output.startswith("Error: ") and "Traceback" not in result.output
    # --debug lets the exception through.
    debug = CliRunner().invoke(cli_app, ["--debug", "info", str(tmp_path / "x.nc")])
    assert isinstance(debug.exception, ValueError)


# --- stack ---------------------------------------------------------------------


def test_stack_matches_library(stack_dir, tmp_path):
    out = tmp_path / "out.zarr"
    run(
        "stack", stack_dir, "--out", out, "--reference", "point", "--row", 5, "--col", 7
    )
    expected = dxr.rereference(
        dxr.stack_disp(
            dxr.mask(
                dxr.correct(
                    dxr.open_disp(stack_dir, corrections=True),
                    ["ionospheric_delay", "solid_earth_tide"],
                ),
                "recommended",
            )
        ),
        "point",
        row=5,
        col=7,
    )
    back = xr.open_zarr(out, mask_and_scale=False)
    np.testing.assert_allclose(
        back.displacement.values, expected.displacement.values, equal_nan=True
    )
    ops = [json.loads(line)["op"] for line in back.attrs["history"].splitlines()]
    assert ops == ["open_disp", "correct", "mask", "stack_disp", "rereference"]


def test_stack_options(stack_dir, tmp_path):
    out = tmp_path / "out.nc"
    run(
        "stack",
        stack_dir,
        "--out",
        out,
        "--corrections",
        "none",
        "--mask",
        "none",
        "--reference",
        "none",
        "--start",
        "2020-03-01",
    )
    back = xr.open_dataset(out)
    ops = [json.loads(line)["op"] for line in back.attrs["history"].splitlines()]
    assert ops == ["open_disp", "stack_disp"]
    assert back.time.values.min() >= np.datetime64("2020-03-01")
    assert "ionospheric_delay" not in back


def test_stack_refuses_existing_and_bad_suffix(stack_dir, tmp_path):
    out = tmp_path / "out.zarr"
    run("stack", stack_dir, "--out", out, "--reference", "none")
    assert "--overwrite" in run("stack", stack_dir, "--out", out, ok=False).output
    run("stack", stack_dir, "--out", out, "--reference", "none", "--overwrite")
    bad = run("stack", stack_dir, "--out", tmp_path / "out.tif", ok=False)
    assert ".zarr or .nc" in bad.output


def test_stack_product_corrections_on_nisar(disp_stacks, tmp_path):
    # DISP-NISAR has the ionosphere applied already; "product" skips it.
    stack = disp_stacks["nisar"]
    out = tmp_path / "out.zarr"
    run("stack", stack.paths[0].parent, "--out", out, "--reference", "none")
    applied = json.loads(xr.open_zarr(out).attrs["corrections_applied"])
    assert "ionospheric_delay" not in applied
    bad = run(
        "stack",
        stack.paths[0].parent,
        "--out",
        tmp_path / "b.zarr",
        "--corrections",
        "ionospheric_delay",
        ok=False,
    )
    assert "already applied" in bad.output or "Error" in bad.output


# --- export --------------------------------------------------------------------


@pytest.fixture
def stored(stack_dir, tmp_path):
    out = tmp_path / "stack.zarr"
    run("stack", stack_dir, "--out", out, "--reference", "none")
    return out


def test_export_geotiff(stored, tmp_path):
    tif = tmp_path / "last.tif"
    run("export", stored, "--format", "geotiff", "--out", tif)
    back = xr.open_zarr(stored, mask_and_scale=False)
    with rasterio.open(tif) as src:
        np.testing.assert_allclose(
            src.read(1), back.displacement.isel(time=-1).values, equal_nan=True
        )
    one = tmp_path / "epoch.tif"
    run("export", stored, "--format", "geotiff", "--out", one, "--epoch", "2020-02-07")
    epochs = tmp_path / "epochs"
    out = run(
        "export", stored, "--format", "geotiff", "--out", epochs, "--per-epoch"
    ).output
    assert f"Wrote {back.sizes['time']} GeoTIFF" in out


def test_export_geotiff_unknown_layer(stored, tmp_path):
    out = run(
        "export",
        stored,
        "--format",
        "geotiff",
        "--out",
        tmp_path / "a.tif",
        "--layer",
        "nope",
        ok=False,
    ).output
    assert "No layer 'nope'" in out


def test_export_zarr_and_netcdf_from_granules(stack_dir, tmp_path):
    expected = dxr.stack_disp(dxr.open_disp(stack_dir))
    run("export", stack_dir, "--format", "zarr", "--out", tmp_path / "a.zarr")
    run("export", stack_dir, "--format", "netcdf", "--out", tmp_path / "a.nc")
    for back in (
        xr.open_zarr(tmp_path / "a.zarr", mask_and_scale=False),
        xr.open_dataset(tmp_path / "a.nc", mask_and_scale=False),
    ):
        np.testing.assert_array_equal(
            back.displacement.values, expected.displacement.values
        )
    bad = run(
        "export", stack_dir, "--format", "zarr", "--out", tmp_path / "a.nc", ok=False
    )
    assert "writes a .zarr output" in bad.output


def test_export_mintpy(stored, tmp_path):
    out = run(
        "export", stored, "--format", "mintpy", "--out", tmp_path / "mintpy"
    ).output
    assert "timeseries.h5" in out
    assert (tmp_path / "mintpy" / "timeseries.h5").exists()


# --- geometry ------------------------------------------------------------------


def test_geometry(stack_dir, tmp_path):
    ds = dxr.open_disp(stack_dir)
    d = tmp_path / "static"
    d.mkdir()
    _raster(ds, [0.6, -0.1, 0.79]).rio.to_raster(
        d / STATIC_NAME.format("line_of_sight_enu")
    )
    _raster(ds, [120.0], nodata=None).rio.to_raster(d / STATIC_NAME.format("dem"))
    out = tmp_path / "static.zarr"
    run("geometry", stack_dir, "--out", out, "--static-dir", d, "--no-download")
    back = xr.open_zarr(out)
    assert back.sizes["y"] == ds.sizes["y"] and back.sizes["x"] == ds.sizes["x"]
    assert "dem" in back


def test_geometry_missing_files(stack_dir, tmp_path):
    out = run(
        "geometry",
        stack_dir,
        "--out",
        tmp_path / "s.zarr",
        "--static-dir",
        tmp_path,
        "--no-download",
        ok=False,
    ).output
    assert "Error: FileNotFoundError" in out and "Traceback" not in out


# --- search --------------------------------------------------------------------


def test_search(monkeypatch, tmp_path):
    calls = {}

    def fake_search(**kwargs):
        calls.update(kwargs)
        return pd.DataFrame({"filename": ["a.nc", "b.nc"]})

    monkeypatch.setattr("disp_xr.search.search", fake_search)
    out = run(
        "search", "--frame-id", 11116, "--start", "2020-01-01", "--end", "2020-06-01"
    ).output
    assert "a.nc" in out and "b.nc" in out
    assert calls["frame_id"] == 11116 and calls["product_version"] == "1.0"
    assert calls["short_name"] == "OPERA_L3_DISP-S1_V1"
    assert calls["start_datetime"].year == 2020

    table = tmp_path / "found.csv"
    run(
        "search",
        "--frame-id",
        1,
        "--product",
        "DISP-NISAR",
        "--version",
        "any",
        "--url-type",
        "s3",
        "--out",
        table,
    )
    assert calls["product_version"] is None and calls["url_type"].value == "s3"
    assert calls["short_name"] == dxr.product.PRODUCTS["DISP-NISAR"].cmr_short_name
    assert len(pd.read_csv(table)) == 2


# --- virtual -------------------------------------------------------------------


def test_virtual(stack_dir, tmp_path):
    pytest.importorskip("icechunk")
    pytest.importorskip("virtualizarr")
    repo = tmp_path / "repo"
    out = run("virtual", stack_dir, "--repo", repo, "--workers", 1).output
    assert "Committed 2 ministacks" in out and "Validated 10 granules" in out
    assert dxr.open_virtual(repo).sizes["time"] == 10


def test_setup_logging_with_file(tmp_path):
    # The config named JsonFormatter, not JSONFormatter, so this always failed.
    import logging

    from disp_xr.log.logging import setup_logging

    log = tmp_path / "logs" / "run.jsonl"
    setup_logging(filename=log)
    logging.getLogger("disp_xr").info("hello")
    for handler in logging.getLogger("disp_xr").handlers:
        handler.flush()
    assert json.loads(log.read_text().splitlines()[-1])["message"] == "hello"
