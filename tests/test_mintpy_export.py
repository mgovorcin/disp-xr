"""MintPy export (plan task T14.3)."""

import h5py
import numpy as np
import pandas as pd
import pytest
import xarray as xr

import disp_xr as dxr
from disp_xr.export.mintpy import mintpy_metadata

INC, AZ = 40.0, 100.0  # clockwise azimuth of a descending ground-to-satellite LOS


@pytest.fixture
def ds(stack_dir):
    return dxr.chain(dxr.open_disp(stack_dir, corrections=True))


@pytest.fixture
def static(ds):
    like = ds.temporal_coherence.isel(time=1, drop=True)
    lsm = xr.zeros_like(like, dtype=np.uint8)
    lsm[3, 3], lsm[4, 4] = 2, 255
    return xr.Dataset(
        {
            "incidence_angle": xr.full_like(like, INC),
            "azimuth_angle": xr.full_like(like, AZ),
            "dem": xr.full_like(like, 120.0),
            "layover_shadow_mask": lsm,
        }
    )


def test_files_and_timeseries(ds, static, tmp_path):
    paths = dxr.to_mintpy(ds, tmp_path, static=static)
    assert set(paths) == {
        "timeseries",
        "avgSpatialCoh",
        "timeseriesDensity",
        "maskReliable",
        "geometryGeo",
    }
    with h5py.File(paths["timeseries"]) as f:
        np.testing.assert_array_equal(f["timeseries"][:], ds.displacement.values)
        dates = [d.decode() for d in f["date"][:]]
        assert dates == [f"{t:%Y%m%d}" for t in pd.DatetimeIndex(ds.time.values)]
        assert f["bperp"].shape == (ds.sizes["time"],)
        assert f.attrs["REF_DATE"] == dates[0] and f.attrs["FILE_TYPE"] == "timeseries"


def test_every_dataset_is_compressed(ds, static, tmp_path):
    """Regression: upstream bug 6, compression was off after the first dataset."""
    paths = dxr.to_mintpy(ds, tmp_path, static=static)
    for path in paths.values():
        with h5py.File(path) as f:
            for name, dset in f.items():
                if dset.ndim >= 1 and dset.dtype.kind != "S":
                    assert dset.compression == "gzip", (path.name, name)


def test_metadata_from_spec_and_grid(ds):
    meta = mintpy_metadata(ds)
    transform = ds.rio.transform()
    assert meta["X_FIRST"] == str(transform.c) and meta["Y_STEP"] == "-30.0"
    assert meta["WAVELENGTH"] == "0.05546576" and meta["PLATFORM"] == "Sentinel-1"
    assert meta["HEIGHT"] == "693000.0"
    assert meta["EPSG"] == str(ds.rio.crs.to_epsg())
    nisar = mintpy_metadata(ds.assign_attrs(product="DISP-NISAR"))
    assert nisar["PLATFORM"] == "NISAR" and nisar["WAVELENGTH"] == "0.2384"


def test_reference_point_recorded(ds, tmp_path):
    out = dxr.to_mintpy(ds.disp.rereference("point", row=40, col=41), tmp_path)
    with h5py.File(out["timeseries"]) as f:
        assert (f.attrs["REF_Y"], f.attrs["REF_X"]) == ("40", "41")


def test_geometry_conventions(ds, static, tmp_path):
    paths = dxr.to_mintpy(ds, tmp_path, static=static)
    with h5py.File(paths["geometryGeo"]) as f:
        assert f["azimuthAngle"][0, 0] == pytest.approx(-100.0)  # anticlockwise
        assert f["incidenceAngle"][0, 0] == pytest.approx(INC)
        assert 850e3 < f["slantRangeDistance"][0, 0] < 900e3
        shadow = f["shadowMask"][:]
        assert shadow[3, 3] == 1 and shadow[4, 4] == 0 and shadow[0, 0] == 0
        water = f["waterMask"][:]
        assert water[:, -1].sum() == 0 and water[:, 0].all()


def test_quality_files(ds, tmp_path):
    paths = dxr.to_mintpy(ds, tmp_path)
    with h5py.File(paths["maskReliable"]) as f:
        mask = f["mask"][:]
        assert mask[40, 40] == 1 and mask[0, 0] == 0
    with h5py.File(paths["avgSpatialCoh"]) as f:
        assert f["avgSpatialCoh"][40, 40] == pytest.approx(0.9)


def test_refuses_unchained_and_existing(stack_dir, ds, tmp_path):
    with pytest.raises(ValueError, match=r"chain first \(disp_xr.chain\)"):
        dxr.to_mintpy(dxr.open_disp(stack_dir), tmp_path)
    dxr.to_mintpy(ds, tmp_path)
    with pytest.raises(ValueError, match="overwrite=True"):
        dxr.to_mintpy(ds, tmp_path)
    dxr.to_mintpy(ds, tmp_path, overwrite=True)


def test_written_in_row_blocks(ds, tmp_path):
    # Enough for one (y, x) layer, a sixth of the cube: about 11 rows per block.
    budget = ds.sizes["y"] * ds.sizes["x"] * 8
    assert budget < ds.displacement.nbytes / 5
    with dxr.config.set({"max_load_bytes": budget}):
        paths = dxr.to_mintpy(ds, tmp_path)
    with h5py.File(paths["timeseries"]) as f:
        np.testing.assert_array_equal(f["timeseries"][:], ds.displacement.values)


def test_mintpy_reads_the_output(ds, static, tmp_path):
    readfile = pytest.importorskip("mintpy.utils.readfile")
    paths = dxr.to_mintpy(ds, tmp_path, static=static)
    data, atr = readfile.read(str(paths["timeseries"]))
    assert data.shape == ds.displacement.shape and atr["FILE_TYPE"] == "timeseries"
    inc, _ = readfile.read(str(paths["geometryGeo"]), datasetName="incidenceAngle")
    assert inc[0, 0] == pytest.approx(INC)
