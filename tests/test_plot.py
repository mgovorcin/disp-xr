"""Quick-look plots (plan task T17): image-free smoke tests on Agg."""

import numpy as np
import pandas as pd
import pytest

import disp_xr as dxr
from disp_xr.geometry import OutOfBoundsError

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
plt = pytest.importorskip("matplotlib.pyplot")


@pytest.fixture(autouse=True)
def _close():
    yield
    plt.close("all")


@pytest.fixture
def raw(stack_dir):
    return dxr.open_disp(stack_dir)


@pytest.fixture
def chained(raw):
    return dxr.chain(raw)


def test_map_last_epoch_in_mm(chained):
    ax = chained.disp.plot_map()
    image = ax.images[0]
    vmin, vmax = image.get_clim()
    assert vmin == -vmax > -np.inf  # symmetric about zero
    last = chained.displacement.isel(time=-1).values * 1000
    np.testing.assert_allclose(np.ma.filled(image.get_array(), np.nan), last, rtol=1e-6)
    assert "(mm)" in image.colorbar.ax.get_ylabel()
    assert f"EPSG:{chained.rio.crs.to_epsg()}" in ax.get_xlabel()
    assert (
        ax.get_title(loc="left") == "displacement, 2020-04-30, relative to 2020-01-01"
    )
    assert ax.get_title() == ""  # xarray's own title is cleared


def test_map_nearest_epoch_and_limits(chained):
    ax = dxr.plot.plot_map(chained, "2020-02-07", vmax=5, units="m")
    assert ax.images[0].get_clim() == (-5, 5)
    assert "(m)" in ax.images[0].colorbar.ax.get_ylabel()
    assert "2020-02-06" in ax.get_title(loc="left")


def test_map_robust_limit_ignores_outliers(chained):
    da = chained.displacement.isel(time=-1).copy()
    da[0, 0] = 1e3
    vmax = dxr.plot.plot_map(da, percentile=98).images[0].get_clim()[1]
    assert vmax < 1e5  # 1e3 m would be 1e6 mm


def test_map_block_averages_large_maps(chained):
    image = dxr.plot.plot_map(chained, max_pixels=16).images[0]
    assert max(image.get_array().shape) <= 16


def test_map_does_not_touch_input(chained):
    before = dict(chained.displacement.attrs)
    chained.disp.plot_map()
    assert chained.displacement.attrs == before


def test_map_on_dataarray_and_errors(chained):
    ax = chained.displacement.disp.plot_map(time=chained.time[0])
    assert ax.images
    with pytest.raises(ValueError, match="No layer"):
        chained.disp.plot_map(layer="nope")
    with pytest.raises(ValueError, match=r"\(y, x\)"):
        dxr.plot.plot_map(chained.displacement.isel(x=0))


def test_point_by_lonlat_matches_pixel(chained):
    row, col = 10, 20
    x, y = float(chained.x[col]), float(chained.y[row])
    from pyproj import Transformer

    lon, lat = Transformer.from_crs(
        chained.rio.crs, "EPSG:4326", always_xy=True
    ).transform(x, y)
    ax = chained.disp.plot_point(lon, lat)
    (line,) = ax.lines[:1]
    expected = chained.displacement.isel(y=row, x=col).values * 1000
    np.testing.assert_allclose(line.get_ydata(), expected, rtol=1e-6)
    assert "lon" in ax.get_title(loc="left")
    assert "(mm)" in ax.get_ylabel()


def test_point_reference_markers(raw, chained):
    refs = pd.unique(chained.reference_time.values)
    ax = chained.disp.plot_point(row=5, col=5)
    rules = [line for line in ax.lines if line.get_linestyle() == "--"]
    assert len(rules) == len(refs) - 1
    assert any("reference change" in t.get_text() for t in ax.texts)
    # Unchained data restarts at zero at each change; markers show where.
    ax = raw.disp.plot_point(row=5, col=5)
    assert len([ln for ln in ax.lines if ln.get_linestyle() == "--"]) == len(refs) - 1


def test_point_buffer_median(chained):
    ax = chained.disp.plot_point(row=10, col=10, buffer_m=30)
    window = chained.displacement.isel(y=slice(9, 12), x=slice(9, 12))
    expected = window.median(dim=("y", "x")).values * 1000
    np.testing.assert_allclose(ax.lines[0].get_ydata(), expected, rtol=1e-6)
    assert "median within 30 m" in ax.get_title(loc="left")


def test_point_errors(chained):
    with pytest.raises(OutOfBoundsError):
        chained.disp.plot_point(0.0, 0.0)
    with pytest.raises(ValueError, match="lon and lat"):
        chained.disp.plot_point()
    with pytest.raises(ValueError, match="time"):
        dxr.plot.plot_point(chained.displacement.isel(time=0), row=1, col=1)


def test_draws_on_given_axes(chained):
    fig, (a, b) = plt.subplots(1, 2)
    assert chained.disp.plot_map(ax=a) is a
    assert chained.disp.plot_point(row=1, col=1, ax=b) is b


def test_core_import_without_matplotlib(stack_dir):
    import subprocess
    import sys

    code = (
        "import sys; sys.modules['matplotlib'] = None\n"
        "import disp_xr as dxr\n"
        f"ds = dxr.open_disp({str(stack_dir)!r})\n"
        "try:\n"
        "    ds.disp.plot_map()\n"
        "except ImportError as e:\n"
        "    print(e)\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert "disp_xr[plot]" in out.stdout


def test_core_does_not_import_interactive_libraries():
    import subprocess
    import sys

    code = (
        "import sys, disp_xr, disp_xr.plot, disp_xr.cli\n"
        "print(sorted(m for m in ('hvplot', 'datashader', 'holoviews', 'bokeh')"
        " if m in sys.modules))\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "[]"
