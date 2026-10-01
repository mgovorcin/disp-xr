"""Old import paths keep working and warn (plan task T4.3; removed in T38)."""

import importlib
import sys

import pytest

import disp_xr.export.raster
import disp_xr.geometry
import disp_xr.quality
import disp_xr.reference

MOVED_MODULES = [
    ("disp_xr.quality_metrics", disp_xr.quality, "get_stack_stat"),
    ("disp_xr.io", disp_xr.export.raster, "write_geotiff"),
]


@pytest.mark.parametrize(("old", "new", "name"), MOVED_MODULES)
def test_old_module_warns_and_forwards(old, new, name):
    sys.modules.pop(old, None)
    with pytest.warns(DeprecationWarning, match=f"{old} moved to {new.__name__}"):
        module = importlib.import_module(old)
    assert getattr(module, name) is getattr(new, name)


@pytest.mark.parametrize(
    ("name", "new"),
    [
        ("latlon_to_utm", disp_xr.geometry),
        ("utm_to_latlon", disp_xr.geometry),
        ("get_extent", disp_xr.geometry),
        ("get_burst_ids", disp_xr.geometry),
        ("find_reference_point", disp_xr.reference),
    ],
)
def test_moved_utils_name_warns_and_forwards(name, new):
    import disp_xr.utils

    with pytest.warns(DeprecationWarning, match=f"utils.{name} moved to"):
        func = getattr(disp_xr.utils, name)
    assert func is getattr(new, name)


def test_utils_unknown_name_raises():
    import disp_xr.utils

    with pytest.raises(AttributeError, match="has no attribute 'nope'"):
        disp_xr.utils.nope  # noqa: B018
