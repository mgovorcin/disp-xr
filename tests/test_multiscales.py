"""Multiscale pyramids in Zarr (plan task T27.7)."""

import numpy as np
import pytest
import xarray as xr
import zarr

import disp_xr as dxr
from disp_xr.export.multiscales import MULTISCALES_CONVENTION, build_overviews


@pytest.fixture
def ds(stack_dir):
    return dxr.stack_disp(dxr.open_disp(stack_dir)).disp.rereference("median")


def test_build_overviews_layout(ds):
    tree = build_overviews(ds, factors=(2, 33))
    assert list(tree.children) == ["0", "1", "2"]
    layout = tree.attrs["multiscales"]["layout"]
    assert [e["asset"] for e in layout] == ["0", "1", "2"]
    assert "derived_from" not in layout[0]
    for entry, factor in zip(layout[1:], (2, 33)):
        level = tree[entry["asset"]].to_dataset()
        assert entry["derived_from"] == "0"
        assert entry["transform"] == {
            "scale": [float(factor)] * 2,
            "translation": [0.0, 0.0],
        }
        t = level.rio.transform()
        assert entry["spatial:transform"] == [t.a, t.b, t.c, t.d, t.e, t.f]
        assert entry["spatial:shape"] == [level.sizes["y"], level.sizes["x"]]
        assert entry["disp_xr:resampling"]["displacement"] == "weighted_mean"
        assert entry["disp_xr:resampling"]["recommended_mask"] == "mode"
    assert layout[2]["spatial:transform"][0] == 30.0 * 33
    # Every level holds the variables of level 0, and its own GeoTransform.
    base = set(tree["0"].data_vars)
    for name in ("1", "2"):
        level = tree[name].to_dataset()
        assert base <= set(level.data_vars)
        gt = [float(v) for v in level.spatial_ref.attrs["GeoTransform"].split()]
        t = level.rio.transform()
        assert gt == [t.c, t.a, t.b, t.f, t.d, t.e]
    # Cell edges start at the first pixel edge: same origin on every level.
    assert len({tuple(e["spatial:transform"][2::3]) for e in layout}) == 1
    assert tree.attrs["zarr_conventions"][0]["uuid"] == MULTISCALES_CONVENTION["uuid"]
    assert tree.attrs["disp_xr:multiscales_convention"] == "v0.1"
    assert tree.attrs["disp_xr:weights"] == "quality"
    with pytest.raises(ValueError, match="increasing"):
        build_overviews(ds, factors=(33, 2))


def test_round_trip(ds, tmp_path):
    store = tmp_path / "pyramid.zarr"
    written = dxr.to_pyramid(ds, store, factors=(2, 33), stat="wmedian")
    tree = dxr.open_pyramid(store)
    assert list(tree.children) == ["0", "1", "2"]
    assert tree.attrs["multiscales"]["resampling_method"] == "weighted_median"
    assert tree.attrs["disp_xr_version"] == dxr.__version__
    for name in ("0", "1", "2"):
        level = tree[name].to_dataset()
        # Each level is self-contained: CRS and transform without the root.
        assert level.rio.crs == ds.rio.crs
        assert level.rio.transform() == written[name].to_dataset().rio.transform()
        xr.testing.assert_allclose(
            level.displacement.load(),
            written[name].to_dataset().displacement.load(),
        )
    group = zarr.open_group(store / "2", mode="r")
    assert group.attrs["proj:code"] == "EPSG:32610"
    assert group.attrs["spatial:shape"] == [
        tree["2"].sizes["y"],
        tree["2"].sizes["x"],
    ]
    assert group["displacement"].metadata.dimension_names == ("time", "y", "x")


def test_small_levels_are_one_chunk(ds, tmp_path):
    store = tmp_path / "pyramid.zarr"
    dxr.to_pyramid(ds, store, factors=(33,), small_mb=1)
    coarse = zarr.open_group(store / "1", mode="r")["displacement"]
    assert coarse.chunks == coarse.shape


def test_open_reads_the_layout_not_group_names(ds, tmp_path):
    store = tmp_path / "pyramid.zarr"
    dxr.to_pyramid(ds, store, factors=(2, 33))
    root = zarr.open_group(store, mode="r+")
    multiscales = dict(root.attrs["multiscales"])
    multiscales["layout"] = [multiscales["layout"][i] for i in (0, 2)]
    root.attrs["multiscales"] = multiscales
    tree = dxr.open_pyramid(store)
    assert list(tree.children) == ["0", "2"]  # group "1" exists, not listed
    with pytest.raises(FileExistsError):
        dxr.to_pyramid(ds, store, factors=(2,))
    dxr.to_pyramid(ds, store, factors=(3,), mode="w")
    assert list(dxr.open_pyramid(store).children) == ["0", "1"]


def test_open_needs_a_layout(ds, tmp_path):
    store = tmp_path / "plain.zarr"
    dxr.to_zarr(ds, store)
    with pytest.raises(ValueError, match="no multiscales"):
        dxr.open_pyramid(store)


def test_written_tree_matches_coarse_values(ds, tmp_path):
    store = tmp_path / "pyramid.zarr"
    dxr.to_pyramid(ds, store, factors=(33,))
    stored = dxr.open_pyramid(store)["1"].to_dataset()
    direct = dxr.downsample(ds, factor=33).compute()
    np.testing.assert_allclose(
        stored.displacement.values, direct.displacement.values, equal_nan=True
    )
