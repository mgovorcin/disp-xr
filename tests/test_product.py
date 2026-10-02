"""Characterization tests for ``disp_xr.product`` (plan task T2.6).

Tests marked ``xfail(strict=True)`` assert the correct behavior for bugs listed
in docs/code_review.md; they turn into failures (XPASS) once T5 fixes them, at
which point the marker is removed.
"""

import shutil

import pandas as pd
import pytest

from disp_xr import product


def test_get_disp_info_one_row_per_granule(two_ministacks, stack_dir):
    df = product.get_disp_info(stack_dir)
    assert len(df) == len(two_ministacks.paths)
    assert df["date12"].is_monotonic_increasing
    assert set(df["date1"]) == {"20200101", "20200301"}


def test_get_disp_info_columns(stack_dir):
    df = product.get_disp_info(stack_dir)
    assert {"frame_id", "polarization", "version", "path", "date1", "date2"} <= set(
        df.columns
    )


def test_get_disp_info_parses_dates(two_ministacks, stack_dir):
    df = product.get_disp_info(stack_dir)
    assert df["end_date"].min() == two_ministacks.times[1]


def test_get_disp_info_keeps_latest_version_of_a_duplicate(disp_stacks):
    """Regression: bug 8, mixed versions crashed under pandas 3."""
    stack = disp_stacks["duplicate"]
    df = product.get_disp_info(stack.paths[0].parent)
    assert len(df) == len(stack.paths)
    assert set(df["version"]) == {"v1.0"}


def test_get_disp_info_orders_versions_numerically(two_ministacks, tmp_path):
    """Regression: bug 1, versions were compared as strings."""
    for path in two_ministacks.paths[:2]:
        shutil.copy(path, tmp_path / path.name.replace("_v1.0_", "_v1.10_"))
        shutil.copy(path, tmp_path / path.name.replace("_v1.0_", "_v1.9_"))
    df = product.get_disp_info(tmp_path)
    assert set(df["version"]) == {"v1.10"}


# --- find_products, DispGranule, group_ministacks (plan task T5) -------------

from disp_xr.product import (  # noqa: E402
    DISP_NISAR,
    DISP_S1,
    DispGranule,
    UnsupportedProductVersionError,
    find_products,
    group_ministacks,
)

S1_NAME = (
    "OPERA_L3_DISP-S1_IW_F11116_VV_20200101T000153Z_20200113T000153Z"
    "_v1.0_20250412T164531Z.nc"
)


def test_granule_from_s1_name():
    g = DispGranule.from_path(S1_NAME)
    assert (g.product, g.frame_id, g.version, g.polarization, g.mode) == (
        "DISP-S1",
        11116,
        "1.0",
        "VV",
        "IW",
    )
    assert g.reference_time == pd.Timestamp("2020-01-01T00:01:53")
    assert g.secondary_time == pd.Timestamp("2020-01-13T00:01:53")
    assert g.production_time == pd.Timestamp("2025-04-12T16:45:31")
    assert g.spec is DISP_S1


def test_granule_from_provisional_nisar_name():
    name = S1_NAME.replace("DISP-S1_IW", "DISP-NI_LSAR").replace("_VV_", "_HH_")
    g = DispGranule.from_path(name)
    assert (g.product, g.mode, g.polarization) == ("DISP-NISAR", "LSAR", "HH")
    assert not g.spec.validated


def test_granule_from_metadata_for_sas_names(disp_stacks):
    stack = disp_stacks["nisar"]
    g = DispGranule.from_path(stack.paths[0])
    assert g.product == "DISP-NISAR"
    assert (g.frame_id, g.version, g.production_time) == (26410, "0.4", None)
    assert g.secondary_time == stack.times[1]
    assert g.spec is DISP_NISAR


def test_granule_rejects_unknown_file(tmp_path):
    other = tmp_path / "other.nc"
    other.write_bytes(b"not hdf5")
    with pytest.raises(ValueError, match="not an OPERA DISP file name"):
        DispGranule.from_path(other)


def test_find_products_one_row_per_granule(two_ministacks, stack_dir):
    df = find_products(stack_dir)
    assert len(df) == len(two_ministacks.paths)
    assert df["secondary_time"].is_monotonic_increasing
    assert list(df["secondary_time"]) == list(two_ministacks.times[1:])
    assert set(df["product"]) == {"DISP-S1"}


def test_find_products_accepts_a_list(two_ministacks):
    assert len(find_products(two_ministacks.paths[:3])) == 3


def test_find_products_keeps_latest_and_reports_dropped(disp_stacks):
    stack = disp_stacks["duplicate"]
    df = find_products(stack.paths[0].parent)
    assert len(df) == len(stack.paths)
    assert set(df["version"]) == {"1.0"}
    dropped = df.attrs["dropped"]
    assert list(dropped["version"]) == ["0.9"]


def test_find_products_orders_versions_numerically(two_ministacks, tmp_path):
    for path in two_ministacks.paths[:2]:
        shutil.copy(path, tmp_path / path.name.replace("_v1.0_", "_v1.10_"))
        shutil.copy(path, tmp_path / path.name.replace("_v1.0_", "_v1.9_"))
    assert set(find_products(tmp_path)["version"]) == {"1.10"}


def test_find_products_min_version(two_ministacks, tmp_path):
    for path in two_ministacks.paths[:2]:
        shutil.copy(path, tmp_path / path.name)
    shutil.copy(
        two_ministacks.paths[2],
        tmp_path / two_ministacks.paths[2].name.replace("_v1.0_", "_v0.10_"),
    )
    assert len(find_products(tmp_path)) == 3
    assert len(find_products(tmp_path, min_version="1.0")) == 2


def test_find_products_rejects_unsupported_version(two_ministacks, tmp_path):
    path = two_ministacks.paths[0]
    shutil.copy(path, tmp_path / path.name.replace("_v1.0_", "_v2.0_"))
    with pytest.raises(UnsupportedProductVersionError, match="version 2.0"):
        find_products(tmp_path)
    assert len(find_products(tmp_path, allow_unsupported=True)) == 1


def test_find_products_requires_frame_when_mixed(disp_stacks):
    root = disp_stacks["two_utm_zones"].paths[0].parent.parent
    with pytest.raises(ValueError, match="span frames \\[11116, 11117\\]"):
        find_products(root)
    assert set(find_products(root, frame_id=11117)["frame_id"]) == {11117}


def test_find_products_nisar_sas_output(disp_stacks):
    df = find_products(disp_stacks["nisar"].paths[0].parent)
    assert set(df["product"]) == {"DISP-NISAR"}
    assert df["production_time"].isna().all()


def test_group_ministacks(two_ministacks, stack_dir):
    groups = group_ministacks(find_products(stack_dir))
    assert list(groups) == list(two_ministacks.reference_times.unique())
    assert [len(g) for g in groups.values()] == [5, 5]


# --- check_granules (plan task T5.8) ------------------------------------------

from disp_xr.product import MixedProductionError, check_granules  # noqa: E402


def test_check_granules_accepts_one_production(stack_dir):
    df = check_granules(find_products(stack_dir))
    assert set(df["layout"]) == {0}


def test_check_granules_refuses_mixed_productions(disp_stacks):
    df = find_products(disp_stacks["mixed_dtype"].paths[0].parent)
    with pytest.raises(MixedProductionError) as err:
        check_granules(df)
    message = str(err.value)
    assert "displacement float32" in message and "displacement float64" in message
    assert "2025-04-12" in message and "2026-09-20" in message


def test_check_granules_reads_one_file_per_production(monkeypatch, disp_stacks):
    import disp_xr.product as product

    opened: list[str] = []
    real = product._layout
    monkeypatch.setattr(
        product,
        "_layout",
        lambda path, layers: opened.append(path) or real(path, layers),
    )
    df = find_products(disp_stacks["mixed_dtype"].paths[0].parent)
    with pytest.raises(MixedProductionError):
        check_granules(df)
    assert len(opened) == 2  # one per production, not one per granule
