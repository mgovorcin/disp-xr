"""Characterization tests for ``disp_xr.product`` (plan task T2.6).

Tests marked ``xfail(strict=True)`` assert the correct behavior for bugs listed
in docs/code_review.md; they turn into failures (XPASS) once T5 fixes them, at
which point the marker is removed.
"""

import shutil

import pytest

from disp_xr import product


def test_get_disp_info_one_row_per_granule(two_ministacks, stack_dir):
    df = product.get_disp_info(stack_dir)
    assert len(df) == len(two_ministacks.paths)
    assert df["date12"].is_monotonic_increasing
    assert set(df["date1"]) == {"20200101", "20200301"}


def test_get_disp_info_columns(stack_dir):
    df = product.get_disp_info(stack_dir)
    # "polarizarion" is a typo in the current code; T5 renames it.
    assert {"frame_id", "polarizarion", "version", "path", "date1", "date2"} <= set(
        df.columns
    )


def test_get_disp_info_parses_dates(two_ministacks, stack_dir):
    df = product.get_disp_info(stack_dir)
    assert df["end_date"].min() == two_ministacks.times[1]


@pytest.mark.xfail(
    strict=True,
    reason="Bug 8 (code_review.md): .unique().max() fails on pandas 3 string arrays",
    raises=AttributeError,
)
def test_get_disp_info_keeps_latest_version_of_a_duplicate(disp_stacks):
    stack = disp_stacks["duplicate"]
    df = product.get_disp_info(stack.paths[0].parent)
    assert len(df) == len(stack.paths)
    assert set(df["version"]) == {"v1.0"}


@pytest.mark.xfail(
    strict=True,
    reason="Bug 1 (code_review.md): versions compared as strings",
    raises=(AttributeError, AssertionError),
)
def test_get_disp_info_orders_versions_numerically(two_ministacks, tmp_path):
    for path in two_ministacks.paths[:2]:
        shutil.copy(path, tmp_path / path.name.replace("_v1.0_", "_v1.10_"))
        shutil.copy(path, tmp_path / path.name.replace("_v1.0_", "_v1.9_"))
    df = product.get_disp_info(tmp_path)
    assert set(df["version"]) == {"v1.10"}
