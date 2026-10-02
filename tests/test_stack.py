"""Ministack chaining (plan task T7).

The kernel cases follow opera-utils tests/test_disp_rebase.py (commit dff4b45),
with explicit secondary dates, since disp-xr takes each offset from the epoch
whose date equals the new reference date.
"""

import dask
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from disp_xr import product, stack
from disp_xr.open import open_disp
from disp_xr.stack import MissingCrossoverError, NaNPolicy, stack_disp

D = pd.Timestamp


def _ds(values, times, refs):
    """A (time, y, x) dataset of constant images."""
    data = np.array(values, dtype=np.float32)[:, None, None] * np.ones((1, 2, 2), "f4")
    return xr.Dataset(
        {"displacement": (("time", "y", "x"), data)},
        coords={
            "time": pd.DatetimeIndex(times),
            "reference_time": ("time", pd.DatetimeIndex(refs)),
        },
    )


# --- kernel cases (ported) -----------------------------------------------------


def test_single_reference_is_identity():
    ds = _ds([1, 2, 3], ["2020-01-13", "2020-01-25", "2020-02-06"], ["2020-01-01"] * 3)
    out = stack_disp(ds, zero_epoch=False)
    np.testing.assert_array_equal(out.displacement.values, ds.displacement.values)


def test_two_references_add_crossover():
    times = ["2020-01-13", "2020-01-25", "2020-02-06", "2020-02-18"]
    refs = ["2020-01-01", "2020-01-01", "2020-01-25", "2020-01-25"]
    out = stack_disp(_ds([1, 2, 1, 2], times, refs), zero_epoch=False)
    np.testing.assert_array_equal(out.displacement.values[:, 0, 0], [1, 2, 3, 4])


def test_three_references():
    times = ["2020-01-13", "2020-01-25", "2020-02-06", "2020-02-18", "2020-03-01"]
    refs = ["2020-01-01", "2020-01-01", "2020-01-25", "2020-01-25", "2020-02-18"]
    out = stack_disp(_ds([1, 2, 1, 2, 5], times, refs), zero_epoch=False)
    np.testing.assert_array_equal(out.displacement.values[:, 0, 0], [1, 2, 3, 4, 9])


@pytest.mark.parametrize("nan_policy", ["propagate", "omit"])
def test_nan_at_crossover(nan_policy):
    times = ["2020-01-13", "2020-01-25", "2020-02-06"]
    refs = ["2020-01-01", "2020-01-01", "2020-01-25"]
    ds = _ds([1, 2, 1], times, refs)
    ds.displacement[1, 0, 0] = np.nan
    out = stack_disp(ds, nan_policy=nan_policy, zero_epoch=False).displacement
    if nan_policy == "propagate":
        assert np.isnan(out.values[2, 0, 0])
    else:
        assert out.values[2, 0, 0] == 1.0  # missing offset taken as zero
    assert out.values[2, 1, 1] == 3.0  # other pixels unaffected


def test_offset_comes_from_crossover_date_not_previous_epoch():
    # The epoch before the reference change is not the crossover here: the
    # 2020-02-06 secondary of ministack 1 comes after the new reference date.
    times = ["2020-01-13", "2020-01-25", "2020-02-06", "2020-02-18"]
    refs = ["2020-01-01", "2020-01-01", "2020-01-01", "2020-01-25"]
    out = stack_disp(_ds([1, 2, 7, 3], times, refs), zero_epoch=False)
    np.testing.assert_array_equal(out.displacement.values[:, 0, 0], [1, 2, 7, 5])


def test_missing_crossover_raises():
    times = ["2020-01-13", "2020-02-18"]
    refs = ["2020-01-01", "2020-01-25"]
    with pytest.raises(MissingCrossoverError, match=r"\['2020-01-25'\]"):
        stack_disp(_ds([1, 2], times, refs))


def test_keeps_input_dtype():
    ds = _ds([1, 2], ["2020-01-13", "2020-01-25"], ["2020-01-01"] * 2)
    ds["displacement"] = ds.displacement.astype("float64")
    assert stack_disp(ds).displacement.dtype == np.float64


def test_nan_policy_enum_values():
    assert NaNPolicy("omit") is NaNPolicy.omit
    with pytest.raises(ValueError, match="'drop' is not a valid NaNPolicy"):
        stack_disp(_ds([1], ["2020-01-13"], ["2020-01-01"]), nan_policy="drop")


# --- stack_disp on synthetic granules ----------------------------------------


@pytest.fixture
def opened(stack_dir):
    return open_disp(stack_dir)


@pytest.mark.parametrize("variant", ["two_ministacks", "three_ministacks"])
def test_chained_matches_truth(disp_stacks, variant):
    truth = disp_stacks[variant]
    out = stack_disp(open_disp(truth.paths[0].parent))
    np.testing.assert_allclose(out.displacement.values, truth.cumulative, atol=1e-6)


def test_zero_epoch(two_ministacks, opened):
    out = stack_disp(opened)
    first = two_ministacks.times[0]
    assert out.time.values[0] == np.datetime64(first)
    assert out.time.to_index().is_unique
    assert out.reference_time.values[0] == np.datetime64(first)
    assert (out.displacement.isel(time=0) == 0).all()
    assert np.isnan(out.temporal_coherence.isel(time=0)).all()
    assert (out.recommended_mask.isel(time=0) == 255).all()
    assert out.recommended_mask.dtype == np.uint8
    assert out.attrs["reference_date"] == str(first)


def test_without_zero_epoch(two_ministacks, opened):
    out = stack_disp(opened, zero_epoch=False)
    assert out.sizes["time"] == len(two_ministacks.paths)


def test_dims_and_chunks(opened):
    out = stack_disp(opened)
    assert out.displacement.dims == ("time", "y", "x")
    assert len(out.displacement.chunks[0]) == 1


def test_quality_layers_unchanged(opened):
    out = stack_disp(opened, zero_epoch=False)
    xr.testing.assert_identical(out.temporal_coherence, opened.temporal_coherence)


def test_start_at_second_ministack(two_ministacks, opened):
    second = two_ministacks.reference_times.unique()[1]
    out = stack_disp(opened, start=second)
    assert out.time.values[0] == np.datetime64(second)
    expected = two_ministacks.cumulative[6:] - two_ministacks.cumulative[5]
    np.testing.assert_allclose(out.displacement.values[1:], expected, atol=1e-6)


def test_start_must_be_a_reference_date(opened):
    with pytest.raises(ValueError, match="start must be a ministack reference date"):
        stack_disp(opened, start="2020-02-06")


def test_end_drops_later_epochs(opened):
    out = stack_disp(opened, end="2020-02-06")
    assert pd.Timestamp(out.time.values[-1]) <= D("2020-02-06")


def test_missing_crossover_in_real_layout(disp_stacks, tmp_path):
    import shutil

    truth = disp_stacks["two_ministacks"]
    crossover = truth.reference_times.unique()[1]
    for p in truth.paths:
        if crossover.strftime("%Y%m%dT%H%M%SZ_v") not in p.name:
            shutil.copy(p, tmp_path / p.name)
    with pytest.raises(MissingCrossoverError):
        stack_disp(open_disp(tmp_path))


def test_is_lazy(opened):
    def fail(*_args, **_kwargs):
        raise AssertionError("stack_disp computed")

    with dask.config.set(scheduler=fail):
        out = stack_disp(opened)
    assert isinstance(out.displacement.data, dask.array.Array)


# --- deprecated combine_disp_product (characterization, plan T2.6) -----------


@pytest.fixture
def chained(stack_dir):
    return stack.combine_disp_product(product.get_disp_info(stack_dir))


def test_combine_matches_truth(two_ministacks, chained):
    np.testing.assert_allclose(
        chained.displacement.values, two_ministacks.cumulative, atol=1e-6
    )


def test_combine_times_start_at_first_reference(two_ministacks, chained):
    np.testing.assert_array_equal(
        chained.time.values, two_ministacks.times.values.astype("datetime64[ns]")
    )


def test_combine_start_date_zero_epoch_is_first_selected_reference(
    two_ministacks, stack_dir
):
    """Regression: bug 3, the zero epoch kept the first product's date."""
    df = product.get_disp_info(stack_dir)
    ds = stack.combine_disp_product(df, start_date="2020-03-01")
    second_reference = two_ministacks.reference_times.unique()[1]
    assert ds.time.values[0] == np.datetime64(second_reference)


def test_combine_zero_epoch_has_no_fake_quality_values(chained):
    """Regression: bug 3, quality layers were set to 0 at the zero epoch."""
    assert np.isnan(chained.temporal_coherence.isel(time=0).values).all()


def test_combine_is_deprecated(stack_dir):
    with pytest.warns(DeprecationWarning, match="stack_disp"):
        stack.combine_disp_product(product.get_disp_info(stack_dir))


def test_create_rebased_displacement_alias():
    ds = _ds(
        [1, 2, 1, 2],
        ["2020-01-13", "2020-01-25", "2020-02-06", "2020-02-18"],
        ["2020-01-01", "2020-01-01", "2020-01-25", "2020-01-25"],
    )
    with pytest.warns(DeprecationWarning, match="stack_disp"):
        out = stack.create_rebased_displacement(
            ds.displacement, ds.reference_time.values
        )
    np.testing.assert_array_equal(out.values[:, 0, 0], [1, 2, 3, 4])
