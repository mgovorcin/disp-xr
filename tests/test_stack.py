"""Characterization tests for ``disp_xr.stack`` (plan task T2.6)."""

import numpy as np
import pytest

from disp_xr import product, stack

# combine_disp_product calls open_mfdataset with xarray's changing data_vars
# default; T6.1 sets it explicitly and removes this filter.
pytestmark = pytest.mark.filterwarnings(
    "ignore:In a future version of xarray the default value for data_vars"
)


@pytest.fixture
def chained(stack_dir):
    return stack.combine_disp_product(product.get_disp_info(stack_dir))


def test_chained_displacement_matches_truth(two_ministacks, chained):
    np.testing.assert_allclose(
        chained.displacement.values, two_ministacks.cumulative, atol=1e-6
    )


def test_chained_times_start_at_first_reference(two_ministacks, chained):
    np.testing.assert_array_equal(
        chained.time.values, two_ministacks.times.values.astype("datetime64[ns]")
    )


def test_three_ministacks_chain(disp_stacks):
    truth = disp_stacks["three_ministacks"]
    ds = stack.combine_disp_product(product.get_disp_info(truth.paths[0].parent))
    np.testing.assert_allclose(ds.displacement.values, truth.cumulative, atol=1e-6)


def test_chained_stack_has_one_time_chunk_per_epoch(chained):
    # Current behavior: chunks={"time": -1} applies per single-epoch file, so the
    # chained stack is split at every epoch. T6.5 and T7 make time one chunk.
    assert chained.displacement.chunks[0] == (1,) * chained.sizes["time"]


@pytest.mark.xfail(
    strict=True,
    reason="Bug 3 (code_review.md): zero epoch dated at the first product, not at "
    "the first selected reference",
)
def test_start_date_zero_epoch_is_first_selected_reference(two_ministacks, stack_dir):
    df = product.get_disp_info(stack_dir)
    ds = stack.combine_disp_product(df, start_date="2020-03-01")
    second_reference = two_ministacks.reference_times.unique()[1]
    assert ds.time.values[0] == np.datetime64(second_reference)


@pytest.mark.xfail(
    strict=True,
    reason="Bug 3 (code_review.md): quality layers are set to 0 at the zero epoch",
)
def test_zero_epoch_has_no_fake_quality_values(chained):
    assert np.isnan(chained.temporal_coherence.isel(time=0).values).all()
