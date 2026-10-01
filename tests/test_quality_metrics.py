"""Characterization tests for ``disp_xr.quality_metrics`` (plan task T2.6)."""

import numpy as np
import pytest
import xarray as xr
from fixtures.make_disp import WATER_COLUMNS

from disp_xr import product, quality_metrics, stack

# combine_disp_product calls open_mfdataset with xarray's changing data_vars
# default; T6.1 sets it explicitly and removes this filter.
pytestmark = pytest.mark.filterwarnings(
    "ignore:In a future version of xarray the default value for data_vars"
)


@pytest.fixture
def epochs(stack_dir):
    ds = stack.combine_disp_product(product.get_disp_info(stack_dir))
    return ds.isel(time=slice(1, None))


def test_get_stack_stat_mean(epochs):
    mean = quality_metrics.get_stack_stat(epochs.temporal_coherence, "mean")
    assert mean.shape == (64, 64)
    assert float(mean[0, 0]) == pytest.approx(0.3)
    assert float(mean[40, 40]) == pytest.approx(0.9)


def test_get_stack_stat_rejects_unknown_mode(epochs):
    with pytest.raises(ValueError, match="Unsupported mode: 'mode'"):
        quality_metrics.get_stack_stat(epochs.temporal_coherence, "mode")


def test_get_stack_stat_rejects_dataset(epochs):
    with pytest.raises(TypeError, match="Input must be"):
        quality_metrics.get_stack_stat(epochs, "mean")


def test_get_value_percentage_of_masked_epochs(epochs):
    pct = quality_metrics.get_value_percentage(
        epochs.recommended_mask.chunk({"time": -1}), value=0
    )
    assert pct[0, 0] == 100.0
    assert pct[40, 40] == 0.0
    assert (pct[:, -WATER_COLUMNS:] == 100.0).all()


def test_get_value_percentage_reverse(epochs):
    pct = quality_metrics.get_value_percentage(
        epochs.recommended_mask.chunk({"time": -1}), value=0, reverse=True
    )
    assert pct[40, 40] == 100.0


def test_get_counts_counts_along_first_axis():
    a = np.array([[[0, 1]], [[0, 0]], [[1, 0]]])
    np.testing.assert_array_equal(quality_metrics.get_counts(a, 0), [[[2, 2]]])
    np.testing.assert_array_equal(
        quality_metrics.get_counts(a, 0, reverse=True), [[[1, 1]]]
    )


def test_get_stack_stat_is_eager(epochs):
    # Characterizes current behavior; T10 makes it lazy.
    mean = quality_metrics.get_stack_stat(epochs.temporal_coherence, "mean")
    assert isinstance(mean.data, np.ndarray)
    assert isinstance(mean, xr.DataArray)
