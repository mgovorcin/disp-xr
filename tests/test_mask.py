"""Quality masks and summaries (plan task T10)."""

import dask
import numpy as np
import pytest
import xarray as xr
from fixtures.make_disp import PS_PIXELS, WATER_COLUMNS

import disp_xr as dxr
from disp_xr.quality import combine_quality_masks, valid_mask

BAD = [(0, 0), (21, 21), (40, 63)]  # low coherence, unwrapping failed, water
GOOD = (40, 40)


@pytest.fixture
def ds(stack_dir):
    return dxr.open_disp(stack_dir)


def test_recommended_masks_exactly_the_bad_pixels(ds):
    out = dxr.mask(ds)
    disp = out.displacement.values
    expected_bad = ds.recommended_mask.values == 0
    assert np.isnan(disp[expected_bad]).all()
    np.testing.assert_array_equal(
        disp[~expected_bad], ds.displacement.values[~expected_bad]
    )
    for y, x in BAD:
        assert np.isnan(disp[:, y, x]).all()


def test_quality_layers_are_not_masked(ds):
    out = dxr.mask(ds)
    xr.testing.assert_identical(out.temporal_coherence, ds.temporal_coherence)
    assert out.short_wavelength_displacement.isnull().sum() > 0


def test_strict_uses_each_layer_even_if_recommended_says_good(ds):
    trusting = ds.assign(recommended_mask=xr.ones_like(ds.recommended_mask))
    valid = valid_mask(trusting, "strict")
    for y, x in BAD:
        assert not valid[:, y, x].any()
    assert valid[:, GOOD[0], GOOD[1]].all()


def test_strict_threshold_from_product_attrs(ds):
    assert '"temporal_coherence": 0.6' in valid_mask(ds, "strict").attrs["criteria"]


def test_explicit_threshold_overrides_preset(ds):
    valid = valid_mask(ds, "strict", temporal_coherence=0.95)
    assert not valid.any()


def test_criterion_can_be_turned_off(ds):
    valid = valid_mask(
        ds,
        "strict",
        recommended=False,
        water=False,
        connected_components=False,
        temporal_coherence=False,
        phase_similarity=0.5,
    )
    assert not bool(valid[0, 0, 0]) and bool(valid[0, 40, 63])


def test_no_criterion_or_unknown_preset_raises(ds):
    with pytest.raises(ValueError, match="No quality criterion chosen"):
        valid_mask(ds, None)
    with pytest.raises(ValueError, match="Unknown preset 'loose'"):
        valid_mask(ds, "loose")


def test_missing_layer_raises(ds):
    with pytest.raises(ValueError, match="needs the 'water_mask' layer"):
        valid_mask(ds.drop_vars("water_mask"), "strict")


def test_nodata_fails_connected_components(ds):
    labels = ds.connected_component_labels.copy()
    labels[0, 40, 40] = 65535
    valid = valid_mask(
        ds.assign(connected_component_labels=labels), None, connected_components=True
    )
    assert not bool(valid[0, 40, 40])


def test_mask_after_chaining_and_history(ds):
    out = ds.disp.chain().disp.mask("strict")
    assert np.isnan(out.displacement.isel(time=1, y=0, x=0))
    assert [h["op"] for h in out.disp.history][-1] == "mask"
    assert "connected_components" in out.attrs["mask_criteria"]


def test_mask_keeps_stacked_zero_epoch(ds):
    out = dxr.mask(dxr.chain(ds), "strict")
    assert float(out.displacement[0, GOOD[0], GOOD[1]]) == 0
    assert not np.isnan(out.displacement[1:, GOOD[0], GOOD[1]]).any()


def test_quality_summary_values(ds):
    q = dxr.quality_summary(ds).compute()
    assert float(q.pct_masked[0, 0]) == 100 and float(q.pct_masked[40, 40]) == 0
    assert (q.pct_masked[:, -WATER_COLUMNS:] == 100).all()
    for y, x in PS_PIXELS:
        assert float(q.pct_ps[y, x]) == 100
    assert float(q.temporal_coherence_hmean[40, 40]) == pytest.approx(0.9)
    assert bool(q.reliable[40, 40]) and not bool(q.reliable[0, 0])


def test_quality_summary_ignores_stacked_zero_epoch(ds):
    q = dxr.quality_summary(dxr.chain(ds)).compute()
    assert float(q.pct_masked[40, 40]) == 0 and float(q.reliability[40, 40]) == 1


def test_quality_summary_by_ministack(ds):
    q = dxr.quality_summary(ds, by_ministack=True)
    assert q.sizes["reference_time"] == 2
    assert float(q.pct_masked.isel(reference_time=1, y=0, x=0)) == 100


def test_mask_and_summary_are_lazy(ds):
    def fail(*_args, **_kwargs):
        raise AssertionError("computed")

    with dask.config.set(scheduler=fail):
        masked = dxr.mask(ds, "strict")
        summary = dxr.quality_summary(ds)
    assert isinstance(masked.displacement.data, dask.array.Array)
    assert isinstance(summary.pct_masked.data, dask.array.Array)


# --- combine_quality_masks, ported from opera-utils tests ----------------------


def _q(values):
    return xr.DataArray(values, dims=["y", "x"], coords={"y": [0, 1], "x": [0, 1]})


def test_combine_logical_or():
    result = combine_quality_masks(
        [_q([[0.3, 0.7], [0.2, 0.8]]), _q([[0.6, 0.4], [0.9, 0.1]])], [0.5, 0.5]
    )
    xr.testing.assert_equal(result, _q([[True, True], [True, True]]))


def test_combine_logical_and():
    result = combine_quality_masks(
        [_q([[0.3, 0.7], [0.6, 0.8]]), _q([[0.6, 0.4], [0.7, 0.9]])],
        [0.5, 0.5],
        np.logical_and,
    )
    xr.testing.assert_equal(result, _q([[False, False], [True, True]]))


def test_combine_length_mismatch_raises():
    with pytest.raises(ValueError, match="zip"):
        combine_quality_masks([_q([[0.1, 0.2], [0.3, 0.4]])], [0.5, 0.5])
