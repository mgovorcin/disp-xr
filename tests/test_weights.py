"""Quality weights from the DISP quality layers (plan task T26)."""

import h5py
import numpy as np
import pytest
import xarray as xr

import disp_xr as dxr
from disp_xr.tutorial import CYCLE_PIXELS, LOW_COHERENCE, WATER_COLUMNS
from disp_xr.weights import (
    max_bandwidth_from_granule,
    network_degree,
    residual_cycle_histogram,
    residual_cycles,
)


@pytest.fixture
def ds(stack_dir):
    return dxr.open_disp(stack_dir)


@pytest.fixture
def weights(ds):
    return dxr.quality_weights(ds).compute()


def test_stretch_for_coherence_and_similarity(ds, weights):
    coherence = ds.temporal_coherence.isel(time=0).values
    expected = np.clip(
        (coherence - 0.6) / 0.4, 0, 1
    )  # floor from recommended_mask attrs
    np.testing.assert_allclose(
        weights.weight_temporal_coherence.isel(time=0), expected, rtol=1e-6
    )
    # The low-coherence patch is at or below both floors: excluded.
    assert (weights.weight.isel(time=1)[LOW_COHERENCE] == 0).all()
    assert (weights.static_weight[LOW_COHERENCE] == 0).all()


def test_residual_cycles_get_the_penalty(weights):
    # Epoch 1 carries the injected cycles (1 and 2 times 2 pi).
    w = weights.weight_timeseries_inversion_residuals.isel(time=1)
    for (row, col), cycles in CYCLE_PIXELS.items():
        assert int(weights.cycles.isel(time=1)[row, col]) == cycles
        assert float(w[row, col]) == pytest.approx(0.5**cycles, rel=1e-6)
    clean = float(w[30, 5])  # residual 0.1 rad: frac 0.016
    assert clean == pytest.approx(np.exp(-0.1 / (2 * np.pi) / 0.25), rel=1e-5)
    # The writer injects the cycles at epoch 1 of each of the two ministacks.
    n = weights.sizes["time"]
    for row, col in CYCLE_PIXELS:
        assert float(weights.jump_fraction[row, col]) == pytest.approx(2 / n)


def test_many_cycle_jumps_keep_a_positive_weight(ds):
    noisy = ds.copy()
    residuals = noisy.timeseries_inversion_residuals.copy()
    residuals[:, 30, 30] = 6 * 2 * np.pi  # six cycles on every date
    noisy["timeseries_inversion_residuals"] = residuals
    w = dxr.quality_weights(noisy, min_residual=0.05).compute()
    assert (w.weight_timeseries_inversion_residuals[:, 30, 30] == 0.05).all()
    assert (w.weight[:, 30, 30] > 0).all() and float(w.static_weight[30, 30]) > 0
    assert float(w.jump_fraction[30, 30]) == 1.0


def test_geometric_mean_and_exponents(ds):
    only = dxr.quality_weights(ds, ["temporal_coherence"]).compute()
    xr.testing.assert_allclose(only.weight, only.weight_temporal_coherence)
    both = dxr.quality_weights(
        ds,
        ["temporal_coherence", "phase_similarity"],
        exponents={"phase_similarity": 3},
    ).compute()
    expected = both.weight_temporal_coherence**0.25 * both.weight_phase_similarity**0.75
    xr.testing.assert_allclose(both.weight, expected)


def test_ambiguous_half_cycles():
    r = xr.DataArray([np.pi, 0.9 * np.pi, 0.5 * np.pi, 2.2 * np.pi], dims="time")
    parts = residual_cycles(r)
    assert parts.ambiguous.values.tolist() == [True, True, False, False]
    np.testing.assert_allclose(parts.frac, [0.5, 0.45, 0.25, 0.1])


def test_network_degree(ds):
    degree = network_degree(ds, max_bandwidth=3)
    # Two ministacks of 5 secondaries plus their reference (6 dates each).
    assert degree.values.tolist() == [4, 5, 5, 4, 3] * 2


def test_max_bandwidth_from_dolphin_config(tmp_path):
    path = tmp_path / "granule.nc"
    config = "interferogram_network:\n  max_bandwidth: 3\n  reference_idx: null\n"
    with h5py.File(path, "w") as f:
        f["metadata/dolphin_workflow_config"] = config
    assert max_bandwidth_from_granule(path) == 3
    with h5py.File(tmp_path / "bare.nc", "w"):
        pass
    assert max_bandwidth_from_granule(tmp_path / "bare.nc") is None


def test_cycle_histogram(ds):
    counts = residual_cycle_histogram(ds)
    # The injected 1 and 2 cycles, once per ministack, at exact multiples.
    assert counts.loc[1.0] == 2 and counts.loc[2.0] == 2
    assert counts.drop([0.0, 1.0, 2.0]).sum() == 0  # bin 0: the 0.1 rad background


def test_weights_feed_fit_trajectory(ds):
    stacked = dxr.chain(ds)
    w = dxr.quality_weights(stacked).weight.fillna(0)
    fit = dxr.fit_trajectory(stacked, weights=w)
    excluded = fit.velocity.values[LOW_COHERENCE]
    assert np.isnan(excluded).all()  # weight 0 on every epoch
    assert np.isfinite(fit.velocity.values[40, :-WATER_COLUMNS]).all()


def test_errors(ds):
    with pytest.raises(ValueError, match="Unknown"):
        dxr.quality_weights(ds, ["shp_counts"])
    with pytest.raises(ValueError, match="no"):
        dxr.quality_weights(ds.drop_vars("phase_similarity"))
    with pytest.raises(ValueError, match="never exclude"):
        dxr.quality_weights(ds, min_residual=0)


def test_quality_weights_stay_lazy(ds):
    # Building the weights must not compute anything: np.round on a dask-backed
    # DataArray once loaded the whole residual stack into memory.
    import dask

    def forbid(*args, **kwargs):  # noqa: ARG001 (dask scheduler signature)
        raise AssertionError("quality_weights computed eagerly")

    lazy = ds.chunk({"time": 1})
    with dask.config.set(scheduler=forbid):
        out = dxr.quality_weights(lazy)
    assert out.weight.chunks is not None
    np.testing.assert_allclose(out.weight.values, dxr.quality_weights(ds).weight.values)
