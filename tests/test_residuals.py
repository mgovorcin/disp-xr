"""Variograms and the correlated-error cell sigma (plan task T27.3b)."""

import numpy as np
import pytest
import xarray as xr

import disp_xr as dxr
from disp_xr.downsample import downsample
from disp_xr.residuals import (
    VariogramModel,
    empirical_variogram,
    fit_variogram,
    mean_block_correlation,
    n_eff_correlated,
)
from disp_xr.tutorial import correlated_noise

TRUE = VariogramModel(1.0, (("exponential", 4.0, 900.0),))


def _field(z, spacing=30.0):
    ny, nx = z.shape
    return xr.DataArray(
        z,
        dims=("y", "x"),
        coords={"y": -spacing * np.arange(ny), "x": spacing * np.arange(nx)},
    )


def _exact_variance(model, weights, spacing):
    """Brute force: sill * sum_ij w_i w_j rho_ij / (sum w)^2 over a block."""
    iy, ix = np.nonzero(weights > 0)
    w = weights[iy, ix]
    d = np.hypot((iy[:, None] - iy[None]) * spacing, (ix[:, None] - ix[None]) * spacing)
    return model.sill * (w @ model.correlation(d) @ w) / w.sum() ** 2


def test_model_shapes():
    m = VariogramModel(0.5, (("spherical", 1.0, 100.0), ("gaussian", 2.0, 1000.0)))
    assert m.sill == 3.5
    assert m.gamma(0.0) == 0.0 and m.correlation(0.0) == 1.0
    assert m.gamma(1e-9) == pytest.approx(0.5)  # the nugget jump
    assert m.gamma(100.0) - m.gamma(99.0) > 0
    assert m.gamma(1e6) == pytest.approx(3.5)
    exp = VariogramModel(0.0, (("exponential", 1.0, 300.0),))
    assert exp.gamma(300.0) == pytest.approx(1 - np.exp(-3))  # practical range
    assert "spherical" in m.to_json()
    with pytest.raises(ValueError, match="unknown"):
        VariogramModel(0.0, (("cubic", 1.0, 1.0),)).gamma(1.0)


def test_correlated_noise_variance():
    z = np.stack(
        [
            correlated_noise((64, 64), correlation_range=300, sill=4, nugget=1, seed=s)
            for s in range(20)
        ]
    )
    assert np.var(z) == pytest.approx(5.0, rel=0.1)


@pytest.mark.parametrize("seed", [0, 1])
def test_variogram_recovers_known_model(seed):
    z = correlated_noise((256, 256), correlation_range=900, sill=4, nugget=1, seed=seed)
    emp = empirical_variogram(_field(z))
    assert emp["count"].min() > 1000  # log-uniform distances fill short lags
    assert emp["distance"].min() < 45
    fit = fit_variogram(emp, ("exponential",))
    assert fit.sill == pytest.approx(5.0, rel=0.25)
    assert fit.nugget == pytest.approx(1.0, rel=0.25)
    assert fit.components[0][2] == pytest.approx(900, rel=0.3)
    two = fit_variogram(emp)  # default: nugget + two spherical models
    ranges = [c[2] for c in two.components]
    assert ranges == sorted(ranges)
    assert two.sill == pytest.approx(5.0, rel=0.25)


def test_variogram_skips_nan_and_checks_input():
    z = correlated_noise((64, 64), correlation_range=300, seed=0)
    z[:, :20] = np.nan
    emp = empirical_variogram(_field(z), n_pairs=20_000, n_bins=10)
    assert np.isfinite(emp["gamma"]).all()
    with pytest.raises(ValueError, match=r"\(y, x\)"):
        empirical_variogram(_field(z).expand_dims(time=2))
    with pytest.raises(ValueError, match="fewer than two"):
        empirical_variogram(_field(np.full((8, 8), np.nan)))


def test_mean_block_correlation_matches_pairs():
    f = 5
    full = np.ones((f, f))
    n = f * f
    exact = _exact_variance(TRUE, full, 30.0)
    rho = mean_block_correlation(TRUE, f, 30.0)
    assert TRUE.sill * (n + n * (n - 1) * rho) / n**2 == pytest.approx(exact)
    assert mean_block_correlation(TRUE, 1, 30.0) == 0.0
    white = VariogramModel(1.0, ())
    assert mean_block_correlation(white, 9, 30.0) == pytest.approx(0.0)


def test_n_eff_limits():
    w = np.random.default_rng(0).random(50)
    s1, s2 = w.sum(), (w**2).sum()
    assert n_eff_correlated(s1, s2, 0.0) == pytest.approx(s1**2 / s2)
    assert n_eff_correlated(s1, s2, 1.0) == pytest.approx(1.0)


def test_uniform_gap_approximation():
    # Random gaps and weights: the block-mean approximation stays close.
    rng = np.random.default_rng(1)
    f = 15
    rho = mean_block_correlation(TRUE, f, 30.0)
    for _ in range(5):
        w = rng.random((f, f)) * (rng.random((f, f)) > 0.3)
        exact = _exact_variance(TRUE, w, 30.0)
        approx = TRUE.sill / n_eff_correlated(w.sum(), (w**2).sum(), rho)
        assert approx == pytest.approx(exact, rel=0.05)


def test_block_mean_variance_monte_carlo():
    f, reps = 9, 150
    means = np.concatenate(
        [
            correlated_noise(
                (4 * f, 4 * f), correlation_range=900, sill=4, nugget=1, seed=s
            )
            .reshape(4, f, 4, f)
            .mean(axis=(1, 3))
            .ravel()
            for s in range(reps)
        ]
    )
    rho = mean_block_correlation(TRUE, f, 30.0)
    predicted = TRUE.sill / n_eff_correlated(f * f, f * f, rho)
    assert np.var(means) == pytest.approx(predicted, rel=0.15)
    assert predicted > 20 * TRUE.sill / f**2  # independent-pixel sigma far too small


def test_downsample_sigma_corr():
    z = correlated_noise((27, 27), correlation_range=900, sill=4, nugget=1, seed=0)
    ds = _field(z).to_dataset(name="displacement").expand_dims(time=2)
    out = downsample(ds, factor=9, weights=None, variogram={"displacement": TRUE})
    out = out.compute()
    exact = np.sqrt(_exact_variance(TRUE, np.ones((9, 9)), 30.0))
    np.testing.assert_allclose(out.displacement_sigma_corr, exact, rtol=1e-10)
    np.testing.assert_allclose(out.n_eff_corr, TRUE.sill / exact**2, rtol=1e-10)
    assert (out.displacement_sigma_corr > out.displacement_sigma).all()
    attrs = out.displacement_sigma_corr.attrs
    assert "exponential" in attrs["variogram"] and attrs["mean_block_correlation"] > 0
    with pytest.raises(ValueError, match="not aggregated"):
        downsample(ds, factor=9, weights=None, variogram={"velocity": TRUE})


def test_top_level_names():
    assert dxr.fit_variogram is fit_variogram
    assert dxr.empirical_variogram is empirical_variogram
    assert dxr.VariogramModel is VariogramModel
