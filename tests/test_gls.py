"""Full-frame GLS with colored noise per group (plan task T23)."""

import json

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import disp_xr as dxr
from disp_xr.timeseries.gls import NoiseShape, _gls_block, coherence_groups
from disp_xr.timeseries.noise import _fractional_diff_coeffs, _sampling_grid, fit_noise
from disp_xr.timeseries.trajectory import Trajectory

TERMS = Trajectory(periods=(1.0,))


def _times(n):
    return pd.date_range("2017-01-01", periods=n, freq="12D").values


def _flicker_frame(n=200, ny=12, nx=12, nan_fraction=0.05, seed=0):
    """Trend + flicker + white; the left half noisier, with lower coherence."""
    rng = np.random.default_rng(seed)
    psi = _fractional_diff_coeffs(0.5, n)
    white = rng.standard_normal((n, ny, nx))
    flicker = np.apply_along_axis(lambda w: np.convolve(w, psi)[:n], 0, white)
    scale = np.where(np.arange(nx) < nx // 2, 3e-3, 1e-3)[None, None, :]
    velocity = np.linspace(-0.01, 0.01, ny)[:, None] * np.ones((1, nx))
    years = np.arange(n) * 12 / 365.25
    data = years[:, None, None] * velocity + scale * flicker
    data += 1e-3 * rng.standard_normal((n, ny, nx))
    data[rng.random(data.shape) < nan_fraction] = np.nan
    coherence = np.where(np.arange(nx) < nx // 2, 0.4, 0.9) * np.ones((n, ny, nx))
    ds = xr.Dataset(
        {
            "displacement": (("time", "y", "x"), data, {"units": "meters"}),
            "temporal_coherence": (("time", "y", "x"), coherence),
        },
        coords={
            "time": _times(n),
            "y": np.arange(ny) * -30.0,
            "x": np.arange(nx) * 30.0,
        },
    )
    return ds, velocity


def _dense_gls(y, times, design, shape):
    """Reference GLS on the valid epochs with a Cholesky of their covariance."""
    ok = np.isfinite(y)
    obs_idx, _ = _sampling_grid(times, None)
    cov = shape.covariance(obs_idx)[np.ix_(ok, ok)]
    chol = np.linalg.cholesky(cov)
    a = np.linalg.solve(chol, design[ok])
    b = np.linalg.solve(chol, y[ok])
    beta, *_ = np.linalg.lstsq(a, b, rcond=None)
    r = b - a @ beta
    sigma2 = r @ r / (ok.sum() - design.shape[1])
    cov_beta = sigma2 * np.linalg.inv(a.T @ a)
    return beta, np.sqrt(np.diag(cov_beta)), np.sqrt(sigma2)


@pytest.mark.parametrize(
    "shape",
    [
        NoiseShape("PLWN", ((-1.0, 0.7), (0.0, 0.3))),
        NoiseShape("RWFNWN", ((-2.0, 0.1), (-1.0, 0.5), (0.0, 0.4))),
    ],
)
def test_schur_complement_matches_dense_gls(shape):
    rng = np.random.default_rng(4)
    n = 60
    times = _times(n)
    design = TERMS.design(times)
    data = rng.normal(0, 1e-3, (40, n)).cumsum(axis=1) * 0.1
    data[rng.random(data.shape) < 0.2] = np.nan  # a different gap pattern per pixel
    data[0] = rng.normal(0, 1e-3, n)  # one complete pixel
    obs_idx, _ = _sampling_grid(times, None)
    inverse = np.linalg.inv(shape.covariance(obs_idx))[None]
    column = TERMS.names.index("linear_trend")
    v, s, amp, count = _gls_block(
        data, np.zeros(40, int), design, inverse, np.array([0]), column, 0
    )
    for i in (0, 1, 7, 23, 39):
        beta, sig, a = _dense_gls(data[i], times, design, shape)
        np.testing.assert_allclose(v[i], beta[column], rtol=1e-7)
        np.testing.assert_allclose(s[i], sig[column], rtol=1e-7)
        np.testing.assert_allclose(amp[i], a, rtol=1e-7)
        assert count[i] == np.isfinite(data[i]).sum()


def test_missing_epoch_blocks_are_solved_in_bounded_pieces(monkeypatch):
    """Issue #94: pixels missing many epochs are solved in pieces whose
    (pixels, k, k) stacks stay under _SCHUR_BYTES; the result is unchanged."""
    from disp_xr.timeseries import gls

    rng = np.random.default_rng(5)
    n = 60
    times = _times(n)
    design = TERMS.design(times)
    data = rng.normal(0, 1e-3, (300, n)).cumsum(axis=1) * 0.1
    data[rng.random(data.shape) < 0.5] = np.nan  # about 30 of 60 epochs missing
    shape = NoiseShape("PLWN", ((-1.0, 0.7), (0.0, 0.3)))
    obs_idx, _ = _sampling_grid(times, None)
    inverse = np.linalg.inv(shape.covariance(obs_idx))[None]
    args = (data, np.zeros(300, int), design, inverse, np.array([0]), 2, 0)
    whole = _gls_block(*args)
    sizes = []
    real = gls._missing_groups

    def spy(missing, valid):
        for sel, idx in real(missing, valid):
            sizes.append((len(sel), len(sel) * idx.shape[1] ** 2 * 8))
            yield sel, idx

    monkeypatch.setattr(gls, "_SCHUR_BYTES", 8 * 30**2 * 3)  # 3 pixels at k=30
    monkeypatch.setattr(gls, "_missing_groups", spy)
    pieces = _gls_block(*args)
    # over the budget only when one pixel alone is (k > 30 here)
    assert all(nbytes <= 8 * 30**2 * 3 or n_px == 1 for n_px, nbytes in sizes)
    assert len(sizes) > len(np.unique(np.isnan(data).sum(axis=1)))
    for a, b in zip(whole, pieces):
        np.testing.assert_array_equal(a, b)


def test_labels_enter_the_graph_as_the_smallest_integer(frame):
    """Issue #94: the in-memory label map is embedded in the dask graph; as
    int64 it was 530 MB on a full frame, as int8 it is 66 MB."""
    from disp_xr.timeseries import gls

    ds, _ = frame
    shapes = {g: NoiseShape("PLWN", ((-1.0, 0.7), (0.0, 0.3))) for g in (0, 1)}
    out = gls.gls_velocity(
        ds.chunk({"y": 6, "x": 6}), terms=TERMS, n_groups=2, shapes=shapes
    )
    graph = dict(out.velocity.__dask_graph__())
    embedded = [
        v for v in graph.values() if isinstance(v, np.ndarray) and v.shape == (6, 6)
    ]
    assert embedded and all(v.dtype == np.int8 for v in embedded)
    assert out.noise_group.dtype == np.int64


@pytest.fixture(scope="module")
def frame():
    return _flicker_frame()


@pytest.fixture(scope="module")
def fit(frame):
    ds, _ = frame
    return dxr.fit_velocity(ds, "gls", terms=TERMS, n_groups=2, n_samples=8).compute()


def test_groups_and_noise_table(fit):
    table = pd.DataFrame(json.loads(fit.attrs["noise_groups"]))
    assert len(table) == 2 and set(table.model) == {"PLWN"}
    # Both halves are flicker plus white.
    assert table.kappa.between(-1.4, -0.6).all()
    assert set(np.unique(fit.noise_group)) == {0, 1}
    assert fit.noise_group.dtype == np.int64  # the type of coherence_groups
    assert fit.attrs["method"] == "gls"


def test_sigmas_are_calibrated_and_exceed_ols(frame, fit):
    ds, velocity = frame
    error = (fit.velocity - velocity) / fit.velocity_sigma
    assert 0.85 < float((abs(error) < 2).mean()) <= 1.0
    ols = dxr.fit_velocity(ds, "ols", terms=TERMS)
    assert float(fit.velocity_sigma.median()) > 3 * float(ols.velocity_sigma.median())
    # The noisier half gets the larger amplitude and sigma.
    left, right = fit.isel(x=slice(0, 6)), fit.isel(x=slice(6, None))
    assert float(left.noise_amplitude.median()) > 2 * float(
        right.noise_amplitude.median()
    )
    assert float(left.velocity_sigma.median()) > 2 * float(
        right.velocity_sigma.median()
    )


def test_one_pixel_group_reproduces_fit_noise(frame):
    # A group of one pixel takes its noise shape from that pixel, so GLS must
    # equal the per-pixel exact fit.
    ds, _ = frame
    row, col = 2, 3
    labels = xr.full_like(ds.displacement.isel(time=0), -1, dtype=int)
    labels[row, col] = 0
    out = dxr.fit_velocity(ds, "gls", terms=TERMS, groups=labels, n_samples=1)
    one = fit_noise(ds.time.values, ds.displacement[:, row, col].values, terms=TERMS)
    assert float(out.velocity[row, col]) == pytest.approx(one.velocity, rel=1e-6)
    assert float(out.velocity_sigma[row, col]) == pytest.approx(
        one.velocity_sigma, rel=1e-6
    )


def test_agrees_with_per_pixel_fit_noise_on_average(frame, fit):
    # Per pixel, the exact fit's own kappa scatters on 200 epochs; the group
    # shape is the median over sampled pixels. On average they agree.
    ds, _ = frame
    ratios = []
    for row, col in [(r, c) for r in (1, 5, 9) for c in (1, 4, 7, 10)]:
        y = ds.displacement[:, row, col].values
        one = fit_noise(ds.time.values, y, terms=TERMS, model="PLWN")
        ratios.append(float(fit.velocity_sigma[row, col]) / one.velocity_sigma)
    assert 0.75 < np.median(ratios) < 1.35


def test_label_map_and_excluded_pixels(frame):
    ds, _ = frame
    labels = xr.zeros_like(ds.displacement.isel(time=0), dtype=int)
    labels[:3] = -1
    out = dxr.fit_velocity(ds, "gls", terms=TERMS, groups=labels, n_samples=6).compute()
    assert np.isnan(out.velocity[:3]).all() and np.isfinite(out.velocity[3:]).all()
    assert (out.noise_group[:3] == -1).all()


def test_auto_noise_selects_by_bic(frame):
    ds, _ = frame
    out = dxr.fit_velocity(
        ds, "gls", terms=TERMS, n_groups=1, n_samples=4, noise="auto"
    )
    table = pd.DataFrame(json.loads(out.attrs["noise_groups"]))
    assert table.model.iloc[0] in {"FNWN", "PLWN", "RWFNWN"}


def test_coherence_groups_quantiles(frame):
    ds, _ = frame
    labels = coherence_groups(ds, 2)
    assert set(np.unique(labels)) == {0, 1}
    assert (labels[:, :6] == 0).all() and (labels[:, 6:] == 1).all()


def test_denoised_data_are_refused(frame):
    ds, _ = frame
    smoothed = ds.assign_attrs(denoised="kalman")
    with pytest.raises(ValueError, match="denoised"):
        dxr.fit_velocity(smoothed, "ols", terms=TERMS)
    with pytest.raises(ValueError, match="denoised"):
        dxr.fit_trajectory(smoothed, TERMS)
    assert dxr.fit_trajectory(smoothed, TERMS, allow_denoised=True) is not None


def test_errors(frame):
    ds, _ = frame
    with pytest.raises(NotImplementedError, match="T28"):
        dxr.fit_velocity(ds, "gls", terms=TERMS, groups="quadtree")
    with pytest.raises(ValueError, match="temporal_coherence"):
        dxr.fit_velocity(ds.displacement, "gls", terms=TERMS)
    with pytest.raises(ValueError, match="noise must be"):
        dxr.fit_velocity(ds, "gls", terms=TERMS, noise="AR1")
