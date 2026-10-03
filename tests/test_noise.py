"""Colored-noise estimation for one series (plan task T21)."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from disp_xr.timeseries.noise import (
    NoiseResult,
    _fractional_diff_coeffs,
    _powerlaw_covariance,
    _softmax_weights,
    fit_noise,
)
from disp_xr.timeseries.trajectory import Trajectory

REFERENCE = json.loads(
    (Path(__file__).parent / "data" / "noise_reference.json").read_text()
)
TERMS = Trajectory(periods=(1.0,))


def _series(name):
    (s,) = [s for s in REFERENCE["series"] if s["name"] == name]
    return s, pd.to_datetime(s["dates"]).values, np.array(s["values"])


# --- covariance builders ------------------------------------------------------


def test_fractional_diff_coeffs():
    # (1 - B)^-1 is the cumulative sum: every coefficient is 1.
    np.testing.assert_allclose(_fractional_diff_coeffs(1.0, 5), np.ones(5))
    # d = 0.5 (flicker): 1, 1/2, 3/8, 5/16.
    np.testing.assert_allclose(_fractional_diff_coeffs(0.5, 4), [1, 0.5, 0.375, 0.3125])
    np.testing.assert_allclose(_fractional_diff_coeffs(0.0, 3), [1, 0, 0])


def test_powerlaw_covariance_known_cases():
    np.testing.assert_allclose(_powerlaw_covariance(0.0, 4), np.eye(4))  # white
    i, j = np.indices((4, 4))
    # Random walk: C[i, j] = min(i, j) + 1.
    np.testing.assert_allclose(_powerlaw_covariance(-2.0, 4), np.minimum(i, j) + 1)
    flicker = _powerlaw_covariance(-1.0, 50)
    np.testing.assert_allclose(flicker, flicker.T)
    assert np.linalg.eigvalsh(flicker).min() > 0


def test_softmax_weights():
    w = _softmax_weights(np.array([0.0, np.log(3.0)]))
    np.testing.assert_allclose(w, [1 / 5, 3 / 5, 1 / 5])


# --- against geepers ------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "model"),
    [
        ("insar_12day", "WN"),
        ("insar_12day", "FNWN"),
        ("insar_12day", "PLWN"),
        ("insar_12day", "RWFNWN"),
        ("insar_12day_gaps", "PLWN"),
        ("insar_12day_gaps", "RWFNWN"),
        ("gnss_flicker_gaps", "FNWN"),
        ("gnss_randomwalk", "PLWN"),
    ],
)
def test_reproduces_geepers(name, model):
    s, times, values = _series(name)
    ref = s["fits"][model]
    res = fit_noise(
        times, values, terms=TERMS, model=model, sampling_days=s["sampling_days"]
    )
    # Plan T21: velocity within 1%, sigma within 5%. disp-xr measures time in
    # calendar decimal years, geepers in grid steps of 365.25 days.
    assert res.velocity == pytest.approx(ref["velocity"], rel=0.01)
    assert res.velocity_sigma == pytest.approx(ref["velocity_sigma"], rel=0.05)
    if np.isfinite(ref["kappa"]):
        assert res.kappa == pytest.approx(ref["kappa"], abs=0.02)
    assert res.log_likelihood == pytest.approx(ref["log_likelihood"], abs=1.0)
    for key in ("sigma_white", "sigma_flicker", "sigma_randomwalk", "sigma_powerlaw"):
        if np.isfinite(ref[key]) and ref[key] > 1e-5:
            assert getattr(res, key) == pytest.approx(ref[key], rel=0.05), key


def test_colored_noise_sigma_exceeds_white():
    s, times, values = _series("insar_12day")
    white = fit_noise(times, values, terms=TERMS, model="WN")
    colored = fit_noise(times, values, terms=TERMS, model="PLWN")
    assert colored.velocity_sigma > 2 * white.velocity_sigma


# --- synthetic power-law noise -----------------------------------------------------


@pytest.mark.parametrize("kappa", [-0.5, -1.0, -1.5])
def test_recovers_kappa(kappa):
    rng = np.random.default_rng(10)
    n = 400
    psi = _fractional_diff_coeffs(-kappa / 2, n)
    noise = 2e-3 * np.convolve(rng.standard_normal(n), psi)[:n]
    times = pd.date_range("2019-01-01", periods=n, freq="12D").values
    years = np.arange(n) * 12 / 365.25
    y = 0.004 * years + noise + 1e-4 * rng.standard_normal(n)
    res = fit_noise(times, y, terms=Trajectory(periods=()), model="PLWN")
    assert res.kappa == pytest.approx(kappa, abs=0.25)


@pytest.mark.parametrize("name", ["insar_12day", "insar_12day_gaps", "gnss_flicker"])
def test_whittle_reproduces_geepers(name):
    s, times, values = _series(name)
    ref = s["fits"]["PLWN_whittle"]
    res = fit_noise(
        times,
        values,
        terms=TERMS,
        model="PLWN",
        method="whittle",
        sampling_days=s["sampling_days"],
    )
    assert res.velocity == pytest.approx(ref["velocity"], rel=0.01)
    assert res.velocity_sigma == pytest.approx(ref["velocity_sigma"], rel=0.05)
    assert res.kappa == pytest.approx(ref["kappa"], abs=0.02)


def test_whittle_departs_from_exact_on_short_series():
    # Whittle is asymptotic: on 200 epochs it overestimates |kappa| here
    # (-1.59 against -1.04), so the velocity sigma is too large. Recorded for
    # plan task T22, where Whittle is the fast full-frame estimator.
    s, times, values = _series("insar_12day")
    fast = fit_noise(times, values, terms=TERMS, model="PLWN", method="whittle")
    exact = s["fits"]["PLWN"]
    assert fast.kappa < exact["kappa"] - 0.3
    assert fast.velocity_sigma > 1.5 * exact["velocity_sigma"]


# --- results and errors ------------------------------------------------------------


def test_result_fields_and_nan_epochs():
    s, times, values = _series("insar_12day")
    values = values.copy()
    values[[5, 40]] = np.nan
    res = fit_noise(times, values, terms=TERMS, model="FNWN")
    assert isinstance(res, NoiseResult)
    assert res.names == TERMS.names
    assert set(res.parameters) == set(TERMS.names)
    assert np.isnan(res.residuals[[5, 40]]).all() and np.isnan(res.fitted[5])
    np.testing.assert_allclose(np.nanstd(res.residuals), np.nanstd(values - res.fitted))
    assert res.aic < res.bic  # n > e^2, so BIC penalizes more
    assert res.sampling_days == 12


def test_design_matrix_input():
    s, times, values = _series("insar_12day")
    design = TERMS.design(times)
    res = fit_noise(times, values, design, names=TERMS.names, model="WN")
    ref = fit_noise(times, values, terms=TERMS, model="WN")
    np.testing.assert_allclose(res.coefficients, ref.coefficients)


def test_errors():
    times = pd.date_range("2020-01-01", periods=30, freq="12D").values
    y = np.arange(30.0)
    with pytest.raises(ValueError, match="model must be"):
        fit_noise(times, y, model="AR1")
    with pytest.raises(ValueError, match="at least 10"):
        fit_noise(times[:8], y[:8])
    with pytest.raises(ValueError, match="rank deficient"):
        fit_noise(times, y, terms=Trajectory(steps=("2030-01-01",)))
    with pytest.raises(ValueError, match="sampling interval"):
        fit_noise(times, y, sampling_days=30)
    with pytest.raises(ValueError, match="whittle"):
        fit_noise(times, y, model="FNWN", method="whittle")
