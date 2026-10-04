"""Estimator base, Chain and cross-validation (plan task T19)."""

import numpy as np
import pytest
import xarray as xr

from disp_xr.models import BaseEstimator, Chain, Mean, coordinates, cross_val_score

from .common import check_estimator, synthetic


class _Plane(BaseEstimator):
    """Weighted least-squares plane (test only; T24 brings the real trend)."""

    def __init__(self, damping: float = 0.0):
        self.damping = damping

    def _fit(self, easting, northing, data, weights):
        self.center_ = (easting.mean(), northing.mean())
        a = np.column_stack(
            [
                np.ones_like(easting),
                easting - self.center_[0],
                northing - self.center_[1],
            ]
        )
        sw = np.sqrt(weights)
        self.coef_ = np.linalg.lstsq(a * sw[:, None], data * sw, rcond=None)[0]

    def _predict(self, easting, northing):
        return (
            self.coef_[0]
            + self.coef_[1] * (easting - self.center_[0])
            + self.coef_[2] * (northing - self.center_[1])
        )


def test_mean_passes_shared_checks():
    check_estimator(Mean)


def test_plane_and_chain_pass_shared_checks():
    check_estimator(_Plane)
    check_estimator(lambda: Chain([("mean", Mean()), ("plane", _Plane())]))


def test_mean_value_and_score():
    coords, data = synthetic()
    weights = np.random.default_rng(1).random(data.size)
    est = Mean().fit(coords, data, weights)
    assert est.mean_ == pytest.approx(np.sum(weights * data) / weights.sum())
    assert est.score(coords, data, weights) == pytest.approx(0.0, abs=1e-12)


def test_chain_fits_residuals():
    coords, data = synthetic()
    chain = Chain([("mean", Mean()), ("plane", _Plane())]).fit(coords, data)
    plane = _Plane().fit(coords, data)
    # The plane absorbs what the mean leaves; the sum equals one plane fit.
    np.testing.assert_allclose(chain.predict(coords), plane.predict(coords), atol=1e-12)
    assert chain.named_steps_["plane"].coef_[0] == pytest.approx(0.0, abs=1e-10)
    assert chain.score(coords, data) > 0.9


def test_grid_reproduces_a_plane_exactly():
    coords, _ = synthetic()
    data = 2e-6 * coords[0] + 0.5
    grid = _Plane().fit(coords, data).grid(spacing=1000.0)
    expected = 2e-6 * grid.x.values[None, :] + 0.5
    np.testing.assert_allclose(grid.values, np.broadcast_to(expected, grid.shape))


def test_coordinates_helpers():
    raster = xr.Dataset(coords={"y": [3.0, 2.0], "x": [10.0, 20.0, 30.0]})
    e, n = coordinates(raster)
    assert e.shape == n.shape == (2, 3) and e[0, 2] == 30.0 and n[1, 0] == 2.0
    points = xr.Dataset(coords={"x": ("point", [1.0, 2.0]), "y": ("point", [5.0, 6.0])})
    e, n = coordinates(points)
    assert e.tolist() == [1.0, 2.0] and n.tolist() == [5.0, 6.0]
    e, n = coordinates(points.to_dataframe().reset_index())
    assert e.tolist() == [1.0, 2.0]
    with pytest.raises(ValueError):
        coordinates(xr.Dataset(coords={"lat": [1.0]}))


def test_cross_validation_separates_blocks():
    coords, data = synthetic(n=600)
    good = cross_val_score(_Plane(), coords, data, spacing=3_000.0)
    bad = cross_val_score(Mean(), coords, data, spacing=3_000.0)
    assert good.mean() > 0.8 > bad.mean()
    with pytest.raises(ValueError, match="blocks"):
        cross_val_score(Mean(), coords, data, spacing=1e6)
