"""Shared checks every disp-xr estimator passes (plan task T19.6).

Later estimators (trends, the GNSS tie, gridders) call :func:`check_estimator`
with a factory and data their model can fit.
"""

import numpy as np
import pytest
import xarray as xr

from disp_xr.models import NotFittedError, clone, cross_val_score


def synthetic(n=400, seed=0):
    """Scattered points over a 30 km square with a plane plus noise."""
    rng = np.random.default_rng(seed)
    easting = 500_000.0 + rng.uniform(0, 30_000, n)
    northing = 4_100_000.0 + rng.uniform(0, 30_000, n)
    data = 1e-6 * (easting - 500_000) - 2e-6 * (northing - 4_100_000) + 0.01
    data = data + 0.001 * rng.standard_normal(n)
    return (easting, northing), data


def check_estimator(make, coords=None, data=None):
    """Run the shared checks on ``make()`` estimators."""
    if coords is None:
        coords, data = synthetic()
    est = make()
    # Parameters round-trip and show in the repr.
    params = est.get_params()
    assert type(est).__name__ in repr(est)
    assert make().set_params(**params).get_params().keys() == params.keys()
    with pytest.raises(ValueError, match="no parameter"):
        est.set_params(not_a_parameter=1)
    with pytest.raises(NotFittedError):
        est.predict(coords)
    assert est.fit(coords, data) is est
    assert [k for k in vars(est) if k.endswith("_")], "no fitted attribute"
    # Predictions keep the shape of the coordinates (1D and 2D).
    assert est.predict(coords).shape == data.shape
    grid2d = tuple(c[:100].reshape(10, 10) for c in coords)
    assert est.predict(grid2d).shape == (10, 10)
    # NaN data and zero weights drop points; negative weights are refused.
    gappy = data.copy()
    gappy[::7] = np.nan
    weights = np.ones_like(data)
    weights[::7] = 0.0
    a = make().fit(coords, gappy).predict(coords)
    b = make().fit(coords, data, weights).predict(coords)
    np.testing.assert_allclose(a, b, rtol=1e-8, atol=1e-12)
    with pytest.raises(ValueError, match="non-negative"):
        make().fit(coords, data, -weights)
    with pytest.raises(ValueError, match="differ"):
        make().fit(coords, data[:-1])
    # Gridding: north up, the requested spacing, the CRS.
    grid = est.grid(spacing=5_000.0, crs="EPSG:32610")
    assert grid.dims == ("y", "x")
    assert grid.y[0] > grid.y[-1]
    assert float(grid.x[1] - grid.x[0]) == 5_000.0
    assert grid.rio.crs.to_epsg() == 32610
    like = xr.Dataset(coords={"y": grid.y.values[::2], "x": grid.x.values[::2]})
    like = like.rio.write_crs("EPSG:32610")
    on_like = est.grid(like=like)
    np.testing.assert_allclose(on_like, grid.values[::2, ::2])
    # Copies are unfitted with the same parameters; cross-validation runs.
    copy = clone(est)
    assert copy.get_params().keys() == params.keys()
    with pytest.raises(NotFittedError):
        copy.predict(coords)
    scores = cross_val_score(make(), coords, data, spacing=5_000.0, n_splits=4)
    assert scores.shape == (4,) and np.isfinite(scores).all()
