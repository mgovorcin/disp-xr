# Spatial models

Estimators with a `fit` / `predict` interface for spatial models: trends, the
GNSS tie (T33) will build on them. verde is the design
reference, not a dependency.

```python
from disp_xr import models

flat = ds.disp.deramp(degree=1)  # a plane removed from every epoch, lazily

coords = models.coordinates(points)  # (easting, northing) of a point table
est = models.Chain([("trend", models.Trend(1)), ("spline", models.Spline(1e-6))])
est.fit(coords, points.velocity.values)
grid = est.grid(like=ds)  # on the frame's grid, with its CRS
scores = models.cross_val_score(est, coords, points.velocity.values, spacing=5000)
```

::: disp_xr.models.base
    options:
      members: false
      show_root_heading: false

::: disp_xr.models.BaseEstimator

::: disp_xr.models.Chain

::: disp_xr.models.Mean

## Trends and de-ramping

::: disp_xr.models.trend
    options:
      members: false
      show_root_heading: false

::: disp_xr.models.Trend

::: disp_xr.models.deramp

::: disp_xr.models.polynomial_terms

## Gridders

::: disp_xr.models.gridding
    options:
      members: false
      show_root_heading: false

::: disp_xr.models.Spline

::: disp_xr.models.Nearest

## GNSS tie

::: disp_xr.models.gnss_tie
    options:
      members: false
      show_root_heading: false

::: disp_xr.models.fit_tie

::: disp_xr.models.apply_tie

::: disp_xr.models.cross_validate_tie

::: disp_xr.models.GNSSTie

::: disp_xr.models.TieResult

## Helpers

::: disp_xr.models.coordinates

::: disp_xr.models.cross_val_score

::: disp_xr.models.clone

::: disp_xr.models.NotFittedError
