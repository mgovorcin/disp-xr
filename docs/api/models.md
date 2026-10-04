# Spatial models

Estimators with a `fit` / `predict` interface for spatial models: trends, the
GNSS tie and gridders will build on it (plan tasks T24, T33); today it holds
the base, `Chain` and the `Mean` baseline. verde is the design
reference, not a dependency.

```python
from disp_xr import models

coords = models.coordinates(points)  # (easting, northing) of a point table
est = models.Mean().fit(coords, points.velocity.values)
grid = est.grid(like=ds)  # on the frame's grid, with its CRS
scores = models.cross_val_score(models.Mean(), coords, points.velocity.values, spacing=5000)
```

::: disp_xr.models.base
    options:
      members: false
      show_root_heading: false

::: disp_xr.models.BaseEstimator

::: disp_xr.models.Chain

::: disp_xr.models.Mean

::: disp_xr.models.coordinates

::: disp_xr.models.cross_val_score

::: disp_xr.models.clone

::: disp_xr.models.NotFittedError
