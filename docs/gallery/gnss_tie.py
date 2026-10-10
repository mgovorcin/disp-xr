# %% [markdown]
# # Tying DISP to GNSS
#
# Long-wavelength errors (orbits, ionosphere, troposphere left by the
# corrections) show up in DISP as smooth ramps that grow with time. GNSS
# stations measure the true motion at a few points, so the difference DISP
# minus GNSS at the stations samples the error. `disp_xr.models.fit_tie` fits
# a smooth field to that difference (a plane by default), and `apply_tie`
# subtracts it from DISP.
#
# The tie holds long-wavelength error only. Stations on local deformation
# (a subsiding basin, a landslide) would pull the field toward their motion,
# so `exclude=` leaves them out of the fit: local signals stay in the product
# (remove-restore).
#
# This page plants a known ramp in a synthetic frame and removes it.

# %%
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import disp_xr as dxr
from disp_xr.models import apply_tie, cross_validate_tie, fit_tie
from disp_xr.tutorial import make_disp_stack, make_gnss_source, vertical_los

folder = Path(tempfile.mkdtemp())
make_disp_stack(
    folder / "granules", "three_ministacks", shape=(96, 96), epochs_per_ministack=20
)
truth = dxr.open_disp(folder / "granules").disp.stack().disp.rereference("median")

# A plane growing with time: about 1.5 cm across the frame after two years.
years = (truth.time - truth.time[0]) / np.timedelta64(1, "D") / 365.25
plane = 3e-6 * (truth.x - truth.x.mean()) - 2e-6 * (truth.y - truth.y.mean())
error = (plane * years).transpose("time", "y", "x").astype("float32")
observed = truth.assign(
    displacement=(truth.displacement + error).assign_attrs(truth.displacement.attrs)
)

pixels = [(8, 8), (8, 88), (88, 8), (88, 88), (48, 48), (25, 65), (70, 30), (48, 12)]
source = make_gnss_source(truth, pixels, folder / "gnss", noise=0.0005)
gps = dxr.gnss.load(truth, source=source)
static = vertical_los(truth)

# %% [markdown]
# ## Fit the tie
#
# The tie is fitted on the output of `compare`. In epoch mode it fits one
# plane per epoch to the displacement residuals (weights from the GNSS
# sigmas); in velocity mode, one plane to the velocity residuals.

# %%
comparison = dxr.residuals.compare(observed, gps, static)
tie = fit_tie(comparison, mode="epoch")
corrected = apply_tie(observed, tie)


def rms_mm(da):
    """Return the RMS of ``da`` in millimeters."""
    return 1e3 * float(np.sqrt(np.nanmean(da**2)))


print(f"error before: {rms_mm(observed.displacement - truth.displacement):.1f} mm")
print(f"error after:  {rms_mm(corrected.displacement - truth.displacement):.1f} mm")
print(corrected.displacement.attrs["gnss_tie"])

# %%
last = {"time": -1}
fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), layout="constrained")
panels = [
    (1e3 * error.isel(last), "planted error"),
    (1e3 * tie.correction(observed).isel(last), "tie correction"),
    (1e3 * (corrected.displacement - truth.displacement).isel(last), "error after"),
]
for ax, (da, title) in zip(axes, panels):
    da.plot.imshow(ax=ax, cmap="RdBu_r", vmin=-8, vmax=8, cbar_kwargs={"label": "mm"})
    ax.set_title(f"{title}, last epoch", loc="left")
    ax.set_aspect("equal")
x, y = static.x.values, static.y.values
for ax in axes:
    ax.plot([x[c] for _, c in pixels], [y[r] for r, _ in pixels], "k^", ms=6)
plt.show()

# %% [markdown]
# ## Does it generalize? Leave one station out
#
# Each station in turn is left out, the tie is fitted on the others and the
# left-out station's residual is scored before and after. Smaller RMSE after
# means the field predicts the error where it has no station. With more
# flexible fields (`damping=` adds a spline) the scores improve partly by
# construction, so compare fields of similar complexity.

# %%
scores = cross_validate_tie(comparison, mode="epoch")
print((1e3 * scores[["rmse_before", "rmse_after"]]).round(2))
before, after = (1e3 * scores.attrs[k] for k in ("rmse_before", "rmse_after"))
print(f"network RMSE, stations left out: {before:.1f} mm before, {after:.1f} mm after")
# One station on local deformation can dominate the RMSE; the median
# describes the typical station.
before, after = (1e3 * scores.attrs[k] for k in ("median_before", "median_after"))
print(f"median per station: {before:.1f} mm before, {after:.1f} mm after")

# %% [markdown]
# On a real frame, use `dxr.gnss.load(ds)` and `dxr.open_static(ds)`, and
# pass `exclude=` with the stations (or a lon/lat polygon) on local
# deformation.
