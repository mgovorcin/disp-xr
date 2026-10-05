# %% [markdown]
# # Denoising time series and regular time grids
#
# `dxr.denoise` smooths displacement series in time, for display and for
# resampling onto a regular grid. It has two methods with the same interface:
#
# - `"kalman"`: a local-linear-trend Kalman filter with a smoother. It follows
#   changes in rate over about `tau_days`, and a chi-square gate rejects
#   outliers such as unwrapping jumps on single dates;
# - `"lowess"`: robust locally weighted regression over `bandwidth_days`.
#
# Neither improves a velocity estimate. Fit velocities on the raw series:
# denoised data carry `attrs["denoised"]`, and the fits refuse them.
#
# The series below has a known signal and irregular sampling like real
# DISP-S1 frames: 12-day revisits, some 24-day ones, and a 200-day hole.
# It also has a one-date spike (an unwrapping error), a step on a known
# date, and a short transient.

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

import disp_xr as dxr
from disp_xr.timeseries import gap_mask, regular_time_grid

rng = np.random.default_rng(1)
days = np.concatenate(
    [np.arange(0, 408, 12), np.arange(420, 700, 24), np.arange(900, 1500, 12)]
)
times = pd.Timestamp("2018-01-01") + pd.to_timedelta(days, "D")
years = days / 365.25
truth = (
    -0.012 * years  # subsidence, 12 mm/yr
    + 0.004 * np.sin(2 * np.pi * years)  # 4 mm annual cycle
    + 0.010 * (days >= 1100)  # 10 mm step
    + 0.008 * np.exp(-(((days - 1300) / 30.0) ** 2))  # 8 mm transient
)
noisy = truth + 0.002 * rng.standard_normal((4, days.size))
noisy[:, 50] += 0.028  # unwrapping error: half a wavelength on one date
series = xr.DataArray(
    noisy.T[:, :, None],
    dims=("time", "y", "x"),
    coords={"time": times, "y": np.arange(4.0), "x": [0.0]},
    attrs={"units": "meters"},
)
intervals = sorted({int(d) for d in np.diff(days)})
print(f"{days.size} acquisitions, intervals {intervals} days")

# %% [markdown]
# ## On the acquisitions
#
# The step is declared with `breaks=`. Automatic break detection is out of
# scope: steps come from catalogs, such as earthquakes or equipment
# changes.

# %%
step = times[days >= 1100][0]
kalman = dxr.denoise(series, "kalman", tau_days=20, breaks=[step]).compute()
lowess = dxr.denoise(series, "lowess", bandwidth_days=60, breaks=[step]).compute()

fig, ax = plt.subplots(figsize=(11, 4), layout="constrained")
ax.plot(times, 1e3 * series.isel(y=0, x=0), ".", color="0.6", label="data")
ax.plot(times, 1e3 * truth, "k-", lw=1, label="truth")
ax.plot(times, 1e3 * kalman.isel(y=0, x=0), label="kalman, tau 20 d")
ax.plot(times, 1e3 * lowess.isel(y=0, x=0), label="lowess, 60 d")
ax.set(ylabel="displacement (mm)")
ax.legend(ncols=4, loc="lower left")
plt.show()


def rms_mm(a):
    """Return the RMS of ``a`` in millimeters."""
    return 1e3 * float(np.sqrt(np.nanmean(np.square(a))))


spike = 50
transient = np.abs(days - 1300) <= 30
base = truth - 0.008 * np.exp(-(((days - 1300) / 30.0) ** 2))
for name, out in (("kalman", kalman), ("lowess", lowess)):
    pixel = out.isel(y=0, x=0).values
    kept = (pixel - base)[transient].mean() / (truth - base)[transient].mean()
    print(
        f"{name}: error {rms_mm(out.values[:, :, 0].T - truth):.1f} mm RMS, "
        f"spike left {1e3 * (pixel[spike] - truth[spike]):.1f} mm of 28, "
        f"transient kept {kept:.0%}"
    )

# %% [markdown]
# Both remove the spike: the Kalman gate rejects the date, and LOWESS
# down-weights it through its robust passes. Both keep the step, because it
# is declared. They differ on the transient, a signal of about two months:
# the Kalman filter keeps more of it than the 60-day LOWESS window. A
# shorter `tau_days` or `bandwidth_days` keeps faster signals at the cost of
# more noise.
#
# ## On a regular grid
#
# `grid="auto"` resamples onto a regular grid. Its step comes from the
# sampling: 15 days for 12-day revisits, 30 days when 24-day intervals are
# common. Grid dates inside holes longer than `max_gap_days` are refused
# (NaN), not interpolated.

# %%
grid = regular_time_grid(times)
print(f"grid every {(grid[1] - grid[0]).days} days, {grid.size} dates")
refused = gap_mask(times, grid)
print(
    f"refused: {refused.sum()} dates, from {grid[refused][0]:%Y-%m-%d} "
    f"to {grid[refused][-1]:%Y-%m-%d}"
)
on_grid = dxr.denoise(series, "kalman", grid="auto", breaks=[step]).compute()

fig, ax = plt.subplots(figsize=(11, 4), layout="constrained")
ax.plot(times, 1e3 * series.isel(y=0, x=0), ".", color="0.6", label="data")
ax.plot(on_grid.time, 1e3 * on_grid.isel(y=0, x=0), "o-", ms=3, label="kalman on grid")
for start, end in zip(grid[refused][:1], grid[refused][-1:]):
    ax.axvspan(start, end, color="C3", alpha=0.1, label="refused gap")
ax.set(ylabel="displacement (mm)")
ax.legend(loc="lower left")
plt.show()

# %% [markdown]
# ## Denoised data are for display
#
# The output records the method, and the fits refuse it unless asked
# explicitly. Smoothing correlates the residuals, so velocity sigmas from a
# denoised series would be far too small.

# %%
print(on_grid.attrs["denoised"])
try:
    dxr.fit_velocity(on_grid, method="ols")
except ValueError as err:
    print("refused:", str(err).split(".")[0])
