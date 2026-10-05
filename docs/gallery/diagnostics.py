# %% [markdown]
# # Checking a velocity fit
#
# `fit_trajectory` and `fit_velocity(method="ols")` give white-noise sigmas.
# Those hold only when the residuals are uncorrelated and Gaussian. DISP
# residuals often are not: troposphere and unwrapping leave correlated
# noise, and an unmodelled step leaves a long run of residuals of one sign.
# `disp_xr.diagnostics` checks this for every pixel:
#
# - `model_diagnostics`: residual mean, spread, skewness, kurtosis,
#   autocorrelation at chosen lags and the Anderson-Darling normality
#   statistic, in one pass over the data;
# - `significance`: z-scores, p-values and confidence intervals of the
#   coefficients.
#
# The frame below has three strips with the same 10 mm/yr velocity:
#
# - left: white noise, where the OLS sigmas are right;
# - middle: a random walk added, a correlated noise;
# - right: a 15 mm step that the model does not include.

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

import disp_xr as dxr
from disp_xr.diagnostics import AD_CRITICAL_5, model_diagnostics, significance

rng = np.random.default_rng(3)
nt, ny, nx = 200, 12, 36
times = pd.Timestamp("2018-01-01") + pd.to_timedelta(np.arange(nt) * 12, "D")
years = np.arange(nt) * 12 / 365.25
signal = 0.010 * years + 0.003 * np.sin(2 * np.pi * years)
data = signal[:, None, None] + 0.003 * rng.standard_normal((nt, ny, nx))
middle, right = slice(12, 24), slice(24, 36)
data[:, :, middle] += np.cumsum(0.0015 * rng.standard_normal((nt, ny, 12)), axis=0)
data[nt // 2 :, :, right] += 0.015
ds = xr.Dataset(
    {"displacement": (("time", "y", "x"), data, {"units": "meters"})},
    coords={"time": times, "y": np.arange(ny) * 30.0, "x": np.arange(nx) * 30.0},
)

# %% [markdown]
# ## Fit and diagnose
#
# The trajectory model has an offset, the velocity and an annual cycle.
# `model_diagnostics` evaluates the model from the coefficients and reads
# the data once.

# %%
fit = dxr.fit_trajectory(ds).compute()
diag = model_diagnostics(ds, fit, lags=(1, 2)).compute()
print(list(diag.data_vars))

fig, axes = plt.subplots(1, 3, figsize=(13, 3), layout="constrained")
(1e3 * fit.velocity).plot.imshow(ax=axes[0], cbar_kwargs={"label": "mm/yr"})
axes[0].set_title("velocity", loc="left")
diag.autocorrelation_lag1.plot.imshow(ax=axes[1], vmin=-0.2, vmax=1)
axes[1].set_title("residual autocorrelation, lag 1", loc="left")
diag.anderson_darling.plot.imshow(ax=axes[2], vmin=0, vmax=5)
axes[2].set_title(f"Anderson-Darling (5% critical {AD_CRITICAL_5})", loc="left")
for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])
plt.show()

# %% [markdown]
# The white-noise strip has residual autocorrelation near zero and passes
# the normality test. The random walk and the step both leave strongly
# autocorrelated residuals: above about 0.3 the white-noise sigmas are
# optimistic. The step also shows in the velocity map itself, as a bias.
#
# ## How optimistic?
#
# A colored-noise method estimates the noise from the residuals' spectrum.
# Compare its velocity sigma with the OLS one in each strip.

# %%
ols = dxr.fit_velocity(ds, method="ols").compute()
whittle = dxr.fit_velocity(ds, method="whittle").compute()
ratio = whittle.velocity_sigma / ols.velocity_sigma
for name, cols in (("white", slice(0, 12)), ("random walk", middle), ("step", right)):
    strip = {"x": ds.x[cols]}
    print(
        f"{name:>12}: lag-1 autocorrelation "
        f"{float(diag.autocorrelation_lag1.sel(strip).median()):.2f}, "
        f"OLS sigma {1e3 * float(ols.velocity_sigma.sel(strip).median()):.2f} mm/yr, "
        f"colored-noise sigma {float(ratio.sel(strip).median()):.1f}x larger"
    )

# %% [markdown]
# ## Significance
#
# `significance` gives each coefficient's z-score, p-value and confidence
# interval from the fit's sigmas. With white-noise sigmas on correlated
# residuals, almost everything looks significant. Use the colored-noise
# sigmas (or MIDAS) before reading the p-values.

# %%
sig = significance(fit)
velocity = sig.sel(coefficient="linear_trend")  # the velocity term
print(f"pixels with a significant velocity: {float(velocity.significant.mean()):.0%}")
print(
    "95% interval at one random-walk pixel: "
    f"{1e3 * float(velocity.ci_low.isel(y=0, x=18)):.1f} to "
    f"{1e3 * float(velocity.ci_high.isel(y=0, x=18)):.1f} mm/yr"
)

# %% [markdown]
# The true velocity is 10 mm/yr everywhere, outside that interval: the
# white-noise interval is too narrow by the factor printed above.
