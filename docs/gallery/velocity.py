# %% [markdown]
# # Velocity from a trajectory fit
#
# `fit_trajectory` fits every pixel with a trajectory model: offset and
# velocity, seasonal sine/cosine pairs, steps and postseismic terms, by least
# squares over each pixel's valid epochs. The synthetic frame has a known
# velocity, a 5 mm annual cycle and a 10 mm step, so the fit can be checked.

# %%
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import disp_xr as dxr
from disp_xr.tutorial import make_disp_stack

folder = Path(tempfile.mkdtemp()) / "granules"
truth = make_disp_stack(
    folder, "three_ministacks", shape=(64, 64), epochs_per_ministack=30
)
ds = dxr.open_disp(folder).disp.mask("recommended").disp.stack()
print(
    f"{ds.sizes['time']} epochs over "
    f"{(ds.time[-1] - ds.time[0]).values.astype('timedelta64[D]')}"
)

# %% [markdown]
# ## Fit
#
# `Trajectory` lists the terms. The coefficient names encode the model, so a
# fit written to a store can be evaluated again with `predict_trajectory`.

# %%
step_date = f"{truth.step_date:%Y-%m-%d}"
terms = dxr.Trajectory(polynomial=1, periods=(1.0,), steps=(step_date,))
fit = dxr.fit_trajectory(ds, terms).compute()
print([str(name) for name in fit.coefficient.values])
print(list(fit.data_vars))

# %% [markdown]
# ## Velocity and its uncertainty
#
# Sigmas assume white noise (plan tasks T21 to T23 add colored noise, which
# makes them larger and more realistic).

# %%
fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.2), layout="constrained")
(fit.velocity * 1000).plot.imshow(
    ax=left, cmap="RdBu_r", cbar_kwargs={"label": "mm/yr"}
)
left.set_title("velocity", loc="left")
(fit.velocity_sigma * 1000).plot.imshow(ax=right, cbar_kwargs={"label": "mm/yr"})
right.set_title("velocity sigma (white noise)", loc="left")
for ax in (left, right):
    ax.set_aspect("equal")
plt.show()

# %% [markdown]
# ## Against the truth

# %%
error = (fit.velocity - truth.velocity) / fit.velocity_sigma
fitted = error.notnull()
print(
    f"velocity within 2 sigma: {float((abs(error) < 2).sum() / fitted.sum()):.0%}"
    " of fitted pixels"
)
print(
    f"annual amplitude: {float(fit.annual_amplitude.median()) * 1000:.2f} mm"
    f" (truth {truth.annual_amplitude * 1000:.0f} mm)"
)
step = fit.coefficients.sel(coefficient=f"step_{step_date}")
print(f"step: {float(step.median()) * 1000:.2f} mm (truth {truth.step * 1000:.0f} mm)")

# %% [markdown]
# ## Model and residuals at a pixel
#
# `predict_trajectory` evaluates the fit at any dates, lazily and in float32;
# the model is never stored unless you write it.

# %%
model = dxr.predict_trajectory(fit, ds.time.values)
row, col = 30, 50
fig, ax = plt.subplots(figsize=(8, 3.2), layout="constrained")
ax.plot(
    ds.time,
    ds.displacement[:, row, col] * 1000,
    "o",
    ms=3,
    color="#2a78d6",
    label="data",
)
ax.plot(ds.time, model[:, row, col] * 1000, color="#e34948", lw=2, label="trajectory")
ax.set_ylabel("displacement (mm)")
ax.legend(frameon=False)
ax.set_title(f"row {row}, col {col}", loc="left")
plt.show()
residual = ds.displacement[:, row, col] - model[:, row, col]
print(f"residual RMS: {float(np.sqrt((residual**2).mean())) * 1000:.2f} mm")
