# %% [markdown]
# # Velocity with colored-noise uncertainty
#
# Least-squares sigmas assume every epoch is independent. InSAR noise is
# correlated in time, so those sigmas are too small. `fit_velocity(method=
# "gls")` estimates the noise shape per group of pixels by exact maximum
# likelihood and fits every pixel by generalized least squares, so each pixel
# gets a colored-noise sigma.
#
# The frame here is synthetic: a velocity gradient, flicker noise plus white
# noise, three times noisier in the left half (where coherence is lower).

# %%
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

import disp_xr as dxr
from disp_xr.timeseries.noise import _fractional_diff_coeffs

n, ny, nx = 200, 24, 24
rng = np.random.default_rng(0)
psi = _fractional_diff_coeffs(0.5, n)  # flicker: (1 - B)^-0.5
flicker = np.apply_along_axis(
    lambda w: np.convolve(w, psi)[:n], 0, rng.standard_normal((n, ny, nx))
)
noisy = np.arange(nx) < nx // 2
scale = np.where(noisy, 3e-3, 1e-3)
velocity = np.linspace(-0.01, 0.01, ny)[:, None] * np.ones((1, nx))
years = np.arange(n) * 12 / 365.25
data = years[:, None, None] * velocity + scale * flicker
data += 1e-3 * rng.standard_normal(data.shape)
ds = xr.Dataset(
    {
        "displacement": (("time", "y", "x"), data, {"units": "meters"}),
        "temporal_coherence": (
            ("time", "y", "x"),
            np.broadcast_to(np.where(noisy, 0.4, 0.9), data.shape),
        ),
    },
    coords={
        "time": pd.date_range("2017-01-01", periods=n, freq="12D").values,
        "y": -30.0 * np.arange(ny),
        "x": 30.0 * np.arange(nx),
    },
)

# %% [markdown]
# ## Fit
#
# Two coherence groups; 16 pixels of each are fitted exactly to estimate the
# group's noise shape (spectral index and white fraction). Every pixel's
# amplitude is its own.

# %%
terms = dxr.Trajectory(periods=(1.0,))
gls = dxr.fit_velocity(ds, "gls", terms=terms, n_groups=2).compute()
ols = dxr.fit_velocity(ds, "ols", terms=terms).compute()
pd.DataFrame(json.loads(gls.attrs["noise_groups"]))[
    ["group", "model", "kappa", "white_fraction", "n_samples"]
]

# %% [markdown]
# ## Sigmas
#
# The OLS sigma is several times smaller everywhere; GLS also sees that the
# left half is noisier.

# %%
fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
vmax = float(gls.velocity_sigma.max()) * 1000
for ax, result, title in zip(
    axes, (ols, gls), ("OLS (white noise)", "GLS (colored noise)")
):
    (result.velocity_sigma * 1000).plot.imshow(
        ax=ax, vmin=0, vmax=vmax, cbar_kwargs={"label": "velocity sigma (mm/yr)"}
    )
    ax.set_title(title, loc="left")
    ax.set_aspect("equal")
plt.show()

# %% [markdown]
# ## Are the sigmas honest?
#
# The truth is known, so count how often the velocity error falls within two
# sigmas: about 95% for an honest sigma.

# %%
for name, result in (("OLS", ols), ("GLS", gls)):
    within = float((abs(result.velocity - velocity) < 2 * result.velocity_sigma).mean())
    print(f"{name}: {within:.0%} of velocities within 2 sigma of the truth")
