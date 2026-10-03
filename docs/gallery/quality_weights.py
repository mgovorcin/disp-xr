# %% [markdown]
# # Quality weights
#
# `quality_weights` turns the DISP quality layers into one weight per pixel
# and epoch, between 0 and 1: a linear stretch of phase similarity and
# temporal coherence above a floor, and a down-weighting of dates whose
# inversion residuals show unwrapping cycle jumps (multiples of 2 pi). The
# weights feed weighted fits and, later, quality-weighted downsampling.

# %%
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt

import disp_xr as dxr
from disp_xr.tutorial import make_disp_stack

folder = Path(tempfile.mkdtemp()) / "granules"
make_disp_stack(folder, "two_ministacks", shape=(64, 64))
ds = dxr.open_disp(folder)
w = dxr.quality_weights(ds).compute()
print(list(w.data_vars))

# %% [markdown]
# ## Each layer and the combination
#
# Epoch 1 has injected cycle jumps at two pixels (one and two cycles), and a
# low-coherence corner. The combined weight is the geometric mean of the
# layer weights, so a pixel is only trusted when every layer is good.

# %%
names = [
    "weight_phase_similarity",
    "weight_temporal_coherence",
    "weight_timeseries_inversion_residuals",
    "weight",
]
fig, axes = plt.subplots(1, 4, figsize=(15, 3.6), layout="constrained")
for ax, name in zip(axes, names):
    w[name].isel(time=1).plot.imshow(ax=ax, vmin=0, vmax=1, add_labels=False)
    ax.set_title(name.removeprefix("weight_") or "weight", loc="left", fontsize=10)
    ax.set_aspect("equal")
plt.show()

# %% [markdown]
# ## Cycle jumps down-weight, never exclude
#
# Each 2 pi cycle multiplies the residual weight by `penalty` (0.5), and the
# weight never drops below `min_residual` (0.05). Low coherence or similarity,
# below the processor's thresholds, does exclude (weight 0).

# %%
residual = w.weight_timeseries_inversion_residuals.isel(time=1)
for (row, col), cycles in {(10, 50): 1, (11, 51): 2}.items():
    print(f"{cycles} cycle(s): residual weight {float(residual[row, col]):.3f}")
print(
    "static weight (whole record) of the low-coherence corner:",
    float(w.static_weight[:8, :8].max()),
)

# %% [markdown]
# ## Weighted fit
#
# Weights enter `fit_trajectory` as `1 / variance` factors; a zero weight
# drops the epoch, so the excluded corner gets no velocity.

# %%
stacked = dxr.stack_disp(ds)
weighted = dxr.fit_trajectory(
    stacked, weights=dxr.quality_weights(stacked).weight.fillna(0)
)
plain = dxr.fit_trajectory(stacked)
fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), layout="constrained")
for ax, fit, title in zip(axes, (plain, weighted), ("unweighted", "quality-weighted")):
    (fit.velocity * 1000).plot.imshow(
        ax=ax, cmap="RdBu_r", cbar_kwargs={"label": "mm/yr"}
    )
    ax.set_title(title, loc="left")
    ax.set_aspect("equal")
plt.show()

# %% [markdown]
# ## Do residuals really cluster at 2 pi?
#
# `residual_cycle_histogram` counts residuals in units of 2 pi. On real
# DISP-S1 granules (frame F44055) the counts peak near 1, 2 and 3 cycles,
# which is what the cycle decomposition assumes; run it on your own frame to
# check.

# %%
print(dxr.weights.residual_cycle_histogram(ds))
