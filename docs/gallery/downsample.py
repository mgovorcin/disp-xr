# %% [markdown]
# # Downsampling with quality weights
#
# `downsample` aggregates 30 m pixels into coarse cells (about 1, 2 or 10 km)
# with quality weights, after chaining and re-referencing. Each cell gets a
# weighted mean or median, a sigma, the effective sample count, the share of
# the block that holds data and a quality score. `to_points` turns the cells
# into a point table for fits and GNSS comparison.

# %%
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt

import disp_xr as dxr
from disp_xr.tutorial import make_disp_stack

folder = Path(tempfile.mkdtemp()) / "granules"
make_disp_stack(folder, "two_ministacks", shape=(128, 128))
ds = dxr.stack_disp(dxr.open_disp(folder)).disp.rereference("median")

# %% [markdown]
# ## Aggregate
#
# Factor 9 turns 30 m into 270 m cells here (on a real frame use 33, 67 or
# 333). Odd factors keep each cell centre on a pixel centre. Weights default to
# `quality_weights` (similarity, coherence and inversion residuals), per epoch.

# %%
coarse = dxr.downsample(ds, factor=9).compute()
print(dict(coarse.sizes), coarse.attrs["resolution"], "m")
print(list(coarse.data_vars)[:10])

# %%
last = {"time": -1}
fig, axes = plt.subplots(1, 4, figsize=(16, 3.6), layout="constrained")
(ds.displacement.isel(**last) * 1000).plot.imshow(
    ax=axes[0], cmap="RdBu_r", cbar_kwargs={"label": "mm"}
)
(coarse.displacement.isel(**last) * 1000).plot.imshow(
    ax=axes[1], cmap="RdBu_r", cbar_kwargs={"label": "mm"}
)
(coarse.displacement_sigma.isel(**last) * 1000).plot.imshow(
    ax=axes[2], cbar_kwargs={"label": "mm"}
)
coarse.valid_fraction.isel(**last).plot.imshow(ax=axes[3], vmin=0, vmax=1)
for ax, title in zip(
    axes, ("30 m", "270 m weighted mean", "cell sigma", "valid fraction")
):
    ax.set_title(title, loc="left")
    ax.set_aspect("equal")
plt.show()

# %% [markdown]
# The low-coherence corner gets weight 0, so its cell has no value; the right
# edge cells extend past the frame (128 is not a multiple of 9) and their
# valid fraction shows how much of the block they hold. The sigma treats
# pixels as independent; spatially correlated errors make it too small at
# kilometre scales (a variogram-based sigma is planned).

# %% [markdown]
# ## Points

# %%
points = dxr.to_points(coarse)
points.isel(time=-1).to_dataframe()[
    [
        "cell",
        "centroid_x",
        "centroid_y",
        "displacement",
        "displacement_sigma",
        "n_eff",
        "quality",
    ]
].head()
