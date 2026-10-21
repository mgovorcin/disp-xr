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
ds = dxr.chain(dxr.open_disp(folder)).disp.rereference("median")

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
    ax.set_title("")  # drop xarray's automatic title
    ax.set_title(title, loc="left")
    ax.set_aspect("equal")
plt.show()

# %% [markdown]
# The low-coherence corner gets weight 0, so its cell has no value; the right
# edge cells extend past the frame (128 is not a multiple of 9) and their
# valid fraction shows how much of the block they hold. The sigma treats
# pixels as independent; spatially correlated errors make it too small at
# kilometre scales, which the next section fixes.

# %% [markdown]
# ## Spatial variation within a cell
#
# `sigma` is the uncertainty of the cell value. How much the pixels differ
# inside the cell is a different question, answered with `spread`: the
# weighted standard deviation (`std`), the weighted RMS about the cell value
# (`rmse`), and the weighted median absolute deviation from the weighted
# median (`mad`), which outlier pixels (unwrapping errors, a building) barely
# move. All use the same weights as the cell value. With a weighted median
# and `mad`, every statistic of the cell is robust. For Gaussian noise the
# MAD is about 0.67 times the std, as in the maps below (1.4826 MAD
# estimates the std).

# %%
robust = dxr.downsample(ds, factor=9, stat="wmedian", spread=["std", "mad"]).compute()
fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), layout="constrained")
(robust.displacement.isel(**last) * 1000).plot.imshow(
    ax=axes[0], cmap="RdBu_r", cbar_kwargs={"label": "mm"}
)
for ax, name in zip(axes[1:], ("std", "mad")):
    (robust[f"displacement_{name}"].isel(**last) * 1000).plot.imshow(
        ax=ax, cmap="magma", cbar_kwargs={"label": "mm"}
    )
for ax, title in zip(
    axes,
    ("270 m weighted median", "within-cell weighted std", "within-cell weighted MAD"),
):
    ax.set_title("")  # drop xarray's automatic title
    ax.set_title(title, loc="left")
    ax.set_aspect("equal")
plt.show()

# %% [markdown]
# ## Sigma with spatially correlated errors
#
# InSAR errors (troposphere, unwrapping) are correlated over kilometres, so
# averaging 81 pixels does not divide the noise by 9. Following xDEM
# (Hugonnet et al. 2022), estimate a variogram on a residual field (stable
# ground, or data minus a model), fit it, and pass it to `downsample`. Here
# the residual is synthetic: correlated noise with a 900 m range, 2 mm
# correlated and 1 mm white.

# %%
import numpy as np  # noqa: E402

from disp_xr.tutorial import correlated_noise  # noqa: E402

noise = correlated_noise(
    (128, 128), correlation_range=900, sill=(2e-3) ** 2, nugget=(1e-3) ** 2
)
residual = ds.displacement.isel(time=-1).copy(data=noise)
empirical = dxr.empirical_variogram(residual)
model = dxr.fit_variogram(empirical, ("exponential",))
print(model)

h = np.linspace(1, float(empirical.distance.max()), 300)
fig, ax = plt.subplots(figsize=(5, 3.4), layout="constrained")
ax.plot(empirical.distance, empirical.gamma * 1e6, "o", label="empirical")
ax.plot(h, model.gamma(h) * 1e6, label="fitted model")
ax.set(xscale="log", xlabel="distance (m)", ylabel="semivariance (mm$^2$)")
ax.legend()
plt.show()

# %%
noisy = ds.assign(displacement=ds.displacement + noise)
corr = dxr.downsample(noisy, factor=9, variogram={"displacement": model}).compute()
print(
    "median cell sigma, pixels independent:",
    f"{float(corr.displacement_sigma.median()) * 1000:.2f} mm;",
    "with the variogram:",
    f"{float(corr.displacement_sigma_corr.median()) * 1000:.2f} mm",
)

# %% [markdown]
# The variogram sigma is several times larger: neighbouring pixels share most
# of their error. `n_eff_corr` is the effective number of independent pixels
# per cell. The model holds for every epoch; estimate it on the epoch or the
# velocity residual you care about.

# %% [markdown]
# ## Hexagonal cells
#
# `downsample_hex` groups pixels into hexagons instead of squares: all six
# neighbors are equidistant, so the sampling is more isotropic. The lattice is
# anchored at the CRS origin, so overlapping frames of one UTM zone share
# cells. A spacing of 1.0746 times a square's side gives the same area; here
# 290 m hexagons next to the 270 m squares. The statistics are the same, and
# the result is a point table like `to_points`.

# %%
hexes = dxr.downsample_hex(ds, spacing=290).compute()
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), layout="constrained")
dxr.plot.plot_map(coarse, ax=axes[0], title="270 m squares")
dxr.plot.plot_hex(hexes, ax=axes[1], title="290 m hexagons")
plt.show()
print(hexes.sizes["point"], "hexagons;", int(hexes.pixels.median()), "pixels each")

# %% [markdown]
# H3 global cells (`grid="h3"`, the `hex` extra) line up across frames and
# UTM zones, so cells from neighboring frames merge by their `h3_cell` id.
# `to_geodataframe` returns the cells as polygons for GIS tools.

# %%
h3_cells = dxr.downsample_hex(ds, grid="h3", resolution=9).compute()
gdf = dxr.to_geodataframe(h3_cells)
print(h3_cells.sizes["point"], "H3 cells at resolution 9")
gdf[["h3_cell", "displacement", "n_eff", "geometry"]].head()

# %% [markdown]
# ## Quadtree
#
# `quadtree` keeps large cells where the data are smooth and splits them
# where they vary. Levels are factor-2 cells from 2^min_level pixels; a cell
# splits while its score is above the threshold and its four children keep
# enough reliable pixels. With `metric="displacement"` the score is the RMS
# weighted spread of the pixels in the cell, in meters. `to_raster` paints
# the leaves back on the 30 m grid.
#
# Which to use: in the T29 trade study (three DISP-S1 frames,
# [report][split-metric]), the regular grid of `downsample`
# reconstructed the 30 m field better than any quadtree with the same number
# of points, and agreed better with GNSS except across a creeping fault. Use
# the grid by default and the quadtree for sharp, local gradients. The
# quadtree's default metric is `"structure"`, which splits where the four
# children differ by more than their noise; the displacement metric used
# here reads in meters and, on real frames, splits noisy cells first.
# For Cal/Val accuracy runs, a stricter coherence floor in the weights
# (`dxr.quality_weights(ds, floors={"temporal_coherence": 0.8})`) agreed
# better with GNSS, at the cost of 12 to 39% of the pixels.
#
# [split-metric]: https://github.com/mgovorcin/disp-xr/blob/main/docs/studies/split_metric.md

# %%
tree = dxr.quadtree(
    ds, threshold=0.002, metric="displacement", min_level=2, max_level=5
)
leaves = tree.points.compute()
print(leaves.level.to_series().value_counts().sort_index().rename("leaves"))
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), layout="constrained")
(ds.displacement.isel(time=-1) * 1000).plot.imshow(
    ax=axes[0], cmap="RdBu_r", cbar_kwargs={"label": "mm"}
)
(tree.to_raster("displacement") * 1000).plot.imshow(
    ax=axes[1], cmap="RdBu_r", cbar_kwargs={"label": "mm"}
)
for ax, title in zip(axes, ("30 m", "quadtree leaves")):
    ax.set_title("")  # drop xarray's automatic title
    ax.set_title(title, loc="left")
    ax.set_aspect("equal")
plt.show()

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
