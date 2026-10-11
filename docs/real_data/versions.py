# %% [markdown]
# # Comparing two processing versions of a frame
#
# **Frame F08622 (New York City and Long Island), 2016-07 to 2024-12: the
# same 239 acquisitions processed as product version 1.0 and as 1.1.** A
# new processing version changes the algorithm, not the data, so the two
# should agree within noise wherever the ground is coherent. This notebook
# puts them on a common footing (same epochs, same reference) and looks at
# where they differ: the maps at the last date, the velocities, a few
# pixel series and the scatter around the reference.
#
# This notebook ran on real granules; its outputs are saved, and the
# documentation build does not run it again.

# %%
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyproj import Transformer

import disp_xr as dxr

warnings.simplefilter("ignore", FutureWarning)
root = Path("/u/operapst/bato/work/CALVAL/disp-s1_official")
versions = {
    name: sorted((root / name / "F08622" / "products").glob("*/*.nc"))
    for name in ("v1.0", "v1.1")
}


def title(ax, text):
    """Title the panel on the left, without xarray's automatic title."""
    ax.set_title("")
    ax.set_title(text, loc="left")


for name, files in versions.items():
    table = dxr.find_products(files)
    print(
        f"{name}: {len(table)} granules, {table.reference_time.nunique()} ministacks,"
        f" produced {table.production_time.min():%Y-%m} to"
        f" {table.production_time.max():%Y-%m}"
    )

# %% [markdown]
# ## Ministacks
#
# Both versions cover the same 239 acquisitions in 16 ministacks. Their
# reference dates agree except the second: v1.0 starts it on 2017-03-13 and
# v1.1 on 2016-09-26, so the two chain the first year differently. The table
# lists the first ministack reference dates of each.

# %%
refs = pd.DataFrame(
    {
        name: pd.Series(sorted(dxr.find_products(files).reference_time.unique()))
        for name, files in versions.items()
    }
)
refs.head(8)

# %% [markdown]
# ## A common footing
#
# A 30 x 30 km window over Brooklyn, Queens and lower Manhattan, chained
# per version, kept to the epochs both have, and referenced to the same
# 1 km box in Brooklyn.

# %%
west, south, east, north = -74.10, 40.60, -73.75, 40.85
ref_lon, ref_lat = -73.9791, 40.7035
stacks = {}
for name, files in versions.items():
    raw = dxr.open_disp(files, layers=["displacement", "recommended_mask"])
    to_grid = Transformer.from_crs("EPSG:4326", raw.rio.crs, always_xy=True)
    (x0, x1), (y0, y1) = to_grid.transform([west, east], [south, north])
    window = raw.sel(x=slice(x0, x1), y=slice(y1, y0))
    stacks[name] = dxr.chain(dxr.mask(window, "recommended"))
common = np.intersect1d(stacks["v1.0"].time.values, stacks["v1.1"].time.values)
box = (ref_lon - 0.005, ref_lat - 0.005, ref_lon + 0.005, ref_lat + 0.005)
for name in stacks:
    stacks[name] = dxr.rereference(stacks[name].sel(time=common), "area", bbox=box)
    stacks[name] = stacks[name].persist()  # read once for the maps, fits and series
print(f"{len(common)} common epochs, {dict(stacks['v1.0'].sizes)}")

# %% [markdown]
# ## The last common date

# %%
last = {
    name: (1e3 * ds.displacement.isel(time=-1)).compute() for name, ds in stacks.items()
}
diff = last["v1.1"] - last["v1.0"]
fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), layout="constrained")
for ax, name in zip(axes[:2], stacks):
    last[name].plot.imshow(
        ax=ax, cmap="RdBu", vmin=-40, vmax=40, cbar_kwargs={"label": "mm"}
    )
    title(ax, f"{name}, {str(common[-1])[:10]}")
diff.plot.imshow(
    ax=axes[2], cmap="PuOr", vmin=-20, vmax=20, cbar_kwargs={"label": "mm"}
)
title(axes[2], "v1.1 minus v1.0")
for ax in axes:
    ax.set_aspect("equal")
plt.show()
both = diff.notnull()
print(
    f"pixels valid in both: {float(both.mean()):.0%}; difference median"
    f" {float(diff.median()):+.1f} mm, 1.4826 MAD"
    f" {1.4826 * float(abs(diff - diff.median()).median()):.1f} mm"
)

# %% [markdown]
# ## Velocities
#
# MIDAS velocity of each version on the pixels valid in at least half the
# common epochs in both.

# %%
enough = None
velocity = {}
for name, ds in stacks.items():
    valid = ds.displacement.isel(time=slice(1, None)).notnull().mean("time") >= 0.5
    enough = valid if enough is None else enough & valid
    velocity[name] = 1e3 * dxr.fit_velocity(ds, method="midas").velocity
velocity = {name: v.where(enough).compute() for name, v in velocity.items()}
vdiff = velocity["v1.1"] - velocity["v1.0"]

fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), layout="constrained")
for ax, name in zip(axes[:2], velocity):
    velocity[name].plot.imshow(
        ax=ax, cmap="RdBu", vmin=-5, vmax=5, cbar_kwargs={"label": "mm/yr"}
    )
    title(ax, f"{name} MIDAS velocity")
vdiff.plot.imshow(
    ax=axes[2], cmap="PuOr", vmin=-2, vmax=2, cbar_kwargs={"label": "mm/yr"}
)
title(axes[2], "v1.1 minus v1.0")
for ax in axes:
    ax.set_aspect("equal")
plt.show()
large = float((abs(vdiff) > 2).sum() / vdiff.notnull().sum())
print(
    f"velocity difference: median {float(vdiff.median()):+.2f} mm/yr,"
    f" 1.4826 MAD {1.4826 * float(abs(vdiff - vdiff.median()).median()):.2f} mm/yr,"
    f" |difference| > 2 mm/yr on {large:.1%} of the pixels"
)

# %% [markdown]
# ## Series at four points
#
# The median of a 30 m radius around each point (about one pixel and its
# neighbors): LaGuardia Airport, lower Manhattan, Woodside and Ridgewood.

# %%
points = {
    "LaGuardia": (-73.871854, 40.771890),
    "Lower Manhattan": (-73.997618, 40.725698),
    "Woodside": (-73.90194, 40.74775),
    "Ridgewood": (-73.91977, 40.71759),
}
fig, axes = plt.subplots(2, 2, figsize=(13, 6.5), layout="constrained", sharex=True)
for ax, (label, (lon, lat)) in zip(axes.flat, points.items()):
    x, y = to_grid.transform(lon, lat)
    for name, ds in stacks.items():
        series = 1e3 * ds.displacement.sel(
            x=slice(x - 30, x + 30), y=slice(y + 30, y - 30)
        ).median(("y", "x"))
        ax.plot(series.time, series, ".-", ms=2, lw=0.6, label=name)
    ax.set(title=label, ylabel="mm")
axes[0, 0].legend()
plt.show()

# %% [markdown]
# ## Scatter around the reference
#
# With the reference box assumed stable, the spread of the pixels within
# 1 km of it, epoch by epoch, is a direct look at each version's noise at
# short distance.

# %%
x, y = to_grid.transform(ref_lon, ref_lat)
fig, ax = plt.subplots(figsize=(11, 3.6), layout="constrained")
for name, ds in stacks.items():
    near = 1e3 * ds.displacement.sel(
        x=slice(x - 1000, x + 1000), y=slice(y + 1000, y - 1000)
    )
    spread = near.std(("y", "x")).compute()
    ax.plot(
        spread.time,
        spread,
        ".-",
        ms=2,
        lw=0.6,
        label=f"{name}: median {float(spread.median()):.1f} mm",
    )
ax.set(ylabel="std within 1 km of the reference (mm)")
ax.legend(loc="upper left")
plt.show()

# %% [markdown]
# ## What changed
#
# - Where the ground is coherent, the two versions agree within their noise.
#   - At the last date, on the 66% of pixels valid in both: a median
#     difference of -0.2 mm, spread 0.4 mm (1.4826 MAD).
#   - In velocity: a median difference of -0.01 mm/yr, spread 0.08 mm/yr.
#     Only 0.2% of the pixels differ by more than 2 mm/yr, isolated pixels
#     scattered across the window.
# - LaGuardia Airport is the one point of the four where the series part:
#   from about 2019, v1.1 sits a few millimetres above v1.0.
# - v1.1 is quieter near the reference. The pixel spread within 1 km of it
#   has a median of 5.1 mm, against 5.6 mm for v1.0, and is lower at almost
#   every epoch.
