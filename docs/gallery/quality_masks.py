# %% [markdown]
# # Quality layers and masking
#
# Every DISP granule carries quality layers next to displacement. This example
# looks at them, compares the `"recommended"` and `"strict"` mask presets, and
# summarizes how often each pixel passes over time.

# %%
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt

import disp_xr as dxr
from disp_xr.tutorial import make_disp_stack

folder = Path(tempfile.mkdtemp()) / "granules"
make_disp_stack(folder, "two_ministacks", shape=(128, 128))
ds = dxr.open_disp(folder)
print([name for name in ds.data_vars if name != "displacement"])

# %% [markdown]
# ## The layers of one epoch
#
# `recommended_mask` is 1 where the processor recommends the pixel; water,
# low temporal coherence and unreliable unwrapping regions are 0.

# %%
epoch = ds.isel(time=-1)
fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), layout="constrained")
for ax, name in zip(axes, ["recommended_mask", "temporal_coherence", "water_mask"]):
    epoch[name].plot.imshow(ax=ax, add_labels=False)
    ax.set_title(name, loc="left", fontsize=10)
    ax.set_aspect("equal")
plt.show()

# %% [markdown]
# ## Mask presets
#
# `mask` sets displacement to NaN where a pixel fails. `"strict"` adds the
# connected-component, water and temporal-coherence criteria to the
# recommended mask. Criteria can also be given one by one, for example
# `dxr.mask(ds, None, temporal_coherence=0.5)`. In this synthetic frame the
# recommended mask already excludes everything `"strict"` adds. A coherence
# threshold alone keeps more: it drops the low-coherence corner (0.3) but not
# the water columns, which the recommended mask removes.

# %%
for label, masked in [
    ("recommended", dxr.mask(ds, "recommended")),
    ("strict", dxr.mask(ds, "strict")),
    ("coherence > 0.5", dxr.mask(ds, None, temporal_coherence=0.5)),
]:
    kept = masked.displacement.notnull().mean().compute()
    print(f"{label:>17}: {float(kept):.1%} of pixel-epochs kept")

# %% [markdown]
# ## Over time
#
# `quality_summary` reduces over time: the share of epochs each pixel was
# masked, the harmonic mean of temporal coherence (which weights bad periods
# more than a plain mean), and whether the pixel is reliable (recommended in
# at least 90% of epochs).

# %%
summary = dxr.quality_summary(ds).compute()
fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), layout="constrained")
for ax, name in zip(axes, ["pct_masked", "temporal_coherence_hmean", "reliable"]):
    summary[name].plot.imshow(ax=ax, add_labels=False)
    ax.set_title(name, loc="left", fontsize=10)
    ax.set_aspect("equal")
plt.show()
print(f"reliable pixels: {float(summary.reliable.mean()):.1%}")
