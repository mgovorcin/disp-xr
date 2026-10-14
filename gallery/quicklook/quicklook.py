# %% [markdown]
# # Quick look at a frame
#
# Open a frame of DISP granules, apply the product corrections and quality
# mask, chain the ministacks, re-reference, and look at a map and a time
# series. The granules are synthetic (`disp_xr.tutorial`), written with the
# layout of real DISP-S1 files and a known truth, so the result can be
# checked. Replace `folder` with your own granules.

# %%
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import disp_xr as dxr
from disp_xr.tutorial import make_disp_stack

folder = Path(tempfile.mkdtemp()) / "granules"
truth = make_disp_stack(folder, "three_ministacks", shape=(128, 128))
print(len(truth.paths), "granules in", folder)

# %% [markdown]
# ## Open, correct, mask, chain
#
# `open_disp` reads metadata only. Corrections and the mask apply per
# granule, before `chain` chains the three ministacks into one series
# from the first reference date.

# %%
ds = dxr.open_disp(folder, corrections=True)
ds = ds.disp.correct().disp.mask("recommended").disp.chain()
ds = ds.disp.rereference("auto")
print("relative to", ds.attrs["reference_date"], "| epochs:", ds.sizes["time"])

# %% [markdown]
# ## Map
#
# `plot_map` shows one epoch (the last by default) in millimeters on a scale
# symmetric about zero. Masked pixels and water are blank.

# %%
fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
ds.disp.plot_map(ds.time[len(ds.time) // 2].values, ax=left)
ds.disp.plot_map(ax=right)
plt.show()

# %% [markdown]
# ## Time series
#
# `plot_point` takes a longitude and latitude (or `row`/`col`) and marks each
# reference-date change, where one ministack is joined to the next.

# %%
row, col = 64, 90
ax = ds.disp.plot_point(row=row, col=col, buffer_m=60)
plt.show()

# %% [markdown]
# ## Check against the truth
#
# `rereference` records the pixel it chose in `attrs["reference"]`. The
# synthetic truth is known, so the chaining itself can be checked: without
# corrections, masking or re-referencing, the chained series must equal the
# displacement the granules were built from.

# %%
print("reference:", ds.attrs["reference"])
chained = dxr.chain(dxr.open_disp(folder))
error = chained.displacement.values - truth.cumulative
print(f"largest difference from the truth: {np.nanmax(np.abs(error)) * 1000:.4f} mm")
