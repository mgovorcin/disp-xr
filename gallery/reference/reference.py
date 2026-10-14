# %% [markdown]
# # Corrections and reference methods
#
# DISP displacement is relative in space and in time. In time, each granule
# is relative to its ministack's reference date, until `chain` chains
# them. In space, values are relative to an arbitrary point, so a map means
# something only after re-referencing. The order follows the product:
#
# 1. `correct`: subtract correction layers, per granule (they are relative
#    to each granule's reference date, so this comes before chaining);
# 2. `mask` (optional);
# 3. `chain`: chain the ministacks;
# 4. `rereference`: choose what is zero in each epoch.
#
# The synthetic frame has a low-coherence corner, a patch where unwrapping
# failed, water columns on the east edge and a few persistent scatterers,
# so the methods can be compared.

# %%
import json
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt

import disp_xr as dxr
from disp_xr.tutorial import make_disp_stack

folder = Path(tempfile.mkdtemp()) / "granules"
make_disp_stack(folder, "three_ministacks", shape=(64, 64), epochs_per_ministack=10)
raw = dxr.open_disp(folder, corrections=True)

# %% [markdown]
# ## Corrections
#
# `correct` subtracts the product's ionospheric delay and solid-earth tide
# (`"product"` in the command line picks those the product has not applied
# already). External corrections, such as a troposphere model, go in
# `external=` on the same grid.

# %%
corrected = dxr.correct(raw, ["ionospheric_delay", "solid_earth_tide"])
change = (corrected.displacement - raw.displacement).isel(time=-1).mean().values
print(f"correction at the last epoch: {1e3 * float(change):+.1f} mm")
ds = dxr.chain(dxr.mask(corrected, "recommended"))

# %% [markdown]
# ## Reference methods
#
# | Method | Zero in each epoch | Use when |
# | --- | --- | --- |
# | `point` | one pixel (`row`/`col`, `lon`/`lat`) | a GNSS station, a stable point |
# | `area` | median of a lon/lat box or polygon | a stable region (robust) |
# | `median` | median of the scene's good pixels | the scene is mostly stable |
# | `border` | median of a border strip | deformation sits in the middle |
# | `high_coherence` | median of a good-pixel mask | you have your own mask |
# | `auto` | a stable pixel chosen from the quality layers | no prior knowledge |
#
# `auto` picks a pixel that is never masked, has no phase jumps, and has
# high harmonic-mean coherence, close to the middle of the good area. It
# prefers persistent scatterers. It reads every quality layer of every
# epoch first, which is slow on full frames (see "Large frames and batch
# jobs").

# %%
west, south, east, north = ds.rio.transform_bounds("EPSG:4326")
box = (west, south, west + 0.2 * (east - west), south + 0.2 * (north - south))
references = {
    "point": dxr.rereference(ds, "point", row=40, col=30),
    "area": dxr.rereference(ds, "area", bbox=box),
    "median": dxr.rereference(ds, "median"),
    "border": dxr.rereference(ds, "border", border_pixels=3),
    "high_coherence": dxr.rereference(
        ds, "high_coherence", good_pixels=dxr.valid_mask(ds, "strict").all("time")
    ),
    "auto": dxr.rereference(ds, "auto"),
}
fig, axes = plt.subplots(2, 3, figsize=(12, 7), layout="constrained")
for ax, (name, out) in zip(axes.flat, references.items()):
    (1e3 * out.displacement.isel(time=-1)).plot.imshow(
        ax=ax, cmap="RdBu_r", vmin=-25, vmax=25, cbar_kwargs={"label": "mm"}
    )
    record = json.loads(out.attrs["reference"])
    where = f" (row {record['row']}, col {record['col']})" if "row" in record else ""
    ax.set_title(f"{name}{where}", loc="left")
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
plt.show()

# %% [markdown]
# The maps differ only by a constant in each epoch: re-referencing never
# changes the relative motion between two pixels. What changes is which
# part of the frame reads as zero, and how much noise the reference carries
# into every pixel. A single pixel carries its own noise; a median over many
# good pixels averages it down. Here `auto` chose pixel (30, 30), one of
# the frame's persistent scatterers.

# %%
pixel = {"y": 50, "x": 20}
fig, ax = plt.subplots(figsize=(10, 3.5), layout="constrained")
for name in ("point", "area", "median", "auto"):
    series = 1e3 * references[name].displacement.isel(pixel)
    ax.plot(series.time, series, label=name)
ax.set(ylabel="displacement at one pixel (mm)")
ax.legend(ncols=4)
plt.show()
print(json.loads(references["auto"].attrs["reference"]))
