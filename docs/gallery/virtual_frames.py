# %% [markdown]
# # Virtual frames and versioned stores with Icechunk
#
# Two ways to keep a frame in [Icechunk](https://icechunk.io), a
# transactional store on top of Zarr with commits, branches and tags
# (`cloud` extra):
#
# - **Virtual frame** (`build_virtual_frame`): references to the chunks
#   inside the original granules, with no data copied. Opening reads only
#   the chunk index, so a whole frame opens in seconds, locally or on S3.
#   Each ministack is one commit, and new granules extend the frame.
# - **Native store** (`to_icechunk`): writes your processed results (a
#   stack, a velocity) as Zarr inside Icechunk, with appends along time and
#   tags for releases.
#
# Both work the same on a local folder and on S3. This page runs locally on
# synthetic granules. Icechunk warns that a local folder is not safe for
# concurrent commits from several processes, which does not apply to one
# script; its log level is raised to errors here.

# %%
import os

os.environ.setdefault("ICECHUNK_LOG", "error")

import tempfile
from pathlib import Path

import numpy as np

import disp_xr as dxr
from disp_xr.tutorial import make_disp_stack

work = Path(tempfile.mkdtemp())
stack = make_disp_stack(work / "F11116", "three_ministacks", shape=(64, 64))
granules = dxr.find_products(work / "F11116")
print(f"{len(granules)} granules in {granules.reference_time.nunique()} ministacks")

# %% [markdown]
# ## A virtual frame
#
# The first two ministacks go in now, and the third later, as if it had just
# been produced. Each call commits one snapshot per ministack.

# %%
first = granules[granules.reference_time < granules.reference_time.max()]
snapshots = dxr.build_virtual_frame(first, work / "virtual", max_workers=1)
print(f"{len(snapshots)} commits:", [s[:8] for s in snapshots])

later = granules[granules.reference_time == granules.reference_time.max()]
snapshots += dxr.build_virtual_frame(later, work / "virtual", max_workers=1)
virtual = dxr.open_virtual(work / "virtual")
print(dict(virtual.sizes), virtual.displacement.dtype)

# %% [markdown]
# `validate_virtual` reads chunks through the references and compares them
# with the granules read directly (h5py). Run it once after building a
# frame.

# %%
report = dxr.validate_virtual(work / "virtual", granules)
print(report)

# %% [markdown]
# The virtual frame is a normal `open_disp`-like dataset: chain it, mask it,
# fit it. Here the same chained displacement comes out of the virtual frame
# and the granules.

# %%
from_virtual = dxr.stack_disp(virtual).displacement.isel(time=-1).values
from_files = dxr.stack_disp(dxr.open_disp(work / "F11116")).displacement.isel(time=-1)
print("identical:", np.allclose(from_virtual, from_files.values, equal_nan=True))

# %% [markdown]
# ## Time travel
#
# Every commit stays readable. Opening the frame as of the second commit
# gives the frame before the last ministack was added.

# %%
before = dxr.open_virtual(work / "virtual", snapshot_id=snapshots[1])
print(f"now {virtual.sizes['time']} epochs, at commit 2: {before.sizes['time']}")

# %% [markdown]
# ## A native, versioned store of results
#
# `to_icechunk` writes processed data with the same Zarr layout presets as
# `to_zarr`. `append_dim="time"` adds new epochs as a new commit, and a tag
# marks a release.

# %%
chained = dxr.stack_disp(dxr.open_disp(work / "F11116")).disp.rereference("median")
cut = chained.sizes["time"] - 5
dxr.to_icechunk(
    chained.isel(time=slice(0, cut)), work / "results", message="first epochs"
)
dxr.to_icechunk(
    chained.isel(time=slice(cut, None)),
    work / "results",
    append_dim="time",
    message="five new epochs",
    tag="v1",
)
stored = dxr.open_icechunk(work / "results", tag="v1")
print(dict(stored.sizes), stored.attrs.get("reference"))
