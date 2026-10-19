# %% [markdown]
# # Icechunk: virtual frames and versioned stores
#
# [Icechunk](https://icechunk.io) is a Zarr store with a version history:
# every write is a commit with a snapshot id, branches hold work in
# progress, and tags name snapshots that never move. disp-xr uses it in two
# ways (`cloud` extra: `pip install "disp-xr[cloud]"`):
#
# - **Virtual frame** (`build_virtual_frame`, `open_virtual`): references
#   to the chunks inside the original granules, with no data copied.
#   Opening reads only the chunk index, so a whole frame opens in under a
#   second instead of reading the metadata of every granule (0.44 s against
#   35 s for 29 granules on S3, `benchmarks/README.md`).
# - **Versioned store** (`to_icechunk`, `open_icechunk`): your processed
#   results (a chained stack, a velocity) written as Zarr, with appends
#   along time, branches and tags, and the provenance of each write in its
#   commit.
#
# | You want to | Use |
# |---|---|
# | Browse, quick-look or serve a frame without copying it | virtual frame |
# | Heavy per-pixel work (velocity, GLS, downsampling) | `to_zarr`, then `open_zarr` |
# | Results you cite, tag, append to or try alternatives on | versioned store |
#
# Both work the same on a local folder and on S3 (last section). This page
# runs locally on synthetic granules. Icechunk warns that a local folder is
# not safe for concurrent commits from several processes, which does not
# apply to one script; its log level is raised to errors here.

# %%
import json
import os

os.environ.setdefault("ICECHUNK_LOG", "error")

import tempfile
from pathlib import Path

import icechunk
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
# been produced. Each call commits one snapshot per ministack, so a frame
# grows as new granules arrive.

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
# The virtual frame is a normal `open_disp`-like dataset: displacement stays
# relative to each ministack's reference, as in the granules, so chain it,
# then mask or fit it. The same chained displacement comes out of the
# virtual frame and the granules.

# %%
from_virtual = dxr.chain(virtual).displacement.isel(time=-1).values
from_files = dxr.chain(dxr.open_disp(work / "F11116")).displacement.isel(time=-1)
print("identical:", np.allclose(from_virtual, from_files.values, equal_nan=True))

# %% [markdown]
# `open_virtual` chunks the frame as `open_disp` does (one epoch by
# `chunks.read_block` pixels), so a chain or a fit stays lazy and runs in
# parallel. The references themselves point at the granules' small HDF5
# chunks, which is why a virtual frame is for opening and exploring; for
# repeated per-pixel work, write a native store once.

# %%
print("dask chunks:", virtual.displacement.data.chunksize)
print("granule chunks:", virtual.displacement.encoding["chunks"])

# %% [markdown]
# ## Time travel
#
# Every commit stays readable. Opening the frame as of the second commit
# gives the frame before the last ministack was added.

# %%
before = dxr.open_virtual(work / "virtual", snapshot_id=snapshots[1])
print(f"now {virtual.sizes['time']} epochs, at commit 2: {before.sizes['time']}")

# %% [markdown]
# ## A versioned store of results
#
# `to_icechunk` writes processed data with the same Zarr layout presets as
# `to_zarr`. `append_dim="time"` adds new epochs as a new commit (the
# `"ministack"` layout, so an append writes new shards and rewrites none),
# and `tag` marks a release, for example the run a Cal/Val report used.

# %%
chained = dxr.chain(dxr.open_disp(work / "F11116")).disp.rereference("median")
cut = chained.sizes["time"] - 5
dxr.to_icechunk(
    chained.isel(time=slice(0, cut)),
    work / "results",
    append_dim="time",
    message="first epochs",
)
release = dxr.to_icechunk(
    chained.isel(time=slice(cut, None)),
    work / "results",
    append_dim="time",
    message="five new epochs",
    tag="v1",
)
stored = dxr.open_icechunk(work / "results", tag="v1")
print(dict(stored.sizes), stored.attrs.get("reference"))
print("read from snapshot", stored.attrs["source_snapshot_id"][:8])

# %% [markdown]
# `source_snapshot_id` names exactly what was read, and every export of
# `stored` carries it, so a figure or a report can be traced back to one
# snapshot.
#
# ## Processing from the store
#
# `open_icechunk` gives one dask chunk per shard, like `open_zarr`: a graph
# on a full frame stays small (`docs/performance.md`). The result is lazy
# until computed.

# %%
print("dask chunks:", stored.displacement.data.chunksize)
print("shards:", stored.displacement.encoding["shards"])
velocity = dxr.fit_velocity(stored, "ols").velocity
print("velocity range (mm/yr):", np.round(np.nanpercentile(velocity, [1, 99]) * 1e3, 1))

# %% [markdown]
# ## History and provenance
#
# The commit log is Icechunk's own. disp-xr puts the provenance of each
# write (package versions, frame, input granules, the steps applied) in the
# commit metadata as well as in the dataset attributes, so the log alone
# says how each snapshot was made.

# %%
repo = icechunk.Repository.open(
    icechunk.local_filesystem_storage(str(work / "results"))
)
for snap in repo.ancestry(branch="main"):
    # history holds one JSON line per step, as in the dataset attributes
    lines = snap.metadata.get("history", "").splitlines()
    steps = [json.loads(line)["op"] for line in lines]
    version = snap.metadata.get("disp_xr_version", "")
    print(snap.id[:8], f"{snap.message!r:24}", version, steps)
print("tags:", sorted(repo.list_tags()))

# %% [markdown]
# ## Branches: try a change without touching the release
#
# A branch starts from `main` and takes its own commits. Here a variant
# referenced to the frame border instead of the frame median goes on a
# branch;
# `main` and the `v1` tag still read the median version.

# %%
variant = dxr.chain(dxr.open_disp(work / "F11116")).disp.rereference("border")
dxr.to_icechunk(
    variant, work / "results", branch="border-ref", mode="w", message="border reference"
)
on_branch = dxr.open_icechunk(work / "results", branch="border-ref")
on_main = dxr.open_icechunk(work / "results")
print("branches:", sorted(repo.list_branches()))
print("main:", on_main.attrs["reference"])
print("border-ref:", on_branch.attrs["reference"])
diff = (on_branch.displacement - on_main.displacement).isel(time=-1)
print(f"largest difference at the last epoch: {float(abs(diff).max()) * 1e3:.2f} mm")

# %% [markdown]
# ## On S3
#
# A virtual frame takes an `s3://` location directly, and reads the granules
# with the `AWS_*` environment variables or temporary ASF credentials
# (renewed automatically; direct S3 access works from AWS `us-west-2`). For
# a versioned store, pass an Icechunk repository on S3:
#
# ```python
# dxr.build_virtual_frame(list(s3_urls), "s3://my-bucket/F11116-virtual")
# frame = dxr.chain(dxr.open_virtual("s3://my-bucket/F11116-virtual"))
#
# storage = icechunk.s3_storage(
#     bucket="my-bucket", prefix="F11116-results", from_env=True
# )
# repo = icechunk.Repository.open_or_create(storage)
# dxr.to_icechunk(frame, repo, message="chained frame", tag="calval-2026-10")
# cube = dxr.open_icechunk(repo, tag="calval-2026-10")
# ```
#
# See [Remote access](../remote.md) for credentials, and
# [Exporting results](export.py) for plain Zarr, NetCDF and GeoTIFF.
