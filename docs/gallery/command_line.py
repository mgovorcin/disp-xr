# %% [markdown]
# # The command line, end to end
#
# `disp_xr` runs the common steps without Python:
#
# | Command | What it does |
# | --- | --- |
# | `search` | find a frame's granules in NASA CMR |
# | `info` | summarize local granules: frame, dates, versions, duplicates |
# | `stack` | correct, mask, chain and re-reference into one store |
# | `export` | write COG, GeoTIFF, Zarr, GeoZarr (for bowser), NetCDF or MintPy |
# | `geometry` | download the static layers (LOS, DEM, masks) on the DISP grid |
# | `virtual` | build a virtual frame in Icechunk without copying data |
#
# This page runs `info`, `stack` and `export` on synthetic granules. Finding
# and downloading real ones looks like this (it needs Earthdata credentials
# in `~/.netrc`):
#
# ```bash
# disp_xr search --frame-id 11116 --start 2024-01-01 --out granules.csv
# disp_xr geometry F11116/ --out static/
# ```

# %%
import json
import subprocess
import tempfile
from pathlib import Path

import disp_xr as dxr
from disp_xr.tutorial import make_disp_stack

work = Path(tempfile.mkdtemp())
make_disp_stack(work / "F11116", "three_ministacks", shape=(64, 64))


def run(*args):
    """Run a disp_xr command in the work folder and print what it prints."""
    print("$ disp_xr " + " ".join(args))
    out = subprocess.run(["disp_xr", *args], capture_output=True, text=True, cwd=work)
    print(out.stdout.strip() or out.stderr.strip())
    out.check_returncode()


# %% [markdown]
# ## What is in a folder of granules

# %%
run("info", "F11116")

# %% [markdown]
# ## One stack
#
# `stack` applies the steps in the product's order: subtract the
# corrections the product has not applied, apply the recommended mask,
# chain the ministacks, then re-reference (`auto` by default). The output
# is a Zarr store, or NetCDF with `.nc`, laid out for time series (`--preset
# timeseries`) or maps (`--preset maps`).

# %%
run("stack", "F11116", "--out", "F11116.zarr", "--reference", "median")

# %% [markdown]
# ## Export
#
# `export` reads a store from `stack` (or granules, chained without
# corrections, masking or re-referencing) and writes other formats. Here,
# a Cloud Optimized GeoTIFF of the last epoch, and a GeoZarr store that the
# bowser viewer opens directly (`--flat`: this frame is too small for the
# display pyramid that full frames get).

# %%
run("export", "F11116.zarr", "--format", "cog", "--out", "last.tif")
run("export", "F11116.zarr", "--format", "geozarr", "--out", "view.zarr", "--flat")

# %% [markdown]
# ## Back in Python
#
# Every output records where it came from: input granules, the steps and
# their parameters, and package versions.

# %%
ds = dxr.open_zarr(work / "F11116.zarr")
print(dict(ds.sizes))
print(ds.attrs["reference"])
print("steps:", [step["op"] for step in ds.disp.history])
# Lists are stored as JSON text, so NetCDF and Zarr outputs read the same.
print("from", len(json.loads(ds.attrs["input_granules"])), "granules")
