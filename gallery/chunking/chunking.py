# %% [markdown]
# # Chunks for maps and time series
#
# DISP stores one file per date. A pixel's time series must open every file;
# a map needs one. disp-xr opens data lazily in the layout of the files, one
# epoch and one block per dask chunk, and each operation reads only what it
# touches. This example counts what each question reads.

# %%
import tempfile
from pathlib import Path

from dask.core import flatten

import disp_xr as dxr
from disp_xr.tutorial import make_disp_stack

folder = Path(tempfile.mkdtemp()) / "granules"
make_disp_stack(folder, "three_ministacks", shape=(64, 64))
raw = dxr.open_disp(folder, layers=["displacement"])
print("chunks:", {dim: sizes[:3] for dim, sizes in raw.displacement.chunksizes.items()})


def epochs_read(result, source):
    """Return how many source epochs a lazy result depends on."""
    data = result.data
    graph = data.__dask_graph__().cull(set(flatten(data.__dask_keys__())))
    name = source.data.name
    return len({key[1] for key in graph if isinstance(key, tuple) and key[0] == name})


# %% [markdown]
# ## A map of one chained epoch
#
# Chaining adds, to each epoch, the displacement accumulated at the crossover
# epochs of earlier ministacks. The last epoch of three ministacks therefore
# reads itself and two crossovers, not the whole stack.

# %%
chained = dxr.chain(raw)
print("source epochs:", raw.sizes["time"])
print(
    "read for the last chained epoch:",
    epochs_read(chained.displacement.isel(time=-1), raw.displacement),
)

# %% [markdown]
# ## A pixel's time series
#
# A series reads one block from every granule: all epochs, but only the block
# that holds the pixel.

# %%
point = chained.displacement.isel(y=10, x=10)
print("read for one pixel's series:", epochs_read(point, raw.displacement))

# %% [markdown]
# ## Choosing a layout
#
# Functions that work along time per pixel (fits, statistics) rechunk to
# whole time series themselves. When one kind of access repeats, rechunk once:

# %%
for preset in ("native", "timeseries", "maps"):
    chunks = dxr.rechunk(raw, preset).displacement.chunksizes
    print(f"{preset:>10}: time {chunks['time'][:3]}, y {chunks['y'][:2]}")

# %% [markdown]
# For data you will analyse many times, write a Zarr copy in the layout of
# the work: `dxr.to_zarr(ds, "frame.zarr", preset="timeseries")`.
