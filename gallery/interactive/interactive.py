# %% [markdown]
# # Interactive browsing of a full frame
#
# A DISP frame is about 7,000 x 9,000 pixels, more than a browser can draw as
# individual pixels. [hvplot](https://hvplot.holoviz.org) with
# [datashader](https://datashader.org) rasterizes the array to the size of the
# plot, so a whole frame stays responsive, and a slider steps through epochs.
# These need the `interactive` extra (`pip install "disp_xr[interactive]"`);
# nothing in disp-xr itself imports them.
#
# This page is static: the slider works, and each epoch is rasterized once
# when the page is built. In a notebook or a `panel serve` app, the plot is
# dynamic: zooming in rasterizes again at full resolution, and only the epoch
# on screen is read.

# %%
import tempfile
from pathlib import Path

import holoviews as hv
import hvplot.xarray  # noqa: F401  (registers .hvplot)
import pandas as pd
from holoviews.operation.datashader import rasterize

import disp_xr as dxr
from disp_xr.plot import DIVERGING
from disp_xr.tutorial import make_disp_stack

folder = Path(tempfile.mkdtemp()) / "granules"
make_disp_stack(folder, "three_ministacks", shape=(1024, 1024), epochs_per_ministack=3)
ds = dxr.open_disp(folder, layers=["displacement", "recommended_mask"])
ds = ds.disp.mask("recommended").disp.chain().disp.rereference("median")
mm = dxr.to_units(ds.displacement, "mm")
print(mm.sizes)

# %% [markdown]
# ## Maps through time
#
# A symmetric color scale centered on zero, from the disp-xr diverging map.
# In a notebook, one call gives a dynamic plot with a time slider; only the
# epoch on screen is read and rasterized:
#
# ```python
# mm.hvplot.image(x="x", y="y", groupby="time", rasterize=True, cmap=..., clim=...)
# ```
#
# This static page embeds four epochs, each rasterized once with datashader
# to 320 x 320 pixels, so the page stays small. (hvplot 0.12 cannot combine
# `groupby` with `dynamic=False`, so the frames are collected in a `HoloMap`.)

# %%
vmax = float(abs(mm.isel(time=-1)).quantile(0.98))
epsg = ds.rio.crs.to_epsg()


def frame(time):
    """Rasterize one epoch to 320 x 320 pixels with datashader."""
    image = hv.Image(mm.sel(time=time).reset_coords(drop=True), kdims=["x", "y"])
    return rasterize(image, width=320, height=320, dynamic=False)


dates = mm.time.values[:: max(1, len(mm.time) // 3)]
maps = hv.HoloMap(
    {pd.Timestamp(t).strftime("%Y-%m-%d"): frame(t) for t in dates}, kdims="date"
)
maps.opts(
    hv.opts.Image(
        cmap=list(DIVERGING),
        clim=(-vmax, vmax),
        colorbar=True,
        clabel="displacement (mm)",
        data_aspect=1,
        frame_width=380,
        xlabel=f"x (m) [EPSG:{epsg}]",
        ylabel=f"y (m) [EPSG:{epsg}]",
        tools=["hover"],
    )
)

# %% [markdown]
# ## A point's time series
#
# Hover over the line for the date and value of each epoch.

# %%
row, col = 512, 700
mm.isel(y=row, x=col).hvplot.line(
    x="time",
    color="#2a78d6",
    line_width=2,
    frame_width=620,
    frame_height=260,
    ylabel="displacement (mm)",
    title=f"row {row}, col {col}",
) * mm.isel(y=row, x=col).hvplot.scatter(x="time", color="#2a78d6", size=30)
