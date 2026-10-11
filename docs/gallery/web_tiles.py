# %% [markdown]
# # Serving results as web map tiles
#
# [xpublish-tiles](https://github.com/earth-mover/xpublish-tiles) serves
# xarray datasets as [OGC Tiles](https://www.ogc.org/standards/ogcapi-tiles/)
# for any web map (MapLibre, Leaflet, QGIS): PNG tiles rendered on request
# from the data, a TileJSON description and a legend. A velocity map, a
# displacement epoch or a quality layer from disp-xr can go straight in. A
# Zarr or GeoZarr store from `to_zarr` or `to_geozarr` works too, with
# multiscale pyramids picked by zoom level.
#
# This page fits a velocity on a synthetic frame, starts a local server
# and fetches a tile and the legend. In a real deployment the server runs
# behind your web map (see the end).

# %%
import io
import socket
import tempfile
import threading
import time
from pathlib import Path

import matplotlib.pyplot as plt
import morecantile
import numpy as np
import requests
import uvicorn
import xpublish
from PIL import Image
from xpublish_tiles.xpublish.tiles import TilesPlugin

import disp_xr as dxr
from disp_xr.tutorial import make_disp_stack

folder = Path(tempfile.mkdtemp()) / "granules"
make_disp_stack(folder, "three_ministacks", shape=(128, 128))
ds = dxr.chain(dxr.open_disp(folder)).disp.rereference("median")
velocity = dxr.fit_velocity(ds, "ols").compute()
velocity["velocity"] = (1e3 * velocity.velocity).assign_attrs(units="mm/yr")
# grid_mapping on every variable, so the server finds the CRS at once.
velocity = velocity.rio.write_crs(ds.rio.crs)

# %% [markdown]
# ## Serve
#
# `xpublish.Rest` serves named datasets; `TilesPlugin` adds the tile
# endpoints under `/datasets/<name>/tiles/`.

# %%
rest = xpublish.Rest({"velocity": velocity}, plugins={"tiles": TilesPlugin()})
with socket.socket() as s:
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
server = uvicorn.Server(
    uvicorn.Config(rest.app, host="127.0.0.1", port=port, log_level="warning")
)
threading.Thread(target=server.run, daemon=True).start()
while not server.started:
    time.sleep(0.1)
base = f"http://127.0.0.1:{port}/datasets/velocity/tiles/WebMercatorQuad"

# %% [markdown]
# ## A tile and its legend
#
# The tile covering the frame center at zoom 13, rendered with a fixed color
# range so tiles match across the map.

# %%
lon, lat = np.mean(np.reshape(ds.rio.transform_bounds("EPSG:4326"), (2, 2)), axis=0)
tile = morecantile.tms.get("WebMercatorQuad").tile(lon, lat, 13)
style = {
    "variables": "velocity",
    "style": "raster/RdBu",
    "colorscalerange": "-30,30",
    "width": "256",
    "height": "256",
}
png = requests.get(f"{base}/{tile.z}/{tile.y}/{tile.x}", params=style, timeout=60)
legend = requests.get(
    base.rsplit("/", 1)[0] + "/legend",
    params={k: style[k] for k in ("variables", "style", "colorscalerange")},
    timeout=60,
)
print(png.status_code, png.headers["content-type"], len(png.content), "bytes")
png.raise_for_status()
rgba = np.asarray(Image.open(io.BytesIO(png.content)).convert("RGBA"))
print(f"{(rgba[..., 3] > 0).mean():.0%} of the tile covered by the frame")
legend.raise_for_status()

fig, (left, right) = plt.subplots(
    1, 2, figsize=(9, 4), layout="constrained", width_ratios=(3, 1)
)
left.imshow(Image.open(io.BytesIO(png.content)))
left.set_title(f"tile {tile.z}/{tile.y}/{tile.x}", loc="left")
right.imshow(Image.open(io.BytesIO(legend.content)))
right.set_title("legend (mm/yr)", loc="left")
for ax in (left, right):
    ax.axis("off")
plt.show()

# %% [markdown]
# ## On a web map
#
# Point a raster source at the tile URL template. With MapLibre GL JS:
#
# ```js
# map.addSource("velocity", {
#   "type": "raster",
#   tiles: ["https://your-server/datasets/velocity/tiles/WebMercatorQuad/{z}/{y}/{x}"
#           + "?variables=velocity&style=raster/RdBu&colorscalerange=-30,30"
#           + "&width=256&height=256"],
#   tileSize: 256,
# });
# map.addLayer({ id: "velocity", type: "raster", source: "velocity" });
# ```
#
# `{base}/tilejson.json` with the same parameters describes the layer
# (bounds, zooms) for clients that read TileJSON. To serve a whole frame,
# open its store lazily (`xr.open_zarr`, or a pyramid from `to_geozarr`):
# tiles read only the chunks they cover.

# %%
server.should_exit = True
