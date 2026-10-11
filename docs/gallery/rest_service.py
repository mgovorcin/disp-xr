# %% [markdown]
# # A REST service for web clients
#
# `disp_xr serve STORE` serves a store read-only over HTTP, for web maps and
# dashboards (the DISP portal, Cal/Val pages) that need more than tiles:
#
# - displacement series at a point,
# - the stored velocity and its sigma,
# - grid or quadtree cells of a bounding box as GeoJSON,
# - GNSS station residuals,
# - the provenance of the store.
#
# Nothing heavy runs on request. Velocities and GNSS residuals are served
# when the store holds them, so they are computed once, by whoever builds the
# store. This page builds such a store from a synthetic frame, starts the
# service and queries every route, then shows a MapLibre client. Deploying
# the service is covered in [Serving stores over HTTP](../serving.md).

# %%
import socket
import tempfile
import threading
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import uvicorn
from matplotlib.collections import PolyCollection
from pyproj import Transformer

import disp_xr as dxr
from disp_xr.serve import make_app
from disp_xr.tutorial import make_disp_stack, make_gnss_source, vertical_los

folder = Path(tempfile.mkdtemp())
make_disp_stack(folder / "granules", "three_ministacks", shape=(128, 128))
ds = dxr.chain(dxr.open_disp(folder / "granules")).disp.rereference("median")

# %% [markdown]
# ## A store with velocity and GNSS residuals
#
# The cube, its velocity and sigma (written into the same store), and the
# GNSS comparison as a `gnss` group next to it (`Comparison.to_zarr`).

# %%
fit = dxr.fit_velocity(ds, "ols")
cube = ds.assign(velocity=fit.velocity, velocity_sigma=fit.velocity_sigma)
store = folder / "demo.zarr"
dxr.to_zarr(cube, store, preset="timeseries")

pixels = [(10, 15), (40, 90), (70, 30), (100, 100), (20, 110), (110, 20)]
gps = dxr.gnss.load(ds, source=make_gnss_source(ds, pixels, folder / "gnss"))
comparison = dxr.residuals.compare(ds, gps, vertical_los(ds), min_span_years=0)
comparison.to_zarr(store)

# %% [markdown]
# ## Serve
#
# On the command line this is `disp_xr serve demo.zarr --port 9000`. Here
# the same application runs in a thread. The API page with every route and
# response schema is at `/docs`.

# %%
rest = make_app({"demo": store}, tiles=False)
with socket.socket() as s:
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
server = uvicorn.Server(
    uvicorn.Config(rest.app, host="127.0.0.1", port=port, log_level="warning")
)
threading.Thread(target=server.run, daemon=True).start()
while not server.started:
    time.sleep(0.1)
api = f"http://127.0.0.1:{port}/datasets/demo/disp"
to_ll = Transformer.from_crs(ds.rio.crs, "EPSG:4326", always_xy=True)
lon, lat = to_ll.transform(float(ds.x[70]), float(ds.y[60]))

# %% [markdown]
# ## A point: series and velocity
#
# `timeseries` returns the median over `buffer_m` at each epoch, with the
# quality layers the store holds. `velocity` reads the stored velocity at
# the same point.

# %%
series = requests.get(
    f"{api}/timeseries", params={"lon": lon, "lat": lat, "buffer_m": 60}, timeout=30
).json()
velocity = requests.get(
    f"{api}/velocity", params={"lon": lon, "lat": lat}, timeout=30
).json()
print(series["crs"], series["units"], "reference", series["reference_date"])
print(series["point"])
print(
    f"velocity {1e3 * velocity['velocity']:.1f} ± "
    f"{1e3 * velocity['velocity_sigma']:.1f} mm/yr ({velocity['units']})"
)

dates = pd.to_datetime(series["dates"])
fig, ax = plt.subplots(figsize=(8, 3), layout="constrained")
ax.plot(dates, 1e3 * np.array(series["displacement"], float), ".-")
ax.set(ylabel="displacement (mm)", title=f"{lon:.3f}, {lat:.3f}")
plt.show()

# %% [markdown]
# ## Cells of a bounding box
#
# `cells` downsamples the box on request, on a regular grid (`size_m`) or a
# quadtree (`n_points`), and returns GeoJSON polygons in longitude and
# latitude. The layer is the stored velocity if there is one, else the last
# displacement epoch (`layer=`, `date=` choose).

# %%
west, south, east, north = ds.rio.transform_bounds("EPSG:4326")
box = f"{west},{south},{east},{north}"
grid = requests.get(f"{api}/cells", params={"bbox": box, "size_m": "600"}, timeout=60)
tree = requests.get(f"{api}/cells", params={"bbox": box, "n_points": "60"}, timeout=60)
grid, tree = grid.json(), tree.json()
print(
    grid["method"],
    len(grid["features"]),
    "cells;",
    tree["method"],
    len(tree["features"]),
    "cells;",
    grid["layer"],
    grid["units"],
)

fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), layout="constrained")
for ax, cells in zip(axes, (grid, tree)):
    polygons = [f["geometry"]["coordinates"][0] for f in cells["features"]]
    values = [1e3 * f["properties"]["velocity"] for f in cells["features"]]
    patches = PolyCollection(polygons, array=values, cmap="RdBu", edgecolor="0.4")
    patches.set_clim(-30, 30)
    ax.add_collection(patches)
    ax.autoscale_view()
    ax.set_aspect(1 / np.cos(np.radians(lat)))
    ax.set_title(f"{cells['method']}: {len(polygons)} cells", loc="left")
fig.colorbar(patches, ax=axes, label="velocity (mm/yr)")
plt.show()

# %% [markdown]
# ## GNSS residuals and provenance

# %%
gnss = requests.get(f"{api}/gnss", timeout=30).json()
table = pd.DataFrame(gnss["stations"]).set_index("station")
print(gnss["units"])
print((1e3 * table[["insar_velocity", "gps_velocity", "velocity_residual"]]).round(2))

provenance = requests.get(f"{api}/provenance", timeout=30).json()
print([step["op"] for step in provenance["history"]])
print(len(provenance["input_granules"]), "input granules")

# %% [markdown]
# ## A MapLibre client
#
# The cells are a GeoJSON source; a click asks for the series at that point.
# With MapLibre GL JS:
#
# ```js
# const api = "https://your-server/datasets/demo/disp";
# map.on("load", async () => {
#   const b = map.getBounds();
#   const bbox = [b.getWest(), b.getSouth(), b.getEast(), b.getNorth()].join(",");
#   const cells = await (await fetch(`${api}/cells?bbox=${bbox}&size_m=600`)).json();
#   map.addSource("cells", { type: "geojson", data: cells });
#   map.addLayer({
#     id: "cells", type: "fill", source: "cells",
#     paint: {
#       "fill-color": ["interpolate", ["linear"], ["get", "velocity"],
#                      -0.03, "#b2182b", 0, "#f7f7f7", 0.03, "#2166ac"],
#       "fill-opacity": 0.7,
#     },
#   });
# });
# map.on("click", async (e) => {
#   const q = `lon=${e.lngLat.lng}&lat=${e.lngLat.lat}&buffer_m=60`;
#   const series = await (await fetch(`${api}/timeseries?${q}`)).json();
#   console.log(series.dates, series.displacement);  // draw it in a chart
# });
# ```
#
# For a whole frame, ask for cells of the current view only, and redraw on
# `moveend`; requests above 4 M pixels are refused (413).

# %%
server.should_exit = True
