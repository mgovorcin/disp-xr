# %% [markdown]
# # Exporting results
#
# disp-xr writes Zarr v3, NetCDF, Cloud Optimized GeoTIFF (COG) and MintPy
# inputs.
# Every output records the disp-xr version, the input granules and the
# operations applied, so a file can be traced back to its sources.

# %%
import json
import tempfile
from pathlib import Path

import h5py
import rasterio
import xarray as xr

import disp_xr as dxr
from disp_xr.tutorial import make_disp_stack

work = Path(tempfile.mkdtemp())
make_disp_stack(work / "granules", "two_ministacks", shape=(96, 96))
ds = dxr.open_disp(work / "granules").disp.mask("recommended").disp.stack()
ds = ds.disp.rereference("median")

# %% [markdown]
# ## Zarr
#
# The `"timeseries"` preset puts the whole time axis in each chunk, for
# per-pixel work; `"maps"` puts one epoch in each chunk, for viewing. Both use
# sharding and zstd compression.

# %%
dxr.to_zarr(ds, work / "frame.zarr", preset="timeseries")
back = xr.open_zarr(work / "frame.zarr", decode_coords="all")
print("CRS:", back.rio.crs, "| chunks:", back.displacement.encoding["chunks"])
for line in back.attrs["history"].splitlines():
    print(json.loads(line)["op"])

# %% [markdown]
# ## NetCDF and Cloud Optimized GeoTIFF
#
# `to_geotiff` writes a COG: tiled, with internal overviews once the image is
# larger than 512 pixels. Pass the Dataset and a `layer` so the file keeps the
# provenance, which lives in the Dataset's attributes. `per_epoch=True` writes
# one COG per epoch.

# %%
dxr.to_netcdf(ds, work / "frame.nc")
dxr.to_geotiff(ds.isel(time=-1), work / "last.tif", layer="displacement")
with rasterio.open(work / "last.tif") as src:
    print(src.tags(ns="IMAGE_STRUCTURE")["LAYOUT"], src.crs)
    print("granules:", len(json.loads(src.tags()["INPUT_GRANULES"])))
    print("disp-xr:", src.tags()["DISP_XR_VERSION"])

# %% [markdown]
# ## MintPy
#
# `to_mintpy` writes `timeseries.h5` and the quality files MintPy reads, with
# the platform constants of the product (wavelength, orbit height).

# %%
files = dxr.to_mintpy(ds, work / "mintpy")
with h5py.File(files["timeseries"]) as f:
    print(list(f), f["timeseries"].shape, f.attrs["WAVELENGTH"])
