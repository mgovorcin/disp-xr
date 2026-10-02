# disp-xr

Xarray tools for OPERA surface displacement products: DISP-S1 (Sentinel-1)
and DISP-NISAR.

disp-xr opens a frame of DISP granules as one lazy `(time, y, x)` dataset,
applies the corrections and quality masks the products ship with, chains the
ministacks into one displacement time series, re-references it, and writes the
result to Zarr, NetCDF, GeoTIFF, MintPy, Icechunk or a virtual store.

```python
import disp_xr as dxr

ds = dxr.open_disp("path/to/granules/", corrections=True)  # lazy, CRS attached
ds = ds.disp.correct().disp.mask("recommended").disp.stack()
ds = ds.disp.rereference("auto")

ds.disp.plot_map()                       # last epoch, in mm
ds.disp.plot_point(lon=-89.93, lat=30.02)
dxr.to_zarr(ds, "frame.zarr")
```

Nothing is read until a value is needed: opening a 209-granule frame reads
metadata only, a map reads one epoch (plus one crossover epoch per earlier
ministack), and a point reads one block per granule. See
[Concepts](concepts.md) for what the steps mean, and the
[gallery](gallery/quicklook.py) for worked examples you can run without
downloading anything.

!!! note "Status"
    disp-xr is pre-alpha. Names may change until the v0.2 API review; the
    [plan](todo.md) lists what is done and what is next.

## Where to go next

- [Install](install.md): pip, conda or pixi, and the optional extras.
- [Getting started](getting-started.md): open, correct, mask, chain,
  re-reference, plot and export a frame.
- [Command line](cli.md): the same steps as `disp_xr` commands.
- [API reference](api/index.md): every public function.
