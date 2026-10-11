# Getting started

This page goes from a folder of DISP granules to a referenced displacement
time series, a map and an export. Every block runs as written: the first one
writes a small synthetic frame with [`disp_xr.tutorial`](api/tutorial.md), so
nothing needs downloading. Replace the path with your own granules.

## Make or find granules

```python
import tempfile
from pathlib import Path

import disp_xr as dxr
from disp_xr.tutorial import make_disp_stack

folder = Path(tempfile.mkdtemp()) / "granules"
make_disp_stack(folder, "three_ministacks", shape=(128, 128))

table = dxr.find_products(folder)  # one row per granule, sorted by date
print(table[["frame_id", "reference_time", "secondary_time", "version"]])
```

`find_products` parses OPERA file names (and the `/identification` group when
the name is not standard), drops duplicate granules keeping the latest
version and production, and refuses unsupported product versions. On your own
data it also accepts a glob, a list of paths or an `s3://` prefix.

## Open

```python
ds = dxr.open_disp(table, corrections=True)
ds
```

`ds` is lazy: dims `(time, y, x)`, coordinates `time` (secondary date) and
`reference_time` (the ministack reference date of each epoch), the CRS from
`spatial_ref`, and one variable per product layer. Integer layers keep their
dtype and their no-data value; float layers use NaN.

## Correct, mask and chain

The order is fixed: corrections and the quality mask are applied per granule,
before the ministacks are chained.

```python
ds = dxr.correct(ds)                  # ionosphere and solid-earth tide
ds = dxr.mask(ds, "recommended")      # NaN where recommended_mask is bad
ds = dxr.chain(ds)               # one series from the first date
ds.attrs["reference_date"]
```

The accessor gives the same operations as methods, so a chain reads in order:
`ds.disp.correct().disp.mask("recommended").disp.chain()`.

## Re-reference

Displacement in DISP is relative to the product's reference point. Pick your
own: a stable pixel chosen automatically, a point, the scene median, or an
area.

```python
ds = dxr.rereference(ds, "auto")          # deterministic stable pixel
# dxr.rereference(ds, "point", lon=-118.1, lat=34.2)
# dxr.rereference(ds, "area", bbox=(xmin, ymin, xmax, ymax))
```

## Plot

```python
ds.disp.plot_map()                     # last epoch, mm, symmetric scale
ds.disp.plot_point(row=64, col=64)     # series with reference changes marked
```

## Export

```python
dxr.to_zarr(ds, folder.parent / "frame.zarr")          # Zarr v3, sharded
dxr.to_geotiff(ds.isel(time=-1), folder.parent / "last.tif", layer="displacement")  # COG
```

Every output records the disp-xr version, the input granules and the
operations applied (`ds.disp.history`). The [gallery](gallery/export.py)
shows each format, including MintPy.

## Large frames

A full frame is about 7,000 x 9,000 pixels and a few hundred epochs.
`open_disp` keeps the layout of the files, one epoch and one 1,024-pixel block
per chunk, so a map or a point reads only what it needs. For repeated
per-pixel work, rechunk once with `ds.disp.rechunk("timeseries")`, or write a
Zarr copy with `preset="timeseries"`. See
[Chunks for maps and time series](gallery/chunking.py).
