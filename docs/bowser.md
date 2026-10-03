# View in bowser

[bowser](https://github.com/opera-adt/bowser) is a web viewer for InSAR
displacement stacks. `to_geozarr` writes a GeoZarr store that bowser opens
directly, without its `tifs-to-geozarr` converter:

```python
import disp_xr as dxr

ds = dxr.stack_disp(dxr.open_disp("granules/")).disp.rereference("auto")
dxr.to_geozarr(ds, "frame.zarr", static=dxr.open_static(ds), factors=(33, 67))
```

or from the command line:

```bash
disp_xr export frame_stack.zarr --format geozarr --out frame.zarr --static-dir static/
bowser run --stack-file frame.zarr
```

## What the store holds

- A multiscale pyramid: level `0` at 30 m, display levels by factor 2 until
  the shorter side is under 256 cells (as bowser's own converter builds
  them), and any analysis levels given in `factors`, all quality-weighted
  means (`downsample`). bowser picks the level whose pixel matches the map
  tile, so a pyramid of only 33, 67 and 333 would make mid zooms read 30 m.
- Every level is a complete dataset with the same variables (plus the cell
  statistics on coarse levels), its own CRS and transform.
- Float32 data, chunks of one epoch by 256 x 256 pixels in shards of four
  epochs by 1,024 x 1,024 (bowser's tile-read layout).

## What bowser reads, and what disp-xr sets for it

| bowser reads | disp-xr writes |
| --- | --- |
| Root `multiscales.layout[].asset`, finest first | The zarr-conventions multiscales v0.1 layout (also `derived_from`, `transform`, `spatial:transform`, `spatial:shape`) |
| CRS from `proj:code`, then `proj:wkt2`, then `spatial_ref.crs_wkt` | All three, on the root and on every level |
| Geometry from pixel-center `x`/`y` (y descending) and `GeoTransform` | Both, recomputed per level |
| Whether to offer a moving reference point: `bowser_uses_spatial_ref`, else the units (meters, radians) | `bowser_uses_spatial_ref` on every variable: true only for `displacement` and `velocity`. Without it, bowser would re-reference sigmas (meters) and inversion residuals (radians) |
| Colorbar title from `long_name`, unit from `units` | Both on every variable |
| Reference marker from `reference_lonlat` | The point chosen by `rereference` (none for a median or area reference) |
| Line-of-sight panel from `heading_angle_deg`, `incidence_angle_deg`, `los_enu_ground_to_satellite` | Near, center and far values from the static layers, when `static` is given; the heading assumes a right-looking radar unless `look_side="left"` |
| Histogram panel from `bowser_histogram` (else it reads a full epoch per request) | Per-epoch bins and percentiles on level `0`, computed once at write time |
| Masks from `temporal_coherence`, `phase_similarity`, `recommended_mask`, on the level being drawn | Present on every level (masks by mode, quality layers by weighted mean) |

## Best practices that do not affect bowser

These are checked against bowser `ops` (`0a61c6a`: zarr 3.1.1, xarray
2025.8, rioxarray 0.19) by `tests/bowser_check.py`, which runs bowser's own
server on a store and requests the dataset list, tiles at every level with
and without a mask, a point, a chart, the histogram and the range:

| Practice | bowser |
| --- | --- |
| `proj:` and `spatial:` attributes on every level, not only level 0 | Ignored; no effect |
| Multiscales `transform` and `derived_from` | Ignored; no effect |
| Consolidated metadata at the root | bowser opens with `consolidated=False`; no effect |
| `spatial_ref` as a coordinate (bowser's converter writes a data variable) | Found as before (`"spatial_ref" in ds.variables`) |
| Integer masks with a `_FillValue` instead of float NaN | xarray decodes the fill to NaN on read; masks and histograms behave as with floats |
| Small levels stored as one chunk | Read as usual |
| Coarse levels with extra variables (sigma, `n_eff`, ...) | Not listed (bowser lists level 0); harmless |
| Dataset attributes as JSON objects (`reference_lonlat`, LOS) | Required: as JSON strings, bowser fails to read the LOS attributes. disp-xr writes JSON objects to Zarr (NetCDF keeps strings) |

Gaps in bowser found on the way, not fixed here: overview levels written by
its converter have no `proj:`/`spatial:` attributes; it never reads
`spatial:*`; it chooses the level by comparing the pixel size in native CRS
units with Web Mercator meters, so a geographic-CRS store always gets the
coarsest level; and it has no Zarr tests. disp-xr stores are UTM, so the
level choice works.

## Testing against bowser

```bash
cd bowser && pixi install -e test --frozen
DISP_XR_BOWSER_PYTHON=$PWD/.pixi/envs/test/bin/python \
    pixi run -e test pytest tests/test_geozarr.py   # in disp-xr
```
