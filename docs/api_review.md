# API review for v0.2

The v0.2 gate (plan task T18) freezes the core names: renaming is cheap now
and expensive once the analysis features (M2) are built on them. This page
lists the public API as of 2026-10-02, the decisions already made, and the
open questions for Cal/Val users. Each open question becomes one GitHub issue
during the review (T18.2); the outcome goes into [api_policy.md](api_policy.md).

## Public API

Importing `disp_xr as dxr` gives these names. Every Dataset and DataArray also
has the `.disp` accessor.

| Area | Functions | Accessor |
|---|---|---|
| Find and open | `find_products`, `open_disp`, `open_disp_tree`, `open_static` | |
| Corrections | `correct`, `remove_plate_motion` | `.disp.correct`, `.disp.remove_plate_motion` |
| Quality | `mask`, `valid_mask`, `quality_summary` | `.disp.mask`, `.disp.quality_summary` |
| Chaining | `stack_disp`, `NaNPolicy` | `.disp.stack` |
| Reference | `rereference` | `.disp.rereference` |
| Units, geometry, checks | `to_units`, `enu_to_los`, `validate_disp` | `.disp.to_units`, `.disp.grid`, `.disp.validate`, `.disp.history` |
| Chunks and settings | `rechunk`, `config` | `.disp.rechunk` |
| Export | `to_zarr`, `to_netcdf`, `to_geotiff`, `to_mintpy` | |
| Icechunk and virtual (`cloud`) | `to_icechunk`, `open_icechunk`, `build_virtual_frame`, `open_virtual`, `validate_virtual` | |
| Plot (`plot`) | `disp_xr.plot.plot_map`, `plot_point`, `diverging_cmap` | `.disp.plot_map`, `.disp.plot_point` |
| Search (`aws`) | `disp_xr.search.search` | |
| Synthetic data | `disp_xr.tutorial.make_disp_stack` | |
| Command line | `disp_xr info`, `stack`, `export`, `geometry`, `search`, `virtual` | |

Deprecated and kept with a `DeprecationWarning` until T38: `get_disp_info`,
`combine_disp_product`, `rebase_disp`, and the old module names `download`,
`io`, `mintpy`, `quality_metrics`, `static_layers`, `utils`. They are no longer
listed in `disp_xr.__all__`.

## Settled

These follow from the PRD or from measurements, and are not up for review
unless users report a problem.

- Functions take a Dataset and return a new one; the accessor only delegates,
  so `ds.disp.mask()` and `dxr.mask(ds)` are the same operation.
- Displacement stays in meters, positive toward the satellite; conversion is
  explicit (`to_units`).
- Processing order is fixed: corrections and the quality mask per granule,
  then chaining, then the spatial reference.
- Data opens lazily in the file layout (`chunks="native"`); operations get the
  chunks they need.
- Every output records the disp-xr version, input granules and history.

## Open questions

### 1. `stack_disp` or `chain`?

The PRD and the docs call the operation chaining, the accessor calls it
`.disp.stack()`, and the function is `stack_disp`. "Stack" also means the
whole `(time, y, x)` dataset. Proposal: `dxr.chain(ds)` and `.disp.chain()`,
with `stack_disp` and `.disp.stack` kept as deprecated aliases.

### 2. Default corrections

`correct(ds)` subtracts the ionosphere and the solid-earth tide by default, so
on DISP-NISAR (ionosphere already applied) the default call raises. The
command line already defaults to "every correction the product has not
applied". Proposal: make that the library default too (`which="product"`).

### 3. One way to overwrite outputs

`to_zarr` and `to_icechunk` take `mode="w-" | "w"`, `to_mintpy` takes
`overwrite=`, and `to_netcdf` and `to_geotiff` overwrite silently. Proposal:
`overwrite: bool = False` on every writer, refusing an existing output unless
it is set (`to_icechunk` keeps `mode` for its append semantics).

### 4. Where `search` lives

The function is `disp_xr.search.search`, a name the module shadows, so it is
not in the top-level namespace. Proposal: `dxr.search_granules(...)`.

### 5. `nan_policy="omit"` at a NaN crossover

With `omit`, a NaN crossover currently resets the offset of later epochs to
zero, discarding what accumulated before. The documented intent ("no motion
over the missing interval") suggests carrying the previous offset instead.
Which do users expect?

### 6. Default of `rechunk`

`rechunk(ds)` defaults to `"timeseries"` while `open_disp` defaults to
`"native"`. Should `rechunk` require the layout (no default), so the call
always says what it does?

### 7. Masks before or after chaining in the accessor chain

`mask` works before or after chaining; the PRD fixes "before". Should
`.disp.mask()` refuse chained data (as `correct` does) to enforce the order?

## Already fixed during the review preparation

- `to_geotiff` no longer shows an internal parameter in its signature.
- Deprecated names left `disp_xr.__all__`.
