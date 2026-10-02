# Existing code review: what to keep, refactor or drop

This reviews every module in `src/disp_xr` (2,110 lines, Sep 30, 2026) against [specs.md](specs.md) and the tasks in [todo.md](todo.md).

## Summary

About half of the current code overlaps functions in `opera_utils.disp`: filename parsing, deduplication, search, ministack rebasing, re-referencing, quality-mask combination and MintPy export. disp-xr does not depend on opera_utils (PRD decision), so the opera_utils versions, which are better tested than the local copies, are ported into disp-xr with attribution and replace the buggy local code. [porting.md](porting.md) lists every function, its target module and the upstream bugs to fix on the way. disp-xr's own value is the xarray layer (accessor, lazy pipelines) and the Cal/Val science in S1–S4. Only part of that exists, and only on the unmerged `timeseries_fitting` branch (OLS trajectory fits and residual statistics; see below).

Of the rest:

- **Keep and adapt:** static layers, logging and coordinate helpers.
- **Rewrite on rioxarray:** the rasterio file I/O.
- **Drop:** one module, `cli/geometry.py` (an empty click group).
- **Keep as the search port:** `download.py`, already a copy of the opera_utils search code; it becomes `search.py`.
- **Fix before reuse:** 7 bugs, listed below.

| Module | Lines | Maps to PRD | Verdict | Where it goes |
| --- | --- | --- | --- | --- |
| `product.py` | 123 | F1 | Replace with a port of opera_utils `DispProduct` / `DispProductStack` + DataFrame view | `product.py` (T5) |
| `stack.py` | 87 | F2, F3 | Rewrite: open via xarray, chain via the ported `create_rebased_displacement` | `open.py`, `stack.py` (T6, T7) |
| `quality_metrics.py` | 179 | F4 | Keep the ideas, rewrite lazy | `quality.py` (T10) |
| `utils.py` | 236 | F5, F7, misc | Split; keep coordinate helpers, rewrite reference picker | `geometry.py`, `reference.py` (T4, T11) |
| `static_layers.py` | 321 | F7 | Keep logic, swap file I/O for rioxarray, return xarray | `geometry.py` (T12) |
| `io.py` | 206 | F12 | Replace with rioxarray (`rio.to_raster`, `rio.reproject_match`) | `export/raster.py` (T14) |
| `mintpy.py` | 516 | F12 | Replace with a port of opera_utils `disp_nc_to_mintpy` adapted to `xr.Dataset`; drop the parallel writer | `export/mintpy.py` (T14) |
| `download.py` | 245 | Search (ported) | Keep as the disp-xr search port: rename the logger, add attribution, add the NISAR short name | `search.py` (T4.5) |
| `cli/geometry.py` | 28 | F14 | Drop; an empty click group, and the entry point does not resolve | `cli/` (T15) |
| `log/` | 169 | Infrastructure | Keep as is | `log/` |

## Overlap with `opera_utils.disp`

Checked against `00_tools/src/opera-utils` (commit `dff4b45`, Sep 15, 2026).

| Need (PRD) | disp-xr today | `opera_utils.disp` already has | Decision (port, no dependency) |
| --- | --- | --- | --- |
| F1 parse filenames, dedupe | `product.get_disp_info` (buggy dedupe, `==` version filter) | `DispProduct.from_filename`, `DispProductStack` (dedupe by version, then generation time; regex covers `S1` and `NI`) | Port to `product.py`; add a DataFrame view and ministack grouping; compare versions with `packaging.version` |
| Search CMR | `download.search` (copied code, logger still named `opera_utils`, S1 only) | `search` | Keep `download.search` as the port (`search.py`); fix the logger, add NISAR |
| Remote (S3/HTTPS) open | none | `open_file`, `open_h5` | Port `open_file`, `open_h5` and the Earthdata credential helpers (T6) |
| F3 ministack chaining | `stack.combine_disp_product` (buggy with `start_date`) | `create_rebased_displacement` (lazy, `NaNPolicy` propagate or omit), `rebase_timeseries` | Port to `stack.py`; disp-xr handles open, coords and the zero first epoch |
| F5 re-reference | `utils.find_reference_point` (random choice) | `get_reference_values` with methods none, point, median, border, high_coherence (private `_reference.py`) | Port `get_reference_values` to `reference.py`; add deterministic "auto" scoring and area polygons |
| F4 combine masks | none | `combine_quality_masks`, `create_reliability_mask` | Port to `quality.py`; `mask()` calls them |
| Corrections names | none | `CorrectionDataset` enum: `/corrections/solid_earth_tide`, `/corrections/ionospheric_delay` | Copy the names into `ProductSpec` (T5, T13) |
| F12 MintPy | `mintpy.py` (516 lines, bugs) | `disp_nc_to_mintpy`, `create_static_layers` | Port to `export/mintpy.py`, taking an `xr.Dataset` |
| F7 incidence | `static_layers.los_unit2inc_azimuth` | `create_incidence_angle_raster` | Keep the disp-xr version (array based); compare outputs once |

Risk: ported code can drift from opera_utils as the DISP product changes. Each ported file records the source commit (`dff4b45`) in its header, the upstream tests are ported with it, and the release checklist includes a diff of the ported functions against current opera_utils. `_reference.py` and `_reformat.py` are private upstream modules, so there was no stable API to depend on in any case.

## Bugs found

| # | Location | Problem | Effect | Fix in |
| --- | --- | --- | --- | --- |
| 1 | `product.py:74` | Filters `version == min_version`; when `min_version` is None it picks the max string version | Drops all newer versions when `min_version` is set; string comparison misorders `1.10` vs `1.9` | Fixed in T5 (`find_products`) |
| 2 | `product.py:116` | `_find_duplicates` appends `input_df.iloc[ix]` for the row it keeps, using a label as a position | The "removed duplicates" list holds the kept rows, from the wrong positions once the index is not a range | T5 |
| 3 | `stack.py:75–79` | First epoch is `min(start_date)` over all products even after `start_date` filtering; the zero epoch uses `full_like(..., 0)` for every variable | With `start_date`, the zero epoch has the wrong date; quality layers (coherence, masks) get fake zeros at that epoch | Fixed in T7 (`stack_disp`) |
| 4 | `mintpy.py:487` | `if (ref_lalo is None) & (ref_yx is not None): ... else: raise ValueError` | Passing `ref_lalo` always raises | T14 (replace) |
| 5 | `mintpy.py:516` | `except Exception as e: print(e)` | All errors are swallowed; the function "succeeds" with no output | T14 (replace) |
| 6 | `mintpy.py:84, 88` | `RLOOkS` typo; `AZIMUTH_PIXEL_SIZE = 14.1` hardcoded | Wrong MintPy metadata key; NISAR would be wrong | T14 (replace) |
| 7 | `utils.py:229` | `random.choice` for the reference pixel | Non-reproducible results run to run | T11 |
| 8 | `product.py:73` | `disp_df.version.unique().max()`; under pandas 3 `unique()` returns an `ArrowStringArray`, which has no `max` | `get_disp_info` raises `AttributeError` whenever a folder holds more than one product version, including the duplicate case | Fixed in T5 (`find_products`) |
| 9 | `search.py` (was `download.py`), `search()` | `params["attribute[]"]` is attached only if the filter list is non-empty before the frame filter is appended | With `product_version=None` the frame filter never reaches CMR, so the search returns every frame | Fixed in T4.5 |

Found while writing the characterization tests (T2.6, 2026-10-01):

- Bug 8 above.
- Bug 2 is not visible to callers: `get_disp_info` discards the wrong list it builds, and the rows it keeps are correct. It has no test.
- The chained stack from `combine_disp_product` has one time chunk per epoch. `chunks={"time": -1}` applies to each single-epoch file, so per-pixel work needs a rechunk (T6.5, T7).
- `combine_disp_product` relies on xarray's `data_vars` default for `open_mfdataset`, which xarray warns will change (T6.1).
- `utils.find_reference_point` emits an all-NaN warning before it raises when there is no candidate.

Each bug has a strict `xfail` test in `tests/` that asserts the correct behavior (bug 7 is open; bugs 1, 3, 8 and 9 are fixed and their tests are regression tests).

Smaller issues: `utils.py` imports `yaml` and `h5py`, but neither PyYAML nor h5py is in `environment.yml` (only `types-PyYAML`). `get_burst_ids` is annotated as returning a tuple but returns a list. `quality_metrics.get_stack_stat` calls `.compute()` and its error messages say "Dataset" for a DataArray check. The `get_disp_info` docstring says it returns three values but it returns one DataFrame. `pyproject.toml` points the entry point at `disp_xr.cli:cli_app`, but `cli/` has no `__init__.py`.

## Module by module

### `product.py`: replace (T5)

- **Useful:** the idea of a DataFrame view of granules with `date1`, `date2` and `date12`, which the notebooks use, and the ministack grouping by reference date (`_get_reference_dates`).
- **Not useful:** the hand-written parsing (column typo `polarizarion`, `iterrows` loops), the version filter (bug 1) and the dedupe (bug 2).
- **Refactor:** `find_products(path)` uses the ported parser and dedupe in `disp_xr.product`, then returns one DataFrame row per granule plus `path`, sorted by secondary time. Keep a `get_disp_info` shim that maps to the new columns (T4.3) because the notebooks call it.

### `stack.py`: rewrite (T6, T7)

- **Useful:** default chunks `{"time": -1, "x": 512, "y": 512}`, `start_date` / `end_date` filtering, and the `@log_runtime` decorator.
- **Not useful:** opening one `open_mfdataset` per ministack and chaining with in-place `+=` in a Python loop, and the zero first epoch built with `full_like` (bug 3).
- **Refactor:** T6 opens all granules once with a `preprocess` hook that adds `time` and `reference_time`. T7 calls `create_rebased_displacement(ds.displacement, ds.reference_time, nan_policy=...)` and adds the zero epoch to displacement only. Quality layers stay per epoch and are not rebased. Keep `combine_disp_product` as a deprecated shim.

### `quality_metrics.py`: rewrite lazily (T10)

- **Useful:** per-pixel statistics over time (`get_stack_stat`) and the percentage of epochs equal to a label (`get_value_percentage`), for example the percentage of time a pixel is masked or is a PS. Notebooks use both for the reference-point choice.
- **Not useful:** the `map_blocks` template trick (`get_template`, `get_counts`). It is equivalent to `(da == value).mean("time") * 100`, which xarray does lazily in one line.
- **Refactor:** `quality_summary(ds)` returns a lazy `xr.Dataset` with `pct_masked`, `pct_ps`, mean coherence and similarity. `get_stack_stat` becomes `da.reduce` in the accessor, with no `.compute()`.

### `utils.py`: split (T4, T11, T12)

| Function | Verdict | Destination |
| --- | --- | --- |
| `latlon_to_utm`, `utm_to_latlon` | Keep (used in notebooks), generalize to any CRS | `geometry.py` |
| `get_extent` | Replace with `ds.rio.bounds()` | Shim only |
| `get_chunks_indices` | Drop; only the MintPy writer uses it | — |
| `get_burst_ids` | Keep; S1-specific (reads `pge_runconfig`) and needed for CSLC-STATIC download | `geometry.py`, private |
| `find_reference_point` | Rewrite deterministic (bug 7) as the "auto" method next to the ported reference methods | `reference.py` |

### `static_layers.py`: keep logic, modernize I/O (T12)

- **Useful:** the whole workflow. It downloads CSLC-STATIC by burst id via `asf_search`, stitches with `opera_utils.geometry.stitch_geometry_layers` (the only live opera_utils import in the repo, replaced by a rioxarray merge in T12.1), derives `los_up = sqrt(1 - e² - n²)`, and converts LOS to incidence and azimuth. It also gets the GLO-30 DEM (ellipsoidal heights) via `dem_stitcher` and an ESA WorldCover 2021 water mask (class 80) via `tile_mate`. Nothing else in the stack provides this for DISP.
- **Not useful:** the round-trips through GeoTIFF files with `open_image`, `write_geotiff` and `reproject_raster`; the bounds-to-geographic code repeated in two functions; and the in-place overwrite of `los_east.tif`.
- **Refactor:** return an `xr.Dataset` aligned with `rio.reproject_match(ds)`. Cache downloads on disk. Factor the bounds-in-EPSG:4326 helper. Put `asf_search`, `dem_stitcher` and `tile_mate` behind the `aws` extra. Note that CSLC-STATIC is S1 only, so DISP-NISAR needs its own geometry source, DISP-NI-STATIC (T12.5).

### `io.py`: replace (T14)

- **Useful:** nothing that rioxarray does not already do. `get_geospatial_info` reads CRS and transform via the GDAL NETCDF driver; after T6 they come from `ds.rio.crs` and `ds.rio.transform()`.
- **Refactor:** delete after `static_layers` and `mintpy` stop using it. Keep `write_geotiff` and `reproject_raster` as private helpers only if an array-only path remains.

### `mintpy.py`: replace (T14)

- **Useful:** the MintPy metadata mapping in `get_metadata` (X_FIRST, Y_STEP, EPSG, CENTER_LINE_UTC and so on), as a reference to check the ported writer against, and `write_geometry` (geometry.h5 from incidence, azimuth, DEM and water mask).
- **Not useful:** the `ThreadPoolExecutor` writer that calls `.compute()` per chunk inside threads; the dask `Client` started inside a library function; and bugs 4–6.
- **Refactor:** `to_mintpy(ds, static, out_dir)` uses the writer ported from opera_utils `disp_nc_to_mintpy`, adapted to an `xr.Dataset`, with platform, wavelength and orbit height taken from `ProductSpec`. It writes `timeseries.h5` with h5py chunk by chunk, with no threads. Check it once against the old output (T14.3). Move it to the `mintpy` extra.

### `download.py`: keep as the search port (T4.5)

A near-verbatim copy of `opera_utils.disp` search code: the same regex and functions, with no attribution, and the logger is still named `opera_utils`. It is S1 only (`short_name = OPERA_L3_DISP-S1_V1`). It becomes `search.py`: add the attribution header, name the logger `disp_xr`, take the CMR short name from `ProductSpec` (S1 and NISAR), and fix the dropped frame filter (porting.md, upstream bug 2). `download.py` stays one release as a deprecated shim.

### `cli/geometry.py`: drop and rebuild (T15)

An empty click group named `disp_geometry` with all commands commented out. Replace it with the T15 CLI package.

### `log/`: keep

`setup_logging` (JSON config, optional file handler), `log_runtime` and `json_formatter` are fine. One small change: `log_runtime` logs under `disp_xr.log.logging`; log under the decorated function's module instead.

## Upstream branch `timeseries_fitting` (not on `main`)

`origin/timeseries_fitting` (last commit 2025-06-25, "ignore ipynb with precommit") adds `timeseries/` (1,554 lines) and `stack.rebase_disp` / `create_rebased_displacement`. The local checkouts `01_OPERA/STAKEHOLDER/disp-xr` and `01_OPERA/VLM/disp-xr` sit on this branch, and the stakeholder notebooks (Brisbane, VLM) import from it. It is the most useful code in the repo for S1 and S4, so it is merged first (T4.0).

| File | What it does | Verdict |
| --- | --- | --- |
| `_design_matrix.py` | `DesignMatrixBuilder`: polynomial, periodic, step, exponential and logarithmic terms. Coefficient names encode the model, and `infer_parameters_from_names` rebuilds it | Keep; align term names with geepers (T20.1) |
| `_numpy_solver.py` | `LeastSquaresSolver` (rank check by SVD, QR solve) and `lscov_2d`, with standard errors | Keep as the OLS core of T20 |
| `_jax_solver.py` | Same fit on JAX, with missing-data handling | Keep as an optional backend |
| `estimate_fit.py` | `fit_timeseries_block` (`map_blocks` + template), `get_predicted`, `sincos_to_amplitude_phase` | Keep the API; move to `apply_ufunc` and a lazy float32 model (T20.3, T20.3b) |
| `_stats_analysis.py` | Residual mean, std, skewness, kurtosis, lag autocorrelation, and Anderson-Darling via `apply_ufunc(vectorize=True)` (commented "problematic") | Keep the statistics; stream them and vectorize Anderson-Darling (T32.6b) |
| `_constants.py`, `_validate.py`, `_datetime.py` | `FitResult`, critical values, input checks, decimal-year helpers | Keep |

Issues: `timeseries/__init__.py` puts function objects in `__all__` instead of names (breaks `from disp_xr.timeseries import *` and the docs). `get_residuals_analysis` leaves out the normality test. Storing `get_predicted` output doubles the storage of the displacement (noted in the Brisbane notebook), so the model is computed on the fly.

The Brisbane notebook also does things that are not in the branch and become disp-xr features: z-scores, p-values and confidence intervals of coefficients (T32.6c); EOF, rotated EOF and complex EOF with `xeofs` (T32.6d); and Zarr export with a Blosc zstd compressor (replaced by Zarr v3 presets, T14.2).

## Notebooks

`notebooks/workflow.ipynb` (36 cells) and `interactive_workflow.ipynb` (14 cells) import `download`, `io`, `product`, `quality_metrics`, `static_layers`, `stack.combine_disp_product` and `utils.latlon_to_utm` / `get_extent`. They are the de facto API today, so:

- the T4.3 shims must cover exactly these names, and
- the notebooks are rewritten as T16 gallery examples against the new API, then deleted from `notebooks/`.

`notebooks/validation/Compare_disp_versions.ipynb` becomes a gallery example for the product-team use case (compare DISP versions).

## Changes to the plan

These updates are applied in [todo.md](todo.md):

- **T1:** add `h5py`, `pyyaml` and `packaging` to the core dependencies; remove `opera_utils` from `environment.yml`; ship the opera_utils BSD-3-Clause notice.
- **T2.6:** characterization tests pin the correct behavior for bugs 1–3, not today's buggy outputs, and mark the fixes.
- **T4.5:** `download.py` becomes the ported search module `search.py`.
- **T4.7:** attribution files, per-file headers and a CI rule that forbids `import opera_utils`.
- **T5:** port `DispProduct` / `DispProductStack` into `product.py`; `ProductSpec` adds layer names, units and NISAR differences.
- **T6:** port `open_file`, `open_h5` and the Earthdata credential helpers.
- **T7:** port `rebase_timeseries` and reimplement `create_rebased_displacement` on `apply_ufunc`.
- **T10, T11:** port `combine_quality_masks`, `create_reliability_mask` and the reference methods; add deterministic "auto" and area polygons.
- **T12:** replace `stitch_geometry_layers` with a rioxarray merge; add the NISAR geometry source.
- **T14:** port the MintPy writer and the Zarr/NetCDF encodings; drop the threaded writer.
- **New T4.0:** merge the `timeseries_fitting` branch before the refactor; T20 and T32 build on it.
