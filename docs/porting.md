# Porting plan: opera_utils functions into disp-xr

disp-xr does not depend on `opera_utils` (PRD decision, Sep 30, 2026). The DISP product logic it needs is ported into disp-xr modules, with attribution, and disp-xr owns and tests it from then on.

- **Source:** `opera-utils` commit `dff4b45` (Sep 15, 2026), checkout at `00_tools/src/opera-utils`.
- **Status of this spec:** written by two readers of the source. The independent verification pass did not run; spot checks of the license, the filename regex, the module locations and upstream bugs 1–3 passed. Confirm every line range when porting.
- **Tasks:** each row names the task in [todo.md](todo.md) that owns the port.

## License and attribution

opera_utils is dual-licensed, BSD-3-Clause or Apache-2.0, at the user's choice (Copyright (c) 2022 California Institute of Technology, U.S. Government sponsorship acknowledged). disp-xr is BSD-3-Clause, so the ports are taken under BSD-3-Clause.

1. Ship the opera_utils BSD-3-Clause text as `LICENSES/opera-utils-BSD-3-Clause.txt`, and list it under `license-files` in `pyproject.toml` so it goes into the sdist, wheel and conda package.
2. Give each ported module a header: `Portions adapted from opera-utils (https://github.com/opera-adt/opera-utils, commit dff4b45, <source path>), Copyright (c) 2022 California Institute of Technology, BSD-3-Clause; modified for disp-xr.`
3. Do not use the Caltech or JPL names to endorse disp-xr.
4. Ported tests carry the same header.
5. `plate_motion.py` follows MintPy code, and MintPy is GPL-3.0. Do not copy it. Reimplement plate motion from the published math (ITRF2014 plate motion model, Altamimi et al. 2017) and cite the paper.
6. A CI rule forbids `import opera_utils` (ruff `TID251` banned-api), so the dependency cannot come back.

## What is ported, and where

Verdicts: **port** = copy the logic with small changes; **simplify** = port the core and drop parts disp-xr does not need; **rewrite** = same behavior, new implementation on xarray.

### Product parsing and search (T4.5, T5)

| opera_utils source | Verdict | disp-xr target | Changes |
| --- | --- | --- | --- |
| `constants.DISP_FILE_REGEX` | Port | `product.py`, one regex per `ProductSpec` | The upstream regex hard-codes `IW`, a 5-digit frame and `VV\|HH`; confirm the NISAR pattern on a real granule |
| `disp/_product.py` `DispProduct`, `from_filename` | Simplify | `product.py` (`DispGranule` frozen dataclass) | Drop the grid properties that read the S1 burst database; the grid comes from the opened file |
| `disp/_product.py` `DispProductStack` (dedupe, sort) | Rewrite | `product.py` (`find_products` returns a DataFrame) | Compare versions with `packaging.version`, not strings |
| `disp/_enums.py` dataset enums | Rewrite | `product.py` (`ProductSpec` layer lists) | Layer names per product; mark NISAR `ionospheric_delay` as already applied |
| `disp/_search.py` `search`, `_cmr.py` `fetch_cmr_pages`, `get_download_url`, `DispProduct.from_umm`, `constants.UrlType` | Port | `search.py` | CMR short name from `ProductSpec` (S1 and NISAR); logger named `disp_xr` |
| `disp/_download.py` subset download | Simplify, opt-in | `search.py` (`aws` extra) | Clip a bbox that leaves the frame instead of raising |

### Remote open and credentials (T6)

| opera_utils source | Verdict | disp-xr target | Changes |
| --- | --- | --- | --- |
| `_remote.py` `open_file`, `open_h5`, `get_s3_fs`, `get_https_fs`, `is_remote_url`, `get_url_str` | Port | `_remote.py` (private, `aws` extra) | Keep `cache_type="first"` and the 4 MiB HDF5 page size, which match the DISP file layout |
| `credentials.py` (Earthdata login, temporary S3 credentials, TTL cache) | Port | `_credentials.py` | Expose a refreshable `get_credentials` callable; Earthdata S3 credentials expire after about 1 hour |
| `disp/_utils.py` `_clamp_chunk_dict` | Port | `_chunks.py` | Shared by open, stack and export |

### Ministack chaining (T7)

| opera_utils source | Verdict | disp-xr target | Changes |
| --- | --- | --- | --- |
| `disp/_rebase.py` `NaNPolicy` | Port | `stack.py` (public) | — |
| `disp/_rebase.py` `rebase_timeseries` | Port | `stack.py` (private kernel) | Keep the numerics so upstream tests apply; keep the input dtype (upstream forces float32); check that each crossover epoch exists |
| `disp/_rebase.py` `create_rebased_displacement` | Rewrite | `stack.py` (`stack_disp`) | `xr.apply_ufunc(dask="parallelized")` over `time`; read reference times from `ds.reference_time`; fix the zero epoch label |

The `timeseries_fitting` branch already holds unattributed copies of these in `stack.py`. T4.0 merges the branch; T7 keeps one implementation and deletes `rebase_disp`.

### Quality and reference (T10, T11)

| opera_utils source | Verdict | disp-xr target | Changes |
| --- | --- | --- | --- |
| `disp/_reformat.py` `combine_quality_masks` | Port | `quality.py` | Backs the `mask()` presets |
| `disp/mintpy.py` `create_reliability_mask` | Simplify | `quality.py` (`reliability`) | Return an `xr.DataArray`; no file output |
| Harmonic-mean temporal coherence (`_reformat.py`, `rebase_reference.py`) | Rewrite | `quality.py` (`quality_summary`) | One value per ministack through `groupby("reference_time")` |
| `disp/_enums.py` `ReferenceMethod` | Port | `reference.py` | Methods: none, point, median, border, high_coherence |
| `disp/_reference.py` `get_reference_values` | Simplify | `reference.py` | Bounds-check the point; require a mask for high_coherence; median honors the good-pixel mask |
| `disp/_reference.py` lon/lat to row/col, `_product.py` `OutOfBoundsError` | Rewrite | `geometry.py` | Through `pyproj` and `ds.rio.transform()` |
| `disp/_reference.py` `_get_border_pixels` | Rewrite | `reference.py` | No `dropna` on a stacked dask array (it forces a compute) |
| `disp/rebase_reference.py` `find_reference_point` | Simplify | `reference.py` (`method="auto"`) | Deterministic pick by center of mass of high-quality candidates |

### Geometry (T12)

| opera_utils source | Verdict | disp-xr target | Changes |
| --- | --- | --- | --- |
| `geometry.py` `stitch_geometry_layers`, `Layer`, nodata table | Rewrite | `geometry/static.py` | Merge the CSLC-STATIC bursts with rioxarray instead of the GDAL stitcher; this removes the only live `opera_utils` import (`static_layers.py:11`) |
| `geometry.py` LOS up and 3-band LOS helpers | Simplify | `geometry/static.py` | Computed in `open_static` |
| `disp/create_incidence_range.py` incidence and slant range | Rewrite | `geometry.py` | Array functions on the LOS dataset; wavelength and orbit height from `ProductSpec` |

### Export (T14)

| opera_utils source | Verdict | disp-xr target | Changes |
| --- | --- | --- | --- |
| `disp/_reformat.py` `_get_zarr_encoding`, `_to_shard_dict` | Simplify | `export/zarr.py` | Presets `timeseries` and `maps`; shards as a multiple of chunks with a divisibility check |
| `disp/_utils.py` `_get_netcdf_encoding` | Port | `export/netcdf.py` | — |
| `disp/_utils.py` `round_mantissa` | Port, opt-in | `export/_encoding.py` | Lossy; `keep_bits=None` by default so Cal/Val outputs stay exact |
| `disp/mintpy.py` `disp_nc_to_mintpy`, metadata, HDF5 writer, static layers | Rewrite | `export/mintpy.py` (`mintpy` extra) | Take an `xr.Dataset`; platform, wavelength and height from `ProductSpec`; write chunk by chunk |

### Plate motion (T13, new)

| opera_utils source | Verdict | disp-xr target | Changes |
| --- | --- | --- | --- |
| `disp/plate_motion.py` | Rewrite from the paper | `corrections.py` | Not copied (GPL lineage); see upstream bugs 7–8 |

### Not ported

`reformat_stack` (its steps become the `disp_xr stack` CLI command), the GeoTIFF rebase loop and stack reader/writer classes in `rebase_reference.py`, `burst_frame_db` lookups, `create_virtual_stack` (HDF5 VDS; VirtualiZarr covers this, PRD C3), `last_per_ministack`, `flatten`, and the rasterio-profile helpers.

## Upstream bugs to fix in the ports

Each gets a regression test in disp-xr.

1. `DispProductStack` sorts versions as strings, so `1.10` ranks below `1.9`.
2. `search()` drops the frame filter when `product_version=None`.
3. `create_rebased_displacement(add_reference_time=True)` labels the zero epoch with the first secondary time, giving a duplicate time coordinate.
4. `rebase_timeseries` accumulates in float32 whatever the input dtype, and never checks that the crossover epoch exists.
5. `get_reference_values`: the point method has no bounds check (negative indices wrap), the median ignores `good_pixel_mask`, and high_coherence with no mask silently uses every pixel.
6. `mintpy._write_hdf5` leaves compression off for every dataset after the first; the metadata hard-codes Sentinel-1 platform, wavelength and height, and the velocity step loads the full cube.
7. `plate_motion.geodetic_to_ecef` passes radians to a transformer that expects degrees.
8. `plate_motion.run` swaps latitude and longitude when unpacking the mesh grid.

## Practices kept from opera_utils

- Processing order: corrections, then quality mask (both per product, before chaining), then chain, then spatial reference. Masked NaNs then propagate through the chain.
- `NaNPolicy.propagate` is the default.
- Chain with the whole time axis in one chunk and spatial chunks clamped to the array shape.
- Use `ds.rio.transform()`, not the `GeoTransform` attribute, after subsetting.
- Read `/identification` and `/metadata` with h5py; xarray cannot decode their byte strings.
- Zarr v3 output with inner chunks plus shards, dask rechunked to the shard shape before writing, and uncompressed coordinates.

## Tests to port

All are synthetic and need no network: `test_disp_rebase.py`, `test_disp_reference.py`, the `combine_quality_masks` and encoding parts of `test_disp_reformat.py`, `test_disp_utils.py` (mantissa rounding, chunk clamping, NetCDF encoding), the filename and dedupe parts of `test_disp_product.py`, `test_disp_search.py` (mocked requests), `test_disp_remote.py` and `test_credentials.py` (monkeypatched).

`mintpy.py`, `plate_motion.py`, `create_incidence_range.py` and `stitch_geometry_layers` have no upstream tests; the ports need new ones.

## DISP-NISAR differences the ports must handle

| Item | DISP-S1 | DISP-NISAR |
| --- | --- | --- |
| Filename mode and polarization | `IW`, `VV` | To confirm on a real granule |
| CMR short name | `OPERA_L3_DISP-S1_V1` | To confirm |
| Static geometry source | CSLC-STATIC bursts | DISP-NI-STATIC |
| Wavelength, orbit height | 0.0555 m, about 693 km | about 0.238 m, about 747 km |
| `/corrections/ionospheric_delay` | Separate layer, not applied | Already applied to displacement |
| ASF S3 credential endpoint | OPERA endpoint | Unverified |

Layer names for the 11 displacement and quality layers are the same in both products.
