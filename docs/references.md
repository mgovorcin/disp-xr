# References and practices

Practices disp-xr follows, where each one comes from, and which task implements it. The PRD ([specs.md](specs.md)) states the rules; this file holds the evidence and the detail.

How the list was built: one reader per repository read the source, docs and examples, and a second reader re-opened every cited file and tried to refute each claim. Only confirmed or corrected practices are listed. Where the reference code has a bug or a gap, it is listed under "Do not copy".

Adoption levels: **core** = disp-xr does this by default; **opt-in** = behind the `cloud` extra or a keyword; **example** = shown in the docs gallery only.

## 1. Earth-mover repositories

Commits read: icechunk `468004a` (Sep 29, 2026), icechunk-nasa `b23165f`, icechunk-multiscales-demo `f7d058d`, topozarr `3445f9b`, serverless-datacube-demo `c71264c`, cloud-tensor-storage-benchmarks `9940ee6`, workshop-cng-2025-zarr `4326e6e`, xpublish-tiles `7372993`.

### 1.1 Zarr v3 writing (PRD C1)

| Practice | Level | Source | Task |
| --- | --- | --- | --- |
| Size dask chunks by bytes, 50 to 200 MB, not by a fixed shape. For a 300-epoch float32 stack this gives 256 x 256 in space for the whole time axis. The workshop does not use sharding; with sharding this size is the shard, and the read chunk inside it is smaller (section 2) | Core | workshop-cng-2025-zarr `appendix.ipynb` | T6.5, T14.2 |
| Offer named presets: `timeseries` (time -1) and `maps` (time 1); rechunk before `apply_ufunc` | Core | workshop-cng-2025-zarr part 2 | T6.5 |
| Call `drop_encoding()` before writing, then build the encoding explicitly | Core | workshop-cng-2025-zarr; topozarr `build_demo_data.py` | T14.2 |
| Shards are a whole multiple of chunks, checked; dask chunks are aligned to the shard shape before writing | Core | topozarr `chunking.py`, `coarsen.py` | T14.2 |
| One chunk grid for every variable in a store | Core | Corrects topozarr, whose per-dtype chunk sizes misalign mixed-dtype datasets | T14.2 |
| Coordinates (`x`, `y`, `time`, `reference_time`) as single chunks | Core | serverless-datacube-demo `lib.py` | T14.2 |
| Pass `consolidated=False` for Icechunk stores | Opt-in | workshop-cng-2025-zarr part 1; icechunk docs | T14.2b |
| Test that `_FillValue` and CRS survive a write | Core | icechunk-nasa (fill values had to be fixed by hand for Zarr v3) | T14.2a |
| Accept any zarr `Store` or a URL; mention obstore as an alternative to s3fs | Example | topozarr; cloud-tensor-storage-benchmarks | T14.2 |

### 1.2 CF and GeoZarr metadata, multiscales (PRD C4, NFR Metadata)

| Practice | Level | Source | Task |
| --- | --- | --- | --- |
| Every dataset carries a `spatial_ref` coordinate with `crs_wkt` and `GeoTransform`, and each variable a `grid_mapping` | Core | xpublish-tiles `datasets.py` | T6.2, T14.2a |
| Write GeoZarr `spatial:` and `proj:` attributes next to the CF ones, from `ds.rio.transform()` and `ds.rio.crs` | Core | topozarr `metadata.py`; xpublish-tiles | T14.2a |
| `decode_coords="all"` moves `grid_mapping` from attrs to encoding; code must check both | Core | xpublish-tiles `utils.py` | T6.2 |
| Mask and label layers carry `flag_values` and `flag_meanings` | Core | xpublish-tiles | T14.2a |
| Multiscales layout on the common parent, levels named `/0`, `/1`, ..., each with `derived_from`, a resampling method and its own transform | Core | topozarr `coarsen.py`, `metadata.py`; icechunk-multiscales-demo `lib.py` | T27.7 |
| Levels as sibling groups, so each keeps `y`/`x` and a DataTree opens without alignment errors | Core | icechunk-multiscales-demo `config.py` | T27.7 |
| Each level is self-contained: full CRS and CF attributes | Core | Corrects icechunk-multiscales-demo, which drops the CRS on overview levels | T27.7 |
| Read levels from the `layout` metadata, never from group names | Core | icechunk-multiscales-demo | T27.7 |
| Keep the convention ids, versions and schema URLs in one module, record the version in provenance, and test conformance. The conventions are still at pilot maturity (section 2), so the writer must be easy to migrate | Core | topozarr `test_specs.py` | T27.7 |
| Small levels (a frame at 1 km is about 250 x 300 pixels) use one spatial chunk over all of time | Core | icechunk-multiscales-demo | T27.7 |
| Keep the 30 m level virtual and materialize only the overviews | Opt-in | topozarr `build_demo_data.py` | T27.7, T6.5b |

Downsampling rules that came out of this reading:

- Odd integer factors (33, 67, 333) keep cell centers on 30 m pixel centers. The demos use factor 2 only.
- A weighted median is not composable; compute each level from 30 m. A mean is composable only with stored `sum(w)`, and with NaNs a mean of block means is not the mean of the pixels.
- Mask and label layers need mode or "any", not a mean.
- Build overviews after chaining and re-referencing.

### 1.3 Icechunk (PRD C2, opt-in)

| Practice | Level | Source | Task |
| --- | --- | --- | --- |
| Write with `icechunk.xarray.to_icechunk`; `ds.to_zarr` on a session store loses writes under dask.distributed or multiprocessing | Opt-in | icechunk `guides/xarray.md` | T14.2b |
| One append commit per ministack along `time`; a new writable session after each commit; check the grid before appending | Opt-in | icechunk `guides/xarray.md` | T14.2b |
| Provenance in commit metadata; set default commit metadata before opening the session. Keep reference dates as a coordinate, since commit metadata is invisible to xarray | Opt-in | icechunk `session.py`, `repository.py` | T14.2b |
| Tags are immutable named versions; read with a read-only session by tag or snapshot id. `as_of` works only together with a branch | Opt-in | icechunk `understanding/version-control.md` | T14.2b |
| Never amend a commit whose snapshot id is already in provenance; amending changes the id | Opt-in | icechunk `version-control.md` | T14.2b |
| No snapshot expiry or garbage collection from library code; tag Cal/Val runs instead | Opt-in | icechunk `expiration.md`; icechunk-nasa | T14.2b |
| Distributed writes: initialize metadata, fork the session, each worker writes a region and returns its session, merge, one commit | Opt-in | icechunk `understanding/parallel.md`, `guides/dask.md` | T37.4 |
| Rebase resolves chunk conflicts only. Two jobs that both append along time change the array shape and cannot be rebased | Opt-in | icechunk `version-control.md` | T37.4 |
| Commit before a distributed read | Opt-in | icechunk `guides/dask.md` | T37.4 |
| `Repository.open_or_create` for idempotent jobs; catch storage errors and conflicts separately | Opt-in | icechunk `repository.py`, `guides/error-handling.md` | T37.4 |
| Branches for comparing variants (Kalman versus LOWESS, weight settings) | Example | icechunk `version-control.md` | T16 |

### 1.4 Virtual datasets (PRD C3, opt-in)

Read from VirtualiZarr `b45b7d9` and icechunk-nasa, and checked against real DISP-S1 granules.

| Practice | Level | Source | Task |
| --- | --- | --- | --- |
| Load `time`, `reference_time`, `x`, `y` and `spatial_ref`; keep only the data layers virtual | Opt-in | VirtualiZarr `usage.md`; icechunk-nasa `write_virtual.ipynb` | T6.5b |
| Concatenate with `data_vars="all"`, `coords="minimal"`, `compat="override"`, `join="exact"`. The DISP layers have no time dimension in the file, so `data_vars="minimal"` would keep only the first granule. Never forward the `open_virtual_mfdataset` defaults | Core | VirtualiZarr `xarray.py`; granule check | T6.1, T6.5b |
| Check dtype, chunk shape and codecs per layer before combining, and split a frame by production version | Core | icechunk-nasa design notes; VirtualiZarr `writers/icechunk.py` | T5.8 |
| Drop the `/identification` and `/metadata` string datasets; their lengths differ per file | Core | Granule check (`dolphin_workflow_config` is `S64571` in one production and `S78086` in another) | T6.5b |
| Open `/corrections` as a separate node; `drop_variables` is not forwarded into subgroups | Opt-in | VirtualiZarr `hdf.py` | T6.5b |
| The HDF parser holds the HDF5 lock, so use processes or batch jobs, not threads | Opt-in | VirtualiZarr `scaling.md` | T6.5b |
| Register a virtual chunk container for the source bucket; credentials for it are passed at open and are not saved with the repo | Opt-in | icechunk `virtual.md`; VirtualiZarr examples | T6.5b |
| Validate after every build: random pixel series and one full epoch against h5py reads | Core | icechunk-nasa design notes | T6.5b |
| Treat the virtual store as an ingest and index layer; write a native `timeseries` store for per-pixel work | Core | icechunk-nasa; VirtualiZarr caveats | T6.5b, T14.2 |
| Replace a reprocessed granule with a region write, not an append | Opt-in | VirtualiZarr `accessor.py` | T6.5b |

Facts about DISP-S1 granules found during this reading (to confirm in T2.2):

- Data layers are chunked 256 x 256 with gzip level 4 and shuffle, which the HDF parser supports.
- One frame has about 1,000 to 1,150 chunks per layer and 14 2D layers, so about 15,000 references per granule and a few million per frame.
- `time` units were per-file in the 2025 production and fixed (`seconds since 2010-01-01`) in the 2026 production.
- Displacement was float32 in a 2025 production granule and float64 in a 2026 one.

### 1.5 Batch builds (PRD C6)

| Practice | Level | Source | Task |
| --- | --- | --- | --- |
| Initialize the store once from a template (`to_zarr(compute=False)`), then fill regions | Core | serverless-datacube-demo `lib.py`; icechunk-multiscales-demo | T37.4 |
| Plan regions from the output encoding, aligned to chunk or shard boundaries. With sharding, the shard is the unit of safe parallel writing | Core | serverless-datacube-demo `lib.py` | T37.4 |
| A job is a small serializable record (frame, region, granules, parameters), so it maps to an AWS Batch array index | Core | serverless-datacube-demo `lib.py` | T37.4 |
| Skip regions that are entirely outside the valid footprint | Core | serverless-datacube-demo (land-tile filter) | T37.4 |
| Each job returns a result record (region, timings, bytes, success, error) with its job identity; the records are the resubmission manifest | Core | Corrects serverless-datacube-demo, whose log has no job identity | T37.4 |
| Resolve inputs on the coordinator, so a rerun sees the same granules | Core | Corrects serverless-datacube-demo, where each worker searches | T37.4 |
| A `--limit` smoke run and a dry plan before a full run | Core | serverless-datacube-demo `main.py` | T37.4 |
| Stop a run after repeated all-failed batches or a high failure rate | Core | icechunk-multiscales-demo `coiled_app.py` | T37.4 |
| Partition in space only when the output has the whole time axis per chunk; chaining is sequential in time | Core | Follows from T7 | T37.4 |

### 1.6 Benchmarks (PRD NFR Benchmarks)

| Practice | Level | Source | Task |
| --- | --- | --- | --- |
| Sweep the chunk size in bytes, with the codec held fixed | Core | cloud-tensor-storage-benchmarks notebook | T37.3b |
| At least 5 trials, interleaved; report median and spread; keep raw per-trial rows | Core | cloud-tensor-storage-benchmarks | T37.3b |
| Record library versions, instance type and region with every row | Core | Extends the repo, which pins almost nothing | T37.2 |
| Measure the DISP access patterns: open, one pixel over all epochs, one epoch map, a window over all epochs | Core | icechunk-nasa design notes; the benchmark repo measures only whole-array reads | T37.3b |
| Set zarr `async.concurrency` for S3 and document it | Core | icechunk-nasa `read_virtual.ipynb`; icechunk `performance.md` | T37.3a |

### 1.7 Configuration, logging and serving

| Practice | Level | Source | Task |
| --- | --- | --- | --- |
| One settings object with environment overrides | Core | xpublish-tiles `config.py` | T4.9 |
| Estimate bytes before loading and raise an error that names the remedy | Core | xpublish-tiles `pipeline.py` | T37.3a |
| Document thread settings for compiled kernels and warm them up before timing | Core | xpublish-tiles README | T37.2, T37.3a |
| A registry of named synthetic datasets for tests | Core | xpublish-tiles tests | T2.5b |
| Serve time-independent 2D layers as tiles from the pyramid; pick levels by pixel size | Example | xpublish-tiles | T37.5 |

### 1.8 Do not copy

Bugs and gaps found in the reference code:

- icechunk-multiscales-demo sets overview coordinates to the first pixel of each block while its transform implies block centers, and its array shapes (ceil) disagree with its coarsening (floor) when a size is not divisible.
- topozarr coarsens by 2 only, with unweighted reductions and `boundary="trim"`.
- serverless-datacube-demo commits whatever sessions came back, even after failed jobs, and uses the pre-1.0 Icechunk distributed API.
- cloud-tensor-storage-benchmarks times Icechunk writes without the commit, leaves one backend uncompressed while others use zstd, and never checks the data it reads.
- workshop-cng-2025-zarr and icechunk-nasa pin pre-1.0 Icechunk and VirtualiZarr 1.x APIs; one workshop cell appends the wrong variable.

## 2. Zarr best practices

Source: [zarr.dev](https://zarr.dev/) and the documents it links: the [zarr-python user guide](https://zarr.readthedocs.io/en/stable/), the [Zarr v3 specification](https://zarr-specs.readthedocs.io/en/latest/v3/core/index.html), the [ZEPs](https://zarr.dev/zeps/) and the [zarr-conventions](https://github.com/zarr-conventions) repositories. Read on Oct 1, 2026 by one reader; these rows were not independently re-checked.

| Practice | Source | Task |
| --- | --- | --- |
| A chunk is the unit of reading and should be at least 1 MB uncompressed. Keep the dimension you read along whole and split the others. Float32 sizes: (300, 64, 64) is 4.9 MB, (300, 128, 128) is 19.7 MB, (1, 1024, 1024) is 4 MB | [performance guide](https://zarr.readthedocs.io/en/stable/user-guide/performance/) | T14.2 |
| A shard is the unit of writing and must be written whole. Use sharding when small chunks would create too many objects. A 30 m frame layer at 300 epochs is about 84 GB, so shards of a few hundred MB keep the object count in the hundreds | [performance guide](https://zarr.readthedocs.io/en/stable/user-guide/performance/), [sharding codec](https://zarr-specs.readthedocs.io/en/latest/v3/codecs/sharding-indexed/index.html) | T14.2 |
| Make one dask chunk equal one shard. The worker memory budget must hold one uncompressed shard plus its encoded copy | [arrays guide](https://zarr.readthedocs.io/en/stable/user-guide/arrays/) | T14.2, T37.1 |
| In xarray, set `encoding[var] = {chunks, shards, compressors}` and rechunk to shard boundaries; never pass `safe_chunks=False` | [xarray I/O guide](https://docs.xarray.dev/en/latest/user-guide/io.html) | T14.2 |
| The whole-time-axis `timeseries` preset cannot be appended to: an append rewrites every shard. Append stores use a fixed time chunk (one ministack) and time shards that are whole multiples of it. `timeseries` stores are frozen snapshots | Follows from the sharding and append docs | T14.2, T14.2b |
| Default compressor is Zstandard. Benchmark plain zstd against Blosc (zstd with shuffle) for float32 with NaN, and pick from the result | [arrays guide](https://zarr.readthedocs.io/en/stable/user-guide/arrays/), [codecs API](https://zarr.readthedocs.io/en/stable/api/zarr/codecs/) | T37.3b |
| Use only spec-defined codecs from `zarr.codecs` by default, so GDAL and other readers work. Lossy codecs are opt-in | [v3 migration](https://zarr.readthedocs.io/en/stable/user-guide/v3_migration/) | T14.2 |
| `fill_value` is required in v3. NaN for float layers means all-NaN chunks (water, off-frame) are not stored. Set the nodata value of integer masks explicitly; the default is 0 | [data types](https://zarr-specs.readthedocs.io/en/latest/v3/data-types/index.html) | T14.2a |
| xarray keeps the CF `_FillValue` separate from the Zarr `fill_value` in v3. Round-trip tests check both, and that integer masks do not come back as float | [xarray I/O guide](https://docs.xarray.dev/en/latest/user-guide/io.html) | T14.2a |
| Consolidated metadata is experimental in v3 and not in the spec. Write it for plain stores, re-consolidate after an append, and never require it on read | [consolidated metadata](https://zarr.readthedocs.io/en/stable/user-guide/consolidated_metadata/) | T14.2 |
| Arrays written with zarr-python directly need `dimension_names`, or xarray cannot open them | [xarray Zarr encoding](https://docs.xarray.dev/en/latest/internals/zarr-encoding-spec.html) | T27.7 |
| `async.concurrency` defaults to 10 and multiplies with dask threads. Set it through `zarr.config` or environment variables per job, not in code | [config guide](https://zarr.readthedocs.io/en/stable/user-guide/config/) | T37.3a |
| Parallel writers are safe only when they write different chunks (shards, when sharded) | [performance guide](https://zarr.readthedocs.io/en/stable/user-guide/performance/) | T37.4 |
| `FsspecStore` is the documented route for object storage; the obstore-backed store is marked experimental | [storage guide](https://zarr.readthedocs.io/en/stable/user-guide/storage/) | T14.2, T37.3b |
| The multiscales, `spatial:` and `proj:` conventions are at version 0.1, pilot maturity, with breaking changes expected. Arbitrary scale factors (33) are allowed. GeoZarr is a suite of these conventions, not yet a finished spec | [multiscales](https://github.com/zarr-conventions/multiscales), [spatial](https://github.com/zarr-conventions/spatial), [proj](https://github.com/zarr-conventions/proj), [geozarr-spec](https://github.com/zarr-developers/geozarr-spec) | T27.7 |
| Do not rely on variable (rectilinear) chunking; that ZEP is a draft and xarray rejects it | [ZEPs](https://zarr.dev/zeps/) | T14.2 |

## 3. xarray ecosystem

Source: the [xarray ecosystem page](https://docs.xarray.dev/en/latest/user-guide/ecosystem.html), every project in every section, plus related projects that are not listed there. Maintenance status was read from GitHub on Oct 1, 2026. Function names for sarxarray, stmtools, xvec, xeofs, flox and xarray-regrid were read from source; the rest comes from project docs. One reader; not independently re-checked.

Nothing InSAR-specific can be a dependency. The two SAR projects work on SLC or point-scatterer data in radar coordinates.

### 3.1 Dependencies

| Project | Status | What disp-xr uses | Level | Task |
| --- | --- | --- | --- | --- |
| [rioxarray](https://github.com/corteva/rioxarray) | 0.23.0, Jul 2026 | CRS and transform. Version 0.23 also writes the Zarr `spatial:` and `proj:` conventions; confirm, then let it write them | Core | T6.2, T14.2a |
| [flox](https://github.com/xarray-contrib/flox) | 0.11.2, Feb 2026 | Grouped reductions (`nanmedian`, `mode`, binning). No weighted reductions: a weighted mean is two sums, a weighted median needs disp-xr's own kernel | Core | T10.3, T23.2, T27.2 |
| [numbagg](https://github.com/numbagg/numbagg) | 0.9.6, Sep 2026 | Fast NaN-aware reductions and moving windows, picked up by xarray when installed | Recommended, not imported | T1.5 |
| [xvec](https://github.com/xarray-contrib/xvec) | 0.5.2, Nov 2025; API may change | GNSS stations as a vector data cube (station geometry by time), point extraction, GeoDataFrame export | `gnss` extra | T25.1 |
| [xeofs](https://github.com/xarray-contrib/xeofs) | 3.0.4, Oct 2024; quiet since Jan 2025 | EOF, rotated and complex EOF for the gallery. Maintenance risk: core code never imports it, and a CI job reports breakage | `eof` extra | T32.6d |
| [odc-geo](https://github.com/opendatacube/odc-geo) | 0.5.3, Jul 2026 | Lazy, dask-backed reprojection to a common grid. Evaluate for the mosaic before relying on `rio.reproject_match`, which loads data | Optional, to evaluate | T36.1 |
| [cf_xarray](https://github.com/xarray-contrib/cf-xarray) | 0.11.3, Jun 2026 | Tests assert that outputs decode as CF | `test` extra | T14.5 |
| [hvplot](https://github.com/holoviz/hvplot) with datashader | Active | Interactive full-frame browsing in one gallery page | `interactive` extra | T17.4 |
| [xpublish](https://github.com/xpublish-community/xpublish) | Active | Tile serving in the gallery (with xpublish-tiles) | Docs only | T37.5 |

### 3.2 Design references

Nothing from these is imported; disp-xr copies an idea.

| Project | Idea copied | Task |
| --- | --- | --- |
| [SARXarray](https://github.com/TUDelftGeodesy/sarxarray) (1.4.0, Aug 2026) | Accessor layout on an SAR stack, and `multi_look` by `coarsen` with chosen statistics | T9.2, T27.1 |
| [H3](https://h3geo.org), matplotlib `hexbin`, PostGIS `ST_HexagonGrid` | Hexagonal binning: a global hierarchical hexagon grid (H3), and the planar hexagon lattice that `hexbin` and `ST_HexagonGrid` build (Carr et al. 1987). They count or average; disp-xr adds quality weights and per-cell sigmas, lazily on the stack. Why hexagons: Birch et al. 2007; global grids: Sahr et al. 2003 | T27.8 |
| [stmtools](https://github.com/TUDelftGeodesy/stmtools) (0.1.4) | Point data as a (space, time) xarray Dataset with spatial ordering of points for chunk locality; enrichment from polygons and rasters | T27.5, T28.4, T25.1 |
| [xDEM](https://github.com/GlacioHack/xdem) (0.2.3, Mar 2026) | Uncertainty of a spatial average with correlated errors: variogram-based effective sample count (Hugonnet et al. 2022, Rolstad et al. 2009). The simple n_eff = (Σw)² / Σw² ignores spatial correlation and gives cell sigmas that are too small at 1 to 10 km | T27.3b, T32.1 |
| [xclim](https://github.com/Ouranosinc/xclim) (0.62.0) | A plain compute function wrapped by one decorator that checks units, sets CF attributes and appends history | T9.3 |
| [xarray-regrid](https://github.com/xarray-contrib/xarray-regrid) (0.4.2) | `nan_threshold` semantics for the valid-fraction cutoff of a cell | T27.3 |
| [xarray-schema](https://github.com/xarray-contrib/xarray-schema) | Component schemas (dims, dtype, chunks) in `validate_disp`, including a chunk check | T9.5 |
| [xproj](https://github.com/xarray-contrib/xproj), [rasterix](https://github.com/xarray-contrib/rasterix) | CRS-aware alignment and a raster index built from the affine transform. Too early to depend on; keep grid and CRS access behind one helper so they can be swapped in, and fail on a CRS mismatch (adjacent frames can be in different UTM zones) | T9.5, T36.1 |
| [xrft](https://github.com/xgcm/xrft) (1.0.2) | Spectrum scaling and window-correction conventions. Not usable directly: it needs even spacing and no NaNs | T22.1, T32.4 |
| [xskillscore](https://github.com/xarray-contrib/xskillscore), [xarray-einstats](https://github.com/arviz-devs/xarray-einstats) | Function signature `(a, b, dim, weights, skipna)`; used as test oracles for RMSE, MAD, skewness and kurtosis | T32.1, T32.6b |
| [xarray-spatial](https://github.com/xarray-contrib/xarray-spatial) (0.10.18) | IDW, kriging and spline used as cross-checks for disp-xr's own estimators | T19.6, T24.4 |
| [ndpyramid](https://github.com/carbonplan/ndpyramid) | Pyramid generation; disp-xr writes its own multiscales metadata | T27.7 |

### 3.3 Checked and not used

- **Units:** pint-xarray and MetPy's unit handling conflict with the rule that units live in attributes.
- **Regridding:** xESMF (ESMF dependency, lat/lon focus), pyinterp.
- **Covered already:** geocube, regionmask, salem, GeoWombat, Open Data Cube core (rioxarray and flox cover the need).
- **Fitting:** xarray-lmfit (per-pixel lmfit is far too slow for 70 million pixels), eofs (no rotation), X-regression (dead).
- **Catalogs:** stackstac, odc-stac, intake-xarray, xstac. DISP discovery goes through CMR.
- **L1 SAR readers:** xarray-sentinel, xsar, pyroSAR.
- **Superseded:** kerchunk formats (VirtualiZarr recommends Icechunk).
- **After 1.0:** TOAD (abrupt-shift detection on (time, y, x) cubes), a candidate when automatic break detection comes into scope.
- **Other domains:** the climate, ocean, radar-meteorology, machine-learning and non-Python projects on the page.
