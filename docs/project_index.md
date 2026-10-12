# disp-xr project index

**Last indexed:** 2026-10-12 · `main` after the v0.4.0 release (tag `v0.4.0`) on `origin` = `mgovorcin/disp-xr`, through PR #86 and branch `bench/t37-results`

How and when this file is rebuilt: see "Project Index (cyclical indexing)" in `CLAUDE.md` at the repository root.

## Documents

| File | Answers | Lines |
| --- | --- | --- |
| [specs.md](specs.md) | What to build: requirements F1–F16, S1–S4, C1–C6, storage rules, non-functional rows, open questions | 324 |
| [todo.md](todo.md) | In what order: tasks T1–T39 (228 subtasks), dependencies, milestones M0–M3 | 622 |
| [code_review.md](code_review.md) | What exists: keep / refactor / drop per module, 7 bugs, the `timeseries_fitting` branch | 161 |
| [porting.md](porting.md) | What is ported from opera_utils, where it goes, license terms, upstream bugs to fix | 132 |
| [references.md](references.md) | Outside practices followed and their sources: earth-mover repositories, Zarr docs, xarray ecosystem | 206 |
| [api_review.md](api_review.md) | v0.2 API review: public API, settled decisions, open questions | — |
| [api_policy.md](api_policy.md) | What is public, deprecation rules, frozen names (draft until v0.2) | — |
| [try-it.md](try-it.md) | Guide for Cal/Val users trying disp-xr on their own frame | — |
| [product_notes.md](product_notes.md) | Differences between DISP productions that affect reading a frame (float32 to float64 displacement in November 2025) | — |
| [serving.md](serving.md) | Running `disp_xr serve` for others: preparing stores, workers, reverse proxy with caching, limits | — |
| [remote.md](remote.md) | Finding and reading granules remotely: credentials, search, download, HTTPS and S3, static layers | — |
| [performance.md](performance.md) | Running full frames: processes against threads, disp-xr settings, environment for batch jobs, memory stalls on shared nodes | — |
| [docs_plan.md](docs_plan.md) | Documentation coverage per feature, stale notebooks, GitHub Pages gaps, tutorial order before going public | — |
| [studies/split_metric.md](studies/split_metric.md) | T29 trade study report: frames, criteria, decision (scripts in `studies/split_metric/` at the root) | — |
| [studies/gnss_validation.md](studies/gnss_validation.md) | T34 GNSS validation against geepers on the T29 frames: double-differenced agreement, tie checks, discrepancies (scripts in `studies/gnss_validation/` at the root) | — |
| [check_plan.py](check_plan.py) | Mechanical check of `todo.md` (run in every indexing cycle) | — |
| [gallery/](gallery/) | 18 executed example pages (`# %%` scripts run by mkdocs-jupyter), one or more per feature | — |
| [real_data/](real_data/) | Pre-executed notebooks on production granules (saved outputs, not run by the docs build): Houston F08882, production change F11116, San Andreas creep F11116, terrain-following troposphere F39362, product versions 1.0 against 1.1 F08622 | — |
| [img/](img/) | `architecture.png`, `roadmap.png`, exported from the PRD doc | — |
| `CLAUDE.md` (repository root) | Coding rules and the indexing cycle | — |

The PRD also lives as a shared doc; `specs.md` is its copy in the repo and must match it.

## Status

| Milestone | Tasks | Done |
| --- | --- | --- |
| M0 Foundation (v0.1) | T1–T8 | 5 / 8 (T2, T4, T5, T6, T7). Released as `v0.1.0` (2026-10-02). Open: T1.4b (name maintainers), T3.4–T3.5 (coverage badge, branch protection: need CI), T8.4 (milestone dates, needs the maintainers) |
| M1 Core API (v0.2) | T9–T18 | 8 / 10 (T9, T10, T11, T13, T14, T15, T17, T18). No `v0.2.0`: released with M2 as `v0.3.0` (decision of 2026-10-10); API review closed by the maintainer and the core API frozen for 0.4.0 (T18, 2026-10-11). Open: T12.5 (needs DISP-NI-STATIC), T16.6 (GitHub Pages, waits for the repository to go public) |
| M2 Analysis (v0.3) | T19–T34 | 16 / 16. Released as `v0.3.0` (2026-10-10); gate T34 passed (`studies/gnss_validation.md`). T28 finished 2026-10-11 (T28.5 benchmark on AWS Batch; T28.6 rejected) |
| M3 Release (v1.0) | T35–T40 | 1 / 5 (T40). T37 5 of 7 subtasks done (T37.1, T37.3a, T37.3b, T37.4, T37.5; T37.2 suite timed on AWS Batch at tile size, frame size running; most of T37.3); T35 waits for beta DISP-NISAR products; T36 dropped (2026-10-11); T40 done (REST service); T38–T39 not started |

Subtasks: 215 done, 20 open. All work is merged in `main`. CI is off (Actions minutes on a private repository); every PR
since #30 was checked locally and says so in its description.

| Requirement | Tasks | Status |
| --- | --- | --- |
| F1 Discover | T5 | Done on `feat/product-registry` |
| F2 Open | T6 | Done on `feat/open-disp` (local and remote) |
| F3 Stack | T7 | Done on `feat/chain-ministacks` |
| F4 Mask | T10 | Done: `mask`, `valid_mask`, `quality_summary` |
| F5 Reference | T11 | Done: `rereference` (point, median, border, high_coherence, area, auto) |
| F6 Correct | T13 | Done: `correct` (product layers, external troposphere) |
| F7 Geometry | T12 | DISP-S1 done (`open_static`, `enu_to_los`); DISP-NISAR waits for a DISP-NI-STATIC granule (T12.5) |
| F8 Time series | T20 | Done: `fit_trajectory`, `Trajectory`, `predict_trajectory` |
| F9 Spatial models | T19, T24, T33 | Done: estimator base (T19), trends, de-ramping and gridders (T24), GNSS tie with gallery (T33) |
| F10 GNSS | T25 | Done with geepers as the engine (`gnss` extra) |
| F11 Multi-frame | — | Dropped 2026-10-11 (out of scope for 1.0) |
| F12 Export | T14 | Done: GeoTIFF, Zarr v3, NetCDF, MintPy, Icechunk, virtual stores |
| F13 Plot | T17 | Done: `plot_map`, `plot_point`, accessor, interactive gallery page |
| F14 CLI | T15 | Done: `info`, `stack`, `export`, `geometry`, `search`, `virtual` |
| F15 Plate motion | T13.5 | Done: `plate_motion_los`, `remove_plate_motion` (ITRF2014-PMM) |
| F16 Search | T4.5 | Done: `search.py` (CMR, DISP-S1 and DISP-NISAR short names), `disp_xr search`; remote reading documented in `remote.md` |
| S1 Colored-noise velocity | T20, T21, T22, T23 | Done: OLS, one-series colored noise, per-pixel Whittle and MIDAS, frame GLS with coherence groups; quadtree groups wait for T28; noise-model policy proposed, awaiting confirmation |
| S2 Downsampling | T26, T27, T28, T29 | T26 and T27 done; quadtree (T28) done except its benchmark numbers; trade study (T29) done 2026-10-09: grid recommended, quadtree default metric `structure`, unmasked downsampling with quality weights |
| S3 Temporal denoising | T30, T31 | Done: grid, gaps and breaks (T30), Kalman and LOWESS `denoise` (T31); the deforming-site check ran in T29 (F11116) |
| S4 Residual analysis | T32 | Done: `compare` over geepers, normalized residuals and flags, spectra, common mode, diagnostics; OPERA DISP-S1 requirement curves by default (T32.6) |
| C1 Zarr v3 output | T14.2, T14.2a | Done: `to_zarr` (Zarr v3, sharded, zstd, presets) |
| C2 Icechunk store (opt-in) | T14.2b | Done: `to_icechunk` / `open_icechunk`, one commit per write, append along time, tags, snapshot id in provenance |
| C3 Virtual datasets (opt-in) | T5.8, T14.2c | Done for local granules: `build_virtual_frame`, `open_virtual`, `validate_virtual`; S3 path not run against S3; `/corrections` not referenced |
| C4 Multiscale pyramids | T27.7, T27.9 | Done: `to_pyramid`, `open_pyramid`, `build_overviews`, `to_geozarr` (bowser) |
| C5 Map serving example | T37.5 | Done: gallery `web_tiles` (xpublish-tiles, docs environment only) |
| C6 Batch builds and benchmarks | T37.3b, T37.4 | Region jobs done (`batch.py`); storage benchmark done (local codecs, integer masks, S3: NetCDF, virtual Icechunk, Zarr with fsspec and obstore; `benchmarks/README.md`) |
| C7 Read-only REST service (opt-in) | T40 | Done: `serve.py`, `disp_xr serve`, gallery page and guide (`serving.md`); Zarr, Icechunk and virtual frames |
| X DataTree | T6.4b | `open_disp_tree` done; `open_frames` (T36.1b) dropped with T36 |
| X flox grouped reductions | T10.3, T23.2, T32.1 | Used in `quality_summary(by_ministack=True)`; others not needed so far |
| X apply_ufunc kernels | T7.2, T20.3, T22.3 | Done: per-pixel fits and filters run through `apply_ufunc` on time-contiguous chunks |
| NFR Packaging, Licensing | T1, T4.7, T38 | pixi environments and extras done; T1.4b (maintainers) and T38 (PyPI, conda-forge) open; ownership of the fork unresolved |
| NFR Testing, typing | T2, T3 | Done locally: 601 tests, laziness and graph-size tests, mypy and ruff in pre-commit; GitHub Actions disabled (billing), local pre-commit and pre-push hooks stand in |
| NFR Compatibility | T5.7, T35 | DISP-S1 across productions done (float precision and time encoding, `product_notes.md`); DISP-NISAR waits for products (T35) |
| NFR Metadata, Provenance | T14.2a, T14.4 | Done: history on every result, provenance attributes in every export |
| NFR Benchmarks, Performance, Memory | T37 | Benchmark suite, laziness and graph-size audits, load guard, `performance.md`; memory budget proposed (8 GB per worker, 4 GB coordinator) |
| NFR Docs | T16, T38.3 | Site with 18 executed gallery pages, a tutorial for every feature, guides (remote, performance, product notes); GitHub Pages workflow ready, publishing waits for the repository to go public; docstring examples incomplete (T38.3) |
| API rules (config, history, order) | T4.9, T9, T15.2 | Config and history done; processing order in T15.2 |

## Modules (`src/disp_xr` on `main`)

| Module | Requirement | State | Next task |
| --- | --- | --- | --- |
| `product.py` | F1 | `ProductSpec`, `DispGranule`, `find_products`, `group_ministacks`, `check_granules`; `get_disp_info` kept for the stack | T7 (deprecate `get_disp_info`) |
| `stack.py` | F3 | `chain`, `NaNPolicy`, `MissingCrossoverError`; deprecated wrappers (`stack_disp` and older) | T18.3, T38 (remove wrappers) |
| `quality.py` (was `quality_metrics.py`) | F4 | Masks, presets, quality summary, lazy statistics | T10 |
| `reference.py` (from `utils.py`) | F5 | `rereference`, `get_reference_values`, `find_stable_point`; bug 7 fixed | T11 |
| `geometry/__init__.py` (from `utils.py`) | F7 | Coordinate helpers, `get_burst_ids` | T12 |
| `geometry/static.py` (was `static_layers.py`) | F7 | `open_static`, DISP-S1-STATIC search and cached download, `stitch_cslc_static` | T12 |
| `corrections.py` | F6, F15 | `correct`, `remove_plate_motion`, ITRF2014-PMM poles | T13 |
| `geometry/los.py` | F7 | `enu_to_los`, incidence/azimuth, slant range | T12.2, T12.5 |
| `export/raster.py` (was `io.py`) | F12 | `to_geotiff` (COG), legacy rasterio helpers | T14.1 |
| `export/zarr.py`, `export/_overwrite.py` | F12, C1 | `to_zarr`, `to_netcdf` (with `overwrite`), chunk/shard scheme (presets `timeseries`, `maps`, `ministack`), `round_mantissa` | T14.2, T14.2b, T37.3b |
| `export/icechunk.py` | F12, C2 | `to_icechunk`, `open_icechunk` (`cloud` extra, imported lazily) | T14.2b |
| `batch.py` | C6 | `init_store`, `plan_jobs`, `RegionJob`, `run_region`, `failed_jobs`, manifest | T37.4 |
| `virtual.py` | C3 | `build_virtual_frame`, `open_virtual`, `validate_virtual` (`cloud` extra) | T14.2c |
| `export/_provenance.py` | NFR Provenance, Metadata | Provenance and CF flag attributes | T14.4 |
| `export/mintpy.py` (was `mintpy.py`) | F12 | `to_mintpy`: timeseries, quality masks, geometry (h5py only) | T14.3 |
| `open.py` | F2 | `open_disp`, `open_disp_tree` | T7 (chaining on top) |
| `_chunks.py`, `_remote.py`, `_credentials.py` | F2, C1 | Chunk presets, `data_chunks`; remote access (ported) | — |
| `_grid.py` | Infrastructure | `with_grid_mapping`, lazy `coordinate_grids` | T37.3 |
| `search.py` (was `download.py`) | F16 | Ported CMR search, session seam, NISAR short name; bug 9 fixed | T5.2 (NISAR spec) |
| `tutorial.py` | NFR Docs, tests | `make_disp_stack` (synthetic granules with a known truth), `correlated_noise`, `make_gnss_source` and `vertical_los` (synthetic GNSS) | T2.3, T16.3, T27.3b, T32.7 |
| `export/geozarr.py` | C4 | `to_geozarr` (bowser and standard profiles), `display_factors`, `bowser_attrs`, `reference_lonlat`, `los_attrs`, `histogram_stats` | T27.9 |
| `export/multiscales.py` | C4 | `build_overviews`, `to_pyramid`, `open_pyramid`, `MULTISCALES_CONVENTION` | T27.7, T27.9 |
| `models/gnss_tie.py` | F9 | `GNSSTie`, `fit_tie`, `apply_tie`, `cross_validate_tie`, `TieResult` | T33 |
| `models/trend.py` | F9 | `Trend`, `deramp`, `polynomial_terms` | T24 |
| `models/gridding.py` | F9 | `Spline`, `Nearest` | T24.4 |
| `diagnostics.py` | S4 | `model_diagnostics`, `significance`, `anderson_darling` | T32.6b–c |
| `gnss.py` | F10 | `gnss.load` (with the relocation screen), `stations`, `get_source`, `to_los`, `sample`, `align`, `centred`, `drop_relocations` (geepers engine) | T25, T34.2 |
| `models/base.py` | F9, API (Estimators) | `BaseEstimator`, `Chain`, `Mean`, `coordinates`, `cross_val_score`, `clone` | T19 |
| `timeseries/denoise.py` | S3 | `denoise`, `kalman_smooth`, `lowess`, `q_from_timescale` | T31 |
| `timeseries/temporal.py` | S3 | `regular_time_grid`, `grid_step_days`, `gap_mask`, `split_at_breaks`, `apply_segmented` | T30 |
| `quadtree.py` | S2 (quadtree) | `quadtree` (by threshold or `n_points`), `Quadtree` (`points`, `leaf_id`, `to_raster`) | T28, T29 |
| `hexagons.py` | S2 (hexagon mode) | `downsample_hex` (planar and H3), `to_geodataframe`, `cell_polygons`, `hex_axial`, `hex_centers`, `hexagon_vertices` | T27.8 |
| `downsample.py` | S2 | `downsample`, `to_points` | T27 |
| `residuals.py` | S2 (cell uncertainty), S4 | Variograms and correlated cell sigma; `compare` (per-station reference epoch, velocity gate), `Comparison` (with `to_zarr`), `Requirement`, `DISP_S1_DISPLACEMENT`, `DISP_S1_VELOCITY` (geepers metrics) | T27.3b, T27.8, T32, T34.2 |
| `serve.py` | C7 | `DispPlugin` (xpublish routes `timeseries`, `velocity`, `cells`, `gnss`, `provenance`), `make_app`, `open_store`; `disp_xr serve` | T40 |
| `weights.py` | S2 | `quality_weights`, `residual_cycles`, `network_degree`, `residual_cycle_histogram` | T26 |
| `plot.py` | F13 | `plot_map`, `plot_point`, `diverging_cmap` (`plot` extra, imported lazily) | T17 |
| `accessor.py`, `_history.py`, `units.py`, `validate.py` | API rules | `.disp` accessor, operation history, unit conversion, validation | — |
| `config.py` | API rules | `donfig` settings, `DISP_XR_*` overrides, `check_load` and `LoadTooLargeError` | T37.3a |
| `cli/__init__.py` | F14, F16 | `disp_xr` commands wrapping library functions | T15 |
| `timeseries/` | F8, S1, S4 | `trajectory.py` (`fit_trajectory`), `noise.py` (`fit_noise`, from geepers), `velocity.py` (`fit_velocity`: OLS, Whittle, MIDAS, GLS), `gls.py` (grouped GLS), design matrix, legacy `fit_timeseries_block`; jax optional | T20, T21, T22, T23, T32 |
| `utils.py` | — | Deprecated aliases only | T38 |
| `quality_metrics.py`, `io.py`, `mintpy.py`, `static_layers.py`, `download.py` | — | Deprecated aliases | T38 |
| `log/` | Infrastructure | Unchanged | — |

## Decisions log

| Date | Decision |
| --- | --- |
| 2026-10-11 | Multi-frame mosaics (F11, T36) are dropped; the owner is the sole maintainer (`CODEOWNERS`); the API review (T18.2) is closed by the maintainer, with seven decisions in `api_review.md` |
| 2026-10-11 | The noise-normalized split metric (T28.6) is dropped: it does not beat the grid and splits like `displacement`; the quadtree default stays `structure` |
| 2026-10-10 | No `v0.2.0`: M1 and M2 are released together as `v0.3.0`; the core API is not frozen until the T18 review with Cal/Val users |
| 2026-10-09 | GNSS against DISP (T34.2): daily GNSS solutions match the acquisition's UTC day; `compare` references InSAR and GNSS per station to the first shared epoch; station velocities need 10 shared epochs over 2 years; a GNSS step above 10 m is a monument relocation |
| 2026-10-05 | A read-only REST service becomes requirement C7 (task T40, M3): an xpublish plugin over existing stores, no processing on request |
| 2026-10-04 | `open_disp` reads productions that differ only in float precision (DISP-S1 displacement float64 from November 2025) with a warning, cast to float32; virtual stores still refuse them |
| 2026-10-04 | Residual plots draw both OPERA DISP-S1 requirements by default: displacement 3 (1 + sqrt(L)) mm, velocity 2 mm/yr, 0.1 to 50 km (values to confirm against the requirements document) |
| 2026-10-04 | Documentation is published on GitHub Pages, not ReadTheDocs |
| 2026-10-02 | `open_disp` keeps the file layout (`chunks="native"`); each operation gets the chunks it needs (chaining keeps chunks, per-pixel functions rechunk themselves, `dxr.rechunk` for explicit control); disp-xr computes with at most 32 dask threads by default |
| 2026-09-30 | 1.0 supports DISP-S1 and DISP-NISAR |
| 2026-09-30 | verde is a design reference only; disp-xr has its own fit/predict estimators |
| 2026-09-30 | Package belongs to OPERA Cal/Val |
| 2026-09-30 | Colored-noise estimator lives in disp-xr |
| 2026-09-30 | Default weight layers: phase similarity, temporal coherence, inversion residuals; cycle jumps down-weight, never mask |
| 2026-09-30 | Quadtree split metric decided by a trade study (displacement, velocity, spatial structure) |
| 2026-09-30 | Temporal denoising ships both Kalman and LOWESS |
| 2026-09-30 | No dependency on opera_utils; needed functions are ported into disp-xr with attribution |
| 2026-09-30 | Follow earth-mover practices; Icechunk and VirtualiZarr are opt-in |
| 2026-10-01 | zarr.dev documentation is the reference for Zarr best practices |
| 2026-10-01 | xarray ecosystem reviewed; libraries adopted and ideas copied are listed in `references.md` section 3 |
| 2026-10-01 | `open_disp` refuses granules of mixed production versions (different dtype or encoding) |
| 2026-10-01 | Mosaic CRS: UTM zone with the largest share of valid pixels by default, explicit CRS beyond two zones; regional products mosaic at the downsampled level |
| 2026-10-01 | MintPy export is the `mintpy` extra; search stays in disp-xr as `search.py` (both followed from earlier decisions) |

## Out of sync

- GitHub Actions is off on `mgovorcin/disp-xr`: the private repository used up its Actions minutes (2026-10-04). The `test` workflow is disabled; the `docs` workflow (GitHub Pages, PR #43) fails at start for the same reason until it is disabled or the repository goes public. Local hooks stand in: pre-commit on commit, pytest (py312) and `mkdocs --strict` on push.
- Local checkouts `01_OPERA/STAKEHOLDER/disp-xr` and `01_OPERA/VLM/disp-xr` are on the old `timeseries_fitting` branch, and the conda envs `disp_xr` and `disp_xr_py311` hold non-editable installs of old commits.
- The OPERA requirement values drawn by default (3 (1 + sqrt(L)) mm, 2 mm/yr) were to be confirmed against the requirements document before v0.3; they are released in `v0.3.0` unconfirmed (`specs.md` S4).
## Not verified

- `code_review.md` and the Context lines of `todo.md` were not checked by a second reader against the code. The 7 bugs were read from the source when the review was written.
- `porting.md` was spot-checked only (license, filename regex, module locations, upstream bugs 1–3).
- `references.md` sections 2 and 3 come from one reader. Section 1 was checked claim by claim.
- DISP granule facts in `references.md` (chunking, dtypes, `time` units) come from a few granules; T2.2 confirms them.

## Checks passed this cycle

- `python docs/check_plan.py`: 39 tasks, 232 subtasks, no duplicate or dangling ids; "Depends on" lines, mermaid graph and dependency table agree; no cycles; every task except T39 unlocks another; headings match task titles.
- All relative links in `docs/*.md` and `CLAUDE.md` resolve.
- No sentence in `specs.md`, `todo.md` or `code_review.md` tells disp-xr to depend on, wrap or vendor opera_utils.
- Every requirement in the table above has at least one task.

## Index log

| Date | Trigger | Changes |
| --- | --- | --- |
| 2026-10-12 | T37.3b finished, #86 merged | S3 storage benchmark on AWS Batch (`benchmarks/storage_s3.py`, `batch/` runs any script); integer masks benchmarked, integer layers compressed with Blosc per epoch (`export/zarr.py`, `specs.md` C1); #86 fixed virtual frames on S3 with icechunk 2.3 (`virtual.py`) |
| 2026-10-12 | T38.3 finished, T38.3b repeated, NISAR to v0.6 | opera-utils `4125484` diff ported (HTTPS auth header, NISAR endpoint, affine `@`); T35 and T12.5 scheduled for v0.6; `serve.py` doctests ignored without the serve extra (collection failed in `test-py312` since #77) |
| 2026-10-12 | Release `v0.4.0` (API frozen) | Changelog section `[0.4.0]`; index stamped |
| 2026-10-11 | T18 finished (T18.1 resolved, T18.3 done) | Seven API decisions implemented; `api_policy.md` frozen list; M1 8 / 10; `stack.py` and `export/zarr.py` rows; a misfiled 0.3.0 changelog entry (the `open_virtual` time fix) moved to Unreleased |
| 2026-10-11 | T28.5 finished, T28 done, M2 complete | Quadtree benchmark on a dedicated AWS Batch instance (`benchmarks/README.md`); M2 16 / 16 |
| 2026-10-11 | Owner decisions: T36 dropped, T1.4b and T18.2 done | T36 out of the plan (graph, table, T38), F11 and X rows, M3 1 / 5, one decision line; `CODEOWNERS`, `maintainers` |
| 2026-10-11 | Virtual-frame times fixed | `open_virtual` floors times to whole seconds like `open_disp`; the `test_virtual` item leaves "Out of sync" |
| 2026-10-11 | `serve` reads virtual frames | C7 row; `test-serve` gains the cloud feature; virtual-frame time offset listed under "Out of sync" |
| 2026-10-11 | T40 finished (T40.5) | `docs/gallery/rest_service.py`, `docs/serving.md` (Documents row), C7 done, M3 1 / 6 |
| 2026-10-11 | T40.1 to T40.4 finished | `serve.py` (module row, C7 in progress), `serve` extra and `test-serve` environment, `docs/api/serve.md`, `cli.md`, `install.md` |
| 2026-10-11 | T28.6 finished (rejected) | `studies/split_metric.md` follow-up section and `results/*_unmasked_normalized.csv`; M2 status; one decision |
| 2026-10-10 | Release `v0.3.0` (milestone M2), T34 branches merged (#68 to #73) | Full cycle: status per milestone recounted (203 / 38 subtasks), `gnss.py` and `residuals.py` rows, two decisions (no `v0.2.0`; GNSS rules from T34.2), unmerged-branch line removed from "Out of sync"; T8.2 and T34 ticked |
| 2026-10-10 | T16.7 finished | `docs/real_data/versions.ipynb` (F08622 v1.0 against v1.1) in the nav and gallery index |
| 2026-10-09 | T34.2 finished | Rerun on the full fix stack (velocity gate, relocation screen, median LOSO): the tie improves on all three frames; 1 mm epoch-RMSE gap on F08882 explained (DEN4 relocation, gap-filling policy) |
| 2026-10-09 | T34.2 rerun with the station-reference fix | More stations in every statistic (F08882 tie 92 to 126); epochs now use the same stations as geepers; new open item: minimum record length for station velocities |
| 2026-10-09 | T34.2 rerun with the GNSS day fixes | Report results from disp-xr `feature/gnss-daily-day` and geepers `feature/gps-start-date`; `validate.py` keys epochs by UTC day; new open item: stations without GNSS on the reference date are dropped |
| 2026-10-09 | T34.1 finished, T34.2 report | `docs/studies/gnss_validation.md`, `studies/gnss_validation/` (`extract.py`, `validate.py`, result tables); double-differenced agreement with geepers on F08882, F11116, F39362; `gnss.align` next-day matching listed under "Out of sync" |
| 2026-10-09 | T27.10 finished | `downsample` and `downsample_hex` take `spread=` (`<layer>_std`, `_rmse`, `_mad`, weighted); gallery section |
| 2026-10-09 | T29 finished, study branch merged | Report results and decision; `studies/split_metric/` (scripts, station lists, result tables, `figure.py`); quadtree default metric `structure`; PRD S2 questions closed; new T28.6; `docs/docs_plan.md` (audit of 2026-10-04) came with the branch |
| 2026-10-06 | Decisions on codec, memory budget, notebooks, home | Blosc (zstd, bit shuffle) default for float layers (T37.3b local half); memory budget 8 GB per worker, 4 GB coordinator (T37.1 done, PRD NFR Memory); `notebooks/` deleted, version comparison moved to new T16.7; the repository stays on `mgovorcin/disp-xr` (PRD decision), so ownership is no longer out of sync |
| 2026-10-06 | T34.0 finished | `residuals.compare`: DISP noise in `chi2` (`disp_sigma`), datum offset and `relative_residual`; Houston notebook now flags 0 of 10 stations (was 9) |
| 2026-10-06 | Real-data notebooks 3 and 4 | `real_data/san_andreas` (creep step -11.9 mm/yr observed against -12.1 expected from 26 mm/yr) and `real_data/troposphere` (Copernicus GLO-30 elevation; per-epoch slope; series scatter above 1300 m 14.7 to 8.2 mm); larger site font |
| 2026-10-05 | Real-data notebooks | `docs/real_data/` (Houston F08882, production change F11116) under Tutorials > Real data; new T34.0 (`compare` outlier flag ignores DISP noise); this file renamed from `INDEX.md`, which clashed with the landing page `index.md` on case-insensitive file systems |
| 2026-10-05 | Full cycle after PRs #35–#51 | Status, requirements, modules and Out of sync rebuilt from the code and plan: production-change reading, laziness and graph fixes, requirement curves, GNSS and tutorial gallery pages (18), region jobs, web tiles, memory budget proposal; stale branch notes removed |
| 2026-10-04 | T32 and T33 finished | Requirement curves (T32.6); gallery pages `gnss_compare` and `gnss_tie` on synthetic stations (`tutorial.make_gnss_source`, `vertical_los`; geepers added to the docs environment) |
| 2026-10-03 | T33.1–T33.4 finished | `models/gnss_tie.py`: tie from `compare` output (velocity or epoch), station exclusion for deforming areas, lazy `apply_tie`, LOSO cross-validation; `gnss.sample` names a missing `units` |
| 2026-10-03 | T31 finished | `denoise` (gated Kalman with RTS or fixed-lag, robust LOWESS) on acquisitions or grids; study scores reproduced (transient 92.8% / 82.2% / 74.5%) |
| 2026-10-03 | T30 finished | `timeseries/temporal.py`: grid step from the 75th-percentile interval (study's pre segment), refused gaps, declared breaks on NumPy and xarray |
| 2026-10-03 | T32.4, T32.6b–d finished | `diagnostics.py`; `residuals.residual_spectra`, `residuals.common_mode` (geepers CME); EOF example and allowed-to-fail CI job; T32 open for the requirement thresholds (T32.6) and the gallery (T32.7) |
| 2026-10-03 | Fix: numba threads | `config["numba_threads"]` (1) sets `NUMBA_NUM_THREADS` on import; numbagg in dask tasks opened a pool per core per thread and crashed on a 224-core node |
| 2026-09-30 | First index | Created from `specs.md`, `todo.md`, `code_review.md` and the source on `main` |
| 2026-10-03 | T32.1–T32.3, T32.5 finished | `residuals.compare` / `Comparison` over geepers analysis and validation (geepers pinned to main until 0.4); normalized residuals, chi2, outlier and bad-epoch flags, cell mode |
| 2026-10-03 | T25 finished; geepers is the GNSS engine | `gnss.py` adapters over geepers sources and reader; PRD GNSS-source question closed; geepers pixel-rounding bug worked around (floor) and flagged |
| 2026-10-03 | T24 finished | `models.Trend`, lazy per-epoch `deramp` (accessor too), `Spline` and `Nearest` gridders; base drops zero-weight points |
| 2026-10-03 | T19 finished | `models/` package: estimator base, `Chain`, `Mean`, coordinates helper, block cross-validation, shared estimator test suite |
| 2026-10-03 | T28.1–T28.4 finished | `quadtree.py`: composed level sums (two data passes), pluggable split metrics, top-down boolean selection, leaf table and leaf-id raster; T28.5 open for benchmark numbers (node allocation stalls) |
| 2026-10-03 | T27 finished (T27.8 H3) | `downsample_hex(grid="h3")`, `to_geodataframe`, `cell_polygons`, `hex` extra; banded median writes by ownership |
| 2026-10-03 | Hexagon median by bands | `_banded`: median and mode per band of rows with a halo, about 128 MB per task instead of a whole epoch of the frame; node allocation stalls explained in `benchmarks/README.md` |
| 2026-10-03 | Downsampling performance | One-pass per-cell sums for squares and hexagons (graph 29% and 38% smaller), windowed hexagon partial sums (13x smaller per block on a full frame); timings reported for T27.8 withdrawn (node order effect), see `benchmarks/README.md` |
| 2026-10-03 | T27.8 planar part finished | `downsample_hex` (planar hexagons, lazily, blockwise bincount), `plot.plot_hex`, `mean_mask_correlation`; `downsample` and `downsample_hex` share `_aggregate_layers`; H3 next |
| 2026-10-03 | T27.9 finished | `to_geozarr` and `export --format geozarr`; Zarr attributes stay JSON objects; checked with bowser's own server (`tests/bowser_check.py`); `docs/bowser.md` |
| 2026-10-03 | T27.7 finished; T27.9 planned | Multiscale pyramid writer and reader (`export/multiscales.py`), `to_zarr(group=..., preset="whole")`; downsample derived variables get their own attributes; T27.9 (GeoZarr export bowser opens directly) planned after analyzing bowser `ops` (`0a61c6a`), comparison in `references.md` |
| 2026-10-03 | T27.3b finished; T27.8 planned | `residuals.py` (variogram from log-uniform random pairs, model fit, block correlation); `downsample(variogram=...)` adds `<layer>_sigma_corr` and `n_eff_corr`; `tutorial.correlated_noise`; T27.8 hexagonal downsampling (planar, then H3) added to `specs.md` S2, `todo.md` and `references.md` |
| 2026-10-03 | T27.1–T27.6 finished | `downsample` and `to_points`; per-chunk block reductions without rechunking; T27.3b and T27.7 follow in their own PRs |
| 2026-10-03 | T26 finished | `quality_weights`; real granules show residual clustering at 2 pi (not pi as planned), so cycles count 2 pi and the fixture follows; network degree from the dolphin config (`max_bandwidth = 3`) |
| 2026-10-03 | T23 finished (coherence groups; quadtree after T28) | Grouped exact GLS with Schur-complement gaps; noise shape per group, amplitude per pixel; denoised-input guard; noise-model policy proposed in the PRD; benchmark and gallery |
| 2026-10-03 | T22 finished | `fit_velocity` (OLS, Whittle, MIDAS); Whittle bias study (reliable from about 200 epochs); MIDAS on numbagg medians; benchmarks recorded |
| 2026-10-02 | T21 finished | `fit_noise` and `NoiseResult` ported from geepers (Apache-2.0 attribution added); committed geepers reference; Whittle bias on short series recorded for T22 |
| 2026-10-02 | T20 finished (M2 started before the T18 review, at the user's request) | `fit_trajectory`, `Trajectory`, `predict_trajectory`; hybrid grouped/batched solver; branch bug fixed (step and postseismic onsets); threadpoolctl dependency; benchmark and velocity gallery example |
| 2026-10-02 | T18 prepared | `api_review.md` (7 open questions), `try-it.md` (guide for Cal/Val users), `api_policy.md` draft, `.github/release.yml`; private parameter out of `to_geotiff`; deprecated names out of `__all__`. Waiting: users, tags (`v0.1.0` on `1c04c1c`), ReadTheDocs |
| 2026-10-02 | T17.4 finished; T17 done | Interactive gallery page (hvplot, datashader); pixi `interactive` feature in `docs` and `dev`; test that core imports neither |
| 2026-10-02 | T16.1–T16.5 finished | mkdocs site (pages, API reference, 4 executed gallery examples, Development section), strict docs CI job, `.readthedocs.yaml`; `disp_xr.tutorial`; `to_geotiff` takes a Dataset and tags full provenance, `--format cog`; provenance of reopened stores fixed; CHANGELOG brought up to date; publishing on ReadTheDocs waits for a connected account |
| 2026-10-02 | T6.7 added and finished | Native default chunks, chunk-preserving `stack_disp`, `rechunk`/`time_contiguous`, bounded dask threads for plots and exports; fit fixed for split time chunks and `open_disp` coordinates; decision logged |
| 2026-10-02 | T17.1–T17.3 finished | `plot.py` (`plot_map`, `plot_point`, diverging colormap), accessor wiring, Agg smoke tests, gallery script; matplotlib-base in the pixi `test` feature; T17.4 waits for T16 |
| 2026-10-02 | T15 finished | `disp_xr info`, `stack`, `export`, `geometry`, `search`, `virtual` with one-line user errors; logging config fixed (formatter class name); `stack_disp(start=)` accepts a plain date |
| 2026-10-02 | T14.2c finished; T14 done | Virtual frames in Icechunk (`virtual.py`): one commit per ministack, container per directory or bucket, validation against h5py; M1 status row corrected (was 1 / 10, now 5 / 10) |
| 2026-10-02 | T14.2b finished | `to_icechunk` / `open_icechunk` with append checks, tags and branches; `"ministack"` chunk preset; pixi `cloud` feature (icechunk 2.2, VirtualiZarr 2.7) and `test-cloud` CI job |
| 2026-10-02 | T14.3, T14.5 finished | `to_mintpy` rewritten on xarray (upstream bug 6 fixed, spec constants, block writes, MintPy azimuth convention); `utils.get_chunks_indices` deleted |
| 2026-10-02 | T14.1, T14.2, T14.2a, T14.4 finished | GeoTIFF (COG), Zarr v3 (shards, zstd, explicit fill values, `proj:`/`spatial:`) and NetCDF writers with provenance; `open_disp` records input granules |
| 2026-10-02 | T13 finished | `correct` subtracts product or external corrections once and refuses DISP-NISAR ionosphere and chained data; plate motion from the ITRF2014-PMM paper; `geometry.match_grid` shared by static layers and corrections |
| 2026-10-02 | T12.1–T12.4 finished | `open_static` on DISP-S1-STATIC (new product found in CMR), CSLC-STATIC stitcher without GDAL, LOS helpers; last `opera_utils` import removed with its ruff and doctest exemptions; T12.5 waits for DISP-NI-STATIC |
| 2026-10-02 | T11 finished | `rereference` with ported point/median/border/high_coherence (upstream bug 5 fixed), area by bbox or polygon, deterministic `auto`; `geometry.lonlat_to_rowcol`; bug 7 fixed |
| 2026-10-02 | T10 finished | `mask` / `valid_mask` presets `recommended` and `strict`, `quality_summary` (per pixel or per ministack), `.disp.mask` and `.disp.quality_summary`; ported `combine_quality_masks` |
| 2026-10-02 | T9 finished; migrated | Repo moved to `mgovorcin/disp-xr` (private), first CI run green; `.disp` accessor, history, units, validation; `stack_disp` keeps one time chunk per layer |
| 2026-10-02 | T8 prepared | Changelog and quick start; tag and milestone dates wait for a merged `main` and named maintainers |
| 2026-10-02 | T7 finished | Crossover offsets by date instead of previous epoch; `get_value_percentage` fixed for coordinates carried by `open_disp` output |
| 2026-10-02 | T6 finished | Real-frame timing drove the 1024-pixel read block and the one-file-per-production check; T6.5b moved to T14.2c |
| 2026-10-02 | T5 finished | Product registry and parser checked on real F44055, F39355 and DISP-NISAR golden data; DISP-NISAR spec provisional (no official granule); `get_disp_info` deprecation moved to T7 |
| 2026-10-02 | T4 finished | 13 commits on `refactor/package-layout`; plan: T4.1 creates modules when filled, `geometry` is a package, T4.8 deletions moved to T10.3 and T14.3; new bug 9 (search frame filter) found and fixed; T1.4 done |
| 2026-10-01 | T3 mostly finished | CI workflow, per-Python pixi environments, banned-import rule; T3.4 badge and T3.5 branch protection left for repo admins |
| 2026-10-01 | T2 finished | Added `tests/` (fixture factory, layout record from real F44055 and F11116 granules, characterization tests); pytest config with doctests; pixi `cov` task. New in `code_review.md`: bug 8 and four smaller findings |
| 2026-10-01 | T1 subtasks finished (T1.1–T1.3, T1.5–T1.7) | Ticked in `todo.md`; added T1.7 (pixi); `check_plan.py` accepts ticked boxes. Verified: `mamba env create`, `pip install -e .`, `pip check`, all pre-commit hooks pass; pixi `default`, `test`, `docs`, `dev` environments install and import `disp_xr` |
| 2026-10-01 | `specs.md` changed (opera_utils decision applied, audit findings, earth-mover, Zarr and xarray-ecosystem research) | Added `porting.md`, `references.md`, `check_plan.py`. PRD: F14–F16, storage rules, ecosystem libraries, 4 new NFR rows, correlated-error cell sigma, odd downsampling factors. Plan: 226 subtasks (was 209); T4.5, T4.7, T5, T6, T7, T10, T11, T12, T14 rewritten as ports; C4 moved from T28 to T27.7; NISAR trade-study frames moved to T35.5. Cleared the four contradictions listed in the first index |
