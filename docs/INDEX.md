# disp-xr project index

**Last indexed:** 2026-10-03 · `main` on `origin` = `mgovorcin/disp-xr`, through T27.1–T27.6 (regular-grid downsampling); branch `feat/variogram-sigma` (on PR #16) adds T27.3b

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
| [check_plan.py](check_plan.py) | Mechanical check of `todo.md` (run in every indexing cycle) | — |
| [img/](img/) | `architecture.png`, `roadmap.png`, exported from the PRD doc | — |
| `CLAUDE.md` (repository root) | Coding rules and the indexing cycle | — |

The PRD also lives as a shared doc; `specs.md` is its copy in the repo and must match it.

## Status

| Milestone | Tasks | Done |
| --- | --- | --- |
| M0 Foundation (v0.1) | T1–T8 | 5 / 8 (T2, T4, T5, T6, T7 done; T1 open only for T1.4b; T3 open for T3.4–T3.5) |
| M1 Core API (v0.2) | T9–T18 | 7 / 10 (T9, T10, T11, T13, T14, T15, T17; T12 open only for T12.5, T16 only for ReadTheDocs publishing) |
| M2 Analysis (v0.3) | T19–T34 | 6 / 16 (T20, T21, T22, T23, T26, T27; started before the T18 review) |
| M3 Release (v1.0) | T35–T39 | 0 / 5 |

Six local branches, stacked, not pushed:

- `feature/t1-packaging`: annotation fix, T1 (packaging, pixi), T2 (tests), T3 (CI), planning docs.
- `chore/release-v0.1` (on top): T8.1 and T8.3 (changelog, README quick start).
- `feat/chain-ministacks`: T7. `stack_disp` chains ministacks from the crossover epoch by date (bug 3 fixed); old entry points (`get_disp_info`, `combine_disp_product`, `rebase_disp`, `create_rebased_displacement`) are deprecated wrappers.
- `feat/open-disp`: T6. `open_disp` and `open_disp_tree` (lazy, CRS attached, integer layers keep dtype, mixed productions refused), chunk presets from a byte target, remote access over S3/HTTPS/fsspec with Earthdata and ASF credentials (ported), faster `check_granules`.
- `feat/product-registry`: T5. `ProductSpec` registry (DISP-S1; DISP-NISAR provisional), `DispGranule` parsing from names or metadata, `find_products`, `group_ministacks`, `check_granules`; `get_disp_info` rebuilt on it (bugs 1 and 8 fixed).
- `refactor/package-layout`: T4. Merges `origin/timeseries_fitting`, makes jax optional, moves modules to the new layout behind deprecated aliases, fixes the CLI entry point, adds the top-level API, opera-utils attribution, the search port (with the bug 9 fix) and `disp_xr.config`.

Still open in M0: T1.4b (name maintainers), T3.4 badge and T3.5 branch protection (GitHub admin), then T8 (release v0.1: needs a merged, pushed `main` and a tag). Bug 7 fixed in T11.

## Requirements to tasks

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
| F9 Spatial models | T19, T24, T33 | Not started |
| F10 GNSS | T25 | Not started |
| F11 Multi-frame | T36 | Not started |
| F12 Export | T14 | Done: GeoTIFF, Zarr v3, NetCDF, MintPy, Icechunk, virtual stores |
| F13 Plot | T17 | Done: `plot_map`, `plot_point`, accessor, interactive gallery page |
| F14 CLI | T15 | Done: `info`, `stack`, `export`, `geometry`, `search`, `virtual` |
| F15 Plate motion | T13.5 | Done: `plate_motion_los`, `remove_plate_motion` (ITRF2014-PMM) |
| F16 Search | T4.5 | Exists, S1 only (`download.py`) |
| S1 Colored-noise velocity | T20, T21, T22, T23 | Done: OLS, one-series colored noise, per-pixel Whittle and MIDAS, frame GLS with coherence groups; quadtree groups wait for T28; noise-model policy proposed, awaiting confirmation |
| S2 Downsampling | T26, T27, T28, T29 | T26 and T27 done (quality weights, regular grid, variogram sigma, pyramids, bowser GeoZarr, planar and H3 hexagons); quadtree (T28) and trade study (T29) open |
| S3 Temporal denoising | T30, T31 | Prototype outside repo (`00_tools/tests/temporal_denoising`) |
| S4 Residual analysis | T32 | Residual statistics on branch |
| C1 Zarr v3 output | T14.2, T14.2a | Not started |
| C2 Icechunk store (opt-in) | T14.2b | Done: `to_icechunk` / `open_icechunk`, one commit per write, append along time, tags, snapshot id in provenance |
| C3 Virtual datasets (opt-in) | T5.8, T14.2c | Done for local granules: `build_virtual_frame`, `open_virtual`, `validate_virtual`; S3 path not run against S3; `/corrections` not referenced |
| C4 Multiscale pyramids | T27.7 | Not started |
| C5 Map serving example | T37.5 | Not started |
| C6 Batch builds and benchmarks | T37.3b, T37.4 | Not started |
| X DataTree | T6.4b, T36.1b | `open_disp_tree` done |
| X flox grouped reductions | T10.3, T23.2, T32.1 | Started (T10.3 `by_ministack`) |
| X apply_ufunc kernels | T7.2, T20.3, T22.3 | Not started |
| NFR Packaging, Licensing | T1, T4.7, T38 | In progress (T1) |
| NFR Testing, typing | T2, T3 | T2 done; T3 written, not yet run on GitHub |
| NFR Compatibility | T5.7, T35 | Not started |
| NFR Metadata, Provenance | T14.2a, T14.4 | Not started |
| NFR Benchmarks, Performance, Memory | T37 | Not started |
| NFR Docs | T16 | mkdocs site with executed gallery and strict CI build done; ReadTheDocs publishing waits for an account |
| API rules (config, history, order) | T4.9, T9, T15.2 | Config and history done; processing order in T15.2 |

## Modules (branch `refactor/package-layout`, `src/disp_xr`)

| Module | Requirement | State | Next task |
| --- | --- | --- | --- |
| `product.py` | F1 | `ProductSpec`, `DispGranule`, `find_products`, `group_ministacks`, `check_granules`; `get_disp_info` kept for the stack | T7 (deprecate `get_disp_info`) |
| `stack.py` | F3 | `stack_disp`, `NaNPolicy`, `MissingCrossoverError`; deprecated wrappers | T38 (remove wrappers) |
| `quality.py` (was `quality_metrics.py`) | F4 | Masks, presets, quality summary, lazy statistics | T10 |
| `reference.py` (from `utils.py`) | F5 | `rereference`, `get_reference_values`, `find_stable_point`; bug 7 fixed | T11 |
| `geometry/__init__.py` (from `utils.py`) | F7 | Coordinate helpers, `get_burst_ids` | T12 |
| `geometry/static.py` (was `static_layers.py`) | F7 | `open_static`, DISP-S1-STATIC search and cached download, `stitch_cslc_static` | T12 |
| `corrections.py` | F6, F15 | `correct`, `remove_plate_motion`, ITRF2014-PMM poles | T13 |
| `geometry/los.py` | F7 | `enu_to_los`, incidence/azimuth, slant range | T12.2, T12.5 |
| `export/raster.py` (was `io.py`) | F12 | `to_geotiff` (COG), legacy rasterio helpers | T14.1 |
| `export/zarr.py` | F12, C1 | `to_zarr`, `to_netcdf`, chunk/shard scheme (presets `timeseries`, `maps`, `ministack`), `round_mantissa` | T14.2, T14.2b |
| `export/icechunk.py` | F12, C2 | `to_icechunk`, `open_icechunk` (`cloud` extra, imported lazily) | T14.2b |
| `virtual.py` | C3 | `build_virtual_frame`, `open_virtual`, `validate_virtual` (`cloud` extra) | T14.2c |
| `export/_provenance.py` | NFR Provenance, Metadata | Provenance and CF flag attributes | T14.4 |
| `export/mintpy.py` (was `mintpy.py`) | F12 | `to_mintpy`: timeseries, quality masks, geometry (h5py only) | T14.3 |
| `open.py` | F2 | `open_disp`, `open_disp_tree` | T7 (chaining on top) |
| `_chunks.py`, `_remote.py`, `_credentials.py` | F2, C1 | Chunk presets; remote access (ported) | — |
| `search.py` (was `download.py`) | F16 | Ported CMR search, session seam, NISAR short name; bug 9 fixed | T5.2 (NISAR spec) |
| `tutorial.py` | NFR Docs, tests | `make_disp_stack`: synthetic granules with a known truth (was `tests/fixtures/make_disp.py`); `correlated_noise`: a field with a known exponential covariance | T2.3, T16.3, T27.3b |
| `export/geozarr.py` | C4 | `to_geozarr` (bowser and standard profiles), `display_factors`, `bowser_attrs`, `reference_lonlat`, `los_attrs`, `histogram_stats` | T27.9 |
| `export/multiscales.py` | C4 | `build_overviews`, `to_pyramid`, `open_pyramid`, `MULTISCALES_CONVENTION` | T27.7, T27.9 |
| `hexagons.py` | S2 (hexagon mode) | `downsample_hex` (planar and H3), `to_geodataframe`, `cell_polygons`, `hex_axial`, `hex_centers`, `hexagon_vertices` | T27.8 |
| `downsample.py` | S2 | `downsample`, `to_points` | T27 |
| `residuals.py` | S2 (cell uncertainty), S4 | `empirical_variogram`, `fit_variogram`, `VariogramModel`, `mean_block_correlation`, `mean_mask_correlation`, `n_eff_correlated` | T27.3b, T27.8, T32.1 |
| `weights.py` | S2 | `quality_weights`, `residual_cycles`, `network_degree`, `residual_cycle_histogram` | T26 |
| `plot.py` | F13 | `plot_map`, `plot_point`, `diverging_cmap` (`plot` extra, imported lazily) | T17 |
| `accessor.py`, `_history.py`, `units.py`, `validate.py` | API rules | `.disp` accessor, operation history, unit conversion, validation | — |
| `config.py` | API rules | `donfig` settings, `DISP_XR_*` overrides | — |
| `cli/__init__.py` | F14, F16 | `disp_xr` commands wrapping library functions | T15 |
| `timeseries/` | F8, S1, S4 | `trajectory.py` (`fit_trajectory`), `noise.py` (`fit_noise`, from geepers), `velocity.py` (`fit_velocity`: OLS, Whittle, MIDAS, GLS), `gls.py` (grouped GLS), design matrix, legacy `fit_timeseries_block`; jax optional | T20, T21, T22, T23, T32 |
| `utils.py` | — | Deprecated aliases only | T38 |
| `quality_metrics.py`, `io.py`, `mintpy.py`, `static_layers.py`, `download.py` | — | Deprecated aliases | T38 |
| `log/` | Infrastructure | Unchanged | — |

## Decisions log

| Date | Decision |
| --- | --- |
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

- Local checkouts `01_OPERA/STAKEHOLDER/disp-xr` and `01_OPERA/VLM/disp-xr` are on `timeseries_fitting`, which `refactor/package-layout` now contains. The conda envs `disp_xr` and `disp_xr_py311` hold non-editable installs of old commits.
- disp-xr now lives at `mgovorcin/disp-xr` (migrated 2026-10-02; `upstream` remote is `opera-adt/disp-xr`, which stays at `ce0d105`). The PRD still says the package belongs to OPERA Cal/Val; ownership is to be settled.
- `feature/t1-packaging` is not pushed and does not follow the `type/summary` branch naming of the `git-commit-pr` skill; rename it when it becomes a PR.
- Open bug with a strict xfail test: 7. Bugs 1, 3, 8 and 9 are fixed. The old notebooks still import the deprecated module names (`download`, `io`, `quality_metrics`, `static_layers`, `utils.*`); they keep working with warnings until T16 rewrites them.
- `.claude/skills` is a symlink to `../skills`, which does not exist in this repo; `.claude/` is excluded through `.git/info/exclude`, as in geepers.

## Not verified

- `code_review.md` and the Context lines of `todo.md` were not checked by a second reader against the code. The 7 bugs were read from the source when the review was written.
- `porting.md` was spot-checked only (license, filename regex, module locations, upstream bugs 1–3).
- `references.md` sections 2 and 3 come from one reader. Section 1 was checked claim by claim.
- DISP granule facts in `references.md` (chunking, dtypes, `time` units) come from a few granules; T2.2 confirms them.

## Checks passed this cycle

- `python docs/check_plan.py`: 39 tasks, 230 subtasks, no duplicate or dangling ids; "Depends on" lines, mermaid graph and dependency table agree; no cycles; every task except T39 unlocks another; headings match task titles.
- All relative links in `docs/*.md` and `CLAUDE.md` resolve.
- No sentence in `specs.md`, `todo.md` or `code_review.md` tells disp-xr to depend on, wrap or vendor opera_utils.
- Every requirement in the table above has at least one task.

## Index log

| Date | Trigger | Changes |
| --- | --- | --- |
| 2026-09-30 | First index | Created from `specs.md`, `todo.md`, `code_review.md` and the source on `main` |
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
