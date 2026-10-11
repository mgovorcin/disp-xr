# Changelog

All notable changes to disp-xr. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.3.0] - 2026-10-10

Science features (milestone M2), released with the core API of milestone M1.
There is no 0.2.0: the M1 API review with Cal/Val users has not taken place,
so the core API is **not frozen yet**. Renames that come out of that review
will land in a later 0.x release with deprecation shims (see
`docs/api_policy.md`).

This release passes the M2 gate, "GNSS tie validated"
(`docs/studies/gnss_validation.md`). On three DISP-S1 frames, disp-xr's GNSS
comparison matches the geepers workflow to rounding once both are
double-differenced to a common GNSS station. The GNSS tie improves the
leave-one-station-out score on every frame: the median after the tie is about
1 mm/yr. Several fixes from that validation change values; they are marked
**changes values** or **changes results** below.

### Added

- `cross_validate_tie` reports `median_before` and `median_after` in
  `attrs`, the median of the per-station leave-one-out scores. The RMSE is
  driven by single stations (local subsidence, short records). On the T29
  frames the median improves on every frame, while the RMSE does not.

- `gnss.drop_relocations`, applied by `gnss.load(max_step_m=10.0)`: a 3-D
  step above 10 m between consecutive solutions marks a monument moved under
  the same station id, not ground motion (near-field coseismic steps reach
  about 5 m). The record is split there and only the longest segment is
  kept. The new `relocated` coordinate flags
  such stations. Found on GINN (UNR, Canberra): a 1 km jump between 2019
  and 2020 that reached `compare` as a 7e5 mm/yr velocity.

- `downsample(spread=...)` and `downsample_hex(spread=...)`: the spatial
  variation of each layer within a cell, weighted like the cell value:
  `<layer>_std`, `<layer>_rmse` (about the cell mean or median) and
  `<layer>_mad` (weighted median absolute deviation, robust to outlier
  pixels).

- `disp_xr.batch`: build a frame cube with independent region jobs aligned
  to shards (`init_store`, `plan_jobs`, `RegionJob` as JSON for AWS Batch
  array jobs, `run_region`, a manifest and `failed_jobs` to resubmit).

- `.disp` accessor on Datasets and DataArrays, with an operation history
  recorded in every result; `to_units`; `validate_disp`.
- Quality: `mask` and `valid_mask` (`recommended`, `strict` or per criterion),
  `quality_summary`.
- `rereference` by point, area, scene median, border, high-coherence pixels
  or a deterministic stable pixel (`auto`).
- Static layers: `open_static` (DISP-S1-STATIC, CSLC-STATIC), LOS helpers,
  `enu_to_los`.
- Corrections: `correct` (ionosphere, solid-earth tide, external layers) and
  `remove_plate_motion` (ITRF2014 plate motion model).
- Export: `to_zarr` (Zarr v3, sharded, zstd), `to_netcdf`, `to_geotiff`
  (Cloud Optimized GeoTIFF), `to_mintpy`, all with provenance; `to_icechunk`,
  `open_icechunk`, and virtual frames (`build_virtual_frame`, `open_virtual`,
  `validate_virtual`) with the `cloud` extra.
- `disp_xr` command line: `info`, `stack`, `export`, `geometry`, `search`,
  `virtual`.
- Quick-look plots: `plot_map`, `plot_point` (`plot` extra).
- `rechunk` and a `native` chunk preset; `disp_xr.tutorial.make_disp_stack`
  for synthetic granules.
- Documentation site (mkdocs) with a gallery of executed examples.
- `fit_trajectory` with `Trajectory` terms (polynomial, seasonal, steps,
  postseismic log/exp) and `predict_trajectory`: velocity, amplitudes and
  their white-noise sigmas for every pixel.
- `fit_noise`: colored-noise trajectory fit for one series (WN, FNWN, PLWN,
  RWFNWN by restricted maximum likelihood, or Whittle), ported from geepers;
  returns realistic velocity sigmas, kappa, noise amplitudes, AIC and BIC.
- `fit_velocity(ds, method="ols" | "whittle" | "midas")` for every pixel:
  OLS, Whittle colored-noise sigmas (reliable from about 200 epochs), or the
  robust MIDAS estimator.
- `fit_velocity(method="gls")`: exact colored-noise GLS for every pixel, with
  the noise shape estimated per group of pixels (coherence bins or a label
  map) by exact maximum likelihood and the amplitude per pixel;
  `noise="auto"` picks the model per group by BIC.
- `fit_velocity` and `fit_trajectory` refuse data marked denoised unless
  `allow_denoised=True`.
- `quality_weights`: per-epoch and static pixel weights from phase
  similarity, temporal coherence and inversion residuals (cycle jumps in
  2 pi down-weight, never exclude); `residual_cycle_histogram`,
  `network_degree`.
- `downsample` (weighted mean or median to a regular coarse grid, with cell
  sigma, effective count, valid fraction and quality; mode or any for masks)
  and `to_points`.
- `quadtree`: quality-weighted quadtree leaves (displacement, velocity or
  structure split metric), by threshold or by a target number of points
  (`n_points`).
- `residuals.compare` scores the OPERA DISP-S1 requirements by default:
  relative displacement 3 (1 + sqrt(L)) mm on the pairwise RMSE profile and
  relative velocity 2 mm/yr on station-pair velocity differences
  (`Requirement`, `DISP_S1_DISPLACEMENT`, `DISP_S1_VELOCITY`,
  `Comparison.velocity_pairs`).
- `disp_xr.tutorial.make_gnss_source` and `vertical_los`: synthetic GNSS
  stations that follow a DISP frame, through the geepers source interface.
- Gallery: comparing DISP with GNSS, and tying DISP to GNSS.
- Gallery: serving results as web map tiles with xpublish-tiles (docs
  environment only, `tiles` pixi feature).

### Changed

- `residuals.compare(min_epochs=10, min_span_years=2.0)`: a station's
  velocities count only with at least 10 epochs shared by DISP and GNSS,
  over at least 2 years. Below that, `insar_velocity`, `gps_velocity` and
  their sigmas are NaN and the new `velocity_usable` column is False. The
  station's displacement series still enters the pairs and the epoch RMSE.
  Without the gate, rates from 2 epochs (94 mm/yr on a DISP-S1 frame)
  entered the velocity residuals, the outlier flag and the GNSS tie.
  **This changes results** for short records. Pass `min_span_years=0` to
  compare short stacks.

- `quadtree` splits by the `"structure"` metric by default (was
  `"displacement"`), the best of the three in the T29 trade study
  (`docs/studies/split_metric.md`). Pass `metric="displacement"` to keep the
  old behavior; thresholds of the two metrics are in different units. The
  study recommends the regular grid of `downsample` as the default
  downsampling.

- `to_zarr` compresses float layers with Blosc (zstd, bit shuffle) instead of
  plain zstd: displacement stores 16% smaller and coherence maps 32% smaller
  at the same speed. Quality layers keep zstd in chunks that span time (it
  compresses them twice as well there), as do integer and boolean layers.
  Lossless; stores written before read as before.

- `residuals.compare` flags fewer stations: the reduced `chi2` now adds a
  DISP noise term (`disp_sigma`, estimated from all stations' residuals by
  default) to the GNSS sigma, which alone flagged almost every station on
  real frames (9 of 10 on F08882). `normalized_residual` is computed from the
  new `relative_residual`, the velocity residual less the datum offset (the
  median over stations, now `summary["datum_offset"]`), since DISP is
  relative to its reference and GNSS is absolute.

- Export and plotting check `max_load_bytes` from shape and dtype before
  loading, and raise `LoadTooLargeError` (a `ValueError`) naming the size,
  the limit and the remedy. New page: large frames and batch jobs.

- `open_disp` keeps the file layout by default (`chunks="native"`), and
  `stack_disp` keeps the input chunks, so a map or a point reads only what it
  needs. Per-pixel functions rechunk themselves.
- Plots and exports compute with at most 32 dask threads
  (`config["num_workers"]`).
- `stack_disp(start=)` accepts a plain date.
- `to_geotiff` accepts a Dataset and a `layer`, and tags the input granules,
  history and package versions.

### Deprecated

- `get_disp_info` and `combine_disp_product` (already warning) are no longer
  in `disp_xr.__all__`; they stay importable until T38.

### Fixed

- `residuals.compare` dropped every station without a GNSS solution on the
  DISP reference date (installed later, or a gap that day): `gnss.align`
  zeroes GNSS there, so the whole series was NaN. On the T29 frames this
  was 12–23 % of the stations with 10 or more GNSS epochs. `compare` now
  references both InSAR and GNSS per station to the first epoch they share
  (new `reference_epoch` column in `stations`). This leaves stations with
  GNSS on the reference date unchanged, and gives the others a double
  difference in time with no offset. `gnss.align` on its own references
  such a station to its first GNSS epoch (new `gnss_reference`
  coordinate). **This changes results:** stations are added to the
  velocities, pairs and epoch statistics.

- `gnss.align` and `residuals.compare` matched each DISP epoch to the GNSS
  solution with the nearest timestamp. Daily solutions are stamped at
  midnight, so every acquisition after 12:00 UTC took the next day's
  solution. Daily series are now treated as stamped at noon (new
  `gnss.centred`), so an epoch takes the solution of its own UTC day, for
  both the `nearest` and `linear` methods and for the GNSS sigma. **This
  changes values** on afternoon frames: on DISP-S1 F11116 (14:07 UTC) and
  F39362 (19:16 UTC) the GNSS at an epoch moves by 2–3 mm (median), and
  velocities by up to 0.4 mm/yr (95th percentile). Frames acquired near
  00:00 UTC are unchanged. Found in the T34 validation
  (`docs/studies/gnss_validation.md`).

- `fit_trajectory` and `fit_velocity` (OLS, Whittle) took hours to build their
  graph on a full frame: default weights were a NumPy array broadcast to the
  frame, which dask hashed whole (2 h on a 389-epoch DISP-S1 frame, now 0.1 s
  on 60 epochs). `downsample`, `downsample_hex` and `quadtree` embedded
  frame-sized x/y grids (about 500 MB each) in their graphs; they are now
  built lazily.

- `fit_velocity`, `fit_trajectory` and `downsample` outputs lacked the
  `grid_mapping` attribute on their map variables, so readers that look up
  the CRS per variable (GDAL, xpublish-tiles) missed it.

- `quality_weights` (and so `downsample` and `quadtree` with
  `weights="quality"`) loaded the whole inversion-residual stack into memory:
  `np.round` on a dask-backed DataArray falls back to `np.asarray`. It now
  stays lazy (53 GB for a 91-epoch DISP-S1 frame before).
- `open_disp` refused DISP-S1 frames that span the November 2025 production
  change (displacement stored as float64 instead of float32, every other
  layer alike). Granules that differ only in float precision are now read
  together with a `MixedPrecisionWarning`, cast to float32 and listed in
  `attrs["mixed_precision"]`; virtual stores still refuse them.
- Ministacks of DISP-S1 productions from November 2025 did not chain to
  earlier ones (`MissingCrossoverError`): their time is float seconds since
  2010 and decodes nanoseconds off. `open_disp` now reads times to the whole
  second, as granule names give them.

- The command line failed at start-up (logging formatter name).
- `fit_timeseries_block` fitted each chunk's epochs separately when time was
  split across chunks, and failed on `open_disp` output.
- Re-exporting a store split its `input_granules` list into characters.
- Trajectory steps and postseismic terms never fired: their onsets were
  compared in absolute years with time relative to the reference.
- `rereference("area")` read every chunk of the frame for the median of a
  small box; it now reads only the rows and columns around the area.
- `quadtree` cells on the frame edge (and cells larger than the frame) never
  split: `min_reliable` counted the children's pixels outside the frame. It
  is now a share of the pixels inside the frame.
- `fit_velocity`, `downsample`, `quadtree` and the other functions that align
  chunks failed on a stack read back from Zarr, whose per-epoch coordinate
  `reference_time` is one chunk while the data have one chunk per epoch.

## [0.1.0] - 2026-10-02

First tagged release: a tested, installable foundation for opening and chaining
OPERA DISP frames.

### Added

- `find_products`: one row per granule from a folder, a file list or a remote
  prefix; duplicates resolve to the highest version, then the latest
  production; unsupported versions and mixed frames are refused.
- `ProductSpec` registry for DISP-S1 and DISP-NISAR (DISP-NISAR provisional:
  identified from file metadata until an official granule name exists).
- `open_disp` and `open_disp_tree`: a lazy `(time, y, x)` dataset or DataTree
  with the CRS attached; integer layers keep their dtype; granules from
  mixed productions or other grids are refused.
- `stack_disp`: ministacks chained to one reference date, with offsets taken
  from the crossover epoch by date; `NaNPolicy` for NaN crossovers.
- Remote access over S3, HTTPS and other fsspec URLs, with Earthdata Login and
  ASF temporary credentials.
- `disp_xr.config` with `DISP_XR_*` environment overrides; chunk presets sized
  from a byte target.
- `search` takes an HTTP session and a CMR collection name.
- `disp_xr.timeseries` (from the `timeseries_fitting` branch), with JAX
  optional.
- pixi environments, a test suite on synthetic granules that copy the real
  layout, CI on Python 3.12 to 3.14.

### Fixed

- Product versions were compared as strings (1.10 lost to 1.9).
- `get_disp_info` crashed under pandas 3 when versions differed.
- With `start_date`, the chained stack's zero epoch had the wrong date, and its
  quality layers were filled with zeros.
- `search` dropped the frame filter when no product version was given.
- The `disp_xr` command-line entry point did not resolve.
- `pip install` installed no dependencies.

### Changed

- Modules moved: `quality_metrics` to `quality`, `io` to `export.raster`,
  `mintpy` to `export.mintpy`, `static_layers` to `geometry.static`,
  `download` to `search`, and `utils` helpers to `geometry` and `reference`.

### Deprecated

- The old module names above, `get_disp_info`, `combine_disp_product`,
  `rebase_disp`, `create_rebased_displacement` and `geometry.get_extent`. They
  still work with a `DeprecationWarning` and will be removed before 1.0.

### Removed

- The dependency on opera-utils. The functions disp-xr needs are ported with
  attribution (see `THIRD_PARTY_NOTICES.md`).
