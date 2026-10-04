# Changelog

All notable changes to disp-xr. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

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

### Changed

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

## [0.1.0] - Unreleased

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
