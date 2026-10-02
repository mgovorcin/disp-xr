# Changelog

All notable changes to disp-xr. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/).

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
