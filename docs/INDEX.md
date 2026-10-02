# disp-xr project index

**Last indexed:** 2026-10-02 · branch `feat/open-disp` (on `feat/product-registry`, on `refactor/package-layout`, on `feature/t1-packaging`, on `main` `ce0d105`); nothing pushed

How and when this file is rebuilt: see "Project Index (cyclical indexing)" in [CLAUDE.md](../CLAUDE.md).

## Documents

| File | Answers | Lines |
| --- | --- | --- |
| [specs.md](specs.md) | What to build: requirements F1–F16, S1–S4, C1–C6, storage rules, non-functional rows, open questions | 324 |
| [todo.md](todo.md) | In what order: tasks T1–T39 (228 subtasks), dependencies, milestones M0–M3 | 622 |
| [code_review.md](code_review.md) | What exists: keep / refactor / drop per module, 7 bugs, the `timeseries_fitting` branch | 161 |
| [porting.md](porting.md) | What is ported from opera_utils, where it goes, license terms, upstream bugs to fix | 132 |
| [references.md](references.md) | Outside practices followed and their sources: earth-mover repositories, Zarr docs, xarray ecosystem | 206 |
| [check_plan.py](check_plan.py) | Mechanical check of `todo.md` (run in every indexing cycle) | — |
| [img/](img/) | `architecture.png`, `roadmap.png`, exported from the PRD doc | — |
| [../CLAUDE.md](../CLAUDE.md) | Coding rules and the indexing cycle | — |

The PRD also lives as a shared doc; `specs.md` is its copy in the repo and must match it.

## Status

| Milestone | Tasks | Done |
| --- | --- | --- |
| M0 Foundation (v0.1) | T1–T8 | 4 / 8 (T2, T4, T5, T6 done; T1 open only for T1.4b; T3 open for T3.4–T3.5) |
| M1 Core API (v0.2) | T9–T18 | 0 / 10 |
| M2 Analysis (v0.3) | T19–T34 | 0 / 16 |
| M3 Release (v1.0) | T35–T39 | 0 / 5 |

Four local branches, stacked, not pushed:

- `feature/t1-packaging`: annotation fix, T1 (packaging, pixi), T2 (tests), T3 (CI), planning docs.
- `feat/open-disp` (on top): T6. `open_disp` and `open_disp_tree` (lazy, CRS attached, integer layers keep dtype, mixed productions refused), chunk presets from a byte target, remote access over S3/HTTPS/fsspec with Earthdata and ASF credentials (ported), faster `check_granules`.
- `feat/product-registry`: T5. `ProductSpec` registry (DISP-S1; DISP-NISAR provisional), `DispGranule` parsing from names or metadata, `find_products`, `group_ministacks`, `check_granules`; `get_disp_info` rebuilt on it (bugs 1 and 8 fixed).
- `refactor/package-layout`: T4. Merges `origin/timeseries_fitting`, makes jax optional, moves modules to the new layout behind deprecated aliases, fixes the CLI entry point, adds the top-level API, opera-utils attribution, the search port (with the bug 9 fix) and `disp_xr.config`.

Still open in M0: T1.4b (name maintainers), T3.4 badge and T3.5 branch protection (GitHub admin), then T7 and T8. Tests: 131 passed, 3 strict xfails for open bugs (3 twice, 7).

## Requirements to tasks

| Requirement | Tasks | Status |
| --- | --- | --- |
| F1 Discover | T5 | Done on `feat/product-registry` |
| F2 Open | T6 | Done on `feat/open-disp` (local and remote) |
| F3 Stack | T7 | Exists, buggy (`stack.py`); to be replaced by a port |
| F4 Mask | T10 | Partial (`quality_metrics.py`) |
| F5 Reference | T11 | Partial (`utils.find_reference_point`) |
| F6 Correct | T13 | Not started |
| F7 Geometry | T12 | Exists (`static_layers.py`), imports opera_utils |
| F8 Time series | T20 | On branch `timeseries_fitting` |
| F9 Spatial models | T19, T24, T33 | Not started |
| F10 GNSS | T25 | Not started |
| F11 Multi-frame | T36 | Not started |
| F12 Export | T14 | Exists, buggy (`io.py`, `mintpy.py`) |
| F13 Plot | T17 | Not started |
| F14 CLI | T15 | Entry point broken |
| F15 Plate motion | T13.5 | Not started |
| F16 Search | T4.5 | Exists, S1 only (`download.py`) |
| S1 Colored-noise velocity | T20, T21, T22, T23 | OLS only, on branch |
| S2 Downsampling | T26, T27, T28, T29 | Not started |
| S3 Temporal denoising | T30, T31 | Prototype outside repo (`00_tools/tests/temporal_denoising`) |
| S4 Residual analysis | T32 | Residual statistics on branch |
| C1 Zarr v3 output | T14.2, T14.2a | Not started |
| C2 Icechunk store (opt-in) | T14.2b | Not started |
| C3 Virtual datasets (opt-in) | T5.8, T14.2c | `check_granules` done; virtual store not started (moved from T6.5b) |
| C4 Multiscale pyramids | T27.7 | Not started |
| C5 Map serving example | T37.5 | Not started |
| C6 Batch builds and benchmarks | T37.3b, T37.4 | Not started |
| X DataTree | T6.4b, T36.1b | `open_disp_tree` done |
| X flox grouped reductions | T10.3, T23.2, T32.1 | Not started |
| X apply_ufunc kernels | T7.2, T20.3, T22.3 | Not started |
| NFR Packaging, Licensing | T1, T4.7, T38 | In progress (T1) |
| NFR Testing, typing | T2, T3 | T2 done; T3 written, not yet run on GitHub |
| NFR Compatibility | T5.7, T35 | Not started |
| NFR Metadata, Provenance | T14.2a, T14.4 | Not started |
| NFR Benchmarks, Performance, Memory | T37 | Not started |
| NFR Docs | T16 | Not started |
| API rules (config, history, order) | T4.9, T9, T15.2 | Not started |

## Modules (branch `refactor/package-layout`, `src/disp_xr`)

| Module | Requirement | State | Next task |
| --- | --- | --- | --- |
| `product.py` | F1 | `ProductSpec`, `DispGranule`, `find_products`, `group_ministacks`, `check_granules`; `get_disp_info` kept for the stack | T7 (deprecate `get_disp_info`) |
| `stack.py` | F2, F3 | Chaining works; bug 3; holds unattributed-source copies of the opera-utils rebasing code (header added) | T6, T7 |
| `quality.py` (was `quality_metrics.py`) | F4 | Lazy statistics | T10 |
| `reference.py` (from `utils.py`) | F5 | `find_reference_point`; bug 7 | T11 |
| `geometry/__init__.py` (from `utils.py`) | F7 | Coordinate helpers, `get_burst_ids` | T12 |
| `geometry/static.py` (was `static_layers.py`) | F7 | Static-layer download; still imports opera_utils | T12.1 |
| `export/raster.py` (was `io.py`) | F12 | rasterio file I/O | T14 |
| `export/mintpy.py` (was `mintpy.py`) | F12 | Old writer; bugs 4–6 | T14.3 |
| `open.py` | F2 | `open_disp`, `open_disp_tree` | T7 (chaining on top) |
| `_chunks.py`, `_remote.py`, `_credentials.py` | F2, C1 | Chunk presets; remote access (ported) | — |
| `search.py` (was `download.py`) | F16 | Ported CMR search, session seam, NISAR short name; bug 9 fixed | T5.2 (NISAR spec) |
| `config.py` | API rules | `donfig` settings, `DISP_XR_*` overrides | — |
| `cli/__init__.py` | F14 | Empty `disp_xr` group; entry point works | T15 |
| `timeseries/` | F8, S1, S4 | From `timeseries_fitting`; jax optional | T20, T32 |
| `utils.py` | — | `get_chunks_indices` plus deprecated aliases | T14.3 |
| `quality_metrics.py`, `io.py`, `mintpy.py`, `static_layers.py`, `download.py` | — | Deprecated aliases | T38 |
| `log/` | Infrastructure | Unchanged | — |

## Decisions log

| Date | Decision |
| --- | --- |
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
- `geometry/static.py` (and its alias `static_layers`) still imports `opera_utils`, so it fails to import in the new environments; the ruff ban exempts it until T12.1.
- `pyproject.toml` URLs point to `opera-adt/disp-xr`, the live repo, while the PRD says the package belongs to OPERA Cal/Val. The old `OPERA-Cal-Val/disp_xr_tools` URL redirects to the personal repo `mgovorcin/disp_xr_tools`.
- `feature/t1-packaging` is not pushed and does not follow the `type/summary` branch naming of the `git-commit-pr` skill; rename it when it becomes a PR.
- Open bugs with strict xfail tests: 3, 7. Bugs 1, 8 and 9 are fixed. The old notebooks still import the deprecated module names (`download`, `io`, `quality_metrics`, `static_layers`, `utils.*`); they keep working with warnings until T16 rewrites them.
- `.claude/skills` is a symlink to `../skills`, which does not exist in this repo; `.claude/` is excluded through `.git/info/exclude`, as in geepers.

## Not verified

- `code_review.md` and the Context lines of `todo.md` were not checked by a second reader against the code. The 7 bugs were read from the source when the review was written.
- `porting.md` was spot-checked only (license, filename regex, module locations, upstream bugs 1–3).
- `references.md` sections 2 and 3 come from one reader. Section 1 was checked claim by claim.
- DISP granule facts in `references.md` (chunking, dtypes, `time` units) come from a few granules; T2.2 confirms them.

## Checks passed this cycle

- `python docs/check_plan.py`: 39 tasks, 228 subtasks, no duplicate or dangling ids; "Depends on" lines, mermaid graph and dependency table agree; no cycles; every task except T39 unlocks another; headings match task titles.
- All relative links in `docs/*.md` and `CLAUDE.md` resolve.
- No sentence in `specs.md`, `todo.md` or `code_review.md` tells disp-xr to depend on, wrap or vendor opera_utils.
- Every requirement in the table above has at least one task.

## Index log

| Date | Trigger | Changes |
| --- | --- | --- |
| 2026-09-30 | First index | Created from `specs.md`, `todo.md`, `code_review.md` and the source on `main` |
| 2026-10-02 | T6 finished | Real-frame timing drove the 1024-pixel read block and the one-file-per-production check; T6.5b moved to T14.2c |
| 2026-10-02 | T5 finished | Product registry and parser checked on real F44055, F39355 and DISP-NISAR golden data; DISP-NISAR spec provisional (no official granule); `get_disp_info` deprecation moved to T7 |
| 2026-10-02 | T4 finished | 13 commits on `refactor/package-layout`; plan: T4.1 creates modules when filled, `geometry` is a package, T4.8 deletions moved to T10.3 and T14.3; new bug 9 (search frame filter) found and fixed; T1.4 done |
| 2026-10-01 | T3 mostly finished | CI workflow, per-Python pixi environments, banned-import rule; T3.4 badge and T3.5 branch protection left for repo admins |
| 2026-10-01 | T2 finished | Added `tests/` (fixture factory, layout record from real F44055 and F11116 granules, characterization tests); pytest config with doctests; pixi `cov` task. New in `code_review.md`: bug 8 and four smaller findings |
| 2026-10-01 | T1 subtasks finished (T1.1–T1.3, T1.5–T1.7) | Ticked in `todo.md`; added T1.7 (pixi); `check_plan.py` accepts ticked boxes. Verified: `mamba env create`, `pip install -e .`, `pip check`, all pre-commit hooks pass; pixi `default`, `test`, `docs`, `dev` environments install and import `disp_xr` |
| 2026-10-01 | `specs.md` changed (opera_utils decision applied, audit findings, earth-mover, Zarr and xarray-ecosystem research) | Added `porting.md`, `references.md`, `check_plan.py`. PRD: F14–F16, storage rules, ecosystem libraries, 4 new NFR rows, correlated-error cell sigma, odd downsampling factors. Plan: 226 subtasks (was 209); T4.5, T4.7, T5, T6, T7, T10, T11, T12, T14 rewritten as ports; C4 moved from T28 to T27.7; NISAR trade-study frames moved to T35.5. Cleared the four contradictions listed in the first index |
