# Documentation plan (before going public)

Audit of 2026-10-04 (`main` at `0781413`). Plan: finish the task plan, then
write one tutorial per feature, publish the site on GitHub Pages, make
`mgovorcin/disp-xr` public and re-enable CI.

## Coverage by feature

| Feature | PRD | Docs page | Gallery or example | Gap |
| --- | --- | --- | --- | --- |
| Discover, search | F1, F16 | api/open.md, cli.md | none | partial |
| Open | F2 | getting-started, api/open.md | all gallery pages | partial: no `open_disp_tree` or S3 example |
| Stack | F3 | getting-started, concepts.md | quicklook, chunking | none |
| Mask, quality | F4 | concepts.md, api/process.md | quality_masks, quality_weights | none |
| Reference | F5 | getting-started, concepts.md | quicklook, export | partial: not every method shown |
| Correct | F6 | getting-started, concepts.md | quicklook | partial |
| Geometry, LOS | F7 | api/open.md, bowser.md | none | missing |
| Time series, velocity | F8, S1 | velocity-methods.md | velocity, colored_noise | partial: no diagnostics example (`examples/eof_residuals.py` is not on the site) |
| Spatial models, GNSS tie | F9 | api/models.md | none (T33.5) | missing |
| GNSS | F10 | api/gnss.md | none | missing |
| Export | F12, C1 | getting-started, api/export.md | export, chunking | none |
| Plot | F13 | getting-started, api/plot.md | quicklook, interactive | none |
| Command line | F14 | cli.md | none | partial: no worked walk-through |
| Plate motion | F15 | api/process.md | none | missing |
| Downsampling | S2 | velocity-methods.md | downsample | partial: quadtree, hexagons, variogram |
| Denoising, time grids | S3 | api/process.md | none | missing |
| Residual analysis | S4 | api/gnss.md | none | missing |
| Icechunk, virtual frames | C2, C3 | api/cloud.md, cli.md | none | missing |
| GeoZarr, pyramids, bowser | C4 | bowser.md, api/export.md | export (pyramid only) | partial |

These modules have no docstring examples at all (T38.3 asks for one per public function):

- `open`, `stack`, `reference`, `corrections`, `search`, `gnss`, `residuals`, `diagnostics`, `virtual`
- `export/icechunk`, `export/mintpy`, `export/raster`
- `geometry/static`
- `models/base`, `models/gnss_tie`, `models/gridding`
- `timeseries/denoise`, `timeseries/temporal`, `timeseries/gls`

## Notebooks

None of the four notebooks uses the current API, and none of them is in the site:

- `notebooks/workflow.ipynb` and `notebooks/interactive_workflow.ipynb` use `download`, `io`, `product`, `quality_metrics`, `static_layers`, `combine_disp_product` and `utils.*`;
- `notebooks/validation/Compare_disp_versions.ipynb` uses `combine_disp_product` and `utils.*`;
- `notebooks/test_time.ipynb` is a scratch notebook on the private `timeseries._jax_solver`.

Port them to gallery pages or retire them.

## Site and GitHub Pages

- `mkdocs.yml` has `repo_url` and `edit_uri` but no `site_url` (`https://mgovorcin.github.io/disp-xr/`).
- No deploy job: `test.yml` only builds with `--strict`. Hosting was set up for ReadTheDocs (`.readthedocs.yaml`), so T16.5 should become "GitHub Pages deploy workflow".
- Internal pages are in the nav (`api_review.md`, `code_review.md`, `todo.md`, `porting.md`, `INDEX.md`). Move them under a "Development" section or out of the nav. `studies/` is not in the nav.
- The README only explains `mkdocs serve`: it needs a link to the hosted site.

## Tutorial order (most user-facing first)

1. Search and discover, then open on S3 (F16, F1, F2).
2. Command line walk-through (F14).
3. Geometry and LOS (F7).
4. GNSS: load, LOS, sample, compare (F10, S4).
5. GNSS tie and de-ramping (F9; closes T33.5).
6. Reference methods and corrections in depth (F5, F6).
7. Denoising and regular time grids (S3).
8. Velocity diagnostics and colored-noise choice (S1).
9. Plate motion (F15).
10. Icechunk and virtual frames (C2, C3).
11. GeoZarr, pyramids and bowser (C4).
12. Quadtree and the T29 defaults (S2), after the study.

## Before going public

- Settle ownership. disp-xr came from `opera-adt/disp-xr` (BSD-3, Caltech headers, other contributors), so publishing it under a personal account may need a JPL determination.
- Re-enable the `test` workflow. It is free on a public repository.
