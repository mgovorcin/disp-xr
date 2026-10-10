# API policy

What disp-xr promises about its interface, its outputs and the products it
reads, and how changes are made. The list of frozen names is filled in after
the [API review](api_review.md) (plan task T18.3); everything else here
applies now. The freeze was planned for v0.2; no v0.2.0 was tagged, and
v0.3.0 shipped before the review, so the core API is not frozen yet.

## Versions

disp-xr follows [Semantic Versioning](https://semver.org/):

| Release | May contain |
| --- | --- |
| Patch (1.2.3 → 1.2.4) | Bug fixes and documentation only. Results change only where they were wrong. |
| Minor (1.2 → 1.3) | New features, new optional parameters, deprecations. Existing code keeps working, with warnings for deprecated names. |
| Major (1 → 2) | Removals of deprecated names and other breaking changes. |

Before 1.0, a minor release (0.2 → 0.3) may also break the API, but only
through the deprecation process below. Defaults chosen by trade studies
(for example the quadtree split metric and quality-weight parameters of
plan task T29) may still change before 1.0; each change is listed in the
changelog with the study that motivated it.

## What is public

These are public, and covered by this policy:

- the names in `disp_xr.__all__`, including the submodules it lists
  (`disp_xr.gnss`, `disp_xr.models`, `disp_xr.diagnostics`);
- the public names of `disp_xr.residuals`, `disp_xr.timeseries`,
  `disp_xr.batch`, `disp_xr.plot`, `disp_xr.search`, `disp_xr.weights`,
  `disp_xr.geometry` and `disp_xr.tutorial` (those without a leading
  underscore, listed in each module's `__all__`);
- the `.disp` accessor methods;
- the `disp_xr` command, its subcommands and options;
- the keys of `disp_xr.config` and their `DISP_XR_*` environment variables;
- the layout of the stores disp-xr writes: variable names, dimension names,
  `units`, and the provenance attributes (`history`, `input_granules`,
  `reference`, `product`, `frame_id`).

Modules and names starting with an underscore are private and may change in
any release. So are the deprecated module aliases (`disp_xr.io`,
`disp_xr.download`, `disp_xr.quality_metrics`, `disp_xr.static_layers`,
`disp_xr.mintpy`, `disp_xr.utils`), which are removed at 1.0.

### Stability levels

| Level | Meaning | Currently |
| --- | --- | --- |
| Stable | Covered by everything above. | Everything public not listed below |
| Provisional | May change in a minor release, with a changelog entry but without a deprecation cycle. | DISP-NISAR support (product spec not yet validated, plan task T35); `disp_xr.batch` (region jobs); the REST service once added (plan task T40) |

## Deprecation

A renamed or removed public name:

1. keeps working for at least one minor release (before 1.0) or until the
   next major release (after 1.0), emitting a `DeprecationWarning` that
   names its replacement;
2. is listed under "Deprecated" in the changelog of the release that
   deprecates it, and under "Removed" in the release that removes it;
3. is covered by a test that checks the warning and the forwarded result.

A change to a default value follows the same process when it changes
results, except that defaults set by a documented trade study may change
before 1.0 with a changelog entry (see "Versions").

## Supported products

| Product | Versions read | Status |
| --- | --- | --- |
| DISP-S1 | 0.10 to below 2 | Validated against production granules. Frames that span the November 2025 production change (float64 displacement, time stored as seconds since 2010) are read together, with a `MixedPrecisionWarning`; see [Product notes](product_notes.md). Version 0.10 (pre-release stakeholder runs) is read but not validated. |
| DISP-NISAR | 0.4 to below 2 | Provisional: the spec follows SAS test output and is validated when real products are available (plan task T35). |

Granules outside these ranges are refused with
`UnsupportedProductVersionError`; `open_disp(min_version=...)` narrows the
range further. A new product version is supported in a minor release, after
its granules pass the test suite; the product notes record what changed.

## Supported Python and dependencies

- Python 3.12 and newer; the test matrix covers 3.12, 3.13 and 3.14.
- Core dependencies follow the
  [SPEC 0](https://scientific-python.org/specs/spec-0000/) schedule of the
  scientific Python ecosystem: a dependency version is supported for at
  least two years after its release.
- Optional features live in extras (`aws`, `cloud`, `gnss`, `plot`,
  `interactive`, `hex`, `eof`,
  `mintpy`, `jax`, and `serve` when added); a missing extra raises an
  `ImportError` that names it.

## Frozen names

To be filled in after the review (planned for v0.2, now the first release
after it): the function names, parameter names and
defaults that are frozen, and the outcome of each open question in the
[API review](api_review.md).
