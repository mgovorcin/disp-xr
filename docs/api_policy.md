# API policy

!!! warning "Draft"
    This policy takes effect with v0.2.0. The frozen-name list is filled in
    after the [API review](api_review.md) (plan task T18.3).

## Versions

disp-xr follows [Semantic Versioning](https://semver.org/). Before 1.0, a
minor release (0.2 → 0.3) may change the API, but only through the
deprecation process below; names frozen at v0.2 change only for a documented
reason.

## What is public

The names in `disp_xr.__all__`, the `.disp` accessor methods, the public
functions of `disp_xr.plot`, `disp_xr.search` and `disp_xr.tutorial`, the
`disp_xr` command and its options, and the keys of `disp_xr.config`. Modules
and names starting with an underscore are private and may change at any time.

## Deprecation

A renamed or removed public name:

1. keeps working for at least one minor release, emitting a
   `DeprecationWarning` that names its replacement;
2. is listed under "Deprecated" in the changelog of the release that
   deprecates it, and under "Removed" in the release that removes it;
3. is covered by a test that checks the warning and the forwarded result.

Changes to default values follow the same process when they change results.

## Frozen at v0.2

To be filled in after the review: the function names, parameter names and
defaults that are frozen, and the outcome of each open question in the
[API review](api_review.md).
