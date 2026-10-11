# Command line

The `disp_xr` command wraps the library for batch runs; each subcommand calls
the functions of the same name. `disp_xr --debug <command>` logs debug
messages and shows tracebacks instead of one-line errors. Every command has
`--help`.

```text
disp_xr info PATHS...      [--frame-id F] [--min-version V] [--json] [--table granules.csv]
disp_xr stack PATHS... --out out.zarr|out.nc
    [--corrections product|none|ionospheric_delay,solid_earth_tide]
    [--mask recommended|strict|none]
    [--reference auto|point|median|border|high_coherence|none]
    [--row R --col C | --lon X --lat Y] [--start DATE --end DATE]
    [--preset timeseries|maps] [--overwrite]
disp_xr export PATHS... --format cog|geotiff|zarr|geozarr|netcdf|mintpy --out OUT
    [--layer displacement] [--epoch DATE | --per-epoch] [--static-dir DIR] [--overwrite]
disp_xr geometry PATHS... --out static.zarr [--static-dir DIR] [--no-download]
disp_xr search --frame-id F [--start DATE] [--end DATE] [--product DISP-S1|DISP-NISAR]
    [--version 1.0|any] [--url-type https|s3] [--out granules.csv]
disp_xr virtual PATHS... --repo DIR [--branch main] [--workers N] [--no-validate]
disp_xr serve STORES...    [--host 127.0.0.1] [--port 9000] [--no-tiles]
```

## A typical run

```bash
disp_xr info granules/                                  # what is there
disp_xr stack granules/ --out frame.zarr --mask strict  # correct, mask, chain, reference
disp_xr export frame.zarr --format cog --out last.tif   # Cloud Optimized GeoTIFF
disp_xr export frame.zarr --format mintpy --out mintpy/
disp_xr export frame.zarr --format geozarr --out view.zarr  # bowser opens it
```

- `stack` runs the steps in the fixed order: corrections and quality mask per
  granule, chaining, re-referencing. `--corrections product` (the default)
  subtracts the corrections the product has not applied already.
- `export` takes a store written by `stack`, or granules, which it chains
  without corrections, masking or re-referencing.
- An existing output is refused unless `--overwrite` is given.
- `geometry` and `search` need the `aws` extra; `virtual` needs `cloud`;
  `serve` needs `serve`.
- `serve` names each dataset after its store (`frame.zarr` is `frame`) and
  serves it read-only under `/datasets/<name>/disp/`: `timeseries`,
  `velocity`, `cells`, `gnss` and `provenance`. The API page is at `/docs`.
