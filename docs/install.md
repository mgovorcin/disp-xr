# Install

disp-xr needs Python 3.12 or newer.

## pip

```bash
pip install "disp_xr @ git+https://github.com/mgovorcin/disp-xr"
```

## pixi (development)

[pixi](https://pixi.sh) reads `pyproject.toml` and installs disp-xr in
editable mode with pinned dependencies:

```bash
git clone https://github.com/mgovorcin/disp-xr.git
cd disp-xr
pixi install -e dev      # tests, linters and every extra
pixi run -e test test    # run the tests
```

## Optional extras

The core package opens, processes and exports local granules. Extras add the
rest; install them with `pip install "disp_xr[extra1,extra2]"`.

| Extra | Adds | Needed for |
|---|---|---|
| `aws` | `asf_search`, `s3fs`, `aiohttp`, `requests`, `shapely`, `dem_stitcher`, `tile_mate` | `search`, remote `s3://` and `https://` granules, static-layer and DEM downloads, polygon references |
| `plot` | `matplotlib` | `plot_map`, `plot_point` |
| `mintpy` | `mintpy` | reading the MintPy export with MintPy (writing it needs only h5py) |
| `cloud` | `virtualizarr`, `icechunk`, `obstore` | `to_icechunk`, virtual stores |
| `jax` | `jax` | `jax_solve_linear_system` |
| `gnss` | `xvec`, `geopandas` | GNSS comparison (planned) |
| `eof` | `xeofs` | EOF analysis (planned) |
| `interactive` | `hvplot`, `datashader` | interactive full-frame browsing |
| `docs` | mkdocs and plugins | building this site |

Remote access needs Earthdata Login credentials in `~/.netrc` (or the
`EARTHDATA_USERNAME` and `EARTHDATA_PASSWORD` environment variables); S3
access inside AWS us-west-2 uses temporary ASF credentials fetched with them.
