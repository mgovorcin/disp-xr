
# Xarray tools for OPERA DISP

Analysis tools using [`xarray`](xarray.dev) for OPERA Surface Displacement from Sentinel-1 (DISP-S1) and Surface Displacement from NISAR (DISP-NISAR).

## Quick start

```python
import disp_xr as dxr

df = dxr.find_products("path/to/granules/")   # or "s3://bucket/prefix/"
ds = dxr.open_disp(df, corrections=True)       # lazy (time, y, x), CRS attached
ds = dxr.stack_disp(ds)                        # displacement from the first date

ds.displacement.isel(x=4000, y=3000).plot()    # one pixel's time series
```

`open_disp` returns displacement as stored in the granules, relative to each
ministack's reference date; `stack_disp` chains the ministacks. See
`docs/specs.md` for the plan and `CHANGELOG.md` for what has changed.

## Development setup

### Prerequisite installs

1. Download source code:
```bash
git clone https://github.com/mgovorcin/disp-xr.git
```
2. Install with [pixi](https://pixi.sh) (recommended). It reads `pyproject.toml` and installs `disp_xr` in editable mode:
```bash
cd disp-xr
pixi install            # core environment
pixi install -e dev     # plus tests, linters, aws, plot, mintpy and notebook extras
pixi run -e test test   # run the tests
pixi run -e test check  # run the pre-commit hooks
```
Environments: `default`, `test`, `docs` and `dev`.

   Or install the dependencies with conda/mamba, either to a new environment:
```bash
mamba env create --name disp_xr --file disp-xr/environment.yml
conda activate disp_xr
```
or install within your existing env with mamba.

3. With conda/mamba only (pixi already did this), install `disp_xr` via pip in editable mode
```bash
python -m pip install --no-deps -e  disp-xr
```

### Setup for contributing

We use [pre-commit](https://pre-commit.com/) to automatically run linting, formatting, and [mypy type checking](https://www.mypy-lang.org/).
Additionally, we follow [`numpydoc` conventions for docstrings](https://numpydoc.readthedocs.io/en/latest/format.html).
To install pre-commit locally, run:

```bash
pre-commit install
```
This adds a pre-commit hooks so that linting/formatting is done automatically. If code does not pass the checks, you will be prompted to fix it before committing.
Remember to re-add any files you want to commit which have been altered by `pre-commit`. You can do this by re-running `git add` on the files.

Since we use [black](https://black.readthedocs.io/en/stable/) for formatting and [flake8](https://flake8.pycqa.org/en/latest/) for linting, it can be helpful to install these plugins into your editor so that code gets formatted and linted as you save.

### Running the unit tests

Tests run on small synthetic DISP-S1 stacks written at run time (`tests/fixtures/make_disp.py`); no data download is needed. See `tests/data/README.md`.

```bash
pixi run -e test test   # pytest, including doctests in src/disp_xr
pixi run -e test cov    # with a coverage report
```

With conda/mamba, install the test extra and run pytest:

```bash
python -m pip install --no-deps -e .[test]
pytest
```

Tests marked `xfail(strict=True)` assert the correct behavior for known bugs listed in `docs/code_review.md`. When a fix lands, the test starts passing, pytest reports it as a failure (XPASS), and the marker is removed in the same change.
