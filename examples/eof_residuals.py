"""EOF analysis of trajectory-fit residuals (plan task T32.6d, ``eof`` extra).

Residuals that share a spatial pattern from epoch to epoch (troposphere,
orbit or unwrapping errors over a whole area) show up as leading EOF modes:
a spatial component, a time series of scores and the share of variance
they explain. This example builds a stack with a known pattern in its
residuals, fits a trajectory, and checks that the first EOF mode (plain and
varimax-rotated) recovers the pattern. It runs in its own CI job, which
reports breakage without failing the core build: xeofs has been quiet since
early 2025.

    pixi run -e eof python examples/eof_residuals.py [figure.png]
"""

import sys

import numpy as np
import pandas as pd
import xarray as xr
import xeofs

import disp_xr as dxr


def stack(nt: int = 80, ny: int = 40, nx: int = 50, seed: int = 0) -> tuple:
    """Trend plus seasonal plus one spatial error pattern with random amplitude."""
    rng = np.random.default_rng(seed)
    times = pd.date_range("2020-01-01", periods=nt, freq="12D")
    years = np.arange(nt) * 12 / 365.25
    yy, xx = np.mgrid[0:ny, 0:nx]
    pattern = np.exp(-((xx - 30) ** 2 + (yy - 15) ** 2) / 120.0)  # a "cell"
    pattern /= np.linalg.norm(pattern)
    amplitude = 0.05 * rng.standard_normal(nt)
    values = (
        0.01 * years[:, None, None]
        + 0.003 * np.sin(2 * np.pi * years)[:, None, None]
        + amplitude[:, None, None] * pattern
        + 0.0005 * rng.standard_normal((nt, ny, nx))
    )
    da = xr.DataArray(
        values,
        dims=("time", "y", "x"),
        coords={"time": times, "y": -30.0 * np.arange(ny), "x": 30.0 * np.arange(nx)},
        name="displacement",
        attrs={"units": "meters"},
    )
    return da, pattern


def main(figure: str | None = None) -> None:
    """Fit, take the residuals, run the EOFs and check the first mode."""
    da, pattern = stack()
    fit = dxr.fit_trajectory(da, dxr.Trajectory(periods=(1.0,)))
    residual = da - dxr.predict_trajectory(fit, da.time.values, dtype="float64")
    model = xeofs.single.EOF(n_modes=3)
    model.fit(residual, dim="time")
    # xeofs returns components with y ascending; DISP grids run north up, so
    # align by coordinate before comparing or plotting with numpy.
    grid = residual.isel(time=0, drop=True)
    first = (
        model.components().isel(mode=0).reindex_like(grid).transpose("y", "x").values
    )
    ratio = model.explained_variance_ratio().values
    match = abs(np.corrcoef(first.ravel(), pattern.ravel())[0, 1])
    rotated = xeofs.single.EOFRotator(n_modes=2).fit(model)
    rot_first = (
        rotated.components().isel(mode=0).reindex_like(grid).transpose("y", "x").values
    )
    rot_match = abs(np.corrcoef(rot_first.ravel(), pattern.ravel())[0, 1])
    print(f"explained variance ratio: {np.round(ratio, 3)}")
    print(f"first mode vs pattern: r = {match:.3f}; varimax: r = {rot_match:.3f}")
    if figure:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), layout="constrained")
        axes[0].imshow(pattern)
        axes[0].set_title("planted pattern", loc="left")
        axes[1].imshow(first)
        axes[1].set_title(f"EOF 1 ({ratio[0]:.0%} of variance)", loc="left")
        model.scores().isel(mode=0).plot(ax=axes[2])
        axes[2].set_title("EOF 1 scores", loc="left")
        fig.savefig(figure, dpi=120)
    if match < 0.95 or rot_match < 0.95:
        raise SystemExit("the first EOF mode did not recover the planted pattern")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
