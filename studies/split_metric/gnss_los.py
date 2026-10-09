"""Write the LOS velocity of each used GNSS station (T29.2, GNSS criterion).

For every DISP-S1 frame in ``frames.yaml`` this projects the daily east,
north and up series of the stations marked ``use`` in
``stations/<frame>.csv`` onto the line of sight (LOS vector at the station's
pixel), keeps the frame's dates and fits a MIDAS velocity, robust to steps
and outliers. Writes ``stations/<frame>_los.csv``: id, x, y, the LOS vector,
``v_los`` and ``v_los_sigma`` (m/yr, positive toward the satellite).

Run with ``pixi run -e test-py312 python studies/split_metric/gnss_los.py``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import rioxarray
import xarray as xr
import yaml
from stations import HERE, frame_files

import disp_xr as dxr
from disp_xr import gnss

LOS = ("los_east", "los_north", "los_up")


def los_layers(grid: xr.Dataset, static: Path) -> xr.Dataset:
    """Return ``los_east``, ``los_north``, ``los_up`` on the frame grid."""
    if all((static / f"{name}.tif").exists() for name in LOS):
        layers = {}
        for name in LOS:
            band = rioxarray.open_rasterio(static / f"{name}.tif").squeeze(
                "band", drop=True
            )
            assert band.shape == (grid.sizes["y"], grid.sizes["x"]), name
            layers[name] = band.assign_coords(x=grid["x"], y=grid["y"])
        return xr.Dataset(layers)
    return dxr.open_static(grid, static_dir=static)[list(LOS)]


def station_los(frame: str, entry: dict) -> pd.DataFrame:
    """Return the LOS velocity of the used stations of one frame."""
    used = pd.read_csv(HERE / "stations" / f"{frame}.csv").query("use")
    first = frame_files(frame, entry)[0]
    grid = dxr.open_disp([first], check=False)
    start, end = (str(pd.Timestamp(d).date()) for d in entry["dates"])
    cube = gnss.load(grid, start=start, end=end)
    cube = cube.sel(station=used["id"].tolist())
    projected = gnss.to_los(cube, los_layers(grid, Path(entry["static"])))
    fit = dxr.fit_velocity(projected["los"].transpose("time", "station"), "midas")
    table = pd.DataFrame(
        {
            "id": projected["station"].values,
            "x": projected["x"].values,
            "y": projected["y"].values,
            **{name: projected[name].values for name in LOS},
            "v_los": fit["velocity"].values,
            "v_los_sigma": fit["velocity_sigma"].values,
        }
    )
    # A LOS vector of zeros is the static layers' no-data: no projection.
    return table[(table[list(LOS)] != 0).any(axis=1)].reset_index(drop=True)


def main() -> None:
    """Write ``stations/<frame>_los.csv`` for every DISP-S1 frame."""
    frames = yaml.safe_load((HERE / "frames.yaml").read_text())["disp_s1"]
    for frame, entry in frames.items():
        table = station_los(frame, entry)
        table.to_csv(HERE / "stations" / f"{frame}_los.csv", index=False)
        v = 1e3 * table["v_los"]
        print(
            f"{frame}: {len(table)} stations, LOS velocity {v.min():.1f} to "
            f"{v.max():.1f} mm/yr, median sigma "
            f"{1e3 * table['v_los_sigma'].median():.2f} mm/yr"
        )


if __name__ == "__main__":
    main()
