"""Write the GNSS station list of each T29 frame (T29.1).

For every DISP-S1 frame in ``frames.yaml`` this opens one granule for the
grid, loads the UNR stations inside it over the frame's dates and writes
``stations/<frame>.csv``: id, lon, lat, x, y, first and last date, number of
daily solutions, the share of the DISP span covered, and ``use``. A station
is used when it covers at least half of the span with at least 2.5 years of
data, enough for a velocity with a sigma well under 1 mm/yr.

Run with ``pixi run -e test-py312 python studies/split_metric/stations.py``.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import yaml

import disp_xr as dxr
from disp_xr import gnss

HERE = Path(__file__).parent
MIN_COVER = 0.5
MIN_YEARS = 2.5


def frame_dir(frame: str, entry: dict) -> Path:
    """Return the granule directory of a frame, honoring DISP_XR_STUDY_DATA."""
    root = os.environ.get("DISP_XR_STUDY_DATA")
    return Path(root) / frame if root else Path(entry["path"])


def station_table(frame: str, entry: dict) -> pd.DataFrame:
    """Return the station list of one frame with coverage and the use flag."""
    first = sorted(frame_dir(frame, entry).glob("*.nc"))[0]
    grid = dxr.open_disp([first], check=False)
    start, end = (pd.Timestamp(d) for d in entry["dates"])
    cube = gnss.load(grid, start=str(start.date()), end=str(end.date()))
    has = cube["up"].notnull().transpose("station", "time").values
    days = pd.DatetimeIndex(cube["time"].values)
    last_index = has.shape[1] - 1 - has[:, ::-1].argmax(axis=1)
    span_days = (end - start).days
    table = pd.DataFrame(
        {
            "id": cube["station"].values,
            "lon": cube["lon"].values,
            "lat": cube["lat"].values,
            "x": cube["x"].values,
            "y": cube["y"].values,
            "first": days[has.argmax(axis=1)].date,
            "last": days[last_index].date,
            "n_days": has.sum(axis=1),
        }
    )
    years = (pd.to_datetime(table["last"]) - pd.to_datetime(table["first"])).dt.days
    table["cover"] = (table["n_days"] / span_days).round(3)
    table["use"] = (table["cover"] >= MIN_COVER) & (years / 365.25 >= MIN_YEARS)
    return table.sort_values("id").reset_index(drop=True)


def main() -> None:
    """Write ``stations/<frame>.csv`` for every DISP-S1 frame."""
    frames = yaml.safe_load((HERE / "frames.yaml").read_text())["disp_s1"]
    out = HERE / "stations"
    out.mkdir(exist_ok=True)
    for frame, entry in frames.items():
        table = station_table(frame, entry)
        table.to_csv(out / f"{frame}.csv", index=False)
        print(f"{frame}: {int(table['use'].sum())} used of {len(table)} stations")


if __name__ == "__main__":
    main()
