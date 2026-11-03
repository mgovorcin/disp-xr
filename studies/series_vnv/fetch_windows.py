"""Read station windows from local granules, for a station list with id, lon and lat.

Same windows as insar_gnss_fetch.py in the smooth-trend validation (5 x 5 at 30 m through
recommended_mask, a second window 500 m east), for the stations inside the frame. Saves the
frame-center line of sight when given the static layer. Needs h5py, pyproj and rasterio.

    python fetch_windows.py f11116x <F11116 granules> <cache>/f38502v/stations.csv
    python fetch_windows.py f08882 <F08882 granules> results/houston_stations.csv <enu.tif>
"""

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from pyproj import CRS, Transformer

HALF, OFFSET = 2, 17
HERE = Path(__file__).resolve().parent


def window(ds, i, j):
    return ds[i - HALF : i + HALF + 1, j - HALF : j + HALF + 1]


def job(task):
    path, xy, folder = task
    out = folder / f"{path.stem}.npz"
    if out.exists():
        return 0
    with h5py.File(path, "r") as h:
        x, y = h["x"][:], h["y"][:]
        dx, dy = x[1] - x[0], y[1] - y[0]
        d, m, c = h["displacement"], h["recommended_mask"], h["temporal_coherence"]
        vals = np.full((len(xy), 2), np.nan, dtype="f4")
        good = np.zeros((len(xy), 2), dtype="i2")
        coh = np.full(len(xy), np.nan, dtype="f4")
        for k, (sx, sy) in enumerate(xy):
            i, j = int(round((sy - y[0]) / dy)), int(round((sx - x[0]) / dx))
            for s, (a, b) in enumerate(((i, j), (i, j + OFFSET))):
                if not (HALF <= a < len(y) - HALF and HALF <= b < len(x) - HALF):
                    continue
                v, mask = window(d, a, b), window(m, a, b) > 0
                use = v[mask & np.isfinite(v)]
                good[k, s] = use.size
                if use.size:
                    vals[k, s] = np.median(use)
                if s == 0:
                    coh[k] = np.nanmedian(window(c, a, b))
    tmp = out.with_suffix(".tmp.npz")
    np.savez(tmp, vals=vals, good=good, coh=coh)
    tmp.replace(out)
    return 1


if __name__ == "__main__":
    name, granules, stations = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
    folder = HERE / "results" / name
    folder.mkdir(parents=True, exist_ok=True)
    st = pd.read_csv(stations)
    paths = sorted(granules.glob("OPERA_L3_DISP-S1_IW*.nc"))
    with h5py.File(paths[0]) as h:
        x, y = h["x"][:], h["y"][:]
        wkt = h["spatial_ref"].attrs["crs_wkt"]
    crs = CRS.from_wkt(wkt.decode() if isinstance(wkt, bytes) else str(wkt))
    px, py = Transformer.from_crs(4326, crs, always_xy=True).transform(st.lon, st.lat)
    keep = (px > x.min() + 100) & (px < x.max() - 700)
    keep &= (py > y.min() + 100) & (py < y.max() - 100)
    st = st.assign(x=px, y=py)[keep].reset_index(drop=True)
    st.to_csv(folder / "stations.csv", index=False)
    if len(sys.argv) > 4:
        import rasterio

        with rasterio.open(sys.argv[4]) as src:
            enu = src.read(out_shape=(src.count, 200, 200)).astype(float)
        enu[enu == 0] = np.nan
        east, north, up = np.nanmedian(enu, axis=(1, 2))[:3]
        los = {"east": float(east), "north": float(north), "up": float(up)}
        (folder / "los.json").write_text(json.dumps(los))
        print(name, "line of sight", los)
    xy = st[["x", "y"]].to_numpy()
    print(len(paths), "granules,", len(st), "stations", flush=True)
    with ProcessPoolExecutor(16) as pool:
        tasks = [(p, xy, folder) for p in paths]
        print("read", sum(pool.map(job, tasks, chunksize=4)), flush=True)
