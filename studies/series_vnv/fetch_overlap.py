"""Read the signal-screened stations' windows from local F11116 granules: a second track.

Same windows as insar_gnss_fetch.py in the smooth-trend validation (5 x 5 at 30 m through
recommended_mask, a second window 500 m east), for the stations inside F11116. Needs h5py.
"""

import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from pyproj import Transformer

HALF, OFFSET = 2, 17
HERE = Path(__file__).resolve().parent
GRANULES = Path("/u/aurora-r0/govorcin/01_OPERA/DISP/data/F11116")
OUT = HERE / "results" / "f11116x"


def window(ds, i, j):
    return ds[i - HALF : i + HALF + 1, j - HALF : j + HALF + 1]


def job(task):
    path, xy = task
    out = OUT / f"{path.stem}.npz"
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
    cache = Path(sys.argv[1])
    OUT.mkdir(parents=True, exist_ok=True)
    st = pd.read_csv(cache / "f38502v" / "stations.csv")
    paths = sorted(GRANULES.glob("OPERA*.nc"))
    with h5py.File(paths[0]) as h:
        x, y = h["x"][:], h["y"][:]
    px, py = Transformer.from_crs(4326, 32610, always_xy=True).transform(st.lon, st.lat)
    keep = (
        (px > x.min() + 100)
        & (px < x.max() - 700)
        & (py > y.min() + 100)
        & (py < y.max() - 100)
    )
    st = st.assign(x=px, y=py)[keep].reset_index(drop=True)
    st.to_csv(OUT / "stations.csv", index=False)
    xy = st[["x", "y"]].to_numpy()
    print(len(paths), "granules,", len(st), "stations", flush=True)
    with ProcessPoolExecutor(16) as pool:
        print(
            "read",
            sum(pool.map(job, [(p, xy) for p in paths], chunksize=4)),
            flush=True,
        )
