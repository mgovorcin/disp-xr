"""fit_series itself on the cached pair series: the verdicts match the direct calls?

fit_series bins from the first epoch, the other scripts on a fixed origin, so this also
shows how much the verdict depends on where the 30-day bins happen to fall.
"""

import time
import warnings

import numpy as np
import pandas as pd
import xarray as xr
from common import CACHE, RESULTS, pixel_series

import disp_xr as dxr
from disp_xr.timeseries.series import LINEARITY_MODELS

warnings.simplefilter("ignore")

if __name__ == "__main__":
    real = pd.read_csv(RESULTS / "real.csv")
    pairs = real[real.role == "pixel"]
    out = []
    for frame in ("f11116", "f38502", "f38502v"):
        folder = CACHE / frame
        st = list(pd.read_csv(folder / "stations.csv").id)
        dates, cube = pixel_series(folder)
        t = pairs[pairs.frame == frame]
        data = np.stack(
            [
                cube[:, st.index(a), 0] - cube[:, st.index(b), 0]
                for a, b in zip(t.a, t.b)
            ],
            axis=1,
        )
        da = xr.DataArray(
            data[:, None, :] / 1000.0,
            dims=("time", "y", "x"),
            coords={"time": dates, "y": [0.0], "x": np.arange(len(t), dtype=float)},
            attrs={"units": "meters"},
            name="displacement",
        )
        start = time.perf_counter()
        fit = dxr.fit_series(da, "linearity", window=1, min_valid=60)
        one_frame = time.perf_counter() - start
        print(
            frame, len(t), "series", round(one_frame / len(t), 3), "s each", flush=True
        )
        code = fit["linearity_model"].values[0]
        row = t.assign(
            fs_model=[LINEARITY_MODELS[c] if c >= 0 else None for c in code],
            fs_significant=fit["significant"].values[0],
            fs_velocity=fit["velocity"].values[0] * 1000.0,
            fs_sigma=fit["velocity_sigma"].values[0] * 1000.0,
            fs_validity=fit["validity_years"].values[0],
            fs_departure=fit["departure"].values[0] * 1000.0,
            fs_p_quad=fit["p_quadratic"].values[0],
            fs_p_hinge=fit["p_piecewise"].values[0],
        )
        columns = "lin_model rw_rate rw_rate_sigma rw_horizon lin_departure lin_p_quad lin_p_hinge".split()
        out.append(
            row[["frame", "a", "b", *columns, *[c for c in row if c.startswith("fs_")]]]
        )
    r = pd.concat(out)
    r.to_csv(RESULTS / "end_to_end.csv", index=False)
    print(r.fs_model.notna().sum(), "of", len(r), "fitted")
