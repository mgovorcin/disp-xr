"""fit_series itself on the cached pair series: the verdicts match the direct calls?

fit_series bins from the first epoch, the other scripts on a fixed origin, so this also
shows how much the verdict depends on where the 30-day bins happen to fall.
"""

import warnings

import numpy as np
import pandas as pd
import xarray as xr
from common import CACHE, RESULTS, pixel_series

import disp_xr as dxr

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
        fit = dxr.fit_series(da, "linearity", window=1, min_valid=60)
        models = np.array(
            fit.attrs.get("linearity_model_flags", ["linear", "quadratic", "piecewise"])
        )
        code = fit["linearity_model"].values[0]
        row = t.assign(
            fs_model=[models[c] if c >= 0 else None for c in code],
            fs_velocity=fit["velocity"].values[0] * 1000.0,
            fs_departure=fit["departure"].values[0] * 1000.0,
            fs_p_quad=fit["p_quadratic"].values[0],
            fs_p_hinge=fit["p_piecewise"].values[0],
        )
        out.append(
            row[
                [
                    "frame",
                    "a",
                    "b",
                    "lin_model",
                    "lin_rate",
                    "lin_departure",
                    "lin_p_quad",
                    "lin_p_hinge",
                    "fs_model",
                    "fs_velocity",
                    "fs_departure",
                    "fs_p_quad",
                    "fs_p_hinge",
                ]
            ]
        )
    r = pd.concat(out)
    r.to_csv(RESULTS / "end_to_end.csv", index=False)
    ok = r.fs_model.notna()
    print("series", len(r), "fitted", int(ok.sum()))
    print("same model", round((r.lin_model == r.fs_model)[ok].mean(), 3))
    print(
        "same flagged",
        round(((r.lin_model != "linear") == (r.fs_model != "linear"))[ok].mean(), 3),
    )

    def gate(p, q, dep):
        return (np.minimum(p, q) < 0.01) & (dep > 10)

    print(
        "same gated",
        round(
            (
                gate(r.lin_p_quad, r.lin_p_hinge, r.lin_departure)
                == gate(r.fs_p_quad, r.fs_p_hinge, r.fs_departure)
            )[ok].mean(),
            3,
        ),
    )
    print(
        "rate rms diff mm/yr",
        round(float(np.sqrt(((r.lin_rate - r.fs_velocity)[ok] ** 2).mean())), 2),
    )
