"""Noise of real 5x5, 30-day pixel differences 500 m apart: sets the synthetic noise."""

from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from common import CACHE, DAYS, RESULTS, grid, pixel_series
from geepers.trend import estimate_trend


def job(t):
    frame, sid, s = t
    if len(s) < 60:
        return None
    y = (s - s.iloc[0]).to_numpy()
    row = {"frame": frame, "id": sid, "months": len(s), "std": float(np.std(y))}
    for model in ("FNWN", "RWFNWN"):
        r = estimate_trend(s.index, y, sampling_days=DAYS, noise_model=model)
        row[f"{model}_white"] = r.sigma_white
        row[f"{model}_flicker"] = r.sigma_flicker
        row[f"{model}_rw"] = r.sigma_randomwalk
        row[f"{model}_rate"] = r.velocity
    return row


if __name__ == "__main__":
    tasks = []
    for frame in ("f11116", "f38502", "f38502v"):
        folder = CACHE / frame
        st = pd.read_csv(folder / "stations.csv")
        dates, cube = pixel_series(folder)
        for k, sid in enumerate(st.id):
            d = pd.Series(cube[:, k, 0] - cube[:, k, 1], index=dates)
            tasks.append((frame, sid, grid(d)))
    with ProcessPoolExecutor(24) as pool:
        out = pd.DataFrame([r for r in pool.map(job, tasks) if r])
    RESULTS.mkdir(exist_ok=True)
    out.to_csv(RESULTS / "noise_levels.csv", index=False)
    print(
        out.drop(columns="id")
        .groupby("frame")
        .quantile([0.25, 0.5, 0.75])
        .round(3)
        .T.to_string()
    )
