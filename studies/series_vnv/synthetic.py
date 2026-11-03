"""Known answers at 5x5, 30-day settings: false alarms, misses, timing, present rate.

Noise comes from the real near-null differences (noise_levels.py): white 0.3 mm and
flicker 0.9 mm per epoch, with a random walk of 0-5 mm/sqrt(yr); 3.5 is the median.
"""

from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from common import DAYS, RESULTS, run_methods
from geepers.synthetic import power_law_noise

REPS = 200
START = pd.Timestamp("2016-07-05")
QUIET, TYPICAL = 1.0, 3.5


def scenarios():
    yield {"kind": "flicker only", "years": 10, "walk": None}
    for walk in (0.0, 1.0, 2.0, 3.5, 5.0):
        yield {"kind": "line", "years": 10, "walk": walk}
    for years in (5, 7):
        yield {"kind": "line", "years": years, "walk": TYPICAL}
    for walk in (QUIET, TYPICAL):
        for change in (1.0, 2.0, 4.0, 8.0):
            for at in (0.25, 0.5, 0.75):
                yield {
                    "kind": "hinge",
                    "years": 10,
                    "walk": walk,
                    "size": change,
                    "at": at,
                }
        for acc in (0.25, 0.5, 1.0, 2.0):
            yield {"kind": "curve", "years": 10, "walk": walk, "size": acc}
        # 28 and 56 mm: one and two unwrapping cycles at C-band
        for step in (5.0, 10.0, 28.0, 56.0):
            yield {"kind": "step", "years": 10, "walk": walk, "size": step, "at": 0.5}
        for amp in (20.0, 60.0):
            yield {"kind": "decay", "years": 10, "walk": walk, "size": amp, "at": 0.5}


def series(sc, rng):
    n = int(sc["years"] * 365.25 / DAYS)
    t = np.arange(n) * DAYS / 365.25
    y = rng.uniform(-5, 5) * t
    y += 3.0 * np.sin(2 * np.pi * t + rng.uniform(0, 2 * np.pi))
    y += 1.0 * np.sin(4 * np.pi * t + rng.uniform(0, 2 * np.pi))
    if sc["walk"] is None:
        y += power_law_noise(n, -1, 2.0, rng) + power_law_noise(n, 0, 0.3, rng)
    else:
        step_walk = sc["walk"] * np.sqrt(DAYS / 365.25)
        y += power_law_noise(n, -1, 0.9, rng) + power_law_noise(n, 0, 0.3, rng)
        y += power_law_noise(n, -2, step_walk, rng)
    late = t - sc.get("at", 0.5) * t[-1]
    if sc["kind"] == "hinge":
        y += sc["size"] * np.maximum(late, 0.0)
    elif sc["kind"] == "curve":
        y += 0.5 * sc["size"] * (t - t.mean()) ** 2
    elif sc["kind"] == "step":
        y += sc["size"] * (late >= 0)
    elif sc["kind"] == "decay":
        y += sc["size"] * (1 - np.exp(-np.maximum(late, 0.0))) * (late >= 0)
    return START + pd.to_timedelta(t * 365.25, unit="D"), y


def job(task):
    i, sc, rep = task
    rng = np.random.default_rng([20261102, i, rep])
    dates, y = series(sc, rng)
    row = dict(scenario=i, rep=rep, **sc)
    row.update(run_methods(dates, y))
    return row


if __name__ == "__main__":
    all_sc = list(scenarios())
    tasks = [(i, sc, r) for i, sc in enumerate(all_sc) for r in range(REPS)]
    print(len(all_sc), "scenarios", len(tasks), "series", flush=True)
    rows = []
    with ProcessPoolExecutor(48) as pool:
        for k, row in enumerate(pool.map(job, tasks, chunksize=8)):
            rows.append(row)
            if k % 1000 == 0:
                print(k, flush=True)
    RESULTS.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(RESULTS / "synthetic.csv", index=False)
    print("done", flush=True)
