"""Where the change points fall: every date, by series kind, against ministack boundaries."""

import itertools
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from common import (
    DAYS,
    END,
    NAME,
    RESULTS,
    detect_changepoints,
    folder_of,
    grid,
    los_of,
    pixel_series,
)
from geepers.collocation import distance_matrix
from geepers.gps_sources.unr import UnrSource
from geepers.smooth import _unique_steps

MAX_KM, MIN_MONTHS = 25.0, 60


def job(task):
    role, frame, a, b, s, steps = task
    try:
        table = detect_changepoints(
            s.index, s.to_numpy(), sampling_days=DAYS, step_dates=steps
        ).changepoints
    except (ValueError, np.linalg.LinAlgError):
        return []
    return [
        dict(role=role, frame=frame, a=a, b=b, **row)
        for row in table.to_dict("records")
    ]


def boundaries(folder):
    return sorted(
        {pd.Timestamp(NAME.search(p.name).group(1)) for p in folder.glob("OPERA*.npz")}
    )


if __name__ == "__main__":
    src, tasks, cuts = UnrSource(), [], []
    catalog = src.steps()
    stem = sys.argv[1] if len(sys.argv) > 1 else "change_dates"
    frames = sys.argv[2:] or ["f11116", "f38502", "f38502v", "f11116x"]
    for frame in frames:
        folder = folder_of(frame)
        st = pd.read_csv(folder / "stations.csv")
        cuts += [{"frame": frame, "date": d} for d in boundaries(folder)]
        dates, cube = pixel_series(folder)
        if frame in END:
            keep = dates <= END[frame]
            dates, cube = dates[keep], cube[keep]
        los = los_of(frame)
        px, off, gn, steps = {}, {}, {}, {}
        for k, row in enumerate(st.itertuples()):
            px[row.id] = grid(pd.Series(cube[:, k, 0], index=dates))
            off[row.id] = grid(pd.Series(cube[:, k, 1], index=dates))
            t = src.timeseries(
                row.id, start_date=str(dates[0].date()), end_date=str(dates[-1].date())
            )
            t = t.set_index(pd.to_datetime(t["date"]))
            gn[row.id] = grid(
                (t.east * los["east"] + t.north * los["north"] + t.up * los["up"])
                * 1000.0
            )
            steps[row.id] = list(
                pd.to_datetime(catalog.loc[catalog["id"] == row.id, "date"])
            )
        for sid in st.id:
            d = (px[sid] - off[sid]).dropna()
            if len(d) >= MIN_MONTHS:
                tasks.append(("near", frame, sid, sid, d, None))
        km = distance_matrix(
            st.lon.to_numpy(), st.lat.to_numpy(), st.lon.to_numpy(), st.lat.to_numpy()
        )
        for i, j in itertools.combinations(range(len(st)), 2):
            if km[i, j] > MAX_KM:
                continue
            a, b = st.id[i], st.id[j]
            p, g = (px[a] - px[b]).dropna(), (gn[a] - gn[b]).dropna()
            both = p.index.intersection(g.index)
            if len(both) >= MIN_MONTHS:
                tasks.append(("pixel", frame, a, b, p[both], None))
                tasks.append(
                    (
                        "gnss",
                        frame,
                        a,
                        b,
                        g[both],
                        _unique_steps(steps[a] + steps[b], both) or None,
                    )
                )
    with ProcessPoolExecutor(32) as pool:
        rows = [r for out in pool.map(job, tasks, chunksize=4) for r in out]
    pd.DataFrame(rows).to_csv(RESULTS / f"{stem}.csv", index=False)
    pd.DataFrame(cuts).to_csv(RESULTS / f"{stem}_ministacks.csv", index=False)
    print(len(rows), "change points;", len(cuts), "ministacks")
