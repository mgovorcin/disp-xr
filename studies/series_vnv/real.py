"""Every method on real series: near-null differences and station pairs, pixel and GNSS.

Pairs are differenced in both, within 25 km, as in the smooth-trend validation. Near-null
windows 500 m apart have no shared-site signal to speak of, so a change there is false. The
holdout fits the first part of each pixel pair and predicts the last three years.
"""

import itertools
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from common import (
    DAYS,
    LOS,
    RESULTS,
    folder_of,
    grid,
    linearity_test,
    pixel_series,
    run_methods,
    validity_horizon,
)
from geepers.collocation import distance_matrix
from geepers.gps_sources.unr import UnrSource
from geepers.smooth import _unique_steps

MAX_KM, MIN_MONTHS, HOLD_YEARS = 25.0, 60, 3.0


def yearly_mean(s):
    return float(s.mean())


def holdout(s, truth=None):
    """Linear and hinge predictions of the last year's mean, three years on from the fit.

    The target is the series itself and, for pixel pairs, GNSS on the same pair: the
    series' own future carries its own errors, GNSS does not.
    """
    cut = s.index[-1] - pd.Timedelta(days=HOLD_YEARS * 365.25)
    train, test = s[s.index <= cut], s[s.index > cut - pd.Timedelta(days=0)]
    if len(train) < MIN_MONTHS or len(test) < 24:
        return {}
    lin = linearity_test(train.index, train.to_numpy(), sampling_days=DAYS)
    horizon = validity_horizon(lin, tolerance=10.0)
    base = train.iloc[-12:]
    last = test.iloc[-12:]
    gap = (last.index.mean() - base.index.mean()).days / 365.25
    actual = last.mean() - base.mean()
    out = {
        "hold_model": lin.model,
        "hold_p_quad": lin.p_values.get("quadratic", np.nan),
        "hold_p_hinge": lin.p_values.get("piecewise", np.nan),
        "hold_departure": lin.departure,
        "hold_horizon": horizon.years,
        "hold_driver": horizon.driver,
        "hold_gap": gap,
        "hold_actual": actual,
        "hold_linear": lin.trend.velocity * gap,
        "hold_hinge": (lin.rate_after * gap) if np.isfinite(lin.rate_after) else np.nan,
    }
    rw = linearity_test(
        train.index, train.to_numpy(), sampling_days=DAYS, noise_model="RWFNWN"
    )
    out.update(
        hold_rw_model=rw.model,
        hold_rw_horizon=validity_horizon(rw, tolerance=10.0).years,
        hold_rw_linear=rw.trend.velocity * gap,
    )
    if truth is not None:
        out["hold_truth"] = truth[last.index].mean() - truth[base.index].mean()
    return out


def job(task):
    role, frame, a, b, km, s, steps, truth = task
    row = {
        "role": role,
        "frame": frame,
        "a": a,
        "b": b,
        "km": km,
        "months": len(s),
        "first": s.index[0],
        "last": s.index[-1],
    }
    row.update(run_methods(s.index, s.to_numpy(), steps=steps))
    if role != "gnss":
        try:
            row.update(holdout(s, truth))
        except (ValueError, np.linalg.LinAlgError) as e:
            row["holdout_error"] = repr(e)[:80]
    return row


if __name__ == "__main__":
    src = UnrSource()
    catalog = src.steps()
    tasks = []
    for frame in ("f11116", "f38502", "f38502v", "f11116x"):
        folder = folder_of(frame)
        st = pd.read_csv(folder / "stations.csv")
        dates, cube = pixel_series(folder)
        px, off, gn, steps = {}, {}, {}, {}
        for k, row in enumerate(st.itertuples()):
            px[row.id] = grid(pd.Series(cube[:, k, 0], index=dates))
            off[row.id] = grid(pd.Series(cube[:, k, 1], index=dates))
            t = src.timeseries(
                row.id, start_date=str(dates[0].date()), end_date=str(dates[-1].date())
            )
            t = t.set_index(pd.to_datetime(t["date"]))
            gn[row.id] = grid(
                (t.east * LOS["east"] + t.north * LOS["north"] + t.up * LOS["up"])
                * 1000.0
            )
            steps[row.id] = list(
                pd.to_datetime(catalog.loc[catalog["id"] == row.id, "date"])
            )
        for sid in st.id:
            d = (px[sid] - off[sid]).dropna()
            if len(d) >= MIN_MONTHS:
                tasks.append(("near", frame, sid, sid, 0.5, d, None, None))
        km = distance_matrix(
            st.lon.to_numpy(), st.lat.to_numpy(), st.lon.to_numpy(), st.lat.to_numpy()
        )
        for i, j in itertools.combinations(range(len(st)), 2):
            if km[i, j] > MAX_KM:
                continue
            a, b = st.id[i], st.id[j]
            p, g = (px[a] - px[b]).dropna(), (gn[a] - gn[b]).dropna()
            both = p.index.intersection(g.index)
            if len(both) < MIN_MONTHS:
                continue
            tasks.append(("pixel", frame, a, b, km[i, j], p[both], None, g[both]))
            union = _unique_steps(steps[a] + steps[b], both)
            tasks.append(("gnss", frame, a, b, km[i, j], g[both], union or None, None))
    print(len(tasks), "series", flush=True)
    rows = []
    with ProcessPoolExecutor(48) as pool:
        for k, row in enumerate(pool.map(job, tasks, chunksize=2)):
            rows.append(row)
            if k % 100 == 0:
                print(k, flush=True)
    RESULTS.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(RESULTS / "real.csv", index=False)
    print("done", flush=True)
