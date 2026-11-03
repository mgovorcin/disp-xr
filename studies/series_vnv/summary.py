"""Every table of docs/studies/series_vnv.md, from the runs' CSV files."""

import numpy as np
import pandas as pd
from common import RESULTS

pd.set_option("display.width", 200, "display.max_columns", 40, "display.max_rows", 200)
TRACKS = ("f11116", "f38502", "f38502v")


def flagged(t, pre="lin", dep=0.0, alpha=0.01):
    p = t[[f"{pre}_p_quad", f"{pre}_p_hinge"]].min(axis=1)
    return ((p < alpha) & (t[f"{pre}_departure"] > dep)).where(
        t[f"{pre}_model"].notna()
    )


def tests(t):
    has_steps = t.det_n.notna()
    return pd.DataFrame(
        {
            "model != linear": (t.lin_model != "linear").where(t.lin_model.notna()),
            "p<.01": flagged(t),
            "p<.01, >5 mm": flagged(t, dep=5),
            "p<.01, >10 mm": flagged(t, dep=10),
            "RWFNWN model": (t.rw_model != "linear").where(t.rw_model.notna()),
            "change point": (t.det_n > 0).where(has_steps),
            "offset >15 mm": (t.det_max_offset > 15).where(has_steps),
            "rate change pt": ((t.det_n_rate + t.det_n_both) > 0).where(has_steps),
            "smooth shape": (t.smooth_shape == "smooth").where(t.smooth_shape.notna()),
        }
    ).astype(float)


def synthetic():
    d = pd.read_csv(RESULTS / "synthetic.csv")
    v = tests(d)
    d["walk"] = d.walk.fillna(-1.0)
    d["size"] = d["size"].fillna(0.0)
    d["at"] = d["at"].fillna(0.5)
    t = v.groupby([d.kind, d.years, d.walk, d["size"], d["at"]]).mean().round(2)
    print("# Synthetic: share flagged, 200 series per row\n")
    print(t.to_string())

    # The true rate of a line is the first draw of its seed (synthetic.series)
    line = d[d.kind == "line"].copy()
    line["true"] = [
        np.random.default_rng([20261102, s, r]).uniform(-5, 5)
        for s, r in zip(line.scenario, line.rep)
    ]
    print("\n# Rate error over its sigma on lines (1 = calibrated)\n")
    print(
        line.groupby(["years", "walk"])
        .apply(
            lambda g: pd.Series(
                {
                    "rms error": np.sqrt(((g.lin_rate - g.true) ** 2).mean()),
                    "FNWN sigma": g.lin_rate_sigma.median(),
                    "FNWN z": ((g.lin_rate - g.true) / g.lin_rate_sigma).std(),
                    "RWFNWN sigma": g.rw_rate_sigma.median(),
                    "RWFNWN z": ((g.rw_rate - g.true) / g.rw_rate_sigma).std(),
                }
            ),
            include_groups=False,
        )
        .round(2)
        .to_string()
    )

    h = d[d.kind == "hinge"].copy()
    n_epochs = np.floor(10 * 365.25 / 30)
    true = pd.Timestamp("2016-07-05") + pd.to_timedelta(
        h["at"] * (n_epochs - 1) * 30, unit="D"
    )
    h["error"] = (pd.to_datetime(h.lin_break) - true).dt.days.abs() / 365.25
    h["change"] = h.lin_after - h.lin_before
    gate = flagged(h, dep=5) == 1
    print(
        "\n# Hinges flagged at p<.01, >5 mm: break error (yr) and rate change (mm/yr)\n"
    )
    print(
        h[gate]
        .groupby(["walk", "size"])
        .agg(
            n=("error", "size"),
            error_median=("error", "median"),
            error_q75=("error", lambda x: x.quantile(0.75)),
            change=("change", "median"),
        )
        .round(2)
        .to_string()
    )
    s = d[(d.kind == "step") & (d.det_n > 0)]
    print("\n# Steps: detect's largest offset, median\n")
    print(
        s.groupby(["walk", "size"])
        .det_max_offset.median()
        .round(1)
        .unstack()
        .to_string()
    )


def cluster_interval(px, stat, n=1000):
    """90% interval, resampling stations: pairs that share a station are not separate tests."""
    rng = np.random.default_rng(0)
    ids = sorted(set(px.a) | set(px.b))
    pos = {s: i for i, s in enumerate(ids)}
    a, b = px.a.map(pos).to_numpy(), px.b.map(pos).to_numpy()
    out = []
    for _ in range(n):
        count = np.bincount(rng.integers(0, len(ids), len(ids)), minlength=len(ids))
        out.append(stat(count[a] * count[b]))
    return np.nanquantile(out, [0.05, 0.95]).round(2)


def real():
    d = pd.read_csv(RESULTS / "real.csv")
    fails = {c: int(d[c].notna().sum()) for c in d if c.endswith("_error")}
    print(
        "\n# Real series\n\nseries",
        d.groupby(["frame", "role"]).size().to_dict(),
        "\nfailures",
        fails,
    )
    main = d[d.frame.isin(TRACKS)]

    near = main[main.role == "near"]
    v = tests(near)
    print("\n## 500 m differences: share flagged\n")
    print(v.groupby(near.frame).mean().T.assign(all=v.mean()).round(2).to_string())

    px = main[main.role == "pixel"].reset_index(drop=True)
    g = main[main.role == "gnss"].reset_index(drop=True)
    assert (px.a == g.a).all() and (px.b == g.b).all() and (px.frame == g.frame).all()
    vp, vg = tests(px), tests(g)
    print(
        "\n## Pairs: flagged in the pixels, in GNSS; pixel share where GNSS is and is not\n"
    )
    rows = []
    for c in vp:
        ok = vp[c].notna() & vg[c].notna()
        a, b = vp[c][ok], vg[c][ok]
        rows.append(
            {
                "test": c,
                "pixel": a.mean(),
                "gnss": b.mean(),
                "hit": a[b == 1].mean(),
                "false": a[b == 0].mean(),
                "n_gnss": int((b == 1).sum()),
                "n": int(ok.sum()),
            }
        )
    print(pd.DataFrame(rows).round(2).to_string(index=False))

    for lim in (5, 10, 20):
        s = (flagged(g, dep=lim) == 1).to_numpy()
        stations = len(set(px.a[s]) | set(px.b[s]))
        print(
            f"\nGNSS non-linear, p<.01, >{lim} mm: {s.sum()} pairs, {stations} stations, {px.frame[s].value_counts().to_dict()}"
        )
        for c in (
            "model != linear",
            "p<.01, >10 mm",
            "RWFNWN model",
            "change point",
            "smooth shape",
        ):
            hit, false = vp[c][s].mean(), vp[c][~s].mean()
            print(f"  {c:18s} hit {hit:.2f} false {false:.2f}")
    s = (flagged(g, dep=10) == 1).to_numpy()
    print("\nStrong pairs: departure in GNSS and pixels")
    print(
        pd.DataFrame(
            {
                "frame": px.frame,
                "a": px.a,
                "b": px.b,
                "gnss": g.lin_departure,
                "pixel": px.lin_departure,
                "rw": px.rw_model,
                "changes": px.det_n,
            }
        )[s]
        .round(1)
        .to_string(index=False)
    )

    gate = vp["p<.01, >10 mm"].to_numpy() == 1
    truth = (flagged(g, dep=10) == 1).to_numpy()

    def weighted(w, m):
        return np.sum(w[m]) and float(np.sum(w[m & gate]) / np.sum(w[m]))

    print("\n90% intervals, resampling stations, gate p<.01 and >10 mm:")
    print(
        "  hit",
        cluster_interval(px, lambda w: weighted(w, truth)),
        "false",
        cluster_interval(px, lambda w: weighted(w, ~truth)),
    )

    ok = px.det_n.notna() & px.sparse_n.notna()
    print(
        "\ndetect and sparse same flag:",
        round(((px.det_n > 0) == (px.sparse_n > 0))[ok].mean(), 3),
    )

    # Rates against GNSS
    e = px.lin_rate - g.lin_rate
    print("\n## Rates against GNSS\n")
    print(
        "median |pixel - GNSS|",
        round(e.abs().median(), 2),
        "rms",
        round(float(np.sqrt((e**2).mean())), 2),
    )
    for pre, name in (("lin", "FNWN"), ("rw", "RWFNWN")):
        z = (px[f"{pre}_rate"] - g[f"{pre}_rate"]) / np.hypot(
            px[f"{pre}_rate_sigma"], g[f"{pre}_rate_sigma"]
        )
        print(
            f"{name}: median pixel sigma {px[f'{pre}_rate_sigma'].median():.2f}; misfit/sigma NMAD {1.4826 * (z - z.median()).abs().median():.2f}"
        )

    # Holdout
    t = px.dropna(subset=["hold_model"]).copy()
    print(f"\n## Three years ahead, {len(t)} pairs; share within 10 mm of the target\n")
    rows = []
    for target, name in (("hold_truth", "GNSS"), ("hold_actual", "own future")):
        line, hinge, rw = (
            (t.hold_linear - t[target]).abs(),
            (t.hold_hinge - t[target]).abs(),
            (t.hold_rw_linear - t[target]).abs(),
        )
        linear = (t.hold_model == "linear").to_numpy()
        short, rw_short = (t.hold_horizon < t.hold_gap).to_numpy(), (
            t.hold_rw_horizon < t.hold_gap
        ).to_numpy()
        rows += [
            {
                "target": name,
                "group": "verdict linear",
                "n": linear.sum(),
                "line": (line[linear] <= 10).mean(),
            },
            {
                "target": name,
                "group": "verdict non-linear",
                "n": (~linear).sum(),
                "line": (line[~linear] <= 10).mean(),
                "rate_after": (hinge[~linear] <= 10).mean(),
            },
            {
                "target": name,
                "group": "FNWN horizon >= 3 yr",
                "n": (~short).sum(),
                "line": (line[~short] <= 10).mean(),
            },
            {
                "target": name,
                "group": "FNWN horizon < 3 yr",
                "n": short.sum(),
                "line": (line[short] <= 10).mean(),
            },
            {
                "target": name,
                "group": "RWFNWN horizon >= 3 yr",
                "n": (~rw_short).sum(),
                "line": (rw[~rw_short] <= 10).mean(),
            },
            {
                "target": name,
                "group": "RWFNWN horizon < 3 yr",
                "n": rw_short.sum(),
                "line": (rw[rw_short] <= 10).mean(),
            },
        ]
    print(pd.DataFrame(rows).round(2).to_string(index=False))
    return d


def two_tracks(d):
    def side(frame, role):
        t = d[(d.frame == frame) & (d.role == role)].copy()
        t["key"] = [tuple(sorted(x)) for x in zip(t.a, t.b)]
        return t.set_index("key")

    for name, fa, fb in (
        ("Quiet stations, longest records", "f11116", "f38502"),
        ("Screened for signal", "f11116x", "f38502v"),
    ):
        a, b, g = side(fa, "pixel"), side(fb, "pixel"), side(fa, "gnss")
        k = a.index.intersection(b.index).intersection(g.index)
        a, b, g = a.loc[k], b.loc[k], g.loc[k]
        va, vb, vg = (
            tests(a).fillna(0) == 1,
            tests(b).fillna(0) == 1,
            tests(g).fillna(0) == 1,
        )
        print(
            f"\n## Two tracks: {name}: {len(k)} pairs, {len({s for p in k for s in p})} stations\n"
        )
        rows = [
            {
                "test": c,
                "first": va[c].sum(),
                "second": vb[c].sum(),
                "both": (va[c] & vb[c]).sum(),
                "chance": round(va[c].mean() * vb[c].mean() * len(k), 1),
                "gnss": vg[c].sum(),
            }
            for c in va
        ]
        print(pd.DataFrame(rows).to_string(index=False))
        print(
            "departure correlation between tracks",
            round(np.corrcoef(a.lin_departure, b.lin_departure)[0, 1], 2),
        )
        s = flagged(g, dep=5) == 1
        print(
            pd.DataFrame(
                {
                    "gnss": g.lin_departure,
                    "gnss_break": g.lin_break,
                    "first": a.lin_departure,
                    "first_break": a.lin_break,
                    "second": b.lin_departure,
                    "second_break": b.lin_break,
                }
            )[s.to_numpy()]
            .round(0)
            .to_string()
        )
        both = flagged(a, dep=5).eq(1).to_numpy() & flagged(b, dep=5).eq(1).to_numpy()
        gap = (
            pd.to_datetime(a.lin_break) - pd.to_datetime(b.lin_break)
        ).dt.days.abs() / 365.25
        print(
            "break dates apart, flagged on both tracks: median yr",
            round(gap[both].median(), 2) if both.any() else None,
            "n",
            int(both.sum()),
        )


def dates():
    c = pd.read_csv(RESULTS / "change_dates.csv", parse_dates=["date"])
    c = c[c.frame.isin(TRACKS)]
    m = pd.read_csv(RESULTS / "ministacks.csv", parse_dates=["date"])
    rng = np.random.default_rng(0)

    def within(days, cuts, limit=30):
        cuts = np.sort(cuts.to_numpy("datetime64[D]"))
        days = days.to_numpy("datetime64[D]")
        i = np.clip(np.searchsorted(cuts, days), 1, len(cuts) - 1)
        d = np.minimum(
            np.abs((days - cuts[i - 1]).astype(int)),
            np.abs((cuts[i] - days).astype(int)),
        )
        return d <= limit

    print(
        "\n## Change points: size, and ministack boundaries within 30 days (chance alongside)\n"
    )
    rows = []
    for role, t in c[c.kind != "rate"].groupby("role"):
        hit, base = [], []
        for frame, part in t.groupby("frame"):
            cuts = m[m.frame == frame].date
            hit += list(within(part.date, cuts))
            days = pd.to_datetime(
                rng.uniform(
                    pd.Timestamp("2017-07-01").value,
                    pd.Timestamp("2026-01-01").value,
                    5000,
                ).astype("int64")
            )
            base += list(within(pd.Series(days), cuts))
        size = t.offset.abs()
        rows.append(
            {
                "role": role,
                "n": len(t),
                "size_q25": size.quantile(0.25),
                "size_median": size.median(),
                "size_q75": size.quantile(0.75),
                "boundary": np.mean(hit),
                "chance": np.mean(base),
            }
        )
    print(pd.DataFrame(rows).round(2).to_string(index=False))
    t = c[c.kind != "rate"].copy()
    t["half"] = t.date.dt.year + (t.date.dt.month > 6) * 0.5
    print(
        "\nby half year\n",
        t.groupby(["half", "role"]).size().unstack(fill_value=0).T.to_string(),
    )
    p = c[(c.role == "pixel") & (c.kind != "rate")]
    share = p[(p.a.isin(["P533", "P552"])) | (p.b.isin(["P533", "P552"]))]
    print(f"\npixel-pair offsets involving P533 or P552: {len(share)} of {len(p)}")
    print(
        "within 24-32 mm, one cycle:", round(p.offset.abs().between(24, 32).mean(), 2)
    )
    tracks = p[p.frame.isin(("f11116", "f38502"))].assign(
        key=lambda t: [tuple(sorted(x)) for x in zip(t.a, t.b)]
    )
    first, second = tracks[tracks.frame == "f11116"], tracks[tracks.frame == "f38502"]
    joined = first.merge(second, on="key", suffixes=("", "_other"))
    joined = joined[(joined.date - joined.date_other).dt.days.abs() <= 90]
    joined = joined.dropna(subset=["offset", "offset_other"])
    print(
        f"offsets within 90 days on both tracks: {len(joined)}, same sign "
        f"{(np.sign(joined.offset) == np.sign(joined.offset_other)).mean():.2f}, "
        f"r {np.corrcoef(joined.offset, joined.offset_other)[0, 1]:.2f}"
    )
    for st, year in (("P533", 2019), ("P552", 2023)):
        t = c[
            ((c.a == st) | (c.b == st)) & (c.date.dt.year == year) & (c.kind != "rate")
        ]
        print(
            st,
            t.groupby(["frame", "role"])
            .agg(
                n=("kind", "size"),
                size=("offset", lambda x: round(x.abs().median(), 1)),
                when=("date", lambda x: str(x.median().date())),
            )
            .to_string(),
        )


def end_to_end():
    r = pd.read_csv(RESULTS / "end_to_end.csv")
    ok = r.fs_model.notna()
    direct = (np.minimum(r.lin_p_quad, r.lin_p_hinge) < 0.01) & (r.lin_departure > 10)
    print("\n## fit_series against direct calls\n")
    print(
        f"{ok.sum()} of {len(r)} fitted; same model {(r.lin_model == r.fs_model)[ok].mean():.3f}; "
        f"same significant {(direct == (r.fs_significant == 1))[ok].mean():.3f}; "
        f"significant {(r.fs_significant == 1)[ok].mean():.3f} against {direct[ok].mean():.3f}"
    )
    for name, a, b in (
        ("rate", r.rw_rate, r.fs_velocity),
        ("rate sigma", r.rw_rate_sigma, r.fs_sigma),
        ("horizon", r.rw_horizon, r.fs_validity),
    ):
        print(
            f"{name} (RWFNWN): median |difference| {(a - b)[ok].abs().median():.3f}, medians {a[ok].median():.2f} and {b[ok].median():.2f}"
        )


if __name__ == "__main__":
    synthetic()
    dates_input = real()
    two_tracks(dates_input)
    dates()
    end_to_end()
