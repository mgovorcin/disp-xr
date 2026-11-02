"""Shared pieces of the fit_series V&V: settings, the methods, the cached series."""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from geepers.changepoints import detect_changepoints, fit_sparse_changepoints
from geepers.linearity import linearity_test, validity_horizon
from geepers.neighbors import _on_grid
from geepers.smooth import fit_smooth_trend

warnings.simplefilter("ignore")
DAYS = 30.0
CACHE = Path(
    os.environ.get(
        "GEEPERS_STUDY_CACHE",
        "/u/aurora-r0/govorcin/00_tools/tests/geepers_changepoints/cache",
    )
)
RESULTS = Path(__file__).resolve().parent / "results"
NAME = re.compile(r"_(\d{8})T\d+Z_(\d{8})T\d+Z_v[\d.]+_(\d{8})T")
# Ground to satellite at the frame's center, as in the smooth-trend validation
LOS = {"east": 0.607, "north": -0.110, "up": 0.787}


def pixel_series(folder):
    """Rebase every cached product to the first reference: mm, per station and window."""
    rows = {}
    for path in sorted(folder.glob("OPERA*.npz")):
        ref, sec, made = NAME.search(path.name).groups()
        if (ref, sec) not in rows or rows[(ref, sec)][1] < made:
            rows[(ref, sec)] = (path, made)
    order = sorted(rows, key=lambda k: k[1])
    n_stations = len(pd.read_csv(folder / "stations.csv"))
    total = {order[0][0]: np.zeros((n_stations, 2))}
    out = {}
    for ref, sec in order:
        if ref not in total:
            continue
        z = np.load(rows[(ref, sec)][0])
        total[sec] = np.where(z["good"] >= 5, z["vals"] * 1000.0, np.nan) + total[ref]
        out[pd.Timestamp(sec)] = total[sec]
    dates = pd.DatetimeIndex(sorted(out))
    return dates, np.stack([out[d] for d in dates])


def folder_of(frame):
    """A cached station set; f11116x is fetch_overlap.py's, on the same stations as f38502v."""
    return RESULTS / frame if frame == "f11116x" else CACHE / frame


def grid(series):
    return _on_grid(series, DAYS)


def _largest(column):
    values = np.abs(column.to_numpy(dtype=float))
    values = values[np.isfinite(values)]
    return float(values.max()) if values.size else 0.0


def _attempt(row, name, fit):
    """One method's columns, or its error: a failure must not read as a verdict."""
    try:
        row.update(fit())
    except (ValueError, np.linalg.LinAlgError) as e:
        row[f"{name}_error"] = repr(e)[:80]


def _linearity(prefix, dates, y, opts, **kwargs):
    result = linearity_test(dates, y, **opts, **kwargs)
    horizon = validity_horizon(result, tolerance=10.0)
    return {
        f"{prefix}_model": result.model,
        f"{prefix}_p_quad": result.p_values.get("quadratic", np.nan),
        f"{prefix}_p_hinge": result.p_values.get("piecewise", np.nan),
        f"{prefix}_departure": result.departure,
        f"{prefix}_break": result.breakpoint,
        f"{prefix}_before": result.rate_before,
        f"{prefix}_after": result.rate_after,
        f"{prefix}_change_sigma": result.rate_change_sigma,
        f"{prefix}_rate": result.trend.velocity,
        f"{prefix}_rate_sigma": result.trend.velocity_uncertainty,
        f"{prefix}_acc": result.acceleration,
        f"{prefix}_acc_sigma": result.acceleration_sigma,
        f"{prefix}_horizon": horizon.years,
        f"{prefix}_driver": horizon.driver,
    }


def _changes(prefix, result):
    table = result.changepoints
    return {
        f"{prefix}_n": len(table),
        f"{prefix}_n_offset": int((table["kind"] == "offset").sum()),
        f"{prefix}_n_rate": int((table["kind"] == "rate").sum()),
        f"{prefix}_n_both": int((table["kind"] == "both").sum()),
        f"{prefix}_first": table["date"].iloc[0] if len(table) else pd.NaT,
        f"{prefix}_max_offset": _largest(table["offset"]),
        f"{prefix}_max_rate": _largest(table["rate_change"]),
        f"{prefix}_rate": result.rate,
        f"{prefix}_rate_sigma": result.rate_sigma,
    }


def run_methods(dates, y, steps=None):
    """Every method's verdict on one binned series, in mm. One flat row."""
    row = {}
    opts = {"sampling_days": DAYS, "step_dates": steps}
    _attempt(row, "lin", lambda: _linearity("lin", dates, y, opts))
    # The same test with a random walk in the noise model
    _attempt(row, "rw", lambda: _linearity("rw", dates, y, opts, noise_model="RWFNWN"))
    _attempt(row, "det", lambda: _changes("det", detect_changepoints(dates, y, **opts)))
    _attempt(
        row,
        "sparse",
        lambda: _changes("sparse", fit_sparse_changepoints(dates, y, **opts)),
    )

    def smooth():
        fit = fit_smooth_trend(dates, y, units="mm", max_random_walk_mm=5.0, **opts)
        return {"smooth_shape": fit.shape, "smooth_evidence": fit.evidence}

    _attempt(row, "smooth", smooth)
    return row
