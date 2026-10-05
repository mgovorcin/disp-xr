"""Run the T29 split-metric comparison (T29.2, T29.3).

For each frame, cell size (1, 2 and 10 km) and method this downsamples the
study stack written by ``prepare.py`` and scores the result:

- ``grid``: the regular grid of :func:`disp_xr.downsample` at that size. Its
  number of cells with data is the point budget of the size;
- ``displacement``, ``velocity``, ``structure``: :func:`disp_xr.quadtree`
  with that split metric and the same budget (``n_points``), so the methods
  are compared at equal point counts.

Criteria: reconstruction error against the 30 m field (each method's cell
values painted back on the 30 m grid, nearest; RMSE over reliable pixels)
for displacement (RMS over epochs, and the last epoch) and velocity; GNSS
agreement (cell velocity at each station of ``gnss_los.py`` against its LOS
velocity, RMS after a plane; the ``full`` row gives the 30 m value); the
gradient preservation (T29.4: RMSE between the binned velocity profiles,
30 m and painted, in a 1 km swath along the ``profiles`` of
``frames.yaml``); the cell count and the runtime.

Writes ``results/<frame>.csv``, one row per size and method. Run with
``pixi run -e test-py312 python studies/split_metric/run.py [FRAME ...]``.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import yaml
from prepare import OUT, WORKERS
from stations import HERE

import disp_xr as dxr

SIZES_KM = {1: 33, 2: 67, 10: 333}
METRICS = ("displacement", "velocity", "structure")
MIN_LEVEL, MAX_LEVEL = 3, 10
PROFILE_HALF_WIDTH = 500.0  # meters: a 1 km swath
PROFILE_BIN = 300.0  # meters along the profile
# Column order of results and checkpoints: appended CSV rows are matched by
# position, so every write uses this order.
COLUMNS = [
    "frame", "size_km", "method", "n_points", "threshold", "seconds", "levels",
    "disp_rmse_mm", "disp_rmse_last_mm", "vel_rmse_mm_yr", "gnss_rms_mm_yr",
    "n_gnss", "profile_rmse_mm_yr", "weights",
]  # fmt: skip


def append_rows(rows: pd.DataFrame, checkpoint: Path) -> None:
    """Append ``rows`` to ``checkpoint`` in the order of ``COLUMNS``."""
    rows.reindex(columns=COLUMNS).to_csv(
        checkpoint, mode="a", header=not checkpoint.exists(), index=False
    )


def grid_leaf_id(shape: tuple[int, int], factor: int) -> np.ndarray:
    """Return the cell index of each pixel for a regular grid of ``factor``."""
    rows = np.arange(shape[0]) // factor
    cols = np.arange(shape[1]) // factor
    n_cols = -(-shape[1] // factor)
    return rows[:, None] * n_cols + cols[None, :]


def paint_rmse(
    leaf_id: np.ndarray, values: np.ndarray, field: np.ndarray, reliable: np.ndarray
) -> float:
    """Return the RMSE of cell values painted on the grid against ``field``.

    ``values`` holds one value per cell; ``leaf_id`` -1 marks pixels in no
    cell, which count as missing.
    """
    painted = np.append(values, np.nan)[leaf_id]
    diff = (painted - field)[reliable]
    diff = diff[np.isfinite(diff)]
    return float(np.sqrt(np.mean(diff**2))) if diff.size else np.nan


def cell_mean(
    leaf_id: np.ndarray, field: np.ndarray, weight: np.ndarray, n: int
) -> np.ndarray:
    """Return the weighted mean of ``field`` in each of ``n`` cells."""
    ok = (leaf_id >= 0) & np.isfinite(field) & (weight > 0)
    ids = leaf_id[ok]
    sum_w = np.bincount(ids, weight[ok], minlength=n)
    sum_wv = np.bincount(ids, weight[ok] * field[ok], minlength=n)
    with np.errstate(invalid="ignore", divide="ignore"):
        return sum_wv / sum_w


def gnss_rms(at_stations: np.ndarray, gnss: pd.DataFrame) -> float:
    """Return the RMS of velocity minus GNSS LOS velocity after a plane.

    DISP is relative to a reference pixel and the GNSS are absolute, and
    plate motion on a LOS vector that changes across the frame is a ramp,
    so a plane in x, y is removed from the residuals first.
    """
    residual = at_stations - gnss["v_los"].to_numpy()
    ok = np.isfinite(residual)
    if ok.sum() < 4:
        return np.nan
    x, y = gnss["x"].to_numpy()[ok], gnss["y"].to_numpy()[ok]
    design = np.column_stack([np.ones(ok.sum()), x - x.mean(), y - y.mean()])
    coef, *_ = np.linalg.lstsq(design, residual[ok], rcond=None)
    return float(np.sqrt(np.mean((residual[ok] - design @ coef) ** 2)))


def station_pixels(ds: xr.Dataset, gnss: pd.DataFrame) -> tuple[np.ndarray, ...]:
    """Return the row and column of the pixel holding each station."""
    rows = np.abs(ds["y"].values[None, :] - gnss["y"].to_numpy()[:, None]).argmin(1)
    cols = np.abs(ds["x"].values[None, :] - gnss["x"].to_numpy()[:, None]).argmin(1)
    return rows, cols


def profile_pixels(
    ds: xr.Dataset, profiles: list[list[float]]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return rows, columns and bin ids of the swath pixels of every profile.

    Each profile ``[lon0, lat0, lon1, lat1]`` is a straight line in the
    frame's CRS; pixels within ``PROFILE_HALF_WIDTH`` of it are binned every
    ``PROFILE_BIN`` along it, with bin ids running on across profiles.
    """
    from pyproj import Transformer

    to_grid = Transformer.from_crs("EPSG:4326", ds.rio.crs, always_xy=True)
    xs, ys = ds["x"].values, ds["y"].values
    rows, cols, bins = [], [], []
    offset = 0
    for lon0, lat0, lon1, lat1 in profiles:
        (x0, x1), (y0, y1) = to_grid.transform([lon0, lon1], [lat0, lat1])
        length = float(np.hypot(x1 - x0, y1 - y0))
        ux, uy = (x1 - x0) / length, (y1 - y0) / length
        pad = PROFILE_HALF_WIDTH
        r = np.nonzero((ys >= min(y0, y1) - pad) & (ys <= max(y0, y1) + pad))[0]
        c = np.nonzero((xs >= min(x0, x1) - pad) & (xs <= max(x0, x1) + pad))[0]
        dx = xs[c][None, :] - x0
        dy = ys[r][:, None] - y0
        along = dx * ux + dy * uy
        across = -dx * uy + dy * ux
        inside = (np.abs(across) <= PROFILE_HALF_WIDTH) & (along >= 0)
        inside &= along <= length
        ri, ci = np.nonzero(inside)
        rows.append(r[ri])
        cols.append(c[ci])
        bins.append(offset + (along[ri, ci] // PROFILE_BIN).astype(np.int64))
        offset += int(length // PROFILE_BIN) + 1
    return np.concatenate(rows), np.concatenate(cols), np.concatenate(bins)


def profile_rmse(values: np.ndarray, field: np.ndarray, bins: np.ndarray) -> float:
    """Return the RMSE between the binned profiles of ``values`` and ``field``."""
    ok = np.isfinite(values) & np.isfinite(field)
    n = np.bincount(bins[ok])
    with np.errstate(invalid="ignore", divide="ignore"):
        a = np.bincount(bins[ok], values[ok]) / n
        b = np.bincount(bins[ok], field[ok]) / n
    diff = (a - b)[n > 0]
    return float(np.sqrt(np.mean(diff**2))) if diff.size else np.nan


def score_all(
    parts: list[dict],
    ds: xr.Dataset,
    velocity: np.ndarray,
    static: np.ndarray,
    reliable: np.ndarray,
    gnss: pd.DataFrame,
    profile: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    """Add the reconstruction errors to every partition, reading each epoch once."""
    per_epoch = np.full((len(parts), ds.sizes["time"]), np.nan)
    for t in range(ds.sizes["time"]):
        field = ds["displacement"].isel(time=t).values
        for i, part in enumerate(parts):
            per_epoch[i, t] = paint_rmse(
                part["leaf_id"], part["cell_disp"][t], field, reliable
            )
    for i, part in enumerate(parts):
        leaf_id = part.pop("leaf_id")
        part.pop("cell_disp")
        cell_vel = cell_mean(leaf_id, velocity, static, int(leaf_id.max()) + 1)
        part["disp_rmse_mm"] = 1e3 * float(np.sqrt(np.nanmean(per_epoch[i] ** 2)))
        part["disp_rmse_last_mm"] = 1e3 * per_epoch[i, -1]
        part["vel_rmse_mm_yr"] = 1e3 * paint_rmse(leaf_id, cell_vel, velocity, reliable)
        ids = leaf_id[station_pixels(ds, gnss)]
        at = np.where(ids >= 0, np.append(cell_vel, np.nan)[ids], np.nan)
        part["gnss_rms_mm_yr"] = 1e3 * gnss_rms(at, gnss)
        part["n_gnss"] = int(np.isfinite(at).sum())
        p_rows, p_cols, p_bins = profile
        ids = leaf_id[p_rows, p_cols]
        painted = np.where(ids >= 0, np.append(cell_vel, np.nan)[ids], np.nan)
        field = np.where(reliable[p_rows, p_cols], velocity[p_rows, p_cols], np.nan)
        part["profile_rmse_mm_yr"] = 1e3 * profile_rmse(painted, field, p_bins)


def weights_for(ds: xr.Dataset, params: dict) -> tuple[xr.DataArray, xr.DataArray]:
    """Return per-epoch and static quality weights for ``params``.

    ``params`` holds :func:`disp_xr.quality_weights` keywords; floors default
    to the product thresholds recorded by ``prepare.py`` (``study_floors``),
    as ``quality_weights`` would read them from ``recommended_mask``. Two
    study keys: ``drop`` (a quality layer to leave out) and ``equal`` (equal
    weights on pixels with data).
    """
    params = dict(params)
    if params.pop("equal", False):
        ones = ds["displacement"].isel(time=-1, drop=True).notnull().astype(float)
        return ones, ones
    drop = params.pop("drop", None)
    quality = [
        n
        for n in (
            "phase_similarity",
            "temporal_coherence",
            "timeseries_inversion_residuals",
        )
        if n in ds and n != drop
    ]
    floors = json.loads(ds.attrs.get("study_floors", "{}")) | params.pop("floors", {})
    qw = dxr.quality_weights(ds, quality, floors=floors, **params)
    static = qw.static_weight.fillna(0.0)
    # As downsample does: epochs without quality layers take the static weight.
    return qw.weight.fillna(static), static


def cached_weights(
    ds: xr.Dataset, frame: str, label: str, params: dict
) -> tuple[xr.DataArray, xr.DataArray]:
    """Return the weights of ``params``, computed once and kept as Zarr.

    Every partition of a run reads the weights again; reading one stored
    layer costs about half of recomputing it from three quality layers.
    """
    store = OUT / frame / "weights" / f"{label}.zarr"
    if not (store / "zarr.json").exists():
        weight, static = weights_for(ds, params)
        out = xr.Dataset(
            {"weight": weight.astype(np.float32), "static": static.astype(np.float32)}
        )
        # Equal weights have no time dimension.
        sizes = {"time": 1, "y": 1024, "x": 1024}
        out = out.chunk({d: n for d, n in sizes.items() if d in out.dims})
        # Written under a temporary name and renamed when complete, so a run
        # killed mid-write never leaves a store that looks finished.
        partial = store.with_name(store.name + ".partial")
        out.drop_vars([c for c in out.coords if c not in ("time", "y", "x")]).to_zarr(
            partial, mode="w"
        )
        partial.rename(store)
    cached = xr.open_zarr(store)
    return cached["weight"], cached["static"]


def run_frame(
    frame: str,
    profiles: list[list[float]],
    *,
    sizes: dict[int, int] = SIZES_KM,
    methods: tuple[str, ...] = METRICS,
    params: dict | None = None,
    label: str = "default",
    checkpoint: Path | None = None,
) -> pd.DataFrame:
    """Score the regular grid and the quadtree ``methods`` at every size.

    With ``checkpoint`` (a CSV), each size's rows are appended as soon as
    they are scored, and sizes already there for ``label`` are skipped, so
    a killed run resumes where it stopped.
    """
    done = pd.DataFrame()
    if checkpoint is not None and checkpoint.exists():
        done = pd.read_csv(checkpoint)
        done = done[done["weights"] == label]
    done_sizes = set(done["size_km"]) if len(done) else set()
    ds = xr.open_zarr(OUT / frame / "stack.zarr")
    velocity_da = xr.open_zarr(OUT / frame / "velocity.zarr")["velocity"]
    weights, static_da = cached_weights(ds, frame, label, params or {})
    static = static_da.values
    velocity = velocity_da.values
    reliable = (static > 0) & ds["displacement"].notnull().any("time").values
    gnss = pd.read_csv(HERE / "stations" / f"{frame}_los.csv")
    profile = profile_pixels(ds, profiles)
    rows = [done] if len(done) else []
    for size_km, factor in sizes.items():
        if size_km in done_sizes:
            print(f"{frame} {label} {size_km} km: done, skipped", flush=True)
            continue
        parts = []
        t0 = time.perf_counter()
        coarse = dxr.downsample(
            ds, factor=factor, layers=["displacement"], weights=weights
        ).compute()
        cell_disp = coarse["displacement"].values.reshape(coarse.sizes["time"], -1)
        n_points = int(np.isfinite(cell_disp).any(axis=0).sum())
        parts.append(
            {
                "frame": frame,
                "size_km": size_km,
                "method": "grid",
                "n_points": n_points,
                "threshold": np.nan,
                "seconds": time.perf_counter() - t0,
                "levels": "",
                "leaf_id": grid_leaf_id(reliable.shape, factor).astype(np.int32),
                "cell_disp": cell_disp,
            }
        )
        print({k: v for k, v in parts[-1].items() if k[0] != "l" and k[0] != "c"})
        for metric in methods:
            t0 = time.perf_counter()
            tree = dxr.quadtree(
                ds,
                n_points=n_points,
                metric=metric,
                min_level=MIN_LEVEL,
                max_level=MAX_LEVEL,
                weights=weights,
                velocity=velocity_da if metric == "velocity" else None,
            )
            cell_disp = tree.points["displacement"].transpose("time", "point").values
            levels = tree.points["level"].to_series().value_counts().sort_index()
            parts.append(
                {
                    "frame": frame,
                    "size_km": size_km,
                    "method": metric,
                    "n_points": tree.points.sizes["point"],
                    "threshold": tree.points.attrs["quadtree_threshold"],
                    "seconds": time.perf_counter() - t0,
                    "levels": " ".join(f"{k}:{v}" for k, v in levels.items()),
                    "leaf_id": tree.leaf_id.values.astype(np.int32),
                    "cell_disp": cell_disp,
                }
            )
            print({k: v for k, v in parts[-1].items() if k[0] != "l" and k[0] != "c"})
        score_all(parts, ds, velocity, static, reliable, gnss, profile)
        size_rows = pd.DataFrame(parts).assign(weights=label)
        if checkpoint is not None:
            append_rows(size_rows, checkpoint)
        rows.append(size_rows)
    if 0.03 in done_sizes:
        return pd.concat(rows, ignore_index=True)
    # The 30 m velocity at the station pixel: the agreement before any
    # downsampling, the floor the cells are compared with.
    at = velocity[station_pixels(ds, gnss)]
    at = np.where(reliable[station_pixels(ds, gnss)], at, np.nan)
    full = pd.DataFrame(
        [
            {
                "frame": frame,
                "size_km": 0.03,
                "method": "full",
                "n_points": int(reliable.sum()),
                "gnss_rms_mm_yr": 1e3 * gnss_rms(at, gnss),
                "n_gnss": int(np.isfinite(at).sum()),
                "weights": label,
            }
        ]
    )
    if checkpoint is not None:
        append_rows(full, checkpoint)
    rows.append(full)
    return pd.concat(rows, ignore_index=True)


# T29.5: one parameter at a time around the defaults of quality_weights
# (penalty 0.5, f0 0.25, min_residual 0.05, floors from the product), plus
# equal weights. Floors are absolute values here, not offsets.
# Eight variants, trimmed from fourteen for run time (2026-10-04): the f0=0.1,
# min_residual and similarity-floor variants are left for a follow-up if
# these show the parameters matter. Labels name the cached weight stores.
SWEEP: dict[str, dict] = {
    "default": {},
    "penalty=0.25": {"penalty": 0.25},
    "penalty=1": {"penalty": 1.0},
    "f0=0.5": {"f0": 0.5},
    "floor_coh=0.5": {"floors": {"temporal_coherence": 0.5}},
    "floor_coh=0.8": {"floors": {"temporal_coherence": 0.8}},
    "no_residuals": {"drop": "timeseries_inversion_residuals"},
    "equal": {"equal": True},
}
SWEEP_SIZES = {2: 67}


def best_metric(frame: str) -> str:
    """Return the split metric ranked best at 2 km in ``results/<frame>.csv``.

    The rank sums each metric's rank on velocity reconstruction, GNSS
    agreement and the gradient profile (lower is better).
    """
    table = pd.read_csv(HERE / "results" / f"{frame}.csv")
    rows = table[(table["size_km"] == 2) & table["method"].isin(METRICS)]
    criteria = ["vel_rmse_mm_yr", "gnss_rms_mm_yr", "profile_rmse_mm_yr"]
    ranks = rows.set_index("method")[criteria].rank().sum(axis=1)
    return str(ranks.idxmin())


def sweep_frame(frame: str, profiles: list[list[float]]) -> pd.DataFrame:
    """Score every weight variant of ``SWEEP`` at ``SWEEP_SIZES``.

    Only the grid and the frame's best split metric are swept: the weights
    change the cell means, not much which metric wins, and the sweep is the
    costly part of the study.
    """
    methods = (best_metric(frame),)
    print(f"{frame}: sweeping the grid and {methods[0]}", flush=True)
    tables = []
    for label, params in SWEEP.items():
        table = run_frame(
            frame,
            profiles,
            sizes=SWEEP_SIZES,
            methods=methods,
            params=params,
            label=label,
            checkpoint=HERE / "results" / f"{frame}_sweep.partial.csv",
        )
        tables.append(table)
    return pd.concat(tables, ignore_index=True)


def main(argv: list[str]) -> None:
    """Run the comparison (``--sweep``: the weight sweep) on the named frames."""
    from distributed import Client, LocalCluster

    sweep = "--sweep" in argv
    names = [a for a in argv if a != "--sweep"]
    out = HERE / "results"
    out.mkdir(exist_ok=True)
    frames = yaml.safe_load((HERE / "frames.yaml").read_text())["disp_s1"]
    # Processes: zarr decodes chunks in one thread per process.
    with (
        LocalCluster(
            n_workers=WORKERS, threads_per_worker=2, memory_limit="8GB"
        ) as cluster,
        Client(cluster),
    ):
        run_frames(names or list(frames), frames, sweep, out)


def run_frames(names: list[str], frames: dict, sweep: bool, out: Path) -> None:
    """Write the results (or the sweep) of each named frame."""
    for frame in names:
        profiles = frames[frame]["profiles"]
        if sweep:
            sweep_frame(frame, profiles).to_csv(out / f"{frame}_sweep.csv", index=False)
        else:
            checkpoint = out / f"{frame}.partial.csv"
            run_frame(frame, profiles, checkpoint=checkpoint).to_csv(
                out / f"{frame}.csv", index=False
            )


if __name__ == "__main__":
    main(sys.argv[1:])
