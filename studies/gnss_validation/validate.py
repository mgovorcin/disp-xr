"""T34 GNSS validation: disp-xr against the geepers workflow on the T29 frames.

For each frame, on the station blocks of the unmasked study stack
(``extract.py``):

1. **geepers baseline**: ``geepers.workflows.main``, the geepers command
   line, writes its station rates, pairwise RMSE and epoch RMSE.
2. **disp-xr**: ``disp_xr.residuals.compare`` on the same stack and
   stations, with the same estimator (least squares), sampling radius and
   no GNSS smoothing.
3. The two are matched by station, by station pair and by epoch, and the
   differences written to ``results/``. The station tables are not
   comparable as they stand: geepers fits GPS with MIDAS over the whole GPS
   record and InSAR only on the dates GPS shares, disp-xr fits both over the
   whole InSAR record.
4. **Double difference**, the like-for-like check: both tools' InSAR and GPS
   series referenced to the GNSS station the stack is referenced to, on the
   epochs both hold, each station's mean removed, and fitted with one
   least-squares rate.
5. **GNSS tie** (no geepers counterpart): the velocity-mode plane of
   ``fit_tie`` against an independent weighted least-squares plane, and the
   leave-one-station-out RMSE before and after.

Run::

    pixi run -e test-py312 python validate.py F08882 F11116 F39362
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import yaml
from extract import DATA, STUDY, store_path

import disp_xr as dxr
from disp_xr.models import cross_validate_tie, fit_tie

sys.path.insert(0, str(STUDY))
from gnss_los import los_layers  # noqa: E402

HERE = Path(__file__).parent
BUFFER_M = 100.0  # sampling radius, both tools
# 3-band east/north/up GeoTIFF on the frame grid, as the geepers CLI takes it.
LOS_ENU = {
    "F08882": (
        "/u/aurora-r0/govorcin/01_OPERA/DISP/data/static/F08882/"
        "OPERA_L3_DISP-S1-STATIC_F08882_20140403_S1A_v1.0_line_of_sight_enu.tif"
    ),
    "F11116": (
        "/u/aurora-r0/govorcin/01_OPERA/DISP/data/static/F11116/"
        "OPERA_L3_DISP-S1-STATIC_F11116_20140403_S1A_v1.0_line_of_sight_enu.tif"
    ),
    "F39362": (
        "/u/aurora-r0/govorcin/01_OPERA/STAKEHOLDER/AUSTRALIA/canberra/los_enu.tif"
    ),
}


def run_geepers(frame: str, reference: str | None) -> Path:
    """Run the geepers workflow on the station blocks; return its output folder."""
    from geepers.workflows import main

    out = DATA / frame / "unmasked" / "geepers"
    if (out / "pairwise_rmse.csv").exists():
        return out
    main(
        los_enu_file=LOS_ENU[frame],
        timeseries_stack=store_path(frame),
        output_dir=out,
        gps_source="unr",
        stack_data_var="displacement",
        reference_station=reference,
        insar_buffer_meters=BUFFER_M,
        gps_time_window=0,
        structure_function=True,
    )
    return out


def run_disp_xr(frame: str, entry: dict):
    """Return the disp-xr comparison on the station blocks."""
    ds = xr.open_zarr(store_path(frame))
    gps = dxr.gnss.load(ds)
    static = los_layers(ds, Path(entry["static"]))
    return dxr.residuals.compare(
        ds, gps, static, buffer_m=BUFFER_M, velocity_method="lsq"
    )


def station_table(geepers_dir: Path, result) -> pd.DataFrame:
    """Velocities of both tools per station (mm/yr) and their differences."""
    theirs = pd.read_csv(geepers_dir / "station_summary.csv")
    theirs = theirs.set_index(theirs.columns[0])
    ours = 1e3 * result.stations[["insar_velocity", "gps_velocity"]]
    table = ours.join(theirs, how="inner", rsuffix="_geepers")
    return table


def pair_table(geepers_dir: Path, result) -> pd.DataFrame:
    """Pairwise RMSE of both tools, matched by station pair."""
    theirs = pd.read_csv(geepers_dir / "pairwise_rmse.csv")
    # Both come from geepers' ``pairwise_differential_rmse``, same columns.
    return result.pairs.merge(
        theirs, on=["station1", "station2"], suffixes=("", "_geepers")
    )


def epoch_table(geepers_dir: Path, result) -> pd.DataFrame:
    """Epoch RMSE of both tools, matched by the acquisition's UTC day.

    geepers labels an epoch by the date of the daily GPS solution it is
    attached to, disp-xr by its acquisition time.
    """
    theirs = pd.read_csv(geepers_dir / "epoch_rmse.csv", parse_dates=["date"])
    theirs["date"] = theirs["date"].astype("datetime64[ns]")
    ours = result.epochs.reset_index()
    ours["date"] = pd.to_datetime(ours["date"]).astype("datetime64[ns]").dt.floor("D")
    return ours.merge(theirs, on="date", suffixes=("", "_geepers"))


def _series(geepers_dir: Path, result) -> dict[str, pd.DataFrame]:
    """Each tool's (station, date) InSAR and GPS series, in meters.

    Both are keyed by the acquisition's UTC day, which geepers uses for
    daily GPS solutions.
    """
    theirs = pd.read_csv(geepers_dir / "combined_data.csv", parse_dates=["date"])
    theirs["date"] = theirs["date"].astype("datetime64[ns]")
    theirs = theirs.pivot_table(index=["id", "date"], columns="measurement")["value"]
    theirs = theirs.rename(columns={"los_insar": "insar", "los_gps": "gps"})
    ours = result.series[["insar", "gnss"]].to_dataframe()[["insar", "gnss"]]
    ours = ours.rename(columns={"gnss": "gps"})
    days = pd.DatetimeIndex(ours.index.get_level_values("time")).floor("D")
    ours.index = pd.MultiIndex.from_arrays(
        [ours.index.get_level_values("station"), days], names=["id", "date"]
    )
    return {"geepers": theirs[["insar", "gps"]], "disp_xr": ours}


def double_difference(series: pd.DataFrame, reference: str) -> pd.DataFrame:
    """Return InSAR and GPS relative to ``reference``.

    Each station keeps only the dates where it, the reference, InSAR and GPS
    all have values, so both quantities share one spatial reference.
    """
    ref = series.xs(reference, level="id")
    return series.sub(ref, level="date").dropna()


def _lsq_rate(group: pd.DataFrame) -> pd.Series:
    """Least-squares line rate (mm/yr) of each column, with its span and count."""
    t = group.index.get_level_values("date")
    years = np.asarray((t - t.min()).days / 365.25)
    out = {"n": len(group), "span_yr": years.max() if len(group) else np.nan}
    for name in ("insar", "gps"):
        out[name] = (
            1e3 * np.polyfit(years, group[name].to_numpy(), 1)[0]
            if len(group) >= 2
            else np.nan
        )
    return pd.Series(out)


MIN_EPOCHS = 10
MIN_SPAN_YR = 2.0


def dd_tables(
    geepers_dir: Path, result, reference: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Double-differenced velocities and series of both tools, matched.

    Both tools' series are referenced to the same GNSS station, InSAR and
    GPS alike, and fitted with the same estimator on the epochs both tools
    hold for the station, so a difference is one of sampling alone.
    """
    tools = {
        k: double_difference(v, reference)
        for k, v in _series(geepers_dir, result).items()
    }
    common = tools["geepers"].index.intersection(tools["disp_xr"].index)
    # Remove each station's mean over the shared epochs rather than zeroing
    # at the first, so that one mismatched epoch cannot offset a whole series
    # (geepers before 6ff4264 matched the first acquisition to the next day).
    tools = {
        k: v.loc[common] - v.loc[common].groupby(level="id").transform("mean")
        for k, v in tools.items()
    }
    series = (
        tools["disp_xr"]
        .loc[common]
        .join(tools["geepers"].loc[common], rsuffix="_geepers")
    )
    rates = {
        k: v.loc[common].groupby(level="id").apply(_lsq_rate) for k, v in tools.items()
    }
    velocities = (
        rates["disp_xr"]
        .join(rates["geepers"][["insar", "gps"]], rsuffix="_geepers")
        .drop(index=reference, errors="ignore")
    )
    velocities["usable"] = (velocities["n"] >= MIN_EPOCHS) & (
        velocities["span_yr"] >= MIN_SPAN_YR
    )
    return velocities, series


def tie_check(result) -> dict:
    """Velocity-mode plane of ``fit_tie`` against an independent weighted fit."""
    tie = fit_tie(result, mode="velocity", degree=1)
    used = tie.stations[~tie.stations.get("excluded", False)]
    stations = result.stations.loc[used.index].dropna(
        subset=["velocity_residual", "velocity_sigma"]
    )
    x = stations.index.map(lambda s: float(result.series["x"].sel(station=s)))
    y = stations.index.map(lambda s: float(result.series["y"].sel(station=s)))
    w = 1.0 / stations["velocity_sigma"].to_numpy() ** 2
    design = np.column_stack([np.ones(len(x)), np.asarray(x), np.asarray(y)])
    sw = np.sqrt(w)
    coef, *_ = np.linalg.lstsq(
        design * sw[:, None], stations["velocity_residual"].to_numpy() * sw, rcond=None
    )
    ours = np.asarray(tie.estimators.predict((np.asarray(x), np.asarray(y))))
    theirs = design @ coef
    scores = cross_validate_tie(result, mode="velocity", degree=1)
    return {
        "tie_stations": int(len(stations)),
        "tie_max_diff_mm_yr": float(1e3 * np.max(np.abs(ours - theirs))),
        "loso_rmse_before_mm_yr": float(1e3 * scores.attrs["rmse_before"]),
        "loso_rmse_after_mm_yr": float(1e3 * scores.attrs["rmse_after"]),
    }


def validate(frame: str, entry: dict) -> dict:
    """Run both tools on one frame and write the matched tables."""
    # The stack is already referenced to a box around this GNSS station.
    reference = (
        json.loads(xr.open_zarr(store_path(frame)).attrs.get("reference", "{}"))
        .get("center", "")
        .removeprefix("GNSS station ")
    )
    geepers_dir = run_geepers(frame, reference or None)
    result = run_disp_xr(frame, entry)
    out = HERE / "results"
    out.mkdir(exist_ok=True)
    stations = station_table(geepers_dir, result)
    stations.to_csv(out / f"{frame}_stations.csv")
    pairs = pair_table(geepers_dir, result)
    pairs.to_csv(out / f"{frame}_pairs.csv", index=False)
    epochs = epoch_table(geepers_dir, result)
    epochs.to_csv(out / f"{frame}_epochs.csv", index=False)
    summary = {"frame": frame, "stations_matched": int(len(stations))}
    summary["pairs_matched"] = int(len(pairs))
    summary["epochs_matched"] = int(len(epochs))
    velocities, series = dd_tables(geepers_dir, result, reference)
    velocities.to_csv(out / f"{frame}_dd_velocities.csv")
    series.to_csv(out / f"{frame}_dd_series.csv")
    usable = velocities[velocities["usable"]]
    summary["reference"] = reference
    summary["dd_stations_usable"] = int(len(usable))
    for name in ("insar", "gps"):
        diff = (usable[name] - usable[f"{name}_geepers"]).abs()
        summary[f"dd_{name}_vel_max_diff_mm_yr"] = float(diff.max())
        diff = 1e3 * (series[name] - series[f"{name}_geepers"]).abs()
        summary[f"dd_{name}_series_p95_diff_mm"] = float(diff.quantile(0.95))
    summary |= tie_check(result)
    print(json.dumps(summary), flush=True)
    return summary


def main(frames: list[str]) -> None:
    """Validate each frame and write ``results/summary.json``."""
    entries = yaml.safe_load((STUDY / "frames.yaml").read_text())["disp_s1"]
    warnings.simplefilter("ignore", FutureWarning)
    summaries = [validate(frame, entries[frame]) for frame in frames]
    (HERE / "results" / "summary.json").write_text(json.dumps(summaries, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:] or ["F08882", "F11116", "F39362"])
