"""GNSS tie estimator (plan task T33)."""

import numpy as np
import pytest

import disp_xr as dxr
from disp_xr.models import GNSSTie, apply_tie, cross_validate_tie, fit_tie

from .common import check_estimator

pytest.importorskip("geepers.validation")
from fixtures.fake_gnss import stations_from_disp, vertical_los  # noqa: E402

PIXELS = [(4, 4), (4, 58), (58, 4), (58, 58), (30, 30), (15, 45), (45, 18)]


@pytest.mark.parametrize("damping", [None, 1e-2])
def test_tie_passes_shared_checks(damping):
    check_estimator(lambda: GNSSTie(1, damping))


def _ramp(ds, rate_x=2e-6, rate_y=-1e-6):
    """A plane growing with time: long-wavelength DISP error (m)."""
    years = (ds.time - ds.time[0]) / np.timedelta64(1, "D") / 365.25
    x0, y0 = float(ds.x.mean()), float(ds.y.mean())
    plane = rate_x * (ds.x - x0) + rate_y * (ds.y - y0)  # per year
    return (plane * years).transpose("time", "y", "x")


@pytest.fixture
def case(stack_dir, tmp_path):
    truth = dxr.stack_disp(dxr.open_disp(stack_dir)).disp.rereference("median")
    src = stations_from_disp(truth, PIXELS, tmp_path / "cache", noise=0.0005)
    gps = dxr.gnss.load(truth, source=src)
    error = _ramp(truth) * 30  # about a centimeter across the frame
    observed = truth.assign(
        displacement=(truth.displacement + error.astype("f4")).assign_attrs(
            truth.displacement.attrs
        )
    )
    static = vertical_los(truth)
    return truth, observed, gps, static, error


def test_epoch_tie_removes_the_ramp(case):
    truth, observed, gps, static, error = case
    comparison = dxr.residuals.compare(observed, gps, static, min_span_years=0)
    tie = fit_tie(comparison, mode="epoch")
    corrected = apply_tie(observed, tie)
    before = float(
        np.sqrt(np.nanmean((observed.displacement - truth.displacement) ** 2))
    )
    after = float(
        np.sqrt(np.nanmean((corrected.displacement - truth.displacement) ** 2))
    )
    assert before > 0.003
    assert after < 0.25 * before
    assert corrected.displacement.attrs["gnss_tie"].startswith("epoch")
    assert "apply_tie" in corrected.attrs["history"]


def test_velocity_tie(case):
    truth, observed, gps, static, _ = case
    comparison = dxr.residuals.compare(observed, gps, static, min_span_years=0)
    tie = fit_tie(comparison, mode="velocity")
    corrected = apply_tie(observed, tie)

    def err(d):
        return float(
            np.sqrt(np.nanmean((d.displacement - truth.displacement) ** 2))
        )  # noqa: E731

    assert err(corrected) < 0.5 * err(observed)
    assert tie.correction(observed).dims == ("y", "x")


def test_loso_reports_the_improvement(case):
    _, observed, gps, static, _ = case
    comparison = dxr.residuals.compare(observed, gps, static, min_span_years=0)
    cv = cross_validate_tie(comparison, mode="epoch")
    assert len(cv) == len(PIXELS)
    assert cv.attrs["rmse_after"] < 0.5 * cv.attrs["rmse_before"]


def test_excluded_station_does_not_tilt_the_tie(case):
    truth, observed, gps, static, error = case
    comparison = dxr.residuals.compare(observed, gps, static, min_span_years=0)
    # A local anomaly at one station (monument motion, local subsidence in
    # DISP only): it must stay out of the long-wavelength field.
    bad = "S004"
    comparison.series["residual"].loc[{"station": bad}] += 0.02
    raw = apply_tie(observed, fit_tie(comparison, mode="epoch"))
    kept = apply_tie(observed, fit_tie(comparison, mode="epoch", exclude=[bad]))

    def err(d):
        return float(
            np.sqrt(np.nanmean((d.displacement - truth.displacement) ** 2))
        )  # noqa: E731

    assert err(kept) < 0.5 * err(raw)
    from shapely.geometry import box

    lon, lat = float(gps.lon.sel(station=bad)), float(gps.lat.sel(station=bad))
    polygon = box(lon - 1e-4, lat - 1e-4, lon + 1e-4, lat + 1e-4)
    by_area = fit_tie(comparison, mode="epoch", exclude=polygon)
    assert by_area.stations["excluded"].tolist().count(True) == 1
    cv = cross_validate_tie(comparison, mode="epoch", exclude=[bad])
    assert bool(cv.loc[bad, "excluded"])


def test_errors(case):
    _, observed, gps, static, _ = case
    comparison = dxr.residuals.compare(observed, gps, static, min_span_years=0)
    with pytest.raises(ValueError, match="mode"):
        fit_tie(comparison, mode="daily")
