"""Regular time grids, refused gaps and declared breaks (plan task T30)."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from disp_xr.timeseries import (
    apply_segmented,
    gap_mask,
    grid_step_days,
    regular_time_grid,
    split_at_breaks,
)
from disp_xr.timeseries.temporal import break_epochs

# The `pre` segment of the temporal-denoising study (Kiritimati, DISP-S1):
# 18 x 12 d, 14 x 24 d, one 48 d and one 288 d interval.
PRE = pd.DatetimeIndex(
    [
        "2016-11-12", "2016-12-06", "2016-12-30", "2017-01-23", "2017-02-16",
        "2017-02-28", "2017-03-12", "2017-04-05", "2017-04-29", "2017-05-23",
        "2017-06-16", "2017-07-10", "2017-08-03", "2017-08-27", "2018-06-11",
        "2018-06-23", "2018-07-05", "2018-08-22", "2018-09-15", "2018-09-27",
        "2018-10-09", "2018-10-21", "2018-11-02", "2018-11-14", "2018-12-08",
        "2018-12-20", "2019-01-01", "2019-01-13", "2019-01-25", "2019-02-18",
        "2019-03-02", "2019-03-14", "2019-03-26", "2019-04-07", "2019-04-19",
    ]
)  # fmt: skip
HOLE = (pd.Timestamp("2017-08-27"), pd.Timestamp("2018-06-11"))


def test_pre_sampling_pattern():
    gaps = np.diff(PRE).astype("timedelta64[D]").astype(int)
    assert (np.sort(gaps)[[0, -2, -1]] == [12, 48, 288]).all()
    assert np.median(gaps) == 12  # a median rule would pick 15 d


def test_grid_step():
    assert grid_step_days(PRE) == 30  # the study: 30 d for pre-like frames
    post = pd.date_range("2022-01-04", periods=36, freq="12D")
    assert grid_step_days(post) == 15
    assert grid_step_days(pd.date_range("2022-01-04", periods=20, freq="36D")) == 45
    with pytest.raises(ValueError, match="two"):
        grid_step_days(PRE[:1])


def test_regular_grid():
    grid = regular_time_grid(PRE)
    assert grid[0] == PRE[0] and grid[-1] <= PRE[-1]
    assert (np.diff(grid) == np.timedelta64(30, "D")).all()
    assert len(regular_time_grid(PRE, step=15)) == 60  # the study's 60 points
    with pytest.raises(ValueError, match="increasing"):
        regular_time_grid(PRE[::-1])


def test_gap_mask_refuses_only_the_hole():
    grid = regular_time_grid(PRE, step=15)
    refused = gap_mask(PRE, grid)  # default: 4 x 24 d
    inside = (grid > HOLE[0]) & (grid < HOLE[1])
    assert refused.any() and not refused[~inside].any()  # the 48 d gap is bridged
    # Refused where more than 48 d (half of 96) from both ends of the hole.
    far = (grid - HOLE[0] > pd.Timedelta("48D")) & (
        HOLE[1] - grid > pd.Timedelta("48D")
    )
    np.testing.assert_array_equal(refused, far)
    assert gap_mask(PRE, grid, max_gap_days=400).sum() == 0
    assert gap_mask(PRE, grid, max_gap_days=40).sum() > refused.sum()


def test_breaks_land_on_the_first_epoch_after():
    mask = break_epochs(PRE, ["2018-01-01"])  # inside the hole
    assert np.flatnonzero(mask).tolist() == [14]  # 2018-06-11
    segments = split_at_breaks(PRE, ["2018-01-01", "2030-01-01", "2000-01-01"])
    assert [s.tolist()[0] for s in segments] == [0, 14]
    assert sum(len(s) for s in segments) == len(PRE)
    assert len(split_at_breaks(PRE, None)) == 1


def test_apply_segmented_numpy_and_xarray():
    rng = np.random.default_rng(0)
    y = (
        rng.normal(size=(len(PRE), 3))
        + np.where(np.arange(len(PRE)) >= 14, 10.0, 0.0)[:, None]
    )

    def demean(t, seg):  # noqa: ARG001 (the NumPy fn signature)
        return seg - seg.mean(axis=0)

    out = apply_segmented(demean, y, ["2018-01-01"], times=PRE)
    np.testing.assert_allclose(out[:14].mean(axis=0), 0, atol=1e-12)
    np.testing.assert_allclose(out[14:].mean(axis=0), 0, atol=1e-12)  # the step is gone
    da = xr.DataArray(y, dims=("time", "pixel"), coords={"time": PRE})
    out_da = apply_segmented(lambda seg: seg - seg.mean("time"), da, ["2018-01-01"])
    np.testing.assert_allclose(out_da.values, out)
    ds = da.to_dataset(name="displacement")
    out_ds = apply_segmented(lambda seg: seg - seg.mean("time"), ds, ["2018-01-01"])
    np.testing.assert_allclose(out_ds.displacement.values, out)
    # Short segments pass through.
    short = apply_segmented(demean, y, [PRE[-2]], times=PRE, min_len=3)
    np.testing.assert_allclose(short[-2:], y[-2:])
    with pytest.raises(ValueError, match="times="):
        apply_segmented(demean, y, None)
