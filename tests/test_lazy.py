"""Core operations stay lazy (plan task T37.3).

Each operation is built on a chunked stack under a dask scheduler that
refuses to compute: a hidden compute (``.values``, ``np.asarray`` or a numpy
function that falls back to it) fails the test instead of silently loading a
frame into memory, as ``np.round`` once did in ``quality_weights``.
"""

import dask
import numpy as np
import pytest

import disp_xr as dxr


class HiddenCompute(AssertionError):
    pass


def _forbid(*args, **kwargs):  # noqa: ARG001 (dask scheduler signature)
    raise HiddenCompute("computed while building a lazy result")


@pytest.fixture(scope="module")
def stack(tmp_path_factory):
    from disp_xr.tutorial import make_disp_stack

    folder = tmp_path_factory.mktemp("lazy") / "granules"
    make_disp_stack(folder, "three_ministacks", shape=(64, 64))
    return dxr.stack_disp(dxr.open_disp(folder, corrections=True)).chunk(
        {"time": 1, "y": 32, "x": 32}
    )


# Computing on purpose, so not listed: to_points (its point count depends on
# which cells hold data; it reads the coarse grid only), quadtree (it computes
# the split scores to choose the leaves; leaf values stay lazy) and
# rereference with check_point (it reads the reference pixel's series).
OPERATIONS = {
    "mask": lambda ds: dxr.mask(ds, "recommended"),
    "valid_mask": lambda ds: dxr.valid_mask(ds, "strict"),
    "rereference median": lambda ds: dxr.rereference(ds, "median"),
    "rereference border": lambda ds: dxr.rereference(ds, "border"),
    "rereference point": lambda ds: dxr.rereference(
        ds, "point", row=40, col=40, check_point=False
    ),
    "to_units": lambda ds: dxr.to_units(ds, "mm"),
    "quality_summary": lambda ds: dxr.quality_summary(ds),
    "quality_weights": lambda ds: dxr.quality_weights(ds),
    "downsample": lambda ds: dxr.downsample(ds, factor=8, layers=["displacement"]),
    "fit_trajectory": lambda ds: dxr.fit_trajectory(ds),
    "fit_velocity ols": lambda ds: dxr.fit_velocity(ds, method="ols"),
    "fit_velocity whittle": lambda ds: dxr.fit_velocity(ds, method="whittle"),
    "fit_velocity midas": lambda ds: dxr.fit_velocity(ds, method="midas"),
    "downsample_hex": lambda ds: dxr.downsample_hex(
        ds, spacing=240.0, layers=["displacement"]
    ),
    "rechunk": lambda ds: dxr.rechunk(ds, "timeseries"),
    "deramp": lambda ds: ds.disp.deramp(),
    "denoise lowess": lambda ds: dxr.denoise(ds, method="lowess"),
    "denoise kalman": lambda ds: dxr.denoise(ds, method="kalman"),
}


@pytest.mark.parametrize("name", list(OPERATIONS))
def test_operation_is_lazy(stack, name):
    with dask.config.set(scheduler=_forbid):
        out = OPERATIONS[name](stack)
    arrays = out.data_vars.values() if hasattr(out, "data_vars") else [out]
    assert any(a.chunks is not None for a in arrays), "result is not a dask array"


def test_open_and_stack_are_lazy(tmp_path):
    from disp_xr.tutorial import make_disp_stack

    folder = tmp_path / "granules"
    make_disp_stack(folder, "two_ministacks", shape=(64, 64))
    # open_disp computes the small reference_time eagerly with the synchronous
    # scheduler (it groups the ministacks); nothing else may compute.
    ds = dxr.open_disp(folder, corrections=True)
    with dask.config.set(scheduler=_forbid):
        corrected = dxr.correct(ds, ["ionospheric_delay", "solid_earth_tide"])
        out = dxr.stack_disp(corrected)
    assert out.displacement.chunks is not None
    assert np.isfinite(out.displacement.isel(time=-1).values).any()
