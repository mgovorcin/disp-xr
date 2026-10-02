"""Operation history in attrs["history"]."""

import json

import xarray as xr

import disp_xr as dxr
from disp_xr._history import history, record, recorded


def test_record_appends_and_does_not_mutate():
    ds = xr.Dataset(attrs={"history": "created by another tool"})
    out = record(ds, "op_a", n=1)
    assert ds.attrs["history"] == "created by another tool"
    lines = out.attrs["history"].splitlines()
    assert lines[0] == "created by another tool"
    assert json.loads(lines[1])["op"] == "op_a"


def test_history_skips_foreign_lines_and_keeps_order():
    ds = record(record(xr.Dataset(attrs={"history": "foreign"}), "a"), "b")
    assert [h["op"] for h in history(ds)] == ["a", "b"]
    assert history(ds)[0]["disp_xr"] == dxr.__version__


def test_record_is_deterministic():
    a = record(xr.Dataset(), "op", x=[1, 2])
    b = record(xr.Dataset(), "op", x=[1, 2])
    assert a.attrs["history"] == b.attrs["history"]


def test_unserializable_params_become_strings():
    out = record(xr.Dataset(), "op", data=xr.DataArray([1, 2], dims="t"), obj=object)
    params = history(out)[0]["params"]
    assert params["data"] == "<DataArray dims={'t': 2}>"
    assert "object" in params["obj"]


def test_recorded_decorator_skips_the_data_argument():
    @recorded
    def scale(ds, factor=2):
        return ds * factor

    out = scale(xr.Dataset({"a": ("t", [1.0])}), factor=3)
    assert history(out)[0] == {
        "disp_xr": dxr.__version__,
        "op": "scale",
        "params": {"factor": 3},
    }


def test_open_and_stack_are_recorded(stack_dir):
    ds = dxr.stack_disp(dxr.open_disp(stack_dir), nan_policy="omit")
    ops = history(ds)
    assert [h["op"] for h in ops] == ["open_disp", "stack_disp"]
    assert ops[1]["params"]["nan_policy"] == "omit"
