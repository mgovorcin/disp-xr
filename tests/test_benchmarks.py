"""The benchmark suite runs (plan task T37.2); timings are not checked."""

import importlib.util
from pathlib import Path

SUITE = Path(__file__).parents[1] / "benchmarks" / "suite.py"


def _suite():
    spec = importlib.util.spec_from_file_location("benchmark_suite", SUITE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_suite_runs_and_records_the_environment(tmp_path):
    suite = _suite()
    out = tmp_path / "rows.csv"
    table = suite.run("smoke", trials=2, out=out)
    assert out.exists()
    assert table["operation"].nunique() == 10
    assert (table.groupby("operation")["trial"].nunique() == 2).all()
    for column in ("disp_xr_version", "zarr_version", "host", "page_touch_s"):
        assert column in table
    summary = suite.summary(table)
    assert list(summary.columns) == ["median_s", "min_s", "max_s", "spread", "trials"]


def test_storage_benchmark_runs(tmp_path):
    import disp_xr as dxr
    from disp_xr.tutorial import make_disp_stack

    spec = importlib.util.spec_from_file_location(
        "storage_benchmark", SUITE.with_name("storage.py")
    )
    storage = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(storage)
    make_disp_stack(tmp_path / "g", "two_ministacks", shape=(64, 64))
    stack = dxr.chain(dxr.open_disp(tmp_path / "g"))
    dxr.to_zarr(stack, tmp_path / "stack.zarr")
    table = storage.run(tmp_path / "stack.zarr", epochs=3, size=32, trials=1)
    configurations = len(storage.CODECS) * len(storage.LAYOUTS)
    assert len(table) == 3 * configurations  # two layers and the lossy one
    assert (table["ratio"] > 0).all()
