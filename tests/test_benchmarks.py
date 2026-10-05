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
