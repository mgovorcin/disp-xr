"""Runtime settings: defaults, environment overrides, scoped changes."""

import pytest

from disp_xr.config import DEFAULTS, config


@pytest.fixture(autouse=True)
def _fresh_config():
    config.refresh()
    yield
    config.refresh()


def test_defaults():
    assert config.get("chunks.target_mb") == DEFAULTS["chunks"]["target_mb"]
    assert config.get("max_load_bytes") == 2 * 1024**3
    assert config.get("threads") is None
    assert config.get("chunks.read_block") == 1024


def test_environment_override(monkeypatch):
    monkeypatch.setenv("DISP_XR_CHUNKS__TARGET_MB", "50")
    monkeypatch.setenv("DISP_XR_THREADS", "4")
    config.refresh()
    assert config.get("chunks.target_mb") == 50
    assert config.get("threads") == 4


def test_set_is_scoped():
    with config.set({"max_load_bytes": 10}):
        assert config.get("max_load_bytes") == 10
    assert config.get("max_load_bytes") == 2 * 1024**3


def test_top_level_alias():
    import disp_xr as dxr

    assert dxr.config is config


def _numba_threads(env):
    import os
    import subprocess
    import sys

    code = (
        "import disp_xr, os, numba; "
        "print(os.environ.get('NUMBA_NUM_THREADS'), numba.config.NUMBA_NUM_THREADS)"
    )
    full = {k: v for k, v in os.environ.items() if k != "NUMBA_NUM_THREADS"} | env
    out = subprocess.run(
        [sys.executable, "-c", code], env=full, capture_output=True, text=True
    )
    return out.stdout.split()


def test_numba_threads_are_capped_on_import():
    assert _numba_threads({}) == ["1", "1"]
    # A variable set by the user wins.
    assert _numba_threads({"NUMBA_NUM_THREADS": "3"}) == ["3", "3"]
    assert _numba_threads({"DISP_XR_NUMBA_THREADS": "2"}) == ["2", "2"]
