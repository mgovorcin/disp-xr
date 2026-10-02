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
