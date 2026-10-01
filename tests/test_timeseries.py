"""Import contract of ``disp_xr.timeseries`` (JAX is optional)."""

import importlib.util

import pytest

import disp_xr.timeseries as ts

HAS_JAX = importlib.util.find_spec("jax") is not None


def test_public_names_resolve():
    for name in ts.__all__:
        if name == "jax_solve_linear_system" and not HAS_JAX:
            continue
        assert callable(getattr(ts, name))


@pytest.mark.skipif(HAS_JAX, reason="checks the message shown when JAX is missing")
def test_jax_solver_without_jax_names_the_extra():
    with pytest.raises(ModuleNotFoundError, match=r"disp_xr\[jax\]"):
        ts.jax_solve_linear_system  # noqa: B018


def test_unknown_attribute_raises():
    with pytest.raises(AttributeError, match="has no attribute 'nope'"):
        ts.nope  # noqa: B018
