"""The top-level ``disp_xr`` namespace."""

from importlib.metadata import version

import disp_xr as dxr


def test_version_matches_installed_metadata():
    assert dxr.__version__ == version("disp_xr")


def test_public_names_resolve():
    for name in dxr.__all__:
        assert hasattr(dxr, name)


def test_star_import_exposes_only_public_names():
    namespace: dict = {}
    exec("from disp_xr import *", namespace)
    assert set(namespace) - {"__builtins__"} == set(dxr.__all__)
