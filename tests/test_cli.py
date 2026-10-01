"""The ``disp_xr`` command group (subcommands come in plan task T15)."""

import shutil
import subprocess
from importlib.metadata import version

from click.testing import CliRunner

from disp_xr.cli import cli_app


def test_help_exits_zero():
    result = CliRunner().invoke(cli_app, ["--help"])
    assert result.exit_code == 0
    assert "Xarray tools for OPERA DISP products" in result.output


def test_version_matches_package():
    result = CliRunner().invoke(cli_app, ["--version"])
    assert result.exit_code == 0
    assert version("disp_xr") in result.output


def test_console_script_resolves():
    exe = shutil.which("disp_xr")
    assert exe is not None
    out = subprocess.run([exe, "--help"], capture_output=True, text=True, check=True)
    assert "Usage: disp_xr" in out.stdout
