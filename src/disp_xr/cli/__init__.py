"""Command-line interface for disp-xr (commands are added in plan task T15)."""

import click

from disp_xr.log.logging import setup_logging


@click.group(name="disp_xr")
@click.version_option(package_name="disp_xr")
@click.option("--debug", is_flag=True, help="Log debug messages.")
@click.pass_context
def cli_app(ctx: click.Context, debug: bool) -> None:
    """Xarray tools for OPERA DISP products."""
    ctx.ensure_object(dict)
    ctx.obj["debug"] = debug
    setup_logging(debug=debug)
