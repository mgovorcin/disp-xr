"""Command-line interface for disp-xr.

Each command wraps library functions and holds no processing logic of its
own: options map to function arguments, and outputs go through the writers in
:mod:`disp_xr.export`. Errors a user can fix (a missing file, mixed frames,
an unknown layer) are reported as one line; ``--debug`` shows the traceback.
"""

from __future__ import annotations

import functools
import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Optional, TypeVar

import click
import pandas as pd
import xarray as xr

from disp_xr.log.logging import setup_logging

F = TypeVar("F", bound=Callable[..., Any])

FORMATS = ("cog", "geotiff", "zarr", "geozarr", "netcdf", "mintpy")
MASKS = ("recommended", "strict", "none")
REFERENCES = ("auto", "point", "median", "border", "high_coherence", "none")


def _user_errors(func: F) -> F:
    """Report expected errors as a short message unless ``--debug`` is set."""

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        except (ValueError, FileNotFoundError, FileExistsError, KeyError) as e:
            ctx = click.get_current_context()
            if ctx.find_root().obj.get("debug"):
                raise
            raise click.ClickException(f"{type(e).__name__}: {e}") from e

    return wrapper  # type: ignore[return-value]


@click.group(name="disp_xr")
@click.version_option(package_name="disp_xr")
@click.option("--debug", is_flag=True, help="Log debug messages and show tracebacks.")
@click.pass_context
def cli_app(ctx: click.Context, debug: bool) -> None:
    """Xarray tools for OPERA DISP products."""
    ctx.ensure_object(dict)
    ctx.obj["debug"] = debug
    setup_logging(debug=debug)


_paths = click.argument(
    "paths", nargs=-1, required=True, type=click.Path(exists=True, path_type=Path)
)
_frame = click.option(
    "--frame-id", type=int, help="Frame to keep when PATHS hold several."
)
_min_version = click.option("--min-version", help="Lowest product version to keep.")


def _granules(paths: Sequence[Path]) -> Any:
    return paths[0] if len(paths) == 1 else list(paths)


def _is_zarr(path: Path) -> bool:
    return path.is_dir() and (
        (path / "zarr.json").exists() or (path / ".zgroup").exists()
    )


def _open_input(
    paths: Sequence[Path], frame_id: Optional[int], min_version: Optional[str]
) -> xr.Dataset:
    """Open a store written by ``disp_xr stack``, or chain DISP granules."""
    import disp_xr as dxr

    if len(paths) == 1 and _is_zarr(paths[0]):
        return xr.open_zarr(paths[0], mask_and_scale=False, decode_coords="all")
    ds = dxr.open_disp(_granules(paths), frame_id=frame_id, min_version=min_version)
    return dxr.stack_disp(ds)


def _write(ds: xr.Dataset, out: Path, preset: str, overwrite: bool) -> None:
    """Write ``ds`` as Zarr (``.zarr``) or NetCDF (``.nc``), chosen by suffix."""
    import disp_xr as dxr

    if out.suffix not in (".zarr", ".nc"):
        raise ValueError(f"Output must end in .zarr or .nc, not {out.name!r}")
    if out.exists() and not overwrite:
        raise FileExistsError(f"{out} exists; pass --overwrite to replace it")
    if out.suffix == ".zarr":
        dxr.to_zarr(ds, out, preset=preset, mode="w")  # type: ignore[arg-type]
    else:
        dxr.to_netcdf(ds, out, preset=preset)  # type: ignore[arg-type]


# --- info ----------------------------------------------------------------------


def _summary(df: pd.DataFrame) -> dict[str, Any]:
    dropped = df.attrs.get("dropped")
    return {
        "product": sorted(df["product"].unique().tolist()),
        "frames": sorted(int(f) for f in df["frame_id"].unique()),
        "granules": len(df),
        "ministacks": int(df["reference_time"].nunique()),
        "reference_dates": sorted(
            f"{t:%Y-%m-%d}" for t in df["reference_time"].unique()
        ),
        "first_date": f"{df['secondary_time'].min():%Y-%m-%d}",
        "last_date": f"{df['secondary_time'].max():%Y-%m-%d}",
        "versions": sorted(df["version"].unique().tolist()),
        "dropped_duplicates": (
            [Path(str(p)).name for p in dropped["path"]] if dropped is not None else []
        ),
    }


@cli_app.command()
@_paths
@_frame
@_min_version
@click.option("--json", "as_json", is_flag=True, help="Print the summary as JSON.")
@click.option(
    "--table",
    type=click.Path(dir_okay=False, path_type=Path),
    help="Also save the granule table as CSV.",
)
@_user_errors
def info(
    paths: tuple[Path, ...],
    frame_id: Optional[int],
    min_version: Optional[str],
    as_json: bool,
    table: Optional[Path],
) -> None:
    """Summarize the DISP granules in PATHS: frame, dates, versions, duplicates."""
    import disp_xr as dxr

    df = dxr.find_products(_granules(paths), frame_id=frame_id, min_version=min_version)
    summary = _summary(df)
    if table is not None:
        df.to_csv(table, index=False)
    if as_json:
        click.echo(json.dumps(summary, indent=2))
        return
    click.echo(f"Product:          {', '.join(summary['product'])}")
    click.echo(f"Frame:            {', '.join(map(str, summary['frames']))}")
    click.echo(
        f"Granules:         {summary['granules']} in {summary['ministacks']} ministacks"
    )
    click.echo(f"Dates:            {summary['first_date']} to {summary['last_date']}")
    click.echo(f"Reference dates:  {', '.join(summary['reference_dates'])}")
    click.echo(f"Versions:         {', '.join(summary['versions'])}")
    dropped = summary["dropped_duplicates"]
    click.echo(f"Dropped duplicates: {len(dropped)}")
    for name in dropped:
        click.echo(f"  {name}")


# --- stack ---------------------------------------------------------------------


@cli_app.command()
@_paths
@click.option(
    "--out",
    required=True,
    type=click.Path(path_type=Path),
    help="Output store: .zarr or .nc.",
)
@click.option(
    "--corrections",
    default="product",
    show_default=True,
    help="Comma-separated correction layers to subtract (ionospheric_delay, "
    "solid_earth_tide); 'product' for those the product has not applied; "
    "'none' to skip.",
)
@click.option(
    "--mask", type=click.Choice(MASKS), default="recommended", show_default=True
)
@click.option(
    "--reference",
    type=click.Choice(REFERENCES),
    default="auto",
    show_default=True,
    help="Spatial reference method.",
)
@click.option("--row", type=int, help="Reference pixel row (--reference point).")
@click.option("--col", type=int, help="Reference pixel column (--reference point).")
@click.option("--lon", type=float, help="Reference longitude (--reference point).")
@click.option("--lat", type=float, help="Reference latitude (--reference point).")
@click.option("--start", help="First date to keep (YYYY-MM-DD).")
@click.option("--end", help="Last date to keep (YYYY-MM-DD).")
@click.option(
    "--preset",
    type=click.Choice(["timeseries", "maps"]),
    default="timeseries",
    show_default=True,
    help="Chunk layout of the output.",
)
@click.option("--overwrite", is_flag=True, help="Replace an existing output.")
@_frame
@_min_version
@_user_errors
def stack(
    paths: tuple[Path, ...],
    out: Path,
    corrections: str,
    mask: str,
    reference: str,
    row: Optional[int],
    col: Optional[int],
    lon: Optional[float],
    lat: Optional[float],
    start: Optional[str],
    end: Optional[str],
    preset: str,
    overwrite: bool,
    frame_id: Optional[int],
    min_version: Optional[str],
) -> None:
    """Build one chained, referenced displacement stack from DISP granules.

    Steps, in the fixed order of the PRD: subtract corrections and apply the
    quality mask (per granule), chain the ministacks, re-reference, write.
    """
    import disp_xr as dxr
    from disp_xr.product import PRODUCTS

    which = [c.strip() for c in corrections.split(",") if c.strip()]
    ds = dxr.open_disp(
        _granules(paths),
        corrections=which != ["none"],
        frame_id=frame_id,
        min_version=min_version,
    )
    if which == ["product"]:
        spec = PRODUCTS[ds.attrs["product"]]
        which = [
            c
            for c in spec.applicable_corrections
            if c not in spec.pre_applied_corrections
        ]
    if which != ["none"]:
        ds = dxr.correct(ds, which)
    if mask != "none":
        ds = dxr.mask(ds, mask)
    ds = dxr.stack_disp(ds, start=start, end=end)
    if reference != "none":
        ds = dxr.rereference(ds, reference, row=row, col=col, lon=lon, lat=lat)
    _write(ds, out, preset, overwrite)
    click.echo(f"Wrote {out} ({ds.sizes['time']} epochs)")


# --- export --------------------------------------------------------------------


@cli_app.command()
@_paths
@click.option(
    "--format", "fmt", type=click.Choice(FORMATS), required=True, help="Output format."
)
@click.option(
    "--out",
    required=True,
    type=click.Path(path_type=Path),
    help="Output file, store or directory (GeoTIFF per epoch, MintPy).",
)
@click.option(
    "--layer",
    default="displacement",
    show_default=True,
    help="Layer to write (cog, geotiff).",
)
@click.option("--epoch", help="Date of the epoch to write (cog); default: last.")
@click.option("--per-epoch", is_flag=True, help="One COG per epoch, next to --out.")
@click.option(
    "--preset",
    type=click.Choice(["timeseries", "maps"]),
    default="timeseries",
    show_default=True,
    help="Chunk layout (zarr, netcdf).",
)
@click.option(
    "--static-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Local static layers: MintPy geometry (mintpy), LOS attributes (geozarr).",
)
@click.option(
    "--profile",
    type=click.Choice(["bowser", "standard"]),
    default="bowser",
    show_default=True,
    help="GeoZarr flavor: ready for bowser, or the plain pyramid (geozarr).",
)
@click.option(
    "--factors",
    default="",
    help="Comma-separated analysis levels, e.g. 33,67,333 (geozarr).",
)
@click.option("--flat", is_flag=True, help="One group, no pyramid (geozarr).")
@click.option(
    "--look-side",
    type=click.Choice(["right", "left"]),
    default="right",
    show_default=True,
    help="Radar look direction for the LOS heading (geozarr).",
)
@click.option("--overwrite", is_flag=True, help="Replace an existing output.")
@_frame
@_min_version
@_user_errors
def export(
    paths: tuple[Path, ...],
    fmt: str,
    out: Path,
    layer: str,
    epoch: Optional[str],
    per_epoch: bool,
    preset: str,
    static_dir: Optional[Path],
    profile: str,
    factors: str,
    flat: bool,
    look_side: str,
    overwrite: bool,
    frame_id: Optional[int],
    min_version: Optional[str],
) -> None:
    """Write a stack to Cloud Optimized GeoTIFF, Zarr, GeoZarr, NetCDF or MintPy.

    ``--format geozarr`` writes a multiscale GeoZarr store; with the default
    ``--profile bowser``, ``bowser run --stack-file OUT.zarr`` opens it.

    PATHS is a store written by ``disp_xr stack``, or DISP granules, which are
    chained without corrections, masking or re-referencing (use ``stack``
    for those).
    """
    import disp_xr as dxr

    ds = _open_input(paths, frame_id, min_version)
    if fmt in ("cog", "geotiff"):
        if layer not in ds:
            raise ValueError(f"No layer {layer!r}; have {sorted(ds.data_vars)}")
        sub = ds[[layer]]
        if "time" in sub[layer].dims and not per_epoch:
            sub = sub.sel(time=epoch, method="nearest") if epoch else sub.isel(time=-1)
        written = dxr.to_geotiff(sub, out, layer=layer, per_epoch=per_epoch)
        click.echo(f"Wrote {len(written)} Cloud Optimized GeoTIFF file(s) to {out}")
    elif fmt == "mintpy":
        static = (
            dxr.open_static(ds, static_dir=static_dir, download=False)
            if static_dir is not None
            else None
        )
        files = dxr.to_mintpy(ds, out, static=static, overwrite=overwrite)
        click.echo(f"Wrote {', '.join(str(p) for p in files.values())}")
    elif fmt == "geozarr":
        if out.suffix != ".zarr":
            raise ValueError(f"--format geozarr writes a .zarr store, not {out.name!r}")
        if out.exists() and not overwrite:
            raise FileExistsError(f"{out} exists; pass --overwrite to replace it")
        static = (
            dxr.open_static(ds, static_dir=static_dir, download=False)
            if static_dir is not None
            else None
        )
        dxr.to_geozarr(
            ds,
            out,
            profile=profile,  # type: ignore[arg-type]
            pyramid=not flat,
            factors=[int(f) for f in factors.split(",") if f.strip()],
            static=static,
            look_side=look_side,  # type: ignore[arg-type]
            mode="w",
        )
        click.echo(f"Wrote {out}")
    else:
        suffix = ".zarr" if fmt == "zarr" else ".nc"
        if out.suffix != suffix:
            raise ValueError(
                f"--format {fmt} writes a {suffix} output, not {out.name!r}"
            )
        _write(ds, out, preset, overwrite)
        click.echo(f"Wrote {out}")


# --- geometry ------------------------------------------------------------------


@cli_app.command()
@_paths
@click.option(
    "--out",
    required=True,
    type=click.Path(path_type=Path),
    help="Output store: .zarr or .nc.",
)
@click.option(
    "--static-dir",
    type=click.Path(file_okay=False, path_type=Path),
    help="Directory of static-layer files (downloads go here).",
)
@click.option(
    "--download/--no-download",
    default=True,
    show_default=True,
    help="Download missing static files (needs the aws extra and Earthdata login).",
)
@click.option("--overwrite", is_flag=True, help="Replace an existing output.")
@_frame
@_user_errors
def geometry(
    paths: tuple[Path, ...],
    out: Path,
    static_dir: Optional[Path],
    download: bool,
    overwrite: bool,
    frame_id: Optional[int],
) -> None:
    """Write the static layers (LOS geometry, DEM, masks) of the frame in PATHS."""
    import disp_xr as dxr

    like = dxr.open_disp(_granules(paths), layers=["displacement"], frame_id=frame_id)
    static = dxr.open_static(like, static_dir=static_dir, download=download)
    _write(static, out, "maps", overwrite)
    click.echo(f"Wrote {out} ({', '.join(map(str, static.data_vars))})")


# --- search --------------------------------------------------------------------


@cli_app.command()
@click.option("--frame-id", type=int, required=True, help="DISP frame number.")
@click.option("--start", help="First acquisition date (YYYY-MM-DD).")
@click.option("--end", help="Last acquisition date (YYYY-MM-DD).")
@click.option(
    "--product",
    type=click.Choice(["DISP-S1", "DISP-NISAR"]),
    default="DISP-S1",
    show_default=True,
)
@click.option(
    "--version",
    "product_version",
    default="1.0",
    show_default=True,
    help="Product version; 'any' for all.",
)
@click.option(
    "--url-type",
    type=click.Choice(["https", "s3"]),
    default="https",
    show_default=True,
)
@click.option(
    "--out",
    type=click.Path(dir_okay=False, path_type=Path),
    help="Save the granule table as CSV instead of printing it.",
)
@_user_errors
def search(
    frame_id: int,
    start: Optional[str],
    end: Optional[str],
    product: str,
    product_version: str,
    url_type: str,
    out: Optional[Path],
) -> None:
    """Find DISP granules of one frame in NASA CMR."""
    from disp_xr import search as search_module
    from disp_xr.product import PRODUCTS

    df = search_module.search(
        frame_id=frame_id,
        product_version=None if product_version == "any" else product_version,
        start_datetime=pd.Timestamp(start).to_pydatetime() if start else None,
        end_datetime=pd.Timestamp(end).to_pydatetime() if end else None,
        url_type=search_module.UrlType(url_type),
        short_name=PRODUCTS[product].cmr_short_name,
    )
    if out is not None:
        df.to_csv(out, index=False)
        click.echo(f"Wrote {len(df)} granules to {out}")
    else:
        click.echo(df.to_string(index=False))


# --- virtual -------------------------------------------------------------------


@cli_app.command()
@_paths
@click.option(
    "--repo",
    required=True,
    type=click.Path(file_okay=False, path_type=Path),
    help="Icechunk repository directory (created if missing).",
)
@click.option("--branch", default="main", show_default=True)
@click.option("--workers", type=int, help="Processes that parse granules.")
@click.option(
    "--validate/--no-validate",
    default=True,
    show_default=True,
    help="Compare the store with the granules after the build.",
)
@_frame
@_user_errors
def virtual(
    paths: tuple[Path, ...],
    repo: Path,
    branch: str,
    workers: Optional[int],
    validate: bool,
    frame_id: Optional[int],
) -> None:
    """Build a virtual frame: granule references in Icechunk (cloud extra)."""
    import disp_xr as dxr

    df = dxr.find_products(_granules(paths), frame_id=frame_id)
    snapshots = dxr.build_virtual_frame(df, repo, branch=branch, max_workers=workers)
    click.echo(f"Committed {len(snapshots)} ministacks to {repo} ({branch})")
    if validate:
        report = dxr.validate_virtual(repo, df, branch=branch)
        click.echo(f"Validated {report['n_granules']} granules against h5py")
