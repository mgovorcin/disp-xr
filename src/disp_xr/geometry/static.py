"""Static layers of a DISP frame: LOS unit vectors, angles, DEM, layover/shadow.

:func:`open_static` reads the frame-level DISP-S1-STATIC product
(``OPERA_L3_DISP-S1-STATIC_V1``: ``line_of_sight_enu.tif``, ``dem.tif`` and
``layover_shadow_mask.tif`` on the DISP frame grid), downloading it once into a
cache directory. Frames without it can use :func:`stitch_cslc_static`, which
merges CSLC-S1-STATIC bursts onto the DISP grid with rioxarray.

Layer names and nodata values of the CSLC-STATIC files follow opera-utils
``geometry.py`` (commit dff4b45), Copyright (c) 2022 California Institute of
Technology, BSD-3-Clause; the GDAL stitcher is replaced by
``rio.reproject_match``. See THIRD_PARTY_NOTICES.md.

Downloads need the ``aws`` extra; reading local files does not.
"""

from __future__ import annotations

import logging
import os
import shutil
import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import rioxarray  # noqa: F401  (registers the .rio accessor)
import xarray as xr
from pyproj import Transformer
from rasterio.warp import Resampling

from ..config import config
from ..export.raster import (
    get_geospatial_info,
    open_image,
    reproject_raster,
    write_geotiff,
)
from . import get_burst_ids
from .los import los_to_incidence_azimuth, los_up

logger = logging.getLogger(__name__)

__all__ = [
    "DISP_S1_STATIC_SHORT_NAME",
    "download_static",
    "find_static_urls",
    "open_static",
    "stitch_cslc_static",
]

DISP_S1_STATIC_SHORT_NAME = "OPERA_L3_DISP-S1-STATIC_V1"
STATIC_FILES = ("line_of_sight_enu", "dem", "layover_shadow_mask")
LOS_BANDS = ("los_east", "los_north", "los_up")

# CSLC-S1-STATIC ``/data`` layers and their nodata values.
CSLC_STATIC_NODATA = {
    "los_east": 0,
    "los_north": 0,
    "local_incidence_angle": 0,
    # 0 none, 1 shadow, 2 layover, 3 both; int8 with 127 as nodata.
    "layover_shadow_mask": 127,
}
# Pixel strides that bring the 5 m x 10 m CSLC grid near the 30 m DISP grid.
CSLC_STRIDES = {"y": 3, "x": 6}


def _cache_dir() -> Path:
    configured = config.get("cache.dir", None)
    if configured:
        return Path(configured)
    base = os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache"
    return Path(base) / "disp_xr"


def find_static_urls(
    frame_id: int,
    *,
    short_name: str = DISP_S1_STATIC_SHORT_NAME,
    session: Any = None,
) -> dict[str, str]:
    """Return the HTTPS URL of each DISP-STATIC file of a frame.

    Returns an empty dict when the frame has no DISP-STATIC granule. Needs the
    ``aws`` extra (``requests``).
    """
    import requests

    http = session or requests.Session()
    params: dict[str, str | int] = {
        "short_name": short_name,
        "attribute[]": f"int,FRAME_NUMBER,{frame_id}",
        "page_size": 10,
    }
    response = http.get(
        "https://cmr.earthdata.nasa.gov/search/granules.umm_json", params=params
    )
    response.raise_for_status()
    items = response.json()["items"]
    if not items:
        return {}

    # Several versions: keep the newest product version.
    def version(item: dict) -> tuple[int, ...]:
        attrs = {a["Name"]: a["Values"][0] for a in item["umm"]["AdditionalAttributes"]}
        return tuple(int(v) for v in attrs.get("PRODUCT_VERSION", "0").split("."))

    umm = max(items, key=version)["umm"]
    urls = {}
    for related in umm["RelatedUrls"]:
        url = related["URL"]
        if related.get("Type") != "GET DATA" or not url.startswith("https"):
            continue
        for name in STATIC_FILES:
            if url.endswith(f"_{name}.tif"):
                urls[name] = url
    return urls


def download_static(
    frame_id: int,
    output_dir: Optional[Union[str, Path]] = None,
    **fs_kwargs: Any,
) -> dict[str, Path]:
    """Download the DISP-S1-STATIC files of a frame, once.

    Files already in ``output_dir`` (default: ``<cache>/static/F<frame>``, with
    ``<cache>`` from ``config["cache.dir"]`` or ``~/.cache/disp_xr``) are not
    downloaded again. Needs the ``aws`` extra and Earthdata credentials.

    Raises
    ------
    FileNotFoundError
        If the frame has no DISP-S1-STATIC granule.

    """
    from .._remote import get_filesystem

    out = (
        Path(output_dir) if output_dir else _cache_dir() / "static" / f"F{frame_id:05d}"
    )
    out.mkdir(parents=True, exist_ok=True)
    local = _local_static(out, frame_id)
    if set(local) == set(STATIC_FILES):
        return local
    urls = find_static_urls(frame_id)
    if not urls:
        raise FileNotFoundError(
            f"No DISP-S1-STATIC granule for frame {frame_id}; "
            "use stitch_cslc_static with CSLC-S1-STATIC bursts"
        )
    fs = get_filesystem(next(iter(urls.values())), **fs_kwargs)
    for name, url in urls.items():
        target = out / url.rsplit("/", 1)[-1]
        if not target.exists():
            logger.info(f"Downloading {target.name}")
            partial = target.with_suffix(".part")
            fs.get(url, str(partial))
            partial.rename(target)
    return _local_static(out, frame_id)


def _local_static(directory: Path, frame_id: int) -> dict[str, Path]:
    found = {}
    for name in STATIC_FILES:
        matches = sorted(
            directory.glob(f"*DISP-S1-STATIC_F{frame_id:05d}_*_{name}.tif")
        )
        if matches:
            found[name] = matches[-1]
    return found


def _align(da: xr.DataArray, like: xr.Dataset, resampling: Resampling) -> xr.DataArray:
    """Put ``da`` on the grid of ``like``: select a subgrid, else resample."""
    if da.rio.crs == like.rio.crs:
        dx = abs(float(like.x[1] - like.x[0])) if like.sizes["x"] > 1 else 1.0
        try:
            out = da.sel(x=like.x, y=like.y, method="nearest", tolerance=dx / 100)
            return out.assign_coords(x=like.x, y=like.y)
        except KeyError:
            pass
    return da.rio.reproject_match(like, resampling=resampling)


def _read_static(paths: Mapping[str, Path], like: xr.Dataset) -> xr.Dataset:
    block = config.get("chunks.read_block")
    chunks = {"y": block, "x": block}
    layers: dict[str, xr.DataArray] = {}
    if "line_of_sight_enu" in paths:
        los = xr.open_dataarray(
            paths["line_of_sight_enu"],
            engine="rasterio",
            chunks=chunks,
            mask_and_scale=False,
        )
        for band, name in enumerate(LOS_BANDS, start=1):
            component = los.sel(band=band, drop=True)
            layers[name] = _align(
                component.where(component != 0), like, Resampling.bilinear
            ).astype("float32")
    if "dem" in paths:
        dem = xr.open_dataarray(paths["dem"], engine="rasterio", chunks=chunks).sel(
            band=1, drop=True
        )
        layers["dem"] = _align(dem, like, Resampling.bilinear)
    if "layover_shadow_mask" in paths:
        mask = xr.open_dataarray(
            paths["layover_shadow_mask"],
            engine="rasterio",
            chunks=chunks,
            mask_and_scale=False,
        ).sel(band=1, drop=True)
        layers["layover_shadow_mask"] = _align(mask, like, Resampling.nearest)
        layers["layover_shadow_mask"].attrs.update(
            _FillValue=255,
            description="0 not masked, 1 shadow, 2 layover, 3 layover and shadow",
        )
    return xr.Dataset(layers)


def _finish(static: xr.Dataset, like: xr.Dataset) -> xr.Dataset:
    if {"los_east", "los_north"} <= set(static):
        if "los_up" not in static:
            static["los_up"] = los_up(static["los_east"], static["los_north"])
        incidence, azimuth = los_to_incidence_azimuth(
            static["los_east"], static["los_north"]
        )
        static["incidence_angle"] = incidence.assign_attrs(units="degrees")
        static["azimuth_angle"] = azimuth.assign_attrs(
            units="degrees",
            description="Ground-to-satellite direction, clockwise from north",
        )
    for name in ("los_east", "los_north", "los_up"):
        if name in static:
            static[name].attrs.update(
                units="unitless",
                description=(
                    f"{name[4:].title()} component of the ground-to-satellite"
                    " LOS unit vector"
                ),
            )
    if "dem" in static:
        static["dem"].attrs.setdefault("units", "meters")
    static = static.drop_vars("spatial_ref", errors="ignore")
    static = (
        static.assign_coords(spatial_ref=like["spatial_ref"])
        if "spatial_ref" in like.coords
        else static
    )
    for name in static.data_vars:
        static[name].attrs["grid_mapping"] = "spatial_ref"
    return static


def open_static(
    like: xr.Dataset,
    *,
    static_dir: Optional[Union[str, Path]] = None,
    files: Optional[Mapping[str, Union[str, Path]]] = None,
    download: bool = True,
) -> xr.Dataset:
    """Open the static layers of a DISP frame on the grid of ``like``.

    Parameters
    ----------
    like : xr.Dataset
        A DISP dataset (from :func:`disp_xr.open_disp`), possibly subset; its
        ``frame_id`` attribute picks the static granule and its grid is the
        output grid.
    static_dir : str or Path, optional
        Where DISP-S1-STATIC files are, or are downloaded to.
    files : mapping, optional
        Explicit ``{"line_of_sight_enu" | "dem" | "layover_shadow_mask": path}``.
    download : bool
        Download missing files (``aws`` extra, Earthdata credentials).

    Returns
    -------
    xr.Dataset
        Lazy ``(y, x)`` layers: ``los_east``, ``los_north``, ``los_up``,
        ``incidence_angle``, ``azimuth_angle`` (degrees), ``dem`` and
        ``layover_shadow_mask``. LOS points from the ground to the satellite.

    Raises
    ------
    FileNotFoundError
        If no static file is found or downloadable for the frame.

    """
    if files is not None:
        paths = {k: Path(v) for k, v in files.items()}
    else:
        frame_id = int(like.attrs["frame_id"])
        directory = (
            Path(static_dir)
            if static_dir
            else _cache_dir() / "static" / f"F{frame_id:05d}"
        )
        paths = _local_static(directory, frame_id) if directory.exists() else {}
        if set(paths) != set(STATIC_FILES) and download:
            paths = download_static(frame_id, directory)
    if not paths:
        raise FileNotFoundError("No static layer files found")
    unknown = set(paths) - set(STATIC_FILES)
    if unknown:
        raise ValueError(f"Unknown static files {sorted(unknown)}; use {STATIC_FILES}")
    static = _finish(_read_static(paths, like), like)
    static.attrs.update(
        source="DISP-S1-STATIC", static_files=[str(p) for p in paths.values()]
    )
    return static


def _open_cslc_static(path: Union[str, Path], layers: Sequence[str]) -> xr.Dataset:
    ds = xr.open_dataset(path, group="data", engine="h5netcdf", mask_and_scale=False)
    epsg = int(ds["projection"].attrs["epsg_code"])
    ds = ds[list(layers)].rename({"x_coordinates": "x", "y_coordinates": "y"})
    ds = ds.isel(
        y=slice(None, None, CSLC_STRIDES["y"]), x=slice(None, None, CSLC_STRIDES["x"])
    )
    return ds.rio.write_crs(epsg)


def stitch_cslc_static(
    files: Sequence[Union[str, Path]],
    like: xr.Dataset,
    layers: Sequence[str] = ("los_east", "los_north", "layover_shadow_mask"),
) -> xr.Dataset:
    """Merge CSLC-S1-STATIC bursts onto the grid of a DISP dataset.

    Each burst is decimated (``CSLC_STRIDES``), resampled to the DISP grid
    with ``rio.reproject_match`` (nearest neighbour, as the opera-utils
    stitcher) and merged; where bursts overlap, the first file wins.

    Returns
    -------
    xr.Dataset
        Same layers as :func:`open_static` that CSLC-STATIC provides (no DEM).

    """
    unknown = set(layers) - set(CSLC_STATIC_NODATA)
    if unknown:
        raise ValueError(f"Unknown CSLC-STATIC layers {sorted(unknown)}")
    merged: dict[str, xr.DataArray] = {}
    for path in files:
        burst = _open_cslc_static(path, layers)
        for name in layers:
            nodata = CSLC_STATIC_NODATA[name]
            da = burst[name].rio.write_nodata(nodata)
            on_grid = da.rio.reproject_match(
                like, resampling=Resampling.nearest, nodata=nodata
            )
            valid = on_grid != nodata
            merged[name] = (
                on_grid.where(valid, nodata)
                if name not in merged
                else merged[name].where(merged[name] != nodata, on_grid)
            )
    static = xr.Dataset(merged).assign_coords(x=like.x, y=like.y)
    for name in ("los_east", "los_north"):
        if name in static:
            static[name] = static[name].where(static[name] != 0).astype("float32")
    if "layover_shadow_mask" in static:
        mask = static["layover_shadow_mask"]
        static["layover_shadow_mask"] = mask.where(mask != 127, 255).astype("uint8")
        static["layover_shadow_mask"].attrs["_FillValue"] = 255
    static = _finish(static, like)
    static.attrs.update(source="CSLC-S1-STATIC", static_files=[str(p) for p in files])
    return static


# --- deprecated file-based helpers (removed in plan task T38) -----------------


def get_static_layers(
    disp_path: Union[str, Path], output_dir: Union[str, Path], n_workers: int = 5
) -> dict:
    """Download CSLC-STATIC bursts and write LOS GeoTIFFs on the DISP grid.

    .. deprecated::
        Use ``open_static(open_disp(paths))``, which reads DISP-S1-STATIC, or
        :func:`stitch_cslc_static`.
    """
    warnings.warn(
        "get_static_layers is deprecated; use disp_xr.open_static",
        DeprecationWarning,
        stacklevel=2,
    )
    import asf_search as asf

    from ..open import open_disp

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    temp_dir = output_dir / "tmp"
    temp_dir.mkdir(exist_ok=True)
    results = asf.search(
        operaBurstID=list(get_burst_ids(disp_path)), processingLevel="CSLC-STATIC"
    )
    logger.info(f"Static layer files to download: {len(results)}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results.download(path=temp_dir, processes=n_workers)
    files = [temp_dir / r.properties["fileName"] for r in results]

    like = open_disp([disp_path], check=False, layers=["displacement"]).isel(time=0)
    static = stitch_cslc_static(files, like).compute()
    out = {}
    for name in ("los_east", "los_north", "los_up", "layover_shadow_mask"):
        out[name] = output_dir / f"{name}.tif"
        da = static[name]
        fill = 0 if name.startswith("los") else 127
        da.fillna(fill).astype(da.dtype).rio.write_nodata(fill).rio.to_raster(out[name])
    shutil.rmtree(temp_dir)
    return out


def los_unit2inc_azimuth(
    los_east: Union[str, Path],
    los_north: Union[str, Path],
    output_dir: Union[str, Path],
) -> dict:
    """Convert line-of-sight (LOS) unit vectors to incidence and azimuth angles.

    This function processes east and north LOS unit vector components to derive
    the incidence and azimuth angles and saves the output to the specified directory.

    Parameters
    ----------
    los_east : str | Path
        Path to the file containing the east component of the LOS unit vector.
    los_north : str | Path
        Path to the file containing the north component of the LOS unit vector.
    output_dir : str | Path
        Directory where the output incidence and azimuth angle files will be saved.

    Returns
    -------
    dict
        Dictionary containing metadata or paths of the generated incidence and
        azimuth angle files.

    """
    warnings.warn(
        "los_unit2inc_azimuth is deprecated; use disp_xr.open_static "
        "(incidence_angle, azimuth_angle)",
        DeprecationWarning,
        stacklevel=2,
    )
    # Read los_east and los_north
    try:
        los_ew, los_east_atr = open_image(los_east)
    except FileNotFoundError:
        raise ValueError(f"File {los_east} not found")

    try:
        los_ns, _ = open_image(los_north)
    except FileNotFoundError:
        raise ValueError(f"File {los_north} not found")

    # Create output directory if does not exist
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Get los_up
    up = np.sqrt(1 - los_ew**2 - los_ns**2)
    # Get azimuth and incidence angles
    az_angle = -1 * np.rad2deg(np.arctan2(los_ew, los_ns)) % 360
    incidence_angle = np.rad2deg(np.arccos(up))

    # Write azimuth and incidence angles to file
    logger.info(f"Writing azimuth and incidence angles to {output_dir}")
    write_geotiff(
        output_dir / "azimuth.tif",
        az_angle,
        los_east_atr["bounds"],
        epsg=los_east_atr["crs"].to_epsg(),
    )
    write_geotiff(
        output_dir / "incidence.tif",
        incidence_angle,
        los_east_atr["bounds"],
        epsg=los_east_atr["crs"].to_epsg(),
    )
    return {
        "inc_angle": output_dir / "incidence.tif",
        "az_angle": output_dir / "azimuth.tif",
    }


def download_dem(disp_path: Union[str, Path], output_dir: Union[str, Path]) -> Path:
    """Download a Digital Elevation Model (DEM) for the given DISP dataset.

    Retrieves the DEM corresponding to the provided DISP dataset and saves it
    in the specified output directory.

    Parameters
    ----------
    disp_path : str | Path
        Path to the DISP dataset used to determine the required DEM coverage.
    output_dir : str | Path
        Directory where the downloaded DEM file will be saved.

    Returns
    -------
    Path
        Path to the downloaded DEM file.

    """
    warnings.warn(
        "download_dem is deprecated; use disp_xr.open_static "
        "(dem, from DISP-S1-STATIC)",
        DeprecationWarning,
        stacklevel=2,
    )
    # Create output directory if does not exist
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Get geospatial info
    geo_info = get_geospatial_info(disp_path)

    # Init projection converter
    transformer = Transformer.from_crs(
        f"EPSG:{geo_info.crs.to_epsg()}", "EPSG:4326", always_xy=True
    )

    # Get bounds in geographic coordinates
    snwe = np.zeros((4))
    left, bottom, right, top = geo_info.bounds
    snwe[2], snwe[1] = transformer.transform(left - 1e3, top + 1e3)
    snwe[3], snwe[0] = transformer.transform(right + 1e3, bottom - 1e3)
    bounds = [snwe[2], snwe[0], snwe[3], snwe[1]]

    # Download dem
    logger.info(f"Downloading DEM for bounds: {np.squeeze(bounds)}")
    atr = {"rows": geo_info.rows, "cols": geo_info.cols}
    import dem_stitcher

    dem_data, dem_attr = dem_stitcher.stitch_dem(
        bounds,
        dem_name="glo_30",
        dst_ellipsoidal_height=True,
        dst_area_or_point="Point",
    )

    logger.info(f"Saving DEM: {output_dir / 'dem.tif'}")
    reproject_raster(
        output_name=output_dir / "dem.tif",
        src_data=dem_data,
        atr=atr,
        src_transform=dem_attr["transform"],
        dst_transform=geo_info.transform,
        src_crs=dem_attr["crs"],
        dst_crs=geo_info.crs,
        dtype="float32",
        resampling_mode=Resampling.bilinear,
    )
    return output_dir / "dem.tif"


def download_water_mask(
    disp_path: Union[str, Path], output_dir: Union[str, Path]
) -> Path:
    """Download a water mask for the given DISP dataset.

    Retrieves a water mask corresponding to the provided DISP dataset and saves it
    in the specified output directory.

    Parameters
    ----------
    disp_path : str | Path
        Path to the DISP dataset used to determine the required water mask coverage.
    output_dir : str | Path
        Directory where the downloaded water mask file will be saved.

    Returns
    -------
    Path
        Path to the downloaded water mask file.

    """
    warnings.warn(
        "download_water_mask is deprecated; use the water_mask layer "
        "of the DISP granules",
        DeprecationWarning,
        stacklevel=2,
    )
    # Create output directory if does not exist
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Get geospatial info
    geo_info = get_geospatial_info(disp_path)

    # Init projection converter
    transformer = Transformer.from_crs(
        f"EPSG:{geo_info.crs.to_epsg()}", "EPSG:4326", always_xy=True
    )

    # Get bounds in geographic coordinates
    snwe = np.zeros((4))
    left, bottom, right, top = geo_info.bounds
    snwe[2], snwe[1] = transformer.transform(left - 1e3, top + 1e3)
    snwe[3], snwe[0] = transformer.transform(right + 1e3, bottom - 1e3)
    bounds = [snwe[2], snwe[0], snwe[3], snwe[1]]

    # Download water mask
    logger.info(f"Downloading water mask for bounds: {np.squeeze(bounds)}")
    atr = {"rows": geo_info.rows, "cols": geo_info.cols}
    import tile_mate

    mask_data, mask_attr = tile_mate.get_raster_from_tiles(
        bounds, tile_shortname="esa_world_cover_2021"
    )

    # Make byte mask
    mask_data[mask_data == 80] = 0
    mask_data[mask_data != 0] = 1
    mask_data = mask_data.astype("byte")

    logger.info(f"Saving Water Mask: {output_dir / 'water_mask.tif'}")
    reproject_raster(
        output_name=output_dir / "water_mask.tif",
        src_data=mask_data,
        atr=atr,
        src_transform=mask_attr["transform"],
        dst_transform=geo_info.transform,
        src_crs=mask_attr["crs"],
        dst_crs=geo_info.crs,
        dtype="uint8",
        resampling_mode=Resampling.nearest,
    )
    return output_dir / "water_mask.tif"
