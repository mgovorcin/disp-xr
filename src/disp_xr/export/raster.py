from __future__ import annotations

from io import BytesIO
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Dict, NamedTuple, Tuple, Union

import numpy as np
import rasterio
import xarray as xr
from rasterio.crs import CRS
from rasterio.transform import Affine, from_bounds
from rasterio.warp import Resampling, reproject


class GeoInfo(NamedTuple):
    """Named tuple to store geospatial metadata."""

    crs: CRS
    bounds: Tuple[float, float, float, float]  # (min_x, min_y, max_x, max_y)
    transform: Affine
    gt: Tuple[float, float, float, float, float, float]  # GDAL transform
    rows: int
    cols: int


def get_geospatial_info(file_path: Union[str, Path, BytesIO]) -> GeoInfo:
    """Get geospatial metadata from a NetCDF displacement file.

    Supports both file paths and in-memory BytesIO objects.

    Parameters
    ----------
    file_path : str, Path, or BytesIO
        Path to the NetCDF file, or an in-memory BytesIO object.

    Returns
    -------
    GeoInfo : namedtuple
        A named tuple containing:
        - crs, bounds, transform, gt, rows, cols

    """
    if isinstance(file_path, BytesIO):
        with NamedTemporaryFile(suffix=".nc") as tmp:
            tmp.write(file_path.getbuffer())
            tmp.flush()
            path_to_use = tmp.name
            with rasterio.open(f'NETCDF:"{path_to_use}":/displacement') as rd:
                return GeoInfo(
                    crs=rd.crs,
                    bounds=rd.bounds,
                    transform=rd.transform,
                    gt=rd.transform.to_gdal(),
                    rows=rd.height,
                    cols=rd.width,
                )
    else:
        with rasterio.open(f'NETCDF:"{file_path}":/displacement') as rd:
            return GeoInfo(
                crs=rd.crs,
                bounds=rd.bounds,
                transform=rd.transform,
                gt=rd.transform.to_gdal(),
                rows=rd.height,
                cols=rd.width,
            )


def open_image(file: Union[str, Path]) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Open a raster image file and extract its data and metadata.

    Parameters
    ----------
    file : str or Path
        Path to the raster image file.

    Returns
    -------
    Tuple[np.ndarray, Dict[str, Any]]
        - The first band of the raster image as a NumPy array.
        - A dictionary containing metadata:
            - "width" (int): Width of the image.
            - "height" (int): Height of the image.
            - "crs" (CRS): Coordinate Reference System of the image.
            - "bounds" (BoundingBox): Spatial bounds of the image.
            - "gt" (Affine): Affine transformation matrix.

    """
    with rasterio.open(file) as dataset:
        # Read the first band of the image into a NumPy array
        data = dataset.read(1)

        # Extract metadata
        metadata = {
            "width": dataset.width,
            "height": dataset.height,
            "crs": dataset.crs,
            "bounds": dataset.bounds,
            "gt": dataset.transform,
        }

    return data, metadata


def write_geotiff(
    output_file: Union[str, Path],
    data: np.ndarray,
    bounds: Tuple[float, float, float, float],
    epsg: int = 4326,
) -> None:
    """Write a NumPy array as a GeoTIFF file with geospatial metadata.

    Parameters
    ----------
    output_file : str or Path
        Path to save the output GeoTIFF file.
    data : np.ndarray
        The 2D NumPy array representing the raster data.
    bounds : Tuple[float, float, float, float]
        Bounding box of the raster in the format (min_x, min_y, max_x, max_y).
    epsg : int, optional
        EPSG code for the coordinate reference system (default is 4326).

    Returns
    -------
    None
        The function writes the raster data to the specified file.

    """
    min_x, min_y, max_x, max_y = bounds
    transform = from_bounds(min_x, min_y, max_x, max_y, data.shape[1], data.shape[0])

    profile = {
        "driver": "GTiff",
        "height": data.shape[0],
        "width": data.shape[1],
        "count": 1,  # Number of bands
        "dtype": data.dtype,
        "crs": f"EPSG:{epsg}",  # Coordinate Reference System
        "transform": transform,  # Affine transformation
    }

    # Write the array to a raster file
    with rasterio.open(output_file, "w", **profile) as dst:
        dst.write(data, 1)  # Write data to the first band


def reproject_raster(
    output_name: Union[str, Path],
    src_data: np.ndarray,
    atr: Dict[str, Any],
    src_transform: Affine,
    dst_transform: Affine,
    src_crs: CRS,
    dst_crs: CRS,
    dtype: str = "float32",
    resampling_mode: Resampling = Resampling.bilinear,
) -> None:
    """Reproject a raster dataset to a new coordinate reference system (CRS).

    Parameters
    ----------
    output_name : str or Path
        Path to save the reprojected raster file.
    src_data : np.ndarray
        The source raster data as a NumPy array.
    atr : Dict[str, Any]
        Dictionary containing raster attributes (e.g., "rows", "cols").
    src_transform : Affine
        The affine transformation matrix for the source raster.
    dst_transform : Affine
        The affine transformation matrix for the destination raster.
    src_crs : CRS
        Coordinate reference system of the source raster.
    dst_crs : CRS
        Coordinate reference system for the output raster.
    dtype : str, optional
        Data type for the output raster (default is "float32").
    resampling_mode : Resampling, optional
        Resampling method used during reprojection (default is bilinear).

    Returns
    -------
    None
        The function writes the reprojected raster to the specified output file.

    """
    with rasterio.open(
        output_name,
        "w",
        height=atr["rows"],
        width=atr["cols"],
        count=1,
        dtype=dtype,
        crs=dst_crs,
        transform=dst_transform,
    ) as dst:
        reproject(
            source=src_data,
            destination=rasterio.band(dst, 1),
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=resampling_mode,
        )


def to_geotiff(
    da: xr.DataArray,
    path: Union[str, Path],
    *,
    per_epoch: bool = False,
    compress: str = "DEFLATE",
) -> list[Path]:
    """Write a layer as Cloud Optimized GeoTIFF (with overviews).

    Parameters
    ----------
    da : xr.DataArray
        A ``(y, x)`` layer with a CRS, or ``(time, y, x)`` with ``per_epoch``.
    path : str or Path
        Output file. With ``per_epoch``, one file per epoch named
        ``<stem>_<YYYYMMDD>.tif`` next to it.
    per_epoch : bool
        Write each epoch of a ``(time, y, x)`` layer to its own file.
    compress : str
        GDAL compression.

    Returns
    -------
    list[Path]
        Files written.

    Raises
    ------
    ValueError
        If the layer has no CRS, has the wrong dims, or one epoch is larger
        than ``config["max_load_bytes"]``.

    """
    import pandas as pd
    import rioxarray  # noqa: F401  (registers the .rio accessor)

    from .._version import __version__
    from ..config import config

    path = Path(path)
    if da.rio.crs is None:
        raise ValueError("The layer has no CRS; open it with disp_xr.open_disp")
    if per_epoch:
        if da.dims != ("time", "y", "x"):
            raise ValueError(f"per_epoch needs (time, y, x), not {da.dims}")
        written: list[Path] = []
        for t in da["time"].values:
            name = f"{path.stem}_{pd.Timestamp(t):%Y%m%d}{path.suffix or '.tif'}"
            written += to_geotiff(
                da.sel(time=t, drop=True), path.with_name(name), compress=compress
            )
        return written
    if da.dims != ("y", "x"):
        raise ValueError(f"A GeoTIFF layer needs (y, x) dims, not {da.dims}")
    if da.nbytes > config.get("max_load_bytes"):
        raise ValueError(
            f"Layer of {da.nbytes / 1e9:.1f} GB is larger than max_load_bytes"
        )
    layer = da.compute()
    nodata = np.nan if layer.dtype.kind == "f" else layer.attrs.get("_FillValue")
    attrs = {
        k: v
        for k, v in layer.attrs.items()
        if k not in ("_FillValue", "grid_mapping") and isinstance(v, (str, int, float))
    }
    layer = (
        layer.rio.write_nodata(nodata, encoded=False) if nodata is not None else layer
    )
    layer.attrs = attrs
    layer.rio.to_raster(
        path,
        driver="COG",
        compress=compress,
        tags={"DISP_XR_VERSION": __version__, "HISTORY": da.attrs.get("history", "")},
    )
    return [path]
