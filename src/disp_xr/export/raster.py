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

from ..config import bounded_threads
from ._provenance import provenance_attrs


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


@bounded_threads
def to_geotiff(
    obj: Union[xr.DataArray, xr.Dataset],
    path: Union[str, Path],
    *,
    layer: str = "displacement",
    per_epoch: bool = False,
    compress: str = "DEFLATE",
) -> list[Path]:
    """Write a layer as a Cloud Optimized GeoTIFF (COG) with overviews.

    The file is tiled with internal overviews (GDAL adds them above 512
    pixels), so it can be read in parts over HTTP. Its metadata holds the
    provenance: ``DISP_XR_VERSION``, ``SOFTWARE_VERSIONS``,
    ``INPUT_GRANULES`` and ``HISTORY``.

    Parameters
    ----------
    obj : xr.DataArray or xr.Dataset
        A ``(y, x)`` layer with a CRS, or ``(time, y, x)`` with ``per_epoch``;
        or a Dataset, from which ``layer`` is written. Pass the Dataset to keep
        its provenance: input granules and history live in Dataset attributes.
    layer : str
        Layer to write when ``obj`` is a Dataset.
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

    path = Path(path)
    if isinstance(obj, xr.Dataset):
        if layer not in obj:
            raise ValueError(f"No layer {layer!r}; have {sorted(obj.data_vars)}")
        provenance = provenance_attrs(obj)
        da = obj[layer]
    else:
        da = obj
        provenance = provenance_attrs(obj)  # type: ignore[arg-type]
    if da.rio.crs is None:
        raise ValueError("The layer has no CRS; open it with disp_xr.open_disp")
    if per_epoch:
        if da.dims != ("time", "y", "x"):
            raise ValueError(f"per_epoch needs (time, y, x), not {da.dims}")
        written: list[Path] = []
        for t in da["time"].values:
            name = f"{path.stem}_{pd.Timestamp(t):%Y%m%d}{path.suffix or '.tif'}"
            written.append(
                _write_cog(
                    da.sel(time=t, drop=True),
                    path.with_name(name),
                    compress,
                    provenance,
                )
            )
        return written
    return [_write_cog(da, path, compress, provenance)]


def _write_cog(
    da: xr.DataArray, path: Path, compress: str, provenance: dict[str, Any]
) -> Path:
    """Write one ``(y, x)`` layer as a COG tagged with ``provenance``."""
    from .._version import __version__
    from ..config import check_load

    if da.dims != ("y", "x"):
        raise ValueError(f"A GeoTIFF layer needs (y, x) dims, not {da.dims}")
    check_load(
        da,
        "the GeoTIFF layer",
        "Write a window (da.isel(y=..., x=...)), or a whole stack with to_zarr.",
    )
    image = da.compute()
    nodata = np.nan if image.dtype.kind == "f" else image.attrs.get("_FillValue")
    attrs = {
        k: v
        for k, v in image.attrs.items()
        if k not in ("_FillValue", "grid_mapping") and isinstance(v, (str, int, float))
    }
    image = (
        image.rio.write_nodata(nodata, encoded=False) if nodata is not None else image
    )
    image.attrs = attrs
    # Write a tiled GTiff, then let GDAL CreateCopy make the COG: writing to
    # the COG driver directly truncates floats to integers with rasterio 1.5.1
    # and GDAL 3.13.
    import rasterio.shutil

    staging = path.with_name(f".{path.stem}.staging.tif")
    try:
        image.rio.to_raster(
            staging,
            tiled=True,
            tags={
                "DISP_XR_VERSION": __version__,
                "SOFTWARE_VERSIONS": provenance["software_versions"],
                "INPUT_GRANULES": provenance["input_granules"],
                "HISTORY": provenance["history"],
            },
        )
        rasterio.shutil.copy(staging, path, driver="COG", compress=compress)
    finally:
        staging.unlink(missing_ok=True)
    return path
