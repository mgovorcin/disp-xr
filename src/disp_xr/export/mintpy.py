"""Write a chained DISP dataset as MintPy HDF5 inputs.

Files follow MintPy's layout (``writefile.layout_hdf5``): ``timeseries.h5``
(``date``, ``bperp``, ``timeseries``), ``avgSpatialCoh.h5``,
``timeseriesDensity.h5`` and ``maskReliable.h5`` from the quality layers, and
``geometryGeo.h5`` from :func:`disp_xr.open_static`. Only h5py is needed to
write them; MintPy reads them.

The file set and metadata keys follow opera-utils ``disp_nc_to_mintpy``
(src/opera_utils/disp/mintpy.py, commit dff4b45), Copyright (c) 2022 California
Institute of Technology, BSD-3-Clause, and the earlier disp-xr ``get_metadata``.
See THIRD_PARTY_NOTICES.md. Changes (upstream bug 6): every dataset is
compressed, platform, wavelength and orbit height come from the product spec,
and the cube is written block by block instead of loaded whole.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Optional, Union

import h5py
import numpy as np
import pandas as pd
import xarray as xr

from .._version import __version__
from ..config import bounded_threads, config
from ..geometry import grid
from ..geometry.los import slant_range
from ..product import PRODUCTS
from ..quality import quality_summary

logger = logging.getLogger(__name__)

__all__ = ["mintpy_metadata", "to_mintpy"]

PLATFORMS = {"DISP-S1": "Sentinel-1", "DISP-NISAR": "NISAR"}
H5_CHUNKS = (1, 512, 512)
COMPRESSION = {"compression": "gzip", "compression_opts": 4, "shuffle": True}


def mintpy_metadata(ds: xr.Dataset) -> dict[str, str]:
    """Return the MintPy attributes of a chained DISP dataset.

    Keys and conventions: corner coordinates (``X_FIRST``, ``Y_FIRST``) and
    signed pixel steps from the grid, wavelength, orbit height and platform
    from the product spec, ``REF_DATE`` from the chaining, and ``REF_Y`` /
    ``REF_X`` when the data were re-referenced to a point.
    """
    spec = PRODUCTS[ds.attrs["product"]]
    crs = grid(ds).crs
    transform = ds.rio.transform()
    dates = pd.DatetimeIndex(ds["time"].values)
    first = dates[0]
    meta: dict[str, Any] = {
        "FILE_TYPE": "timeseries",
        "UNIT": "m",
        "LENGTH": ds.sizes["y"],
        "WIDTH": ds.sizes["x"],
        "X_FIRST": transform.c,
        "Y_FIRST": transform.f,
        "X_STEP": transform.a,
        "Y_STEP": transform.e,
        "X_UNIT": "meters",
        "Y_UNIT": "meters",
        "EPSG": crs.to_epsg() if crs else "",
        "UTM_ZONE": (crs.utm_zone or "") if crs else "",
        "PLATFORM": PLATFORMS.get(spec.name, spec.name),
        "PROCESSOR": "opera",
        "WAVELENGTH": spec.wavelength_m,
        "HEIGHT": spec.orbit_height_m,
        "EARTH_RADIUS": 6371000.0,
        "ALOOKS": 1,
        "RLOOKS": 1,
        "REF_DATE": f"{pd.Timestamp(ds.attrs['reference_date']):%Y%m%d}",
        "START_DATE": f"{dates[0]:%Y%m%d}",
        "END_DATE": f"{dates[-1]:%Y%m%d}",
        "CENTER_LINE_UTC": first.hour * 3600 + first.minute * 60 + first.second,
        "DISP_PRODUCT": spec.name,
        "DISP_FRAME_ID": ds.attrs.get("frame_id", ""),
        "DISP_XR_VERSION": __version__,
    }
    reference = json.loads(ds.attrs.get("reference", "{}"))
    if "row" in reference:
        meta.update(REF_Y=reference["row"], REF_X=reference["col"])
    return {k: str(v) for k, v in meta.items()}


def _write(path: Path, attrs: Mapping[str, str], datasets: Mapping[str, Any]) -> None:
    """Write datasets (arrays or lazy DataArrays) with compression on every one."""
    max_bytes = config.get("max_load_bytes")
    with h5py.File(path, "w") as f:
        f.attrs.update(attrs)
        for name, data in datasets.items():
            arr = data.data if isinstance(data, xr.DataArray) else np.asarray(data)
            if arr.ndim == 0 or arr.dtype.kind == "S":
                f.create_dataset(name, data=np.asarray(arr))
                continue
            chunks = tuple(min(c, s) for c, s in zip(H5_CHUNKS[-arr.ndim :], arr.shape))
            dset = f.create_dataset(
                name, shape=arr.shape, dtype=arr.dtype, chunks=chunks, **COMPRESSION
            )
            if arr.ndim < 3:
                if arr.nbytes > max_bytes:
                    raise ValueError(f"{name} is larger than max_load_bytes")
                dset[...] = np.asarray(arr)
                continue
            # Row blocks of whole epochs: bounded memory, one write per block.
            rows = max(
                1, int(max_bytes // (arr.shape[0] * arr.shape[2] * arr.itemsize))
            )
            rows = min(rows, 2048)
            for start in range(0, arr.shape[1], rows):
                block = arr[:, start : start + rows, :]
                dset[:, start : start + rows, :] = np.asarray(block)


def _mintpy_azimuth(azimuth_clockwise: xr.DataArray) -> xr.DataArray:
    # MintPy: LOS ground to satellite, from north, anticlockwise positive,
    # in (-180, 180]. disp-xr: clockwise in [0, 360).
    az = -azimuth_clockwise
    return az - np.round(az / 360.0) * 360.0


@bounded_threads
def to_mintpy(
    ds: xr.Dataset,
    outdir: Union[str, Path],
    *,
    static: Optional[xr.Dataset] = None,
    reliability_threshold: float = 0.9,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Write MintPy inputs from a chained DISP dataset.

    Parameters
    ----------
    ds : xr.Dataset
        Output of :func:`disp_xr.stack_disp` (optionally corrected, masked and
        re-referenced): ``displacement`` relative to one reference date.
    outdir : str or Path
        Output directory, created if needed.
    static : xr.Dataset, optional
        Output of :func:`disp_xr.open_static`; writes ``geometryGeo.h5``.
    reliability_threshold : float
        Fraction of valid epochs for ``maskReliable.h5`` (see
        :func:`disp_xr.quality_summary`).
    overwrite : bool
        Replace existing files.

    Returns
    -------
    dict[str, Path]
        File name stem to path, for each file written.

    Raises
    ------
    ValueError
        If ``ds`` is not chained, or a file exists and ``overwrite`` is False.

    """
    if "reference_date" not in ds.attrs:
        raise ValueError("MintPy needs one reference date; chain with stack_disp first")
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    meta = mintpy_metadata(ds)
    names = ["timeseries", "avgSpatialCoh", "timeseriesDensity", "maskReliable"]
    if static is not None:
        names.append("geometryGeo")
    paths = {name: outdir / f"{name}.h5" for name in names}
    existing = [p.name for p in paths.values() if p.exists()]
    if existing and not overwrite:
        raise ValueError(f"{existing} exist in {outdir}; pass overwrite=True")

    dates = np.array(
        [f"{t:%Y%m%d}" for t in pd.DatetimeIndex(ds["time"].values)], dtype="S8"
    )
    if "perpendicular_baseline" in ds:
        bperp = ds["perpendicular_baseline"].mean(("y", "x")).fillna(0).values
    else:
        bperp = np.zeros(len(dates))
    _write(
        paths["timeseries"],
        meta,
        {
            "date": dates,
            "bperp": bperp.astype(np.float32),
            "timeseries": ds["displacement"].astype(np.float32),
        },
    )

    summary = quality_summary(ds, reliability_threshold=reliability_threshold)
    mask_meta = {**meta, "FILE_TYPE": "mask", "UNIT": "1"}
    _write(
        paths["avgSpatialCoh"],
        {**mask_meta, "FILE_TYPE": "temporalCoherence"},
        {"avgSpatialCoh": summary["temporal_coherence_hmean"].astype(np.float32)},
    )
    _write(
        paths["timeseriesDensity"],
        {**mask_meta, "FILE_TYPE": "timeseriesDensity"},
        {"timeseriesDensity": summary["reliability"].astype(np.float32)},
    )
    _write(
        paths["maskReliable"],
        mask_meta,
        {"mask": summary["reliable"].astype(np.int8)},
    )

    if static is not None:
        geometry = {
            "incidenceAngle": static["incidence_angle"].astype(np.float32),
            "azimuthAngle": _mintpy_azimuth(static["azimuth_angle"]).astype(np.float32),
            "slantRangeDistance": (
                slant_range(static["incidence_angle"], float(meta["HEIGHT"])).astype(
                    np.float32
                )
            ),
        }
        if "dem" in static:
            geometry["height"] = static["dem"].astype(np.float32)
        if "layover_shadow_mask" in static:
            lsm = static["layover_shadow_mask"]
            geometry["shadowMask"] = ((lsm > 0) & (lsm != 255)).astype(np.int8)
        if "water_mask" in ds:
            # 1 land, 0 water, as MintPy; the mask is the same every epoch.
            water = ds["water_mask"].isel(time=-1, drop=True)
            geometry["waterMask"] = (water == 1).astype(np.int8)
        _write(paths["geometryGeo"], {**meta, "FILE_TYPE": "geometry"}, geometry)

    logger.info(f"Wrote MintPy inputs to {outdir}")
    return {name: paths[name] for name in names}
