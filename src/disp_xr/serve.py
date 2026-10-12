"""Read-only REST service over disp-xr stores (PRD C7, plan task T40).

An `xpublish <https://github.com/xpublish-community/xpublish>`_ plugin that
serves, under ``/datasets/{dataset_id}/disp/``:

- ``timeseries``: the displacement series at a point (median over a buffer),
  with the quality layers the store holds and its reference date;
- ``velocity``: the stored velocity and its sigma at a point;
- ``cells``: a regular grid or a quadtree of a bounding box, as GeoJSON;
- ``gnss``: the station residuals and summary of :func:`residuals.compare`,
  written next to the cube by :meth:`Comparison.to_zarr`;
- ``provenance``: operation history, input granules and versions.

Nothing heavy runs on request (PRD C7): velocities and GNSS residuals are
served only when the store holds them, and ``cells`` downsamples at most
``MAX_CELL_PIXELS`` pixels. xpublish adds its own routes (Zarr over HTTP,
dataset info) and, when installed, ``xpublish-tiles`` adds web map tiles.
Every response carries the CRS and the units.

Start a service with ``disp_xr serve STORE`` or :func:`make_app`.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, Optional, Union

import numpy as np
import pandas as pd
import xarray as xr

try:
    from fastapi import APIRouter, Depends, HTTPException, Query
    from pydantic import BaseModel, ConfigDict, Field
    from xpublish import Dependencies, Plugin, Rest, hookimpl
except ImportError as err:  # pragma: no cover - exercised without the extra
    raise ImportError(
        "disp_xr.serve needs the serve extra: pip install 'disp-xr[serve]'"
    ) from err

from ._history import history

__all__ = ["DispPlugin", "make_app", "open_store"]

# Largest bounding box ``cells`` downsamples on request, in pixels.
MAX_CELL_PIXELS = 4_000_000

QUALITY_LAYERS = ("temporal_coherence", "phase_similarity")


# Responses ---------------------------------------------------------------


class Point(BaseModel):
    """Where a point query was read."""

    lon: float
    lat: float
    x: float = Field(description="Easting in the store CRS (pixel center)")
    y: float = Field(description="Northing in the store CRS (pixel center)")
    buffer_m: float
    n_pixels: int = Field(description="Pixels inside the buffer")


class TimeseriesResponse(BaseModel):
    """Displacement at a point, one value per epoch."""

    dataset: str
    crs: str
    units: str
    reference_date: Optional[str]
    point: Point
    dates: list[str]
    displacement: list[Optional[float]]
    quality: dict[str, list[Optional[float]]] = Field(
        description="Median of each quality layer the store holds, per epoch"
    )


class VelocityResponse(BaseModel):
    """Stored velocity at a point."""

    dataset: str
    crs: str
    units: str
    method: Optional[str]
    point: Point
    velocity: Optional[float]
    velocity_sigma: Optional[float]


class GnssStation(BaseModel):
    """One station's velocities and residual (see ``residuals.compare``)."""

    model_config = ConfigDict(extra="allow")
    station: str
    lon: Optional[float] = None
    lat: Optional[float] = None


class GnssResponse(BaseModel):
    """Station residuals and the comparison summary."""

    dataset: str
    crs: str
    units: str
    summary: dict[str, Any]
    stations: list[GnssStation]


class ProvenanceResponse(BaseModel):
    """How the store was made."""

    dataset: str
    crs: str
    history: list[dict[str, Any]]
    input_granules: list[str]
    product_versions: list[str]
    software_versions: dict[str, str]
    reference_date: Optional[str]
    attrs: dict[str, str] = Field(description="Every other global attribute")


class FeatureCollection(BaseModel):
    """GeoJSON FeatureCollection in lon/lat (RFC 7946)."""

    model_config = ConfigDict(extra="allow")
    type: Literal["FeatureCollection"] = "FeatureCollection"
    features: list[dict[str, Any]]


# Reading -----------------------------------------------------------------


def crs_of(ds: xr.Dataset) -> Any:
    """Return the pyproj CRS of ``ds`` (rioxarray or GeoZarr ``proj:`` attrs)."""
    from pyproj import CRS

    crs = ds.rio.crs if hasattr(ds, "rio") else None
    if crs is not None:
        return CRS.from_user_input(crs)
    for key in ("proj:wkt2", "proj:code", "crs_wkt"):
        if key in ds.attrs:
            return CRS.from_user_input(ds.attrs[key])
    raise ValueError("the store has no CRS (no spatial_ref or proj: attributes)")


def _crs_name(ds: xr.Dataset) -> str:
    crs = crs_of(ds)
    epsg = crs.to_epsg()
    return f"EPSG:{epsg}" if epsg else crs.to_wkt()


def _to_grid(ds: xr.Dataset, lon: Any, lat: Any) -> tuple[Any, Any]:
    from pyproj import Transformer

    to_grid = Transformer.from_crs("EPSG:4326", crs_of(ds), always_xy=True)
    return to_grid.transform(lon, lat)


def _to_lonlat(ds: xr.Dataset, x: Any, y: Any) -> tuple[Any, Any]:
    from pyproj import Transformer

    to_ll = Transformer.from_crs(crs_of(ds), "EPSG:4326", always_xy=True)
    return to_ll.transform(x, y)


def _window(ds: xr.Dataset, lon: float, lat: float, buffer_m: float):
    """Return the pixel window around a point and its :class:`Point`.

    The window is the square of pixels whose centers lie within
    ``buffer_m`` of the point's pixel (one pixel for 0). Raises 404 for a
    point outside the frame.
    """
    x, y = _to_grid(ds, lon, lat)
    xs, ys = ds["x"].values, ds["y"].values
    dx, dy = abs(float(xs[1] - xs[0])), abs(float(ys[1] - ys[0]))
    inside = (
        min(xs) - dx / 2 <= x <= max(xs) + dx / 2
        and min(ys) - dy / 2 <= y <= max(ys) + dy / 2
    )
    if not inside:
        raise HTTPException(404, f"({lon}, {lat}) is outside the frame")
    col = int(np.abs(xs - x).argmin())
    row = int(np.abs(ys - y).argmin())
    rx, ry = int(buffer_m // dx), int(buffer_m // dy)
    sel = {
        "x": slice(max(col - rx, 0), col + rx + 1),
        "y": slice(max(row - ry, 0), row + ry + 1),
    }
    window = ds.isel(sel)
    point = Point(
        lon=lon,
        lat=lat,
        x=float(xs[col]),
        y=float(ys[row]),
        buffer_m=buffer_m,
        n_pixels=int(window.sizes["x"] * window.sizes["y"]),
    )
    return window, point


def _median(da: xr.DataArray) -> xr.DataArray:
    """Return the median over the window, which is small enough to load.

    NumPy, not numbagg: numbagg compiles a kernel per dtype on first use,
    which costs seconds on a service's first request and gains nothing on a
    few pixels.
    """
    with xr.set_options(use_numbagg=False):
        return da.load().median(("y", "x"), skipna=True)


def _values(da: xr.DataArray) -> list[Optional[float]]:
    return [None if not np.isfinite(v) else float(v) for v in np.asarray(da)]


def _scalar(da: xr.DataArray) -> Optional[float]:
    value = float(da)
    return value if math.isfinite(value) else None


def _units(da: xr.DataArray) -> str:
    return str(da.attrs.get("units", ""))


def _parse_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, np.ndarray)):
        return [str(v) for v in value]
    try:
        parsed = json.loads(str(value).replace("'", '"'))
    except json.JSONDecodeError:
        return [str(value)]
    return [str(v) for v in parsed] if isinstance(parsed, list) else [str(parsed)]


def _velocity_method(ds: xr.Dataset) -> Optional[str]:
    for entry in reversed(history(ds)):
        if entry.get("op") == "fit_velocity":
            return entry.get("params", {}).get("method")
    return ds["velocity"].attrs.get("method")


def _cell_features(
    ds: xr.Dataset,
    x: np.ndarray,
    y: np.ndarray,
    size: np.ndarray,
    properties: Mapping[str, np.ndarray],
) -> list[dict[str, Any]]:
    """Return one square GeoJSON polygon per cell, corners in lon/lat."""
    half = np.asarray(size, dtype=float) / 2
    corners_x = np.stack([x - half, x + half, x + half, x - half, x - half], axis=1)
    corners_y = np.stack([y - half, y - half, y + half, y + half, y - half], axis=1)
    lon, lat = _to_lonlat(ds, corners_x, corners_y)
    features = []
    for k in range(len(x)):
        props = {}
        for name, values in properties.items():
            v = values[k]
            v = v.item() if isinstance(v, np.generic) else v
            props[name] = None if isinstance(v, float) and not math.isfinite(v) else v
        ring = [[float(a), float(b)] for a, b in zip(lon[k], lat[k])]
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [ring]},
                "properties": props,
            }
        )
    return features


def _cells(
    ds: xr.Dataset,
    bbox: Sequence[float],
    layer: str,
    date: Optional[str],
    size_m: Optional[float],
    n_points: Optional[int],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Downsample a bounding box of ``layer``; return features and metadata."""
    from .downsample import downsample
    from .quadtree import quadtree

    if (size_m is None) == (n_points is None):
        raise HTTPException(422, "pass size_m (grid) or n_points (quadtree)")
    if layer not in ds:
        raise HTTPException(404, f"the store has no {layer!r}")
    west, south, east, north = bbox
    xs, ys = _to_grid(ds, [west, east, east, west], [south, south, north, north])
    sub = ds.sel(
        x=slice(min(xs), max(xs)),
        y=slice(max(ys), min(ys)) if ds.y[0] > ds.y[-1] else slice(min(ys), max(ys)),
    )
    if sub.sizes["x"] == 0 or sub.sizes["y"] == 0:
        raise HTTPException(404, "the bounding box does not overlap the frame")
    n_pixels = sub.sizes["x"] * sub.sizes["y"]
    if n_pixels > MAX_CELL_PIXELS:
        raise HTTPException(
            413, f"{n_pixels} pixels; the limit is {MAX_CELL_PIXELS} per request"
        )
    field = sub[layer]
    when = None
    if "time" in field.dims:
        field = field.sel(time=date, method="nearest") if date else field.isel(time=-1)
        when = str(pd.Timestamp(field["time"].values).date())
    field = field.drop_vars([c for c in field.coords if c not in ("y", "x")])
    data = xr.Dataset({layer: field})
    for name in ("spatial_ref",):
        if name in sub.coords:
            data = data.assign_coords({name: sub[name]})
    meta: dict[str, Any] = {"layer": layer, "units": _units(sub[layer]), "date": when}
    if size_m is not None:
        cells = downsample(data, spacing=size_m, weights=None, layers=[layer])
        stacked = cells.stack(cell=("y", "x")).dropna("cell", subset=[layer])
        x = stacked["x"].values
        y = stacked["y"].values
        size = np.full(x.size, float(size_m))
        props = {layer: stacked[layer].values}
        meta |= {"method": "grid", "size_m": float(size_m)}
    else:
        # The coarsest cell fits the box, so the tree can reach n_points.
        side = max(data.sizes["y"], data.sizes["x"])
        max_level = int(min(8, max(1, math.floor(math.log2(side)))))
        tree = quadtree(
            data,
            n_points=n_points,
            weights=None,
            layer=layer,
            min_level=min(3, max_level - 1),
            max_level=max_level,
        )
        points = tree.points.compute()
        x, y = points["x"].values, points["y"].values
        size = points["size"].values
        props = {layer: points[layer].values, "level": points["level"].values}
        meta |= {"method": "quadtree", "n_points": int(points.sizes["point"])}
    props["size_m"] = size
    return _cell_features(ds, x, y, size, props), meta


# Opening -----------------------------------------------------------------


def open_store(path: Union[str, Path]) -> tuple[xr.Dataset, Optional[xr.Dataset]]:
    """Open a store lazily, with its GNSS group when it has one.

    Zarr stores (written by ``disp_xr stack`` or ``to_zarr``), Icechunk
    repositories (``to_icechunk``) and virtual frames (``disp_xr virtual``)
    are supported. A virtual frame holds its granules as they are, one
    reference per ministack, so it is chained on open
    (:func:`disp_xr.chain`, lazily: nothing is read until a request).

    Parameters
    ----------
    path : str or Path
        The store.

    Returns
    -------
    xr.Dataset
        The cube.
    xr.Dataset or None
        The ``gnss`` group of :meth:`Comparison.to_zarr`, if present.

    Examples
    --------
    >>> ds, gnss = open_store("frame.zarr")  # doctest: +SKIP

    """
    import warnings

    path = Path(path)
    if (path / "zarr.json").exists() or (path / ".zgroup").exists():
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Consolidated metadata")
            ds = xr.open_zarr(path)
            gnss = None
            if (path / "gnss").is_dir():
                gnss = xr.open_zarr(path, group="gnss")
        return ds, gnss
    import icechunk

    from .export.icechunk import open_icechunk

    repo = icechunk.Repository.open(icechunk.local_filesystem_storage(str(path)))
    if repo.config.virtual_chunk_containers:
        from .stack import chain
        from .virtual import open_virtual

        return chain(open_virtual(path)), None
    return open_icechunk(path), None


# Plugin ------------------------------------------------------------------


class DispPlugin(Plugin):
    """xpublish plugin: the disp-xr routes for every dataset.

    :func:`make_app` adds it; add it yourself to combine it with other
    xpublish plugins.

    Examples
    --------
    >>> import xpublish  # doctest: +SKIP
    >>> plugins = {"disp": DispPlugin()}
    >>> rest = xpublish.Rest({"frame": ds}, plugins=plugins)  # doctest: +SKIP

    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = "disp"
    dataset_router_prefix: str = "/disp"
    dataset_router_tags: Sequence[str] = ["disp"]
    gnss: dict[str, Any] = Field(
        default_factory=dict, description="GNSS group per dataset id"
    )

    @hookimpl
    def dataset_router(self, deps: Dependencies) -> APIRouter:  # noqa: D102
        router = APIRouter(
            prefix=self.dataset_router_prefix, tags=list(self.dataset_router_tags)
        )
        gnss_groups = self.gnss

        @router.get("/timeseries", response_model=TimeseriesResponse)
        def timeseries(
            dataset_id: str,
            lon: float = Query(..., ge=-180, le=180),
            lat: float = Query(..., ge=-90, le=90),
            buffer_m: float = Query(0.0, ge=0, le=5000),
            ds: xr.Dataset = Depends(deps.dataset),
        ) -> TimeseriesResponse:
            """Return the displacement series at a point: median over ``buffer_m``."""
            if "displacement" not in ds:
                raise HTTPException(404, "the store has no displacement")
            window, point = _window(ds, lon, lat, buffer_m)
            layers = ["displacement"] + [q for q in QUALITY_LAYERS if q in window]
            medians = xr.Dataset({q: _median(window[q]) for q in layers})
            quality = {}
            for q in layers[1:]:
                da = medians[q]
                if "time" not in da.dims:
                    da = da.expand_dims(time=medians["time"])
                quality[q] = _values(da)
            return TimeseriesResponse(
                dataset=dataset_id,
                crs=_crs_name(ds),
                units=_units(ds["displacement"]),
                reference_date=ds.attrs.get("reference_date"),
                point=point,
                dates=[str(pd.Timestamp(t)) for t in medians["time"].values],
                displacement=_values(medians["displacement"]),
                quality=quality,
            )

        @router.get("/velocity", response_model=VelocityResponse)
        def velocity(
            dataset_id: str,
            lon: float = Query(..., ge=-180, le=180),
            lat: float = Query(..., ge=-90, le=90),
            buffer_m: float = Query(0.0, ge=0, le=5000),
            ds: xr.Dataset = Depends(deps.dataset),
        ) -> VelocityResponse:
            """Return the stored velocity and sigma at a point (window median)."""
            if "velocity" not in ds:
                raise HTTPException(
                    404,
                    "the store has no velocity; write fit_velocity output to it",
                )
            window, point = _window(ds, lon, lat, buffer_m)
            sigma = (
                _scalar(_median(window["velocity_sigma"]))
                if "velocity_sigma" in window
                else None
            )
            return VelocityResponse(
                dataset=dataset_id,
                crs=_crs_name(ds),
                units=_units(ds["velocity"]),
                method=_velocity_method(ds),
                point=point,
                velocity=_scalar(_median(window["velocity"])),
                velocity_sigma=sigma,
            )

        @router.get("/cells", response_model=FeatureCollection)
        def cells(
            dataset_id: str,
            bbox: str = Query(..., description="west,south,east,north in degrees"),
            layer: Optional[str] = None,
            date: Optional[str] = None,
            size_m: Optional[float] = Query(None, gt=0),
            n_points: Optional[int] = Query(None, gt=0, le=100_000),
            ds: xr.Dataset = Depends(deps.dataset),
        ) -> FeatureCollection:
            """Return grid (``size_m``) or quadtree (``n_points``) cells as GeoJSON."""
            try:
                box = [float(v) for v in bbox.split(",")]
            except ValueError as err:
                raise HTTPException(422, "bbox is west,south,east,north") from err
            if len(box) != 4:
                raise HTTPException(422, "bbox is west,south,east,north")
            layer = layer or ("velocity" if "velocity" in ds else "displacement")
            features, meta = _cells(ds, box, layer, date, size_m, n_points)
            return FeatureCollection(
                features=features, dataset=dataset_id, crs="EPSG:4326", **meta
            )

        @router.get("/gnss", response_model=GnssResponse)
        def gnss(
            dataset_id: str,
            _dataset: xr.Dataset = Depends(deps.dataset),  # 404 for unknown ids
        ) -> GnssResponse:
            """Return the stored station residuals of ``residuals.compare``."""
            table = gnss_groups.get(dataset_id)
            if table is None:
                raise HTTPException(
                    404,
                    "the store has no GNSS residuals; write them with "
                    "Comparison.to_zarr(store)",
                )
            frame = table.to_dataframe().reset_index()
            stations = []
            for row in frame.to_dict(orient="records"):
                clean = {}
                for k, v in row.items():
                    if isinstance(v, (pd.Timestamp, np.datetime64)):
                        v = None if pd.isna(v) else str(pd.Timestamp(v))
                    elif isinstance(v, float) and not math.isfinite(v):
                        v = None
                    elif isinstance(v, np.generic):
                        v = v.item()
                    clean[str(k)] = v
                stations.append(GnssStation(**clean))
            return GnssResponse(
                dataset=dataset_id,
                crs="EPSG:4326 (lon, lat)",
                units=str(table.attrs.get("units", "")),
                summary=json.loads(table.attrs.get("summary", "{}")),
                stations=stations,
            )

        @router.get("/provenance", response_model=ProvenanceResponse)
        def provenance(
            dataset_id: str, ds: xr.Dataset = Depends(deps.dataset)
        ) -> ProvenanceResponse:
            """Return the operation history, input granules and versions."""
            known = {
                "history",
                "input_granules",
                "product_versions",
                "software_versions",
                "reference_date",
            }
            software = ds.attrs.get("software_versions", "{}")
            try:
                software = json.loads(str(software).replace("'", '"'))
            except json.JSONDecodeError:
                software = {"text": str(software)}
            if "disp_xr_version" in ds.attrs:
                software = {"disp_xr": str(ds.attrs["disp_xr_version"]), **software}
            return ProvenanceResponse(
                dataset=dataset_id,
                crs=_crs_name(ds),
                history=history(ds),
                input_granules=_parse_list(ds.attrs.get("input_granules")),
                product_versions=_parse_list(ds.attrs.get("product_versions")),
                software_versions={k: str(v) for k, v in software.items()},
                reference_date=ds.attrs.get("reference_date"),
                attrs={k: str(v) for k, v in ds.attrs.items() if k not in known},
            )

        return router


def make_app(
    datasets: Mapping[str, Union[str, Path, xr.Dataset]],
    *,
    gnss: Optional[Mapping[str, xr.Dataset]] = None,
    tiles: bool = True,
) -> Rest:
    """Return the xpublish service for ``datasets``.

    Parameters
    ----------
    datasets : mapping
        Dataset id to a store path (opened lazily with :func:`open_store`)
        or an open Dataset.
    gnss : mapping, optional
        GNSS groups (:meth:`Comparison.to_dataset`) for datasets passed open;
        stores opened from a path bring their own.
    tiles : bool
        Add web map tiles when ``xpublish-tiles`` is installed.

    Returns
    -------
    xpublish.Rest
        Serve with ``.serve(host=..., port=...)``; ``.app`` is the FastAPI
        application (OpenAPI page at ``/docs``).

    Examples
    --------
    >>> rest = make_app({"houston": "F08882.zarr"})  # doctest: +SKIP
    >>> rest.serve(port=9000)  # doctest: +SKIP

    """
    opened: dict[str, xr.Dataset] = {}
    groups: dict[str, Any] = dict(gnss or {})
    for name, value in datasets.items():
        if isinstance(value, xr.Dataset):
            opened[name] = value
            continue
        opened[name], group = open_store(value)
        if group is not None:
            groups[name] = group
    plugins: dict[str, Plugin] = {"disp": DispPlugin(gnss=groups)}
    if tiles:
        try:
            from xpublish_tiles.xpublish.tiles import TilesPlugin
        except ImportError:
            pass
        else:
            plugins["tiles"] = TilesPlugin()
    return Rest(opened, plugins=plugins)
