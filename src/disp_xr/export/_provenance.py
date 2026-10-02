"""Provenance attributes written into every disp-xr output."""

from __future__ import annotations

import json
from importlib.metadata import PackageNotFoundError, version
from typing import Any

import xarray as xr

from .._version import __version__

__all__ = ["provenance_attrs", "with_provenance"]

PACKAGES = ("xarray", "zarr", "dask", "numpy", "rioxarray")

# Product layer meanings, written as CF flag attributes.
FLAGS: dict[str, tuple[list[int], str]] = {
    "recommended_mask": ([0, 1], "bad good"),
    "water_mask": ([0, 1], "water land"),
    "persistent_scatterer_mask": ([0, 1], "not_ps ps"),
    "layover_shadow_mask": ([0, 1, 2, 3], "none shadow layover layover_and_shadow"),
}


def _version(package: str) -> str:
    try:
        return version(package)
    except PackageNotFoundError:
        return "not installed"


def _granules(value: Any) -> list:
    # A store read back from Zarr or NetCDF holds the list as a JSON string.
    return json.loads(value) if isinstance(value, str) else list(value)


def provenance_attrs(ds: xr.Dataset) -> dict[str, Any]:
    """Return the provenance attributes for an output written from ``ds``.

    ``disp_xr_version``, ``software_versions`` (JSON), ``input_granules``
    (JSON list of granule names, from :func:`disp_xr.open_disp`) and the
    ``history`` of disp-xr operations.
    """
    return {
        "disp_xr_version": __version__,
        "software_versions": json.dumps({p: _version(p) for p in PACKAGES}),
        "input_granules": json.dumps(_granules(ds.attrs.get("input_granules", []))),
        "history": ds.attrs.get("history", ""),
    }


def _serializable(value: Any) -> Any:
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value)
    return value


def with_provenance(ds: xr.Dataset) -> xr.Dataset:
    """Return ``ds`` with provenance, CF flag and grid-mapping attributes set.

    List and dict attributes become JSON strings, which every format can store.
    """
    out = ds.copy()
    out.attrs = {k: _serializable(v) for k, v in ds.attrs.items()}
    out.attrs.update(provenance_attrs(ds))
    for name, var in out.data_vars.items():
        attrs = dict(var.attrs)
        if name in FLAGS:
            values, meanings = FLAGS[name]
            attrs.setdefault("flag_values", values)
            attrs.setdefault("flag_meanings", meanings)
        if "spatial_ref" in out.coords and {"y", "x"} <= set(var.dims):
            attrs["grid_mapping"] = "spatial_ref"
        out[name].attrs = attrs
    return out
