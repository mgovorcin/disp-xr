"""Find OPERA DISP granules and describe them.

Each product (DISP-S1, DISP-NISAR) is a :class:`ProductSpec` in :data:`PRODUCTS`.
A granule is identified from its file name, or, when the name is not an OPERA
name (DISP-NISAR SAS output is written as ``<ref>_<sec>.nc``), from the file's
``/identification`` group.

Filename parsing and the version/production deduplication are adapted from
opera-utils (src/opera_utils/disp/_product.py, constants.py, commit dff4b45),
Copyright (c) 2022 California Institute of Technology, BSD-3-Clause. See
THIRD_PARTY_NOTICES.md.
"""

from __future__ import annotations

import logging
import re
import warnings
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Union

import pandas as pd
from packaging.specifiers import SpecifierSet
from packaging.version import Version

from ._remote import is_remote_url, list_granules, open_h5

logger = logging.getLogger(__name__)

__all__ = [
    "DISP_NISAR",
    "DISP_S1",
    "PRODUCTS",
    "DispGranule",
    "ProductSpec",
    "MixedProductionError",
    "UnsupportedProductVersionError",
    "check_granules",
    "find_products",
    "get_disp_info",
    "group_ministacks",
]

DATETIME_FORMAT = "%Y%m%dT%H%M%SZ"

DISPLACEMENT_LAYERS = ("displacement", "short_wavelength_displacement")
QUALITY_LAYERS = (
    "temporal_coherence",
    "phase_similarity",
    "estimated_phase_quality",
    "timeseries_inversion_residuals",
    "recommended_mask",
    "water_mask",
    "persistent_scatterer_mask",
    "connected_component_labels",
    "shp_counts",
)
CORRECTION_LAYERS = ("ionospheric_delay", "solid_earth_tide", "perpendicular_baseline")


class UnsupportedProductVersionError(ValueError):
    """A granule has a product version disp-xr does not support."""


class MixedProductionError(ValueError):
    """Granules of one frame differ in dtype or encoding and cannot be combined."""


@dataclass(frozen=True)
class ProductSpec:
    """What disp-xr needs to know about one DISP product.

    ``validated`` is False until the spec has been checked against an official
    granule of the product.
    """

    name: str
    title: str
    """Value of the root ``title`` attribute in a granule."""
    filename_regex: re.Pattern
    cmr_short_name: str
    supported_versions: SpecifierSet
    wavelength_m: float
    orbit_height_m: float
    static_source: str
    """Product holding the static geometry layers (LOS, DEM)."""
    pre_applied_corrections: frozenset[str] = frozenset()
    """Correction layers already applied to ``displacement`` by the processor."""
    displacement_layers: tuple[str, ...] = DISPLACEMENT_LAYERS
    quality_layers: tuple[str, ...] = QUALITY_LAYERS
    correction_layers: tuple[str, ...] = CORRECTION_LAYERS
    validated: bool = False


_OPERA_NAME = (
    r"_F(?P<frame_id>\d+)_(?P<polarization>[HV]{{2}})_"
    r"(?P<reference>\d{{8}}T\d{{6}}Z)_(?P<secondary>\d{{8}}T\d{{6}}Z)_"
    r"v(?P<version>[\d.]+)_(?P<production>\d{{8}}T\d{{6}}Z)\.nc$"
)

DISP_S1 = ProductSpec(
    name="DISP-S1",
    title="OPERA_L3_DISP-S1 Product",
    filename_regex=re.compile(r"^OPERA_L3_DISP-S1_(?P<mode>IW)" + _OPERA_NAME.format()),
    cmr_short_name="OPERA_L3_DISP-S1_V1",
    # 0.10 is the pre-release seen in stakeholder data; 1.0 is the release.
    supported_versions=SpecifierSet(">=0.10,<2"),
    wavelength_m=0.05546576,
    orbit_height_m=693_000.0,
    static_source="CSLC-S1-STATIC",
    validated=True,
)

# Provisional: no official DISP-NISAR granule name has been seen yet. The
# pre-release SAS output (v0.4) is named <ref>_<sec>.nc and identified from its
# metadata instead. Wavelength is the nominal L-band value.
DISP_NISAR = ProductSpec(
    name="DISP-NISAR",
    title="OPERA_L3_DISP-NISAR Product",
    filename_regex=re.compile(
        r"^OPERA_L3_DISP-NI_(?P<mode>[A-Z0-9]+)" + _OPERA_NAME.format()
    ),
    cmr_short_name="OPERA_L3_DISP-NI_V1",
    supported_versions=SpecifierSet(">=0.4,<2"),
    wavelength_m=0.2384,
    orbit_height_m=747_000.0,
    static_source="DISP-NI-STATIC",
    pre_applied_corrections=frozenset({"ionospheric_delay"}),
)

PRODUCTS: dict[str, ProductSpec] = {s.name: s for s in (DISP_S1, DISP_NISAR)}
_BY_TITLE = {s.title: s for s in PRODUCTS.values()}


@dataclass(frozen=True)
class DispGranule:
    """One DISP granule: the product it belongs to and its dates."""

    path: str
    product: str
    frame_id: int
    reference_time: pd.Timestamp
    secondary_time: pd.Timestamp
    version: str
    production_time: Optional[pd.Timestamp]
    """None when the name carries no production time (SAS output)."""
    polarization: Optional[str] = None
    mode: Optional[str] = None

    @property
    def spec(self) -> ProductSpec:
        """Return the :class:`ProductSpec` of this granule."""
        return PRODUCTS[self.product]

    @classmethod
    def from_path(cls, path: Union[str, Path]) -> DispGranule:
        """Identify a granule from its name, or from its metadata if needed.

        Raises
        ------
        ValueError
            If the name is not an OPERA DISP name and the file carries no DISP
            ``/identification`` group.

        """
        name = Path(path).name
        for spec in PRODUCTS.values():
            m = spec.filename_regex.match(name)
            if m:
                return cls(
                    path=str(path),
                    product=spec.name,
                    frame_id=int(m["frame_id"]),
                    reference_time=pd.Timestamp(
                        pd.to_datetime(m["reference"], format=DATETIME_FORMAT)
                    ),
                    secondary_time=pd.Timestamp(
                        pd.to_datetime(m["secondary"], format=DATETIME_FORMAT)
                    ),
                    version=m["version"],
                    production_time=pd.Timestamp(
                        pd.to_datetime(m["production"], format=DATETIME_FORMAT)
                    ),
                    polarization=m["polarization"],
                    mode=m["mode"],
                )
        return cls._from_metadata(path)

    @classmethod
    def _from_metadata(cls, path: Union[str, Path]) -> DispGranule:
        try:
            with open_h5(path) as f:
                title = _text(f.attrs.get("title", b""))
                ident = f["identification"]
                values = {
                    k: _text(ident[k][()])
                    for k in (
                        "frame_id",
                        "product_version",
                        "reference_datetime",
                        "secondary_datetime",
                    )
                }
        except (OSError, KeyError) as e:
            raise ValueError(
                f"{Path(path).name} is not an OPERA DISP file name and has no"
                " readable /identification group"
            ) from e
        if title not in _BY_TITLE:
            raise ValueError(f"{Path(path).name}: unknown product title {title!r}")
        return cls(
            path=str(path),
            product=_BY_TITLE[title].name,
            frame_id=int(values["frame_id"]),
            reference_time=pd.Timestamp(values["reference_datetime"]).floor("s"),
            secondary_time=pd.Timestamp(values["secondary_datetime"]).floor("s"),
            version=values["product_version"],
            production_time=None,
        )


def _text(value) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


def _expand(
    paths: Union[str, Path, Iterable[Union[str, Path]]],
) -> list[Union[str, Path]]:
    if isinstance(paths, (str, Path)):
        if is_remote_url(paths):
            url = str(paths)
            return [url] if url.endswith(".nc") else list(list_granules(url))
        p = Path(paths)
        return sorted(p.rglob("*.nc")) if p.is_dir() else [p]
    return [p if is_remote_url(p) else Path(p) for p in paths]


def find_products(
    paths: Union[str, Path, Iterable[Union[str, Path]]],
    *,
    frame_id: Optional[int] = None,
    min_version: Optional[str] = None,
    allow_unsupported: bool = False,
) -> pd.DataFrame:
    """Return one row per DISP granule, deduplicated and sorted by date.

    When the same frame and date pair appears more than once, the highest
    product version is kept, then the latest production time. The dropped rows
    are in ``df.attrs["dropped"]``. Versions are checked against the product
    spec after this deduplication.

    Parameters
    ----------
    paths : str, Path or iterable of them
        A directory or remote prefix (``s3://``, ``https://``, any fsspec URL;
        searched recursively for ``*.nc``), one file, or a list of files.
        Remote access needs the ``aws`` extra and credentials (see
        :mod:`disp_xr._remote`).
    frame_id : int, optional
        Keep only this frame. Required when the granules span several frames.
    min_version : str, optional
        Keep only granules with at least this product version.
    allow_unsupported : bool
        Keep granules whose version the product spec does not support, instead
        of raising.

    Returns
    -------
    pd.DataFrame
        Columns ``product``, ``frame_id``, ``reference_time``,
        ``secondary_time``, ``version``, ``production_time``, ``polarization``,
        ``mode``, ``path``.

    Raises
    ------
    ValueError
        If no granule is found, or the granules span several frames and
        ``frame_id`` is not given.
    UnsupportedProductVersionError
        If a granule's version is outside its product's supported versions.

    Examples
    --------
    >>> find_products([])
    Traceback (most recent call last):
    ...
    ValueError: No DISP granules found

    """
    granules = [DispGranule.from_path(p) for p in _expand(paths)]
    if not granules:
        raise ValueError("No DISP granules found")
    df = pd.DataFrame([asdict(g) for g in granules])

    if frame_id is not None:
        df = df[df["frame_id"] == frame_id]
    elif df["frame_id"].nunique() > 1:
        raise ValueError(
            f"Granules span frames {sorted(df['frame_id'].unique().tolist())};"
            " pass frame_id"
        )
    if min_version is not None:
        df = df[[Version(v) >= Version(min_version) for v in df["version"]]]

    key = ["product", "frame_id", "reference_time", "secondary_time"]
    ranked = df.assign(
        _version=[Version(v) for v in df["version"]],
        _production=df["production_time"].fillna(pd.Timestamp.min),
    ).sort_values([*key, "_version", "_production"])
    keep = ~ranked.duplicated(key, keep="last")
    dropped = ranked[~keep].drop(columns=["_version", "_production"])
    if len(dropped):
        logger.info(f"Dropped {len(dropped)} older duplicates")
    out = (
        ranked[keep]
        .drop(columns=["_version", "_production"])
        .sort_values(["secondary_time", "reference_time"])
        .reset_index(drop=True)
    )
    # Checked after deduplication: an old copy that a newer one replaces is fine.
    if not allow_unsupported:
        for product, version in out[["product", "version"]].drop_duplicates().values:
            supported = PRODUCTS[product].supported_versions
            if Version(version) not in supported:
                raise UnsupportedProductVersionError(
                    f"{product} version {version} is not supported"
                    f" (supported: {supported})"
                )
    out.attrs["dropped"] = dropped.reset_index(drop=True)
    return out


def group_ministacks(df: pd.DataFrame) -> dict[pd.Timestamp, pd.DataFrame]:
    """Split a :func:`find_products` table into ministacks, ordered by date.

    A ministack is the set of granules that share a reference date.
    """
    return {
        ref: group.sort_values("secondary_time").reset_index(drop=True)
        for ref, group in df.sort_values("reference_time").groupby(
            "reference_time", sort=True
        )
    }


def get_disp_info(
    products_path: Path | str, min_version: Optional[str] = None
) -> pd.DataFrame:
    """Return the granule table in the original disp-xr column layout.

    Built on :func:`find_products`, so duplicates resolve per date pair
    (highest version, then latest production) and ``min_version`` keeps
    versions at or above it.

    .. deprecated::
        Use :func:`find_products`, then ``stack_disp(open_disp(df))``.

    Parameters
    ----------
    products_path : Union[Path, str]
        Directory with OPERA DISP granules.
    min_version : str, optional
        Lowest product version to keep, for example ``"1.0"``.

    Returns
    -------
    pd.DataFrame
        One row per granule, sorted by ``date12``.

    """
    warnings.warn(
        "get_disp_info is deprecated; use find_products",
        DeprecationWarning,
        stacklevel=2,
    )
    df = find_products(products_path, min_version=min_version)
    logger.info(f"Found OPERA DISP: {len(df)} products")
    out = pd.DataFrame(
        {
            "project": "OPERA",
            "level": "L3",
            "product": df["product"],
            "mode": df["mode"],
            "frame_id": [f"F{f:05d}" for f in df["frame_id"]],
            "polarization": df["polarization"],
            "start_date": df["reference_time"],
            "end_date": df["secondary_time"],
            "version": "v" + df["version"],
            "production_date": [
                None if pd.isna(t) else t.strftime(DATETIME_FORMAT)
                for t in df["production_time"]
            ],
            "path": [Path(p) for p in df["path"]],
        }
    )
    out["date1"] = out["start_date"].dt.strftime("%Y%m%d")
    out["date2"] = out["end_date"].dt.strftime("%Y%m%d")
    out["date12"] = out["date1"] + "_" + out["date2"]
    logger.info(f" Number of reference dates: {out['date1'].nunique()}")
    return out.sort_values(by="date12", ignore_index=True)


def _layout(path: str, layers: tuple[str, ...]) -> tuple:
    with open_h5(path) as f:
        return tuple(
            (
                name,
                str(f[name].dtype),
                f[name].chunks,
                f[name].compression,
                f[name].compression_opts,
                f[name].shuffle,
            )
            for name in layers
            if name in f
        )


def check_granules(df: pd.DataFrame) -> pd.DataFrame:
    """Check that all granules share one dtype and encoding per layer.

    DISP productions differ: the 2026 DISP-S1 production stores displacement as
    float64, the 2025 one as float32. Such granules cannot be stacked or
    virtualized together, so they are refused rather than cast.

    One granule per production run is read (a run writes one layout), so the
    check costs a few file opens per frame.

    Parameters
    ----------
    df : pd.DataFrame
        Output of :func:`find_products`.

    Returns
    -------
    pd.DataFrame
        ``df`` with a ``layout`` column (an integer id; 0 when all agree).

    Raises
    ------
    MixedProductionError
        If more than one layout is found. The message lists each group with
        its displacement dtype, product versions and production dates, so the
        caller can select one production.

    """
    layers_by_product = {
        name: spec.displacement_layers + spec.quality_layers
        for name, spec in PRODUCTS.items()
    }
    # A production run writes one layout, so one file per (product, version,
    # production time) is enough. Files without a production time in the name
    # (SAS output) are each read.
    run = df["production_time"].astype("string").fillna(df["path"])
    sample = df.assign(_run=run).drop_duplicates(["product", "version", "_run"])
    layout_of_run = {
        (prod, ver, r): _layout(path, layers_by_product[prod])
        for prod, ver, r, path in sample[["product", "version", "_run", "path"]].values
    }
    layouts = [
        layout_of_run[(prod, ver, r)]
        for prod, ver, r in zip(df["product"], df["version"], run)
    ]
    ids = {layout: i for i, layout in enumerate(dict.fromkeys(layouts))}
    out = df.assign(layout=[ids[layout] for layout in layouts])
    if len(ids) > 1:
        lines = []
        for layout, i in ids.items():
            group = out[out["layout"] == i]
            dtype = {name: dt for name, dt, *_ in layout}.get("displacement")
            productions = sorted(
                {
                    "unknown" if pd.isna(t) else f"{t:%Y-%m-%d}"
                    for t in group["production_time"]
                }
            )
            lines.append(
                f"  {len(group)} granules: displacement {dtype}, versions"
                f" {sorted(set(group['version']))}, produced {productions}"
            )
        raise MixedProductionError(
            "Granules differ in dtype or encoding; select one production:\n"
            + "\n".join(lines)
        )
    return out
