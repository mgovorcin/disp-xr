"""Virtual DISP frames: Icechunk stores that point at the granules (``cloud`` extra).

A virtual store holds no pixels. Each layer chunk is a reference to a byte
range in a granule, read through VirtualiZarr's HDF parser, so a whole frame
opens as one Zarr dataset without copying data. ``time``, ``reference_time``,
``x``, ``y`` and ``spatial_ref`` are loaded into the store, because their
encoding differs between granules; the ``/identification`` and ``/metadata``
groups are not read.

Displacement stays relative to each ministack reference, as in the granules:
chain and re-reference on top of :func:`open_virtual`. A virtual store is an
index for ingest and exploration; for per-pixel work write a native store
(:func:`disp_xr.to_zarr` with the ``"timeseries"`` preset).

The store registers a virtual chunk container for the granule location (the
S3 bucket, or the local directory). Credentials for it are given when the
repository is opened and are never saved with it.
"""

from __future__ import annotations

import json
import logging
import multiprocessing
import os
from collections.abc import Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Union
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import xarray as xr

from ._credentials import AWSCredentials
from ._remote import is_remote_url, open_h5
from .config import bounded_threads
from .export._provenance import with_provenance
from .export.icechunk import (
    _check_append,
    _commit_metadata,
    _import_icechunk,
    _read,
    open_icechunk,
)
from .product import PRODUCTS, check_granules, find_products, group_ministacks

if TYPE_CHECKING:
    import icechunk

logger = logging.getLogger(__name__)

__all__ = ["build_virtual_frame", "open_virtual", "validate_virtual"]

# Loaded into the store: small, and their encoding differs between granules.
LOADED = ("time", "reference_time", "x", "y", "spatial_ref")
# Region of the ASF buckets that hold OPERA DISP granules.
S3_REGION = "us-west-2"

GranulesLike = Union[str, Path, Iterable[Union[str, Path]], pd.DataFrame]
RepoLike = Union["icechunk.Repository", "icechunk.Storage", str, Path]


def _products(granules: GranulesLike) -> pd.DataFrame:
    return granules if isinstance(granules, pd.DataFrame) else find_products(granules)


def _url(path: Union[str, Path]) -> str:
    """Return the URL of a granule, ``file://`` for local paths."""
    if is_remote_url(path):
        return str(path)
    return Path(path).resolve().as_uri()


def _container_prefix(urls: Sequence[str]) -> str:
    """Return the container prefix: the S3 bucket, or the common local directory."""
    schemes = {urlparse(u).scheme for u in urls}
    if len(schemes) != 1:
        raise ValueError(f"Granules mix URL schemes: {sorted(schemes)}")
    (scheme,) = schemes
    if scheme == "s3":
        buckets = {urlparse(u).netloc for u in urls}
        if len(buckets) != 1:
            raise ValueError(f"Granules span several buckets: {sorted(buckets)}")
        return f"s3://{buckets.pop()}/"
    if scheme != "file":
        raise ValueError(
            f"Virtual stores support s3:// and local granules, not {scheme}"
        )
    parent = Path(os.path.commonpath([urlparse(u).path for u in urls]))
    if len(urls) == 1:
        parent = parent.parent
    return parent.as_uri().rstrip("/") + "/"


def _aws(credentials: Optional[AWSCredentials]) -> AWSCredentials:
    if credentials is not None:
        return credentials
    try:
        return AWSCredentials.from_env()
    except KeyError:
        return AWSCredentials.from_asf()


def _chunk_access(prefix: str, credentials: Optional[AWSCredentials]) -> Any:
    """Return the Icechunk credentials that authorize reads under ``prefix``."""
    icechunk = _import_icechunk()
    if prefix.startswith("file://"):
        return icechunk.Credentials.LocalFileSystemAccess()
    creds = _aws(credentials)
    return icechunk.s3_credentials(
        access_key_id=creds.access_key_id,
        secret_access_key=creds.secret_access_key,
        session_token=creds.session_token,
    )


def _container(prefix: str) -> Any:
    icechunk = _import_icechunk()
    if prefix.startswith("file://"):
        store = icechunk.local_filesystem_store(urlparse(prefix).path)
    else:
        store = icechunk.s3_store(region=S3_REGION)
    return icechunk.VirtualChunkContainer(prefix, store)


def _storage(repo: Union[icechunk.Storage, str, Path]) -> icechunk.Storage:
    icechunk = _import_icechunk()
    if isinstance(repo, icechunk.Storage):
        return repo
    return icechunk.local_filesystem_storage(str(repo))


def _open_repository(
    repo: RepoLike,
    *,
    prefix: Optional[str] = None,
    credentials: Optional[AWSCredentials] = None,
) -> icechunk.Repository:
    """Open (with ``prefix``: or create) a repository with virtual chunk access.

    A :class:`icechunk.Repository` is returned as given: the caller opened it
    and authorized its containers. Otherwise every container in the stored
    config is authorized, and with ``prefix`` a container for it is added (and
    saved) if missing.
    """
    icechunk = _import_icechunk()
    if isinstance(repo, icechunk.Repository):
        return repo
    storage = _storage(repo)
    exists = icechunk.Repository.exists(storage)
    config = (
        icechunk.Repository.fetch_config(storage) if exists else None
    ) or icechunk.RepositoryConfig.default()
    containers = config.virtual_chunk_containers or {}
    prefixes = set(containers)
    added = prefix is not None and prefix not in containers
    if prefix is not None and added:
        config.set_virtual_chunk_container(_container(prefix))
        prefixes.add(prefix)
    access = {p: _chunk_access(p, credentials) for p in prefixes}
    if not exists:
        if prefix is None:
            raise FileNotFoundError(f"No Icechunk repository at {repo}")
        return icechunk.Repository.create(
            storage, config, authorize_virtual_chunk_access=access
        )
    repository = icechunk.Repository.open(
        storage, config, authorize_virtual_chunk_access=access
    )
    if added:
        repository.save_config()
    return repository


def _check_container(repository: icechunk.Repository, prefix: str) -> None:
    containers = repository.config.virtual_chunk_containers or {}
    if not any(prefix.startswith(p) for p in containers):
        raise ValueError(
            f"The repository has no virtual chunk container for {prefix}. Pass a "
            "path or icechunk.Storage instead, so one is registered, or add "
            f"icechunk.VirtualChunkContainer({prefix!r}, ...) to its config."
        )


def _open_granule(
    url: str, drop: list[str], credentials: Optional[dict[str, Optional[str]]]
) -> xr.Dataset:
    """Open one granule as a virtual dataset (runs in a worker process)."""
    from obspec_utils.registry import ObjectStoreRegistry
    from virtualizarr import open_virtual_dataset
    from virtualizarr.parsers import HDFParser

    parsed = urlparse(url)
    if parsed.scheme == "s3":
        from obstore.store import S3Store

        store: Any = S3Store(parsed.netloc, region=S3_REGION, **(credentials or {}))
        root = f"s3://{parsed.netloc}"
    else:
        from obstore.store import LocalStore

        store, root = LocalStore(), "file://"
    vds = open_virtual_dataset(
        url,
        registry=ObjectStoreRegistry({root: store}),
        parser=HDFParser(),
        drop_variables=drop,
        loadable_variables=list(LOADED),
    )
    vds = vds.set_coords(["reference_time", "spatial_ref"])
    for var in vds.data_vars.values():
        var.attrs["grid_mapping"] = "spatial_ref"
    return vds


def _open_granules(
    urls: list[str],
    drop: list[str],
    credentials: Optional[AWSCredentials],
    max_workers: Optional[int],
) -> list[xr.Dataset]:
    creds = None
    if urls and urls[0].startswith("s3://"):
        aws = _aws(credentials)
        creds = {
            "access_key_id": aws.access_key_id,
            "secret_access_key": aws.secret_access_key,
            "session_token": aws.session_token,
        }
    workers = max_workers or min(len(urls), os.cpu_count() or 1)
    if workers <= 1:
        return [_open_granule(u, drop, creds) for u in urls]
    # The HDF parser holds the HDF5 lock, so threads would run one at a time.
    # "spawn" because forking a process that has h5py loaded is unsafe.
    with ProcessPoolExecutor(
        workers, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        return list(
            pool.map(_open_granule, urls, [drop] * len(urls), [creds] * len(urls))
        )


@bounded_threads
def build_virtual_frame(
    granules: GranulesLike,
    repo: RepoLike,
    *,
    branch: str = "main",
    layers: Optional[Sequence[str]] = None,
    max_workers: Optional[int] = None,
    credentials: Optional[AWSCredentials] = None,
) -> list[str]:
    """Write a frame of granules to an Icechunk repository as virtual references.

    One commit per ministack, in reference-date order. A branch that already
    holds a frame is extended: every new ministack must start after its last
    epoch and be on the same grid, with the same layers and dtypes.

    Parameters
    ----------
    granules : str, Path, iterable of paths, or DataFrame
        Local or ``s3://`` granules of one frame and one production, or the
        output of :func:`disp_xr.find_products`.
    repo : icechunk.Repository, icechunk.Storage, str or Path
        Repository to write to. For a storage or a local path, the repository
        is created if needed and a virtual chunk container for the granule
        location is registered. A ``Repository`` must already have one and be
        opened with access to it.
    branch : str
        Branch to commit to.
    layers : sequence of str, optional
        Layers to reference. Defaults to all displacement and quality layers.
    max_workers : int, optional
        Processes that parse granules; ``1`` parses in this process.
    credentials : AWSCredentials, optional
        For ``s3://`` granules. Defaults to ``AWS_*`` environment variables,
        then ASF temporary credentials.

    Returns
    -------
    list of str
        Snapshot id of each commit, one per ministack.

    Raises
    ------
    MixedProductionError
        If the granules mix productions (dtype or encoding differ).
    ValueError
        If a ministack is on another grid or does not follow the branch's
        last epoch.

    Examples
    --------
    >>> snapshots = dxr.build_virtual_frame(
    ...     "F11116/", "F11116-virtual/"
    ... )  # doctest: +SKIP
    >>> dxr.build_virtual_frame(new_granules, "F11116-virtual/")  # doctest: +SKIP

    """
    df = _products(granules)
    check_granules(df)
    (product,) = df["product"].unique()
    spec = PRODUCTS[product]
    available = spec.displacement_layers + spec.quality_layers
    keep = tuple(layers) if layers is not None else available
    unknown = set(keep) - set(available)
    if unknown:
        raise ValueError(f"Unknown {product} layers: {sorted(unknown)}")
    drop = [v for v in available if v not in keep]

    urls = [_url(p) for p in df["path"]]
    prefix = _container_prefix(urls)
    repository = _open_repository(repo, prefix=prefix, credentials=credentials)
    _check_container(repository, prefix)
    if branch not in repository.list_branches():
        repository.create_branch(branch, repository.lookup_branch("main"))

    by_url = dict(zip(urls, _open_granules(urls, drop, credentials, max_workers)))
    existing = _read(repository.readonly_session(branch=branch))
    granted = (
        json.loads(existing.attrs["input_granules"]) if existing is not None else []
    )
    snapshots = []
    for ref, group in group_ministacks(df).items():
        names = [Path(str(p)).name for p in group["path"]]
        ds = xr.concat(
            [by_url[_url(p)] for p in group["path"]],
            dim="time",
            data_vars="all",
            coords="minimal",
            compat="override",
            join="exact",
        )
        ds.attrs = {
            "product": product,
            "frame_id": int(group["frame_id"].iloc[0]),
            "product_versions": sorted(set(group["version"])),
            "input_granules": granted + names,
            "virtual": "true",
        }
        out = with_provenance(ds)
        session = repository.writable_session(branch)
        if existing is None:
            out.vz.to_icechunk(session.store)
        else:
            _check_append(out, existing)
            out.vz.to_icechunk(session.store, append_dim="time")
        meta = _commit_metadata(out)
        meta["input_granules"] = names
        meta["virtual"] = True
        message = (
            f"Reference {len(names)} granules of ministack "
            f"{pd.Timestamp(ref):%Y-%m-%d}"
        )
        snapshots.append(session.commit(message, metadata=meta))
        granted += names
        existing = _read(repository.readonly_session(branch=branch))
        logger.info(f"{message} ({snapshots[-1]})")
    return snapshots


def open_virtual(
    repo: RepoLike,
    *,
    branch: Optional[str] = None,
    tag: Optional[str] = None,
    snapshot_id: Optional[str] = None,
    chunks: Any = None,
    credentials: Optional[AWSCredentials] = None,
) -> xr.Dataset:
    """Open a virtual frame read-only, with access to its granules.

    As :func:`disp_xr.open_icechunk`, but the repository is opened with read
    access to every virtual chunk container in its config: local directories,
    or S3 with ``credentials`` (default: environment, then ASF).

    Examples
    --------
    >>> ds = dxr.open_virtual("F11116-virtual/")  # doctest: +SKIP
    >>> before = dxr.open_virtual("F11116-virtual/", snapshot_id=sid)  # doctest: +SKIP

    """
    repository = _open_repository(repo, credentials=credentials)
    return open_icechunk(
        repository, branch=branch, tag=tag, snapshot_id=snapshot_id, chunks=chunks
    )


def _same(a: np.ndarray, b: np.ndarray) -> bool:
    return bool(np.array_equal(a, b, equal_nan=a.dtype.kind == "f"))


@bounded_threads
def validate_virtual(
    repo: RepoLike,
    granules: GranulesLike,
    *,
    n_samples: int = 10,
    seed: int = 0,
    branch: str = "main",
    credentials: Optional[AWSCredentials] = None,
) -> dict[str, Any]:
    """Check a virtual frame against the granules, read with h5py.

    Compares the time axis, ``n_samples`` random pixel series of every layer,
    and one random epoch in full.

    Returns
    -------
    dict
        ``n_granules``, ``pixels`` (``(row, col)`` pairs), ``epoch`` and
        ``layers`` checked.

    Raises
    ------
    ValueError
        If any value or the time axis differs, naming what differs.

    Examples
    --------
    >>> report = dxr.validate_virtual("F11116-virtual/", "F11116/")  # doctest: +SKIP
    >>> report["n_granules"]  # doctest: +SKIP

    """
    df = _products(granules)
    ds = open_virtual(repo, branch=branch, credentials=credentials)
    times = pd.DatetimeIndex(df["secondary_time"]).tz_localize(None).values
    if ds.sizes["time"] != len(df) or not np.array_equal(
        ds["time"].values.astype("datetime64[s]"), times.astype("datetime64[s]")
    ):
        raise ValueError(
            f"The store has {ds.sizes['time']} epochs; the granules have {len(df)} "
            "or different dates"
        )
    rng = np.random.default_rng(seed)
    rows = rng.integers(0, ds.sizes["y"], n_samples)
    cols = rng.integers(0, ds.sizes["x"], n_samples)
    epoch = int(rng.integers(0, len(df)))
    layers = [str(v) for v in ds.data_vars]

    try:
        pixels = {
            name: (
                ds[name]
                .isel(
                    y=xr.DataArray(rows, dims="pixel"),
                    x=xr.DataArray(cols, dims="pixel"),
                )
                .values
            )
            for name in layers
        }
        full = {name: ds[name].isel(time=epoch).values for name in layers}
    except Exception as e:
        # A granule rewritten after the build moves its chunks, so references
        # land on other bytes and fail to decode: the store is out of date.
        raise ValueError(
            "Virtual store differs from the granules: its references cannot be "
            f"read ({type(e).__name__}: {e}). Were granules replaced after the build?"
        ) from e
    bad = []
    for i, path in enumerate(df["path"]):
        with open_h5(path) as f:
            for name in layers:
                truth = np.array([f[name][r, c] for r, c in zip(rows, cols)])
                if not _same(pixels[name][i], truth):
                    bad.append(f"{name} pixel series at {Path(str(path)).name}")
                if i == epoch and not _same(full[name], f[name][()]):
                    bad.append(f"{name} full epoch {Path(str(path)).name}")
    if bad:
        raise ValueError("Virtual store differs from the granules: " + "; ".join(bad))
    return {
        "n_granules": len(df),
        "pixels": list(zip(rows.tolist(), cols.tolist())),
        "epoch": str(ds["time"].values[epoch]),
        "layers": layers,
    }
