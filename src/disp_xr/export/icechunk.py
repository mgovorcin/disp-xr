"""Write and read DISP datasets in Icechunk repositories (``cloud`` extra).

An Icechunk repository is a versioned Zarr v3 store: every write is a commit
with a snapshot id, and tags name snapshots that must not move (a Cal/Val
run). ``to_icechunk`` wraps :func:`icechunk.xarray.to_icechunk`, which, unlike
``ds.to_zarr`` on a session store, keeps writes made under dask.distributed or
multiprocessing.

Provenance (disp-xr and package versions, product, frame, input granules and
the history of operations) goes in the commit metadata as well as in the
dataset attributes. A stack built by appending one ministack per commit uses
the ``"ministack"`` chunk preset, so an append writes new shards instead of
rewriting old ones. Library code never amends a commit or expires snapshots.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Optional, Union

import numpy as np
import xarray as xr

from ..config import bounded_threads
from .zarr import (
    Preset,
    _prepare,
    _with_zarr_conventions,
    chunk_by_shard,
    zarr_encoding,
)

if TYPE_CHECKING:
    import icechunk

__all__ = ["open_icechunk", "to_icechunk"]

RepoLike = Union["icechunk.Repository", str, Path]


def _import_icechunk():
    try:
        import icechunk
        import icechunk.xarray
    except ImportError as e:
        raise ImportError(
            "Icechunk stores need the cloud extra: pip install 'disp_xr[cloud]'"
        ) from e
    return icechunk


def _repository(repo: RepoLike, create: bool) -> icechunk.Repository:
    """Return ``repo``, or open (``create``: or create) a local repository at it."""
    icechunk = _import_icechunk()
    if isinstance(repo, icechunk.Repository):
        return repo
    storage = icechunk.local_filesystem_storage(str(repo))
    if create:
        return icechunk.Repository.open_or_create(storage)
    return icechunk.Repository.open(storage)


def _read(session: icechunk.Session) -> Optional[xr.Dataset]:
    """Open the root group of ``session`` raw, or ``None`` if it has no data."""
    from zarr.errors import GroupNotFoundError

    try:
        return xr.open_zarr(
            session.store, consolidated=False, mask_and_scale=False, decode_coords="all"
        )
    except GroupNotFoundError:
        return None


def _check_append(ds: xr.Dataset, existing: xr.Dataset) -> None:
    """Raise ``ValueError`` unless ``ds`` extends ``existing`` along time."""
    if set(ds.data_vars) != set(existing.data_vars):
        raise ValueError(
            "Cannot append: variables differ. Store has "
            f"{sorted(existing.data_vars)}, dataset has {sorted(ds.data_vars)}"
        )
    for dim in ("y", "x"):
        if not np.array_equal(ds[dim].values, existing[dim].values):
            raise ValueError(
                f"Cannot append: the {dim} coordinate differs from the store"
            )
    if ds.rio.crs != existing.rio.crs:
        raise ValueError(
            f"Cannot append: CRS {ds.rio.crs} differs from the store's "
            f"{existing.rio.crs}"
        )
    for name, var in ds.data_vars.items():
        if var.dtype != existing[name].dtype:
            raise ValueError(
                f"Cannot append: {name} is {var.dtype}, the store has "
                f"{existing[name].dtype}"
            )
    last = existing["time"].values.max()
    if ds["time"].values.min() <= last:
        raise ValueError(
            f"Cannot append: the dataset starts at {ds['time'].values.min()}, not "
            f"after the store's last epoch {last}. Replace epochs with a region "
            "write instead."
        )


def _merged_granules(new: str, existing: xr.Dataset) -> str:
    """Return the JSON list of granules in ``existing`` followed by ``new`` ones."""
    old = json.loads(existing.attrs.get("input_granules", "[]"))
    return json.dumps(old + [g for g in json.loads(new) if g not in old])


COMMIT_KEYS = (
    "disp_xr_version",
    "software_versions",
    "input_granules",
    "history",
    "product",
    "frame_id",
    "product_versions",
    "source_snapshot_id",
)
JSON_KEYS = ("software_versions", "input_granules", "product_versions")


def _commit_metadata(ds: xr.Dataset) -> dict[str, Any]:
    """Return the provenance attrs of a prepared ``ds``, JSON decoded."""
    return {
        key: json.loads(ds.attrs[key]) if key in JSON_KEYS else ds.attrs[key]
        for key in COMMIT_KEYS
        if key in ds.attrs
    }


@bounded_threads
def to_icechunk(
    ds: xr.Dataset,
    repo: RepoLike,
    *,
    branch: str = "main",
    message: Optional[str] = None,
    append_dim: Optional[str] = None,
    tag: Optional[str] = None,
    preset: Optional[Preset] = None,
    mode: Literal["w-", "w"] = "w-",
    keep_bits: Optional[int] = None,
    compression_level: int = 3,
    target_mb: Optional[float] = None,
) -> str:
    """Write ``ds`` to an Icechunk repository as one commit.

    Parameters
    ----------
    ds : xr.Dataset
        DISP dataset with ``(time, y, x)`` and ``(y, x)`` variables.
    repo : icechunk.Repository, str or Path
        Repository, or a local directory (created if missing).
    branch : str
        Branch to commit to. A missing branch is created from ``main``.
    message : str, optional
        Commit message; defaults to a summary of the write.
    append_dim : {"time"}, optional
        Append along time. The dataset must be on the store's grid, with the
        same variables and dtypes, and start after the store's last epoch. On
        an empty branch this creates the store with the ``"ministack"`` layout.
    tag : str, optional
        Tag the new snapshot (for example a Cal/Val run). Tags never move.
    preset : {"timeseries", "maps", "ministack"}, optional
        Chunk layout of a new store; ignored when appending to an existing
        one. Defaults to ``"timeseries"``, or ``"ministack"`` with
        ``append_dim``. ``"timeseries"`` cannot be appended to, since each
        shard spans the whole time axis and every append would rewrite it.
    mode : {"w-", "w"}
        Without ``append_dim``: fail if the branch has data (default), or
        replace it in a new commit (older snapshots stay readable).
    keep_bits, compression_level, target_mb
        As in :func:`disp_xr.to_zarr`.

    Returns
    -------
    str
        Snapshot id of the commit. Record it in downstream provenance;
        :func:`open_icechunk` does so as ``attrs["source_snapshot_id"]``.

    Examples
    --------
    >>> dxr.to_icechunk(ds, "results/", message="first epochs")  # doctest: +SKIP
    >>> dxr.to_icechunk(
    ...     new_epochs, "results/", append_dim="time", tag="v1"
    ... )  # doctest: +SKIP

    """
    icechunk = _import_icechunk()
    if append_dim not in (None, "time"):
        raise ValueError(f"Only appends along 'time' are supported, not {append_dim!r}")
    if append_dim and preset == "timeseries":
        raise ValueError(
            "The 'timeseries' preset cannot be appended to: every append would "
            "rewrite every shard. Use 'ministack' (the default) or 'maps'."
        )
    repository = _repository(repo, create=True)
    if branch not in repository.list_branches():
        repository.create_branch(branch, repository.lookup_branch("main"))
    existing = _read(repository.readonly_session(branch=branch))
    layout = preset or ("ministack" if append_dim else "timeseries")

    out, _ = _prepare(ds, layout, keep_bits, target_mb)
    out = _with_zarr_conventions(out)
    session = repository.writable_session(branch)
    appending = append_dim is not None and existing is not None
    if append_dim is not None and existing is not None:
        _check_append(out, existing)
        out.attrs["input_granules"] = _merged_granules(
            out.attrs["input_granules"], existing
        )
        # The layout is fixed by the first write: align dask chunks to the
        # store's shards (align_chunks fills a partial last time shard).
        stored = next(iter(existing.data_vars.values()))
        out = out.chunk(
            {
                d: s
                for d, s in zip(stored.dims, stored.encoding["shards"])
                if d != "time"
            }
        )
        for var in out.data_vars.values():
            var.encoding.pop("_FillValue", None)
        icechunk.xarray.to_icechunk(
            out, session, append_dim=append_dim, align_chunks=True
        )
    else:
        encoding = zarr_encoding(
            out, layout, compression_level=compression_level, target_mb=target_mb
        )
        for var in out.data_vars.values():
            var.encoding.pop("_FillValue", None)
        icechunk.xarray.to_icechunk(
            out, session, mode=mode if existing is not None else "w-", encoding=encoding
        )

    if message is None:
        times = out["time"].values
        verb = "Append" if appending else "Write"
        message = (
            f"{verb} {out.sizes['time']} epochs "
            f"({np.datetime_as_string(times.min(), 'D')} to "
            f"{np.datetime_as_string(times.max(), 'D')})"
        )
    snapshot_id = session.commit(message, metadata=_commit_metadata(out))
    if tag is not None:
        repository.create_tag(tag, snapshot_id)
    return snapshot_id


def open_icechunk(
    repo: RepoLike,
    *,
    branch: Optional[str] = None,
    tag: Optional[str] = None,
    snapshot_id: Optional[str] = None,
    chunks: Any = "shards",
    mask_and_scale: bool = False,
) -> xr.Dataset:
    """Open a DISP dataset from an Icechunk repository, read-only.

    Give at most one of ``branch`` (default ``"main"``, its latest commit),
    ``tag`` or ``snapshot_id``. The snapshot read is recorded in
    ``attrs["source_snapshot_id"]``, which exports carry as provenance.

    Parameters
    ----------
    repo : icechunk.Repository, str or Path
        Repository, or a local directory holding one.
    branch, tag, snapshot_id : str, optional
        Version to read.
    chunks : "shards" or any :func:`xarray.open_zarr` value
        ``"shards"`` (default): one dask chunk per shard, as
        :func:`disp_xr.open_zarr` (xarray's own default is one chunk per
        64 x 64 read chunk; ``None`` gives no dask at all, so any
        computation loads the whole store). Anything else is passed on.
    mask_and_scale : bool
        Decode fill values to NaN. Off by default, so integer layers keep
        their dtype, as :func:`disp_xr.open_disp` does.

    Examples
    --------
    >>> ds = dxr.open_icechunk("results/", tag="v1")  # doctest: +SKIP
    >>> earlier = dxr.open_icechunk("results/", snapshot_id=sid)  # doctest: +SKIP

    """
    given = [v for v in (branch, tag, snapshot_id) if v is not None]
    if len(given) > 1:
        raise ValueError("Give at most one of branch, tag or snapshot_id")
    repository = _repository(repo, create=False)
    if not given:
        branch = "main"
    session = repository.readonly_session(
        branch=branch, tag=tag, snapshot_id=snapshot_id
    )
    by_shard = isinstance(chunks, str) and chunks == "shards"
    ds = xr.open_zarr(
        session.store,
        consolidated=False,
        chunks=None if by_shard else chunks,
        mask_and_scale=mask_and_scale,
        decode_coords="all",
    )
    if by_shard:
        ds = chunk_by_shard(ds)
    ds.attrs["source_snapshot_id"] = session.snapshot_id
    return ds
