"""A DISP frame as one Icechunk repository: virtual granules plus chain offsets.

A virtual frame (:func:`disp_xr.build_virtual_frame`) references the granules
as they are: displacement relative to each ministack's reference date, and,
when the frame mixes productions that store layers in different dtypes, one
group per production (``"production/<name>"``). :func:`add_chain_offsets`
stores, once, the arithmetic :func:`disp_xr.chain` would redo on every read:
for each ministack, the chained displacement at its reference date (an
*offset* map), as a native float32 group ``chain`` of the same repository.
:func:`open_frame` then returns the continuous series, ``raw + offset`` of
each epoch's ministack with the zero epoch at the start date, equal to
``chain(open_disp(granules))`` bit for bit.

Every epoch reads two maps (its own and its ministack's offset) instead of
one plus a crossover per earlier ministack, and the ``chain`` group lets
plain xarray users rebase without disp-xr::

    continuous = raw + chain.displacement_offset.sel(ministack=raw.reference_time)

Offsets are computed from the unmasked granules, so masks still apply after
opening, and ``nan_policy`` (``chain``'s) is fixed per repository and
recorded in the group's attributes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

import numpy as np
import pandas as pd
import xarray as xr

from ._history import record
from .stack import DISPLACEMENT_LAYERS, NaNPolicy, _prepend_zero_epoch

if TYPE_CHECKING:
    from .virtual import AWSCredentials, RepoLike

__all__ = ["add_chain_offsets", "open_frame"]

CHAIN_GROUP = "chain"
PRODUCTION_PREFIX = "production"
# One offset map per chunk in time. The shard is the unit of writing (one dask
# chunk each, so parallel writes never share a shard); the chunk inside it is
# the unit of reading.
OFFSET_READ_CHUNK = (1, 1024, 1024)
OFFSET_SHARD = (1, 2048, 2048)


def _frame_groups(session: Any) -> list[Optional[str]]:
    """Return the groups holding granules: every ``production/*`` group, else root."""
    import zarr

    root = zarr.open_group(session.store, mode="r")
    if PRODUCTION_PREFIX in root.group_keys():
        children = sorted(root[PRODUCTION_PREFIX].group_keys())
        return [f"{PRODUCTION_PREFIX}/{name}" for name in children]
    return [None]


def _open_raw(
    repo: RepoLike,
    *,
    branch: Optional[str],
    tag: Optional[str],
    snapshot_id: Optional[str],
    credentials: Optional[AWSCredentials],
    chunks: Any,
) -> xr.Dataset:
    """Every production in time order, mixed float layers cast to float32."""
    from .virtual import _open_repository, open_virtual

    repository = _open_repository(repo, credentials=credentials)
    session = repository.readonly_session(
        branch=None if (tag or snapshot_id) else (branch or "main"),
        tag=tag,
        snapshot_id=snapshot_id,
    )
    version = {"snapshot_id": session.snapshot_id}
    parts = [
        open_virtual(repo, chunks=chunks, credentials=credentials, group=g, **version)
        for g in _frame_groups(session)
    ]
    parts.sort(key=lambda ds: ds["time"].values[0])
    # As open_disp does for mixed granules: a float layer stored at several
    # precisions is read as float32.
    mixed = sorted(
        name
        for name in parts[0].data_vars
        if np.issubdtype(parts[0][name].dtype, np.floating)
        and len({p[name].dtype for p in parts if name in p}) > 1
    )
    for part in parts:
        for name in mixed:
            part[name] = part[name].astype(np.float32)
    raw = (
        parts[0]
        if len(parts) == 1
        else xr.concat(
            parts,
            dim="time",
            data_vars="all",
            coords="minimal",
            compat="override",
            join="exact",
        )
    )
    if mixed:
        raw.attrs["mixed_precision"] = mixed
    raw.attrs["input_granules"] = [
        g for p in parts for g in _granule_names(p.attrs.get("input_granules"))
    ]
    raw.attrs["source_snapshot_id"] = session.snapshot_id
    return raw


def _granule_names(value: Any) -> list[str]:
    import json

    if value is None:
        return []
    return list(json.loads(value)) if isinstance(value, str) else list(value)


def add_chain_offsets(
    repo: RepoLike,
    *,
    branch: str = "main",
    nan_policy: str = "propagate",
    message: Optional[str] = None,
    tag: Optional[str] = None,
    credentials: Optional[AWSCredentials] = None,
) -> Optional[str]:
    """Store the chain offsets of a virtual frame, or extend them, in one commit.

    For each ministack, the chained displacement at its reference date (the
    first ministack's is zero), for every displacement layer, from the
    unmasked granules of all production groups in time order. Ministacks
    already stored are kept: after new granules are added with
    :func:`disp_xr.build_virtual_frame`, a call computes only the new ones,
    from the stored offsets (one crossover map read per new ministack).

    Parameters
    ----------
    repo : icechunk.Repository, icechunk.Storage, str or Path
        Repository of a virtual frame.
    branch : str
        Branch to read and commit to.
    nan_policy : {"propagate", "omit"}
        As in :func:`disp_xr.chain`; fixed by the first call.
    message : str, optional
        Commit message; defaults to a summary.
    tag : str, optional
        Tag the new snapshot. Tags never move.
    credentials : AWSCredentials, optional
        For granules on S3, as in :func:`disp_xr.open_virtual`.

    Returns
    -------
    str or None
        Snapshot id of the commit, or ``None`` if every ministack already had
        its offsets.

    Examples
    --------
    >>> dxr.build_virtual_frame(granules, "F11116-frame/")  # doctest: +SKIP
    >>> dxr.add_chain_offsets("F11116-frame/", tag="v1")  # doctest: +SKIP

    """
    import icechunk
    import icechunk.xarray

    from .export.icechunk import _read
    from .virtual import _open_repository

    policy = NaNPolicy(nan_policy)
    raw = _open_raw(
        repo, branch=branch, tag=None, snapshot_id=None, credentials=credentials,
        chunks="native",
    )  # fmt: skip
    layers = [name for name in DISPLACEMENT_LAYERS if name in raw]
    times = raw["time"].values
    refs = raw["reference_time"].values
    order = list(np.unique(refs))
    repository = _open_repository(repo, credentials=credentials)
    stored = _read(repository.readonly_session(branch=branch), CHAIN_GROUP)
    have: list = []
    if stored is not None:
        if stored.attrs.get("nan_policy") != policy.value:
            raise ValueError(
                f"The stored offsets use nan_policy={stored.attrs.get('nan_policy')!r},"
                f" not {policy.value!r}"
            )
        have = list(stored["ministack"].values)
        if have != order[: len(have)]:
            raise ValueError(
                "The frame's ministacks no longer start with the stored ones "
                "(granules replaced?); rebuild the offsets on a new branch"
            )
    new_refs = order[len(have) :]
    if not new_refs:
        return None

    session = repository.writable_session(branch)
    # One ministack at a time, keeping only the latest offset map in memory:
    # an offset is its crossover epoch plus the offset of the ministack that
    # epoch belongs to (usually the one just written; else read back).
    latest: dict[str, tuple] = {}

    def base(name: str, ref: Any) -> np.ndarray:
        if name in latest and latest[name][0] == ref:
            return latest[name][1]
        group = xr.open_zarr(
            session.store, group=CHAIN_GROUP, consolidated=False, chunks=None
        )
        return np.asarray(group[f"{name}_offset"].sel(ministack=ref).values)

    # Every write carries the group attributes: an append replaces them.
    group_attrs = {
        "chain_reference_date": str(pd.Timestamp(order[0])),
        "nan_policy": policy.value,
        "usage": "continuous = raw + <layer>_offset.sel(ministack=raw.reference_time)",
    }
    written = stored is not None
    for ref in new_refs:
        variables = {}
        for name in layers:
            if ref == order[0]:  # chained values start at zero
                value = np.zeros(raw[name].shape[1:], dtype=raw[name].dtype)
            else:
                (i,) = np.flatnonzero(times == ref)[:1]
                step = np.asarray(raw[name].isel(time=i).values)
                if policy == NaNPolicy.omit:
                    step = np.nan_to_num(step, nan=0.0)
                value = (step + base(name, refs[i])).astype(raw[name].dtype)
            latest[name] = (ref, value)
            variables[f"{name}_offset"] = xr.Variable(
                ("ministack", "y", "x"),
                value[np.newaxis],
                attrs={
                    "units": raw[name].attrs.get("units", "meters"),
                    "long_name": f"chain offset of {name} for each ministack",
                },
            )
        ds = xr.Dataset(variables, coords={"ministack": [ref]}, attrs=group_attrs)
        ds = ds.chunk(dict(zip(("ministack", "y", "x"), OFFSET_SHARD)))
        if not written:
            ds = ds.assign_coords(y=raw["y"].values, x=raw["x"].values)
            for dim in ("y", "x"):
                attrs = raw[dim].attrs.items()
                ds[dim].attrs = {k: v for k, v in attrs if k != "_FillValue"}
            encoding: dict[Any, dict[str, Any]] = {
                v: {"chunks": OFFSET_READ_CHUNK, "shards": OFFSET_SHARD}
                for v in ds.data_vars
            }
            # Whole seconds: xarray would pick units from the first date
            # ("days since" it), which later reference dates (other times of
            # day) do not fit when appended.
            encoding["ministack"] = {
                "units": "seconds since 1970-01-01",
                "dtype": "int64",
            }
            icechunk.xarray.to_icechunk(
                ds, session, group=CHAIN_GROUP, mode="w-", encoding=encoding
            )
            written = True
        else:
            icechunk.xarray.to_icechunk(
                ds, session, group=CHAIN_GROUP, append_dim="ministack"
            )
    dates = f"{pd.Timestamp(new_refs[0]):%Y-%m-%d}"
    if len(new_refs) > 1:
        dates += f" to {pd.Timestamp(new_refs[-1]):%Y-%m-%d}"
    snapshot = session.commit(
        message or f"Chain offsets of {len(new_refs)} ministacks ({dates})",
        metadata={"chain_ministacks": [str(r) for r in new_refs]},
    )
    if tag is not None:
        repository.create_tag(tag, snapshot)
    return snapshot


def open_frame(
    repo: RepoLike,
    *,
    branch: Optional[str] = None,
    tag: Optional[str] = None,
    snapshot_id: Optional[str] = None,
    zero_epoch: bool = True,
    chunks: Any = "native",
    credentials: Optional[AWSCredentials] = None,
) -> xr.Dataset:
    """Open a frame repository as one continuous, chained series.

    The production groups in time order, plus the stored chain offsets
    (:func:`add_chain_offsets`) of each epoch's ministack: equal to
    ``chain(open_disp(granules), nan_policy=...)``, lazily, with every epoch
    read as itself and one offset map.

    Parameters
    ----------
    repo : icechunk.Repository, icechunk.Storage, str or Path
        Frame repository.
    branch, tag, snapshot_id : str, optional
        Version to read (default: the latest commit of ``main``). Give a tag
        for reproducible work.
    zero_epoch : bool
        Prepend the start date as an epoch of zero displacement, as
        :func:`disp_xr.chain` does.
    chunks : optional
        As in :func:`disp_xr.open_virtual`.
    credentials : AWSCredentials, optional
        For granules on S3.

    Returns
    -------
    xr.Dataset
        Chained displacement layers and the quality layers; ``attrs``
        ``reference_date``, ``nan_policy`` and ``source_snapshot_id``.

    Raises
    ------
    ValueError
        If the repository has no chain offsets, or not for every ministack.

    Examples
    --------
    >>> ds = dxr.open_frame("F11116-frame/", tag="v1")  # doctest: +SKIP
    >>> ds.displacement.isel(time=-1).plot()  # doctest: +SKIP

    """
    from zarr.errors import GroupNotFoundError

    from .export.zarr import chunk_by_shard
    from .virtual import _open_repository

    raw = _open_raw(
        repo, branch=branch, tag=tag, snapshot_id=snapshot_id,
        credentials=credentials, chunks=chunks,
    )  # fmt: skip
    repository = _open_repository(repo, credentials=credentials)
    session = repository.readonly_session(snapshot_id=raw.attrs["source_snapshot_id"])
    try:
        chain = xr.open_zarr(
            session.store, group=CHAIN_GROUP, consolidated=False, chunks=None
        )
    except (GroupNotFoundError, FileNotFoundError, KeyError) as error:
        raise ValueError(
            "This repository has no chain offsets; run add_chain_offsets"
        ) from error
    chain = chunk_by_shard(chain)
    missing = set(np.unique(raw["reference_time"].values)) - set(
        chain["ministack"].values
    )
    if missing:
        raise ValueError(
            f"No chain offsets for {len(missing)} ministacks "
            f"({sorted(str(pd.Timestamp(m).date()) for m in missing)}); "
            "run add_chain_offsets"
        )
    out = raw.copy()
    for name in DISPLACEMENT_LAYERS:
        if name not in raw or f"{name}_offset" not in chain:
            continue
        offset = chain[f"{name}_offset"].sel(ministack=raw["reference_time"])
        out[name] = (
            (raw[name] + offset.drop_vars("ministack"))
            .transpose(*raw[name].dims)
            .astype(raw[name].dtype, copy=False)
        )
        out[name].attrs = raw[name].attrs
    first = np.datetime64(pd.Timestamp(chain.attrs["chain_reference_date"]), "ns")
    if zero_epoch:
        out = _prepend_zero_epoch(out, first)
    out.attrs["reference_date"] = str(pd.Timestamp(first))
    out.attrs["nan_policy"] = chain.attrs["nan_policy"]
    return record(
        out,
        "open_frame",
        source_snapshot_id=raw.attrs["source_snapshot_id"],
        tag=tag,
        zero_epoch=zero_epoch,
    )
