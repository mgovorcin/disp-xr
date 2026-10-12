"""Storage benchmark on S3 (plan task T37.3b, S3 half).

Run in AWS (us-west-2, next to the data) on real DISP-S1 granules uploaded
to a scratch prefix. For each storage:

- ``netcdf-s3``: the granules themselves (``open_disp`` on ``s3://`` URLs);
- ``virtual``: a virtual Icechunk store on S3 that references them
  (``build_virtual_frame``; the build, commit included, is timed once);
- ``zarr-<preset>-<MB>``: native Zarr v3 on S3 written by ``to_zarr`` from
  the same granules, for both presets and a sweep of shard sizes, read
  through fsspec (s3fs), and ``zarr-...-obstore`` through an obstore store;

it times four reads of ``displacement``: open (lazy), one pixel's series,
one epoch's map and a 512 x 512 window over all epochs. Trials are
interleaved across storages, values are checked against the NetCDF read,
and every row records the trial. The scratch stores are deleted at the end.

Run on AWS Batch with the image in ``batch/`` (job command)::

    storage_s3.py s3://BUCKET/PREFIX/granules/ s3://BUCKET/PREFIX/work/ \
        --trials 5 --out /tmp/out.csv
"""

from __future__ import annotations

import argparse
import os
import time
from collections.abc import Callable
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import xarray as xr

import disp_xr as dxr

LAYERS = ["displacement"]
PRESETS = ("maps", "timeseries")
SHARD_MB = (8, 32, 128)
WINDOW = 512


def export_role_credentials() -> None:
    """Put the job role's credentials in ``AWS_*`` variables.

    disp-xr's S3 access reads them from the environment (else it asks ASF),
    while Batch serves the role through the container credential endpoint.
    """
    import botocore.session

    creds = botocore.session.get_session().get_credentials().get_frozen_credentials()
    os.environ["AWS_ACCESS_KEY_ID"] = creds.access_key
    os.environ["AWS_SECRET_ACCESS_KEY"] = creds.secret_key
    if creds.token:
        os.environ["AWS_SESSION_TOKEN"] = creds.token
    os.environ.setdefault("AWS_REGION", "us-west-2")


def s3fs_options() -> dict:
    """Return s3fs options with the exported credentials."""
    return {
        "key": os.environ["AWS_ACCESS_KEY_ID"],
        "secret": os.environ["AWS_SECRET_ACCESS_KEY"],
        "token": os.environ.get("AWS_SESSION_TOKEN"),
    }


def obstore_zarr(url: str):
    """Return a zarr ObjectStore over obstore's S3Store for ``url``."""
    from obstore.store import S3Store
    from zarr.storage import ObjectStore

    parsed = urlparse(url)
    store = S3Store(
        parsed.netloc,
        prefix=parsed.path.lstrip("/"),
        region=os.environ["AWS_REGION"],
        access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        session_token=os.environ.get("AWS_SESSION_TOKEN"),
    )
    return ObjectStore(store, read_only=True)


def timed(func: Callable) -> tuple[float, object]:
    """Return the wall time of ``func()`` and its result."""
    start = time.perf_counter()
    out = func()
    return time.perf_counter() - start, out


def build(granules: str, work: str) -> tuple[dict[str, Callable], pd.DataFrame]:
    """Write the derived stores; return their openers and the build rows."""
    import icechunk

    urls = sorted(dxr.find_products(granules).path)
    rows = []
    openers: dict[str, Callable] = {
        "netcdf-s3": lambda: dxr.open_disp(urls, layers=LAYERS),
    }
    t, raw = timed(lambda: dxr.open_disp(urls, layers=LAYERS))
    rows.append({"storage": "netcdf-s3", "step": "open for writing", "seconds": t})

    parsed = urlparse(work)
    bucket, root = parsed.netloc, parsed.path.strip("/")
    storage = icechunk.s3_storage(
        bucket=bucket,
        prefix=f"{root}/virtual",
        region=os.environ["AWS_REGION"],
        from_env=True,
    )
    t, _ = timed(lambda: dxr.build_virtual_frame(urls, storage, layers=LAYERS))
    rows.append({"storage": "virtual", "step": "build and commit", "seconds": t})

    def open_virtual_store():
        storage_again = icechunk.s3_storage(
            bucket=bucket,
            prefix=f"{root}/virtual",
            region=os.environ["AWS_REGION"],
            from_env=True,
        )
        return dxr.open_virtual(storage_again)

    openers["virtual"] = open_virtual_store

    for preset in PRESETS:
        for mb in SHARD_MB:
            name = f"zarr-{preset}-{mb}"
            url = f"{work.rstrip('/')}/{name}.zarr"
            t, _ = timed(
                lambda url=url, preset=preset, mb=mb: dxr.to_zarr(
                    raw,
                    url,
                    preset=preset,
                    target_mb=mb,
                    storage_options=s3fs_options(),
                    overwrite=True,
                )
            )
            rows.append({"storage": name, "step": "write", "seconds": t})
            openers[name] = lambda url=url: xr.open_zarr(
                url, storage_options=s3fs_options(), consolidated=False
            )
            openers[f"{name}-obstore"] = lambda url=url: xr.open_zarr(
                obstore_zarr(url), consolidated=False
            )
    return openers, pd.DataFrame(rows)


def reads(ds: xr.Dataset) -> dict[str, Callable]:
    """Return the timed reads of ``displacement``."""
    da = ds["displacement"]
    ny, nx = da.sizes["y"], da.sizes["x"]
    y0, x0 = ny // 2 - WINDOW // 2, nx // 2 - WINDOW // 2
    return {
        "pixel series": lambda: da.isel(y=ny // 2, x=nx // 2).values,
        "epoch map": lambda: da.isel(time=da.sizes["time"] // 2).values,
        "window all epochs": lambda: (
            da.isel(y=slice(y0, y0 + WINDOW), x=slice(x0, x0 + WINDOW)).values
        ),
    }


def run(openers: dict[str, Callable], trials: int) -> pd.DataFrame:
    """Time open and the reads, trials interleaved; check values."""
    rows = []
    reference: dict[str, np.ndarray] = {}
    for trial in range(trials):
        for storage, opener in openers.items():
            t, ds = timed(opener)
            rows.append(
                {"storage": storage, "op": "open", "trial": trial, "seconds": t}
            )
            for op, read in reads(ds).items():
                t, values = timed(read)
                ok = True
                if op in reference:
                    ok = bool(np.array_equal(values, reference[op], equal_nan=True))
                else:
                    reference[op] = values
                rows.append(
                    {
                        "storage": storage,
                        "op": op,
                        "trial": trial,
                        "seconds": t,
                        "values_match": ok,
                    }
                )
    return pd.DataFrame(rows)


def cleanup(work: str) -> None:
    """Delete everything under the scratch prefix ``work``."""
    import s3fs

    fs = s3fs.S3FileSystem(**s3fs_options())
    if fs.exists(work):
        fs.rm(work, recursive=True)


def main() -> None:
    """Build the stores, time the reads, print and write the rows."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("granules", help="s3:// prefix of the DISP granules")
    parser.add_argument("work", help="s3:// scratch prefix, deleted at the end")
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--out", default="/tmp/out.csv")
    args = parser.parse_args()
    export_role_credentials()
    try:
        openers, built = build(args.granules, args.work)
        print(built.to_string(index=False), flush=True)
        rows = run(openers, args.trials)
    finally:
        cleanup(args.work)
    rows = pd.concat([built.assign(op="build", trial=0), rows], ignore_index=True)
    rows.to_csv(args.out, index=False)
    timing = rows[rows.op != "build"].groupby(["storage", "op"])["seconds"]
    summary = timing.agg(["median", "min", "max"]).round(3)
    print(summary.to_string(), flush=True)
    mismatches = int((rows["values_match"] == False).sum())  # noqa: E712
    print(f"value mismatches: {mismatches}", flush=True)


if __name__ == "__main__":
    main()
