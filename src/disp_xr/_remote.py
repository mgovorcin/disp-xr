"""Open local or remote (S3, HTTPS, any fsspec URL) granules.

Adapted from opera-utils (src/opera_utils/_remote.py, commit dff4b45),
Copyright (c) 2022 California Institute of Technology, BSD-3-Clause. See
THIRD_PARTY_NOTICES.md. Changes: any fsspec protocol is accepted (tests use
``memory://``), listing a remote prefix is supported, and the cloud libraries
are imported only when a remote URL is used (``aws`` extra).
"""

from __future__ import annotations

from os import PathLike, fsdecode
from pathlib import Path
from typing import Any, Optional, Union
from urllib.parse import urlparse

import fsspec
import h5py

from ._credentials import (
    ENDPOINT_TO_HOST,
    ASFEndpoint,
    AWSCredentials,
    get_earthdata_username_password,
)

__all__ = ["get_filesystem", "is_remote_url", "list_granules", "open_file", "open_h5"]

# DISP granules are written with a 4 MiB HDF5 file-space page size.
H5_PAGE_SIZE = 4 * 1024 * 1024


def is_remote_url(path: Union[str, Path]) -> bool:
    """Return True for a URL with a scheme other than ``file``.

    Examples
    --------
    >>> is_remote_url("s3://bucket/a.nc"), is_remote_url("/data/a.nc")
    (True, False)

    """
    scheme = urlparse(str(path)).scheme
    return bool(scheme) and scheme != "file" and len(scheme) > 1


def get_filesystem(
    url: str,
    *,
    asf_endpoint: ASFEndpoint = ASFEndpoint.OPERA,
    earthdata_username: Optional[str] = None,
    earthdata_password: Optional[str] = None,
    storage_options: Optional[dict[str, Any]] = None,
) -> fsspec.AbstractFileSystem:
    """Return an authenticated fsspec filesystem for ``url``.

    ``s3://`` uses ``AWS_*`` environment variables or, failing that, ASF
    temporary credentials. ``https://`` uses Earthdata Login. Other schemes go
    to fsspec with ``storage_options``.
    """
    scheme = urlparse(url).scheme or "file"
    options = dict(storage_options or {})
    if scheme == "s3" and not options:
        try:
            creds = AWSCredentials.from_env()
        except KeyError:
            creds = AWSCredentials.from_asf(asf_endpoint)
        options = creds.to_s3fs_kwargs()
    elif scheme in {"http", "https"} and not options:
        import aiohttp

        user, password = get_earthdata_username_password(
            earthdata_username, earthdata_password, host=ENDPOINT_TO_HOST[asf_endpoint]
        )
        options = {"client_kwargs": {"auth": aiohttp.BasicAuth(user, password)}}
    return fsspec.filesystem(scheme, **options)


def list_granules(prefix: str, **fs_kwargs: Any) -> list[str]:
    """List ``*.nc`` files under a remote prefix, as full URLs."""
    fs = get_filesystem(prefix, **fs_kwargs)
    scheme = urlparse(prefix).scheme
    found = fs.glob(prefix.rstrip("/") + "/**/*.nc")
    return sorted(p if "://" in p else f"{scheme}://{p.lstrip('/')}" for p in found)


def open_file(url: Union[str, PathLike[str]], **fs_kwargs: Any) -> Any:
    """Open a local or remote file for reading, as a file-like object."""
    url_str = fsdecode(url)
    if not is_remote_url(url_str):
        return open(url_str, "rb")  # noqa: SIM115
    fs = get_filesystem(url_str, **fs_kwargs)
    return fs.open(url_str, mode="rb", cache_type="first")


def open_h5(url: Union[str, PathLike[str]], **fs_kwargs: Any) -> h5py.File:
    """Open a local or remote HDF5 file with cloud-friendly page caching."""
    if not is_remote_url(fsdecode(url)):
        return h5py.File(url, "r")
    return h5py.File(
        open_file(url, **fs_kwargs),
        mode="r",
        fs_page_size=H5_PAGE_SIZE,
        rdcc_nbytes=1024**3,
    )
