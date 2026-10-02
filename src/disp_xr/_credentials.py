"""Earthdata Login and ASF temporary S3 credentials.

Adapted from opera-utils (src/opera_utils/credentials.py, commit dff4b45),
Copyright (c) 2022 California Institute of Technology, BSD-3-Clause. See
THIRD_PARTY_NOTICES.md. Changes: the HTTP session is injectable for tests, and
the credential cache is a plain dict with an expiry time.
"""

from __future__ import annotations

import netrc
import os
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

__all__ = [
    "ASFEndpoint",
    "AWSCredentials",
    "EarthdataLoginError",
    "get_earthdata_username_password",
    "get_temporary_aws_credentials",
]

# ASF temporary credentials last one hour; refresh before that.
CREDENTIAL_TTL_SECONDS = 50 * 60


class EarthdataLoginError(Exception):
    """No Earthdata Login credentials were found."""


class ASFEndpoint(Enum):
    """ASF endpoints that issue temporary S3 credentials."""

    OPERA = "https://cumulus.asf.alaska.edu/s3credentials"
    OPERA_UAT = "https://cumulus-test.asf.alaska.edu/s3credentials"
    SENTINEL1 = "https://sentinel1.asf.alaska.edu/s3credentials"


ENDPOINT_TO_HOST = {
    ASFEndpoint.OPERA: "urs.earthdata.nasa.gov",
    ASFEndpoint.OPERA_UAT: "uat.urs.earthdata.nasa.gov",
    ASFEndpoint.SENTINEL1: "urs.earthdata.nasa.gov",
}


@dataclass(frozen=True)
class AWSCredentials:
    """AWS credentials for direct S3 access."""

    access_key_id: str
    secret_access_key: str
    session_token: Optional[str] = None

    @classmethod
    def from_env(cls) -> AWSCredentials:
        """Read ``AWS_ACCESS_KEY_ID``, ``AWS_SECRET_ACCESS_KEY``, ``AWS_SESSION_TOKEN``.

        Raises
        ------
        KeyError
            If the key id or the secret is not set.

        """
        return cls(
            access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
            secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
            session_token=os.environ.get("AWS_SESSION_TOKEN"),
        )

    @classmethod
    def from_asf(
        cls, endpoint: ASFEndpoint = ASFEndpoint.OPERA, *, session: Any = None
    ) -> AWSCredentials:
        """Request (or reuse cached) temporary credentials from ASF."""
        data = get_temporary_aws_credentials(endpoint, session=session)
        return cls(
            access_key_id=data["accessKeyId"],
            secret_access_key=data["secretAccessKey"],
            session_token=data["sessionToken"],
        )

    def to_s3fs_kwargs(self) -> dict[str, Optional[str]]:
        """Return the keyword arguments of ``s3fs.S3FileSystem``."""
        return {
            "key": self.access_key_id,
            "secret": self.secret_access_key,
            "token": self.session_token,
        }


_cache: dict[ASFEndpoint, tuple[float, dict[str, str]]] = {}


def get_temporary_aws_credentials(
    endpoint: ASFEndpoint = ASFEndpoint.OPERA,
    earthdata_username: Optional[str] = None,
    earthdata_password: Optional[str] = None,
    *,
    session: Any = None,
) -> dict[str, str]:
    """Return ASF temporary S3 credentials, cached for 50 minutes.

    Parameters
    ----------
    endpoint : ASFEndpoint
        Which ASF endpoint to ask.
    earthdata_username, earthdata_password : str, optional
        Earthdata Login; otherwise read from ``~/.netrc`` or the environment.
    session : requests.Session, optional
        HTTP session to use; a new one is created when omitted.

    Returns
    -------
    dict
        The endpoint's JSON response (``accessKeyId``, ``secretAccessKey``,
        ``sessionToken``, ``expiration``).

    """
    cached = _cache.get(endpoint)
    if cached and time.monotonic() < cached[0]:
        return cached[1]
    if session is None:
        import requests

        session = requests.Session()
    resp = session.get(endpoint.value)
    if resp.status_code == 401 and "nasa.gov/oauth/authorize?" in resp.url:
        auth = get_earthdata_username_password(
            earthdata_username, earthdata_password, host=ENDPOINT_TO_HOST[endpoint]
        )
        resp = session.get(resp.url, auth=auth)
    resp.raise_for_status()
    data = resp.json()
    _cache[endpoint] = (time.monotonic() + CREDENTIAL_TTL_SECONDS, data)
    return data


def get_earthdata_username_password(
    earthdata_username: Optional[str] = None,
    earthdata_password: Optional[str] = None,
    host: str = "urs.earthdata.nasa.gov",
) -> tuple[str, str]:
    """Return Earthdata Login credentials.

    Looked up in order: the arguments, ``~/.netrc`` for ``host``, then the
    ``EARTHDATA_USERNAME`` and ``EARTHDATA_PASSWORD`` environment variables.

    Raises
    ------
    EarthdataLoginError
        If none of them provides both a username and a password.

    """
    if earthdata_username and earthdata_password:
        return earthdata_username, earthdata_password
    try:
        auth = netrc.netrc().authenticators(host)
        if auth and auth[0] and auth[2]:
            return auth[0], auth[2]
    except (FileNotFoundError, netrc.NetrcParseError):
        pass
    username = os.environ.get("EARTHDATA_USERNAME", "")
    password = os.environ.get("EARTHDATA_PASSWORD", "")
    if username and password:
        return username, password
    raise EarthdataLoginError(
        f"No Earthdata credentials: pass them, add a {host!r} entry to ~/.netrc,"
        " or set EARTHDATA_USERNAME and EARTHDATA_PASSWORD"
    )
