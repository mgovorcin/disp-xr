"""Remote access and credentials, offline (memory filesystem, fake sessions)."""

import netrc as netrc_module

import fsspec
import numpy as np
import pytest

from disp_xr import _credentials, _remote
from disp_xr._credentials import (
    ASFEndpoint,
    AWSCredentials,
    EarthdataLoginError,
    get_earthdata_username_password,
    get_temporary_aws_credentials,
)
from disp_xr.open import open_disp
from disp_xr.product import check_granules, find_products


@pytest.fixture
def memory_frame(two_ministacks):
    """The two_ministacks granules copied into fsspec's memory filesystem."""
    fs = fsspec.filesystem("memory")
    for p in two_ministacks.paths:
        fs.pipe(f"/frame/sub/{p.name}", p.read_bytes())
    yield "memory://frame/"
    fs.rm("/frame", recursive=True)


@pytest.mark.parametrize(
    ("path", "remote"),
    [
        ("s3://bucket/a.nc", True),
        ("https://host/a.nc", True),
        ("memory://a.nc", True),
        ("/data/a.nc", False),
        ("a.nc", False),
        ("file:///data/a.nc", False),
    ],
)
def test_is_remote_url(path, remote):
    assert _remote.is_remote_url(path) is remote


def test_find_products_lists_a_remote_prefix(two_ministacks, memory_frame):
    df = find_products(memory_frame)
    assert len(df) == len(two_ministacks.paths)
    assert df["path"].str.startswith("memory://frame/sub/").all()


def test_check_and_open_remote_granules(stack_dir, memory_frame):
    df = find_products(memory_frame)
    assert set(check_granules(df)["layout"]) == {0}
    remote = open_disp(df, layers=["displacement"])
    local = open_disp(stack_dir, layers=["displacement"])
    np.testing.assert_array_equal(remote.displacement.values, local.displacement.values)


def test_earthdata_from_arguments():
    assert get_earthdata_username_password("u", "p") == ("u", "p")


def test_earthdata_from_netrc(tmp_path, monkeypatch):
    rc = tmp_path / ".netrc"
    rc.write_text("machine urs.earthdata.nasa.gov login alice password secret\n")
    rc.chmod(0o600)
    real_netrc = netrc_module.netrc
    monkeypatch.setattr(netrc_module, "netrc", lambda: real_netrc(str(rc)))
    assert get_earthdata_username_password() == ("alice", "secret")


def test_earthdata_from_environment(monkeypatch):
    monkeypatch.setattr(netrc_module, "netrc", _no_netrc)
    monkeypatch.setenv("EARTHDATA_USERNAME", "bob")
    monkeypatch.setenv("EARTHDATA_PASSWORD", "pw")
    assert get_earthdata_username_password() == ("bob", "pw")


def test_earthdata_missing_raises(monkeypatch):
    monkeypatch.setattr(netrc_module, "netrc", _no_netrc)
    monkeypatch.delenv("EARTHDATA_USERNAME", raising=False)
    monkeypatch.delenv("EARTHDATA_PASSWORD", raising=False)
    with pytest.raises(EarthdataLoginError, match="No Earthdata credentials"):
        get_earthdata_username_password()


def _no_netrc():
    raise FileNotFoundError


def test_aws_credentials_from_env(monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "id")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "key")
    monkeypatch.delenv("AWS_SESSION_TOKEN", raising=False)
    creds = AWSCredentials.from_env()
    assert creds.to_s3fs_kwargs() == {"key": "id", "secret": "key", "token": None}


class FakeResponse:
    def __init__(self, status_code, url, payload=None):
        self.status_code, self.url, self._payload = status_code, url, payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)

    def json(self):
        return self._payload


class FakeSession:
    """Answers the first call with an Earthdata login redirect."""

    def __init__(self):
        self.calls = []

    def get(self, url, auth=None):
        self.calls.append((url, auth))
        if auth is None:
            return FakeResponse(401, "https://urs.earthdata.nasa.gov/oauth/authorize?x")
        return FakeResponse(
            200,
            url,
            {"accessKeyId": "A", "secretAccessKey": "S", "sessionToken": "T"},
        )


@pytest.fixture
def empty_cache(monkeypatch):
    monkeypatch.setattr(_credentials, "_cache", {})


@pytest.mark.usefixtures("empty_cache")
def test_temporary_credentials_follow_login_and_cache():
    session = FakeSession()
    first = get_temporary_aws_credentials(ASFEndpoint.OPERA, "u", "p", session=session)
    second = get_temporary_aws_credentials(ASFEndpoint.OPERA, session=session)
    assert (
        first
        == second
        == {"accessKeyId": "A", "secretAccessKey": "S", "sessionToken": "T"}
    )
    assert session.calls == [
        (ASFEndpoint.OPERA.value, None),
        ("https://urs.earthdata.nasa.gov/oauth/authorize?x", ("u", "p")),
    ]


@pytest.mark.usefixtures("empty_cache")
def test_cached_credentials_expire(monkeypatch):
    session = FakeSession()
    get_temporary_aws_credentials(ASFEndpoint.OPERA, "u", "p", session=session)
    later = _credentials.time.monotonic() + _credentials.CREDENTIAL_TTL_SECONDS + 1
    monkeypatch.setattr(_credentials.time, "monotonic", lambda: later)
    get_temporary_aws_credentials(ASFEndpoint.OPERA, "u", "p", session=session)
    assert len(session.calls) == 4


def test_s3_filesystem_uses_environment_credentials(monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "id")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "key")
    fs = _remote.get_filesystem("s3://bucket/a.nc")
    assert (fs.key, fs.secret) == ("id", "key")


def test_https_filesystem_sends_a_basic_auth_header():
    # aiohttp 3.14 dropped auth= from client_kwargs (opera-utils 4125484).
    pytest.importorskip("aiohttp")
    fs = _remote.get_filesystem(
        "https://example.com/a.nc", earthdata_username="u", earthdata_password="p"
    )
    assert fs.client_kwargs["headers"]["Authorization"] == "Basic dTpw"
    assert "auth" not in fs.client_kwargs
