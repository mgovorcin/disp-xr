"""CMR search for DISP granules, with a fake HTTP session (no network)."""

from datetime import datetime, timezone

import pytest

from disp_xr import search

NAME = "OPERA_L3_DISP-S1_IW_F11116_VV_{ref}_{sec}_v1.0_20250412T164531Z.nc"


def _item(ref: str, sec: str) -> dict:
    name = NAME.format(ref=ref, sec=sec)
    return {
        "umm": {
            "GranuleUR": name,
            "RelatedUrls": [
                {"Type": "GET DATA", "URL": f"https://example.com/{name}"},
                {"Type": "GET DATA VIA DIRECT ACCESS", "URL": f"s3://bucket/{name}"},
            ],
            "DataGranule": {"ArchiveAndDistributionInformation": [{"SizeInBytes": 7}]},
            "SpatialExtent": {
                "HorizontalSpatialDomain": {
                    "Geometry": {
                        "GPolygons": [
                            {
                                "Boundary": {
                                    "Points": [
                                        {"Longitude": -122.0, "Latitude": 37.0},
                                        {"Longitude": -121.0, "Latitude": 37.0},
                                        {"Longitude": -121.0, "Latitude": 38.0},
                                    ]
                                }
                            }
                        ]
                    }
                }
            },
        }
    }


class FakeResponse:
    def __init__(self, items: list[dict], search_after: str | None = None):
        self._items = items
        self.headers = {"CMR-Search-After": search_after} if search_after else {}

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict:
        return {"items": self._items}


class FakeSession:
    """Records every GET and replays the given pages in order."""

    def __init__(self, pages: list[FakeResponse]):
        self.pages = list(pages)
        self.calls: list[dict] = []

    def get(self, url, params=None, headers=None):
        self.calls.append(
            {"url": url, "params": dict(params), "headers": dict(headers)}
        )
        return self.pages.pop(0)


@pytest.fixture
def one_page():
    return FakeSession(
        [
            FakeResponse(
                [
                    _item("20200101T000153Z", "20200113T000153Z"),
                    _item("20200101T000153Z", "20200125T000153Z"),
                ]
            )
        ]
    )


def test_search_returns_one_row_per_granule(one_page):
    df = search.search(frame_id=11116, session=one_page)
    assert len(df) == 2
    assert set(df["frame_id"]) == {11116}
    assert df["filename"].str.startswith("https://example.com/").all()


def test_search_sends_collection_and_filters(one_page):
    search.search(frame_id=11116, product_version="1.0", session=one_page)
    params = one_page.calls[0]["params"]
    assert params["short_name"] == "OPERA_L3_DISP-S1_V1"
    assert params["attribute[]"] == [
        "float,PRODUCT_VERSION,1.0",
        "int,FRAME_NUMBER,11116",
    ]
    assert "uat" not in one_page.calls[0]["url"]


def test_search_uses_given_short_name(one_page):
    search.search(frame_id=11116, short_name="OPERA_L3_DISP-NI_V1", session=one_page)
    assert one_page.calls[0]["params"]["short_name"] == "OPERA_L3_DISP-NI_V1"


def test_search_s3_urls(one_page):
    df = search.search(frame_id=11116, url_type=search.UrlType.S3, session=one_page)
    assert df["filename"].str.startswith("s3://bucket/").all()


def test_search_follows_pagination():
    session = FakeSession(
        [
            FakeResponse([_item("20200101T000153Z", "20200113T000153Z")], "token-1"),
            FakeResponse([_item("20200101T000153Z", "20200125T000153Z")]),
        ]
    )
    df = search.search(frame_id=11116, session=session)
    assert len(df) == 2
    assert session.calls[1]["headers"] == {"CMR-Search-After": "token-1"}


def test_search_filters_by_secondary_date(one_page):
    df = search.search(
        frame_id=11116,
        start_datetime=datetime(2020, 1, 20, tzinfo=timezone.utc),
        session=one_page,
    )
    assert list(df["secondary_datetime"].dt.day) == [25]


def test_search_without_frame_warns(one_page):
    with pytest.warns(UserWarning, match="No `frame_id` specified"):
        search.search(session=one_page)


@pytest.mark.xfail(
    strict=True,
    reason="Bug 9 (code_review.md): frame filter dropped when product_version=None",
)
def test_search_keeps_frame_filter_without_version(one_page):
    search.search(frame_id=11116, product_version=None, session=one_page)
    assert one_page.calls[0]["params"]["attribute[]"] == ["int,FRAME_NUMBER,11116"]


def test_from_filename_rejects_other_names():
    with pytest.raises(ValueError, match="Invalid filename format"):
        search.from_filename("not_a_disp_file.nc")
