"""Read-only REST service (plan task T40)."""

import time

import numpy as np
import pytest

import disp_xr as dxr

pytest.importorskip("xpublish")
pytest.importorskip("httpx2")
pytest.importorskip("geepers.validation")
from fastapi.testclient import TestClient  # noqa: E402
from fixtures.fake_gnss import stations_from_disp, vertical_los  # noqa: E402
from pyproj import Transformer  # noqa: E402

from disp_xr import serve  # noqa: E402
from disp_xr.residuals import compare  # noqa: E402

PIXELS = [(8, 10), (20, 40), (35, 15), (50, 50), (12, 55)]


@pytest.fixture(scope="module")
def frame(disp_stacks):
    stack_dir = disp_stacks["two_ministacks"].paths[0].parent
    ds = dxr.chain(dxr.open_disp(stack_dir))
    fit = dxr.fit_velocity(ds, method="ols")
    return ds.assign(velocity=fit.velocity, velocity_sigma=fit.velocity_sigma)


@pytest.fixture(scope="module")
def store(frame, tmp_path_factory):
    """A time-series-chunked store with velocity and a GNSS group."""
    path = tmp_path_factory.mktemp("serve") / "demo.zarr"
    dxr.to_zarr(frame, path, preset="timeseries")
    cache = tmp_path_factory.mktemp("gnss")
    gps = dxr.gnss.load(frame, source=stations_from_disp(frame, PIXELS, cache))
    compare(frame, gps, vertical_los(frame), min_span_years=0).to_zarr(path)
    return path


@pytest.fixture(scope="module")
def client(store, frame, tmp_path_factory):
    bare = tmp_path_factory.mktemp("bare") / "bare.zarr"
    dxr.to_zarr(frame.drop_vars(["velocity", "velocity_sigma"]), bare)
    return TestClient(serve.make_app({"demo": store, "bare": bare}).app)


def lonlat(ds, row, col):
    to_ll = Transformer.from_crs(ds.rio.crs, "EPSG:4326", always_xy=True)
    return to_ll.transform(float(ds.x[col]), float(ds.y[row]))


def test_timeseries(client, frame):
    lon, lat = lonlat(frame, 30, 20)
    r = client.get(
        "/datasets/demo/disp/timeseries",
        params={"lon": lon, "lat": lat, "buffer_m": 30},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["crs"] == f"EPSG:{frame.rio.crs.to_epsg()}"
    assert body["units"] == frame.displacement.attrs["units"]
    assert body["point"]["n_pixels"] == 9  # 30 m buffer: the pixel and its 8 neighbors
    window = frame.displacement.isel(y=slice(29, 32), x=slice(19, 22))
    expected = window.median(("y", "x")).values
    np.testing.assert_allclose(
        np.array(body["displacement"], float), expected, rtol=1e-6
    )
    assert len(body["dates"]) == frame.sizes["time"]
    assert set(body["quality"]) == {"temporal_coherence", "phase_similarity"}
    assert body["reference_date"] == frame.attrs["reference_date"]


def test_point_outside_frame(client):
    r = client.get("/datasets/demo/disp/timeseries", params={"lon": 0, "lat": 0})
    assert r.status_code == 404
    assert "outside the frame" in r.json()["detail"]


def test_velocity(client, frame):
    lon, lat = lonlat(frame, 12, 40)
    r = client.get("/datasets/demo/disp/velocity", params={"lon": lon, "lat": lat})
    assert r.status_code == 200
    body = r.json()
    assert body["velocity"] == pytest.approx(float(frame.velocity[12, 40]), rel=1e-6)
    assert body["velocity_sigma"] == pytest.approx(
        float(frame.velocity_sigma[12, 40]), rel=1e-6
    )
    assert body["units"] == frame.velocity.attrs["units"]
    missing = client.get(
        "/datasets/bare/disp/velocity", params={"lon": lon, "lat": lat}
    )
    assert missing.status_code == 404 and "fit_velocity" in missing.json()["detail"]


def _bbox(ds, rows, cols):
    (w, n), (e, s) = lonlat(ds, rows[0], cols[0]), lonlat(ds, rows[1], cols[1])
    return f"{min(w, e)},{min(s, n)},{max(w, e)},{max(s, n)}"


def test_cells_grid_and_quadtree(client, frame):
    bbox = _bbox(frame, (2, 61), (2, 61))
    grid = client.get("/datasets/demo/disp/cells", params={"bbox": bbox, "size_m": 300})
    assert grid.status_code == 200
    body = grid.json()
    assert body["type"] == "FeatureCollection" and body["crs"] == "EPSG:4326"
    assert body["method"] == "grid" and body["layer"] == "velocity"
    feature = body["features"][0]
    assert feature["geometry"]["type"] == "Polygon"
    assert len(feature["geometry"]["coordinates"][0]) == 5
    assert set(feature["properties"]) == {"velocity", "size_m"}
    assert 20 <= len(body["features"]) <= 49  # about 60 x 60 pixels in 10 x 10 cells

    tree = client.get(
        "/datasets/demo/disp/cells", params={"bbox": bbox, "n_points": 12}
    )
    assert tree.status_code == 200
    assert tree.json()["method"] == "quadtree"
    assert 6 <= len(tree.json()["features"]) <= 20

    disp = client.get(
        "/datasets/bare/disp/cells", params={"bbox": bbox, "size_m": 300}
    ).json()
    assert disp["layer"] == "displacement"
    assert disp["date"] == str(np.datetime_as_string(frame.time[-1], unit="D"))


@pytest.mark.parametrize(
    ("params", "status"),
    [
        ({"bbox": "1,2,3"}, 422),
        ({"bbox": "a,b,c,d", "size_m": 300}, 422),
        ({"size_m": 300, "n_points": 10}, 422),
        ({"size_m": 300, "layer": "nope"}, 404),
        ({"bbox": "0,0,1,1", "size_m": 300}, 404),
    ],
)
def test_cells_errors(client, frame, params, status):
    params = {"bbox": _bbox(frame, (2, 61), (2, 61)), **params}
    assert client.get("/datasets/demo/disp/cells", params=params).status_code == status


def test_cells_size_cap(client, frame, monkeypatch):
    monkeypatch.setattr(serve, "MAX_CELL_PIXELS", 100)
    r = client.get(
        "/datasets/demo/disp/cells",
        params={"bbox": _bbox(frame, (2, 61), (2, 61)), "size_m": 300},
    )
    assert r.status_code == 413


def test_gnss(client):
    r = client.get("/datasets/demo/disp/gnss")
    assert r.status_code == 200
    body = r.json()
    assert len(body["stations"]) == len(PIXELS)
    station = body["stations"][0]
    assert {"station", "lon", "lat", "insar_velocity", "gps_velocity"} <= set(station)
    assert "datum_offset" in body["summary"]
    assert client.get("/datasets/bare/disp/gnss").status_code == 404


def test_provenance(client, frame):
    body = client.get("/datasets/demo/disp/provenance").json()
    assert [h["op"] for h in body["history"]][0] == "open_disp"
    assert body["input_granules"]
    assert body["reference_date"] == frame.attrs["reference_date"]


def test_unknown_dataset(client):
    assert client.get("/datasets/nope/disp/provenance").status_code == 404


def test_openapi_documents_the_routes(client):
    spec = client.get("/openapi.json").json()
    for route in ("timeseries", "velocity", "cells", "gnss", "provenance"):
        assert f"/datasets/{{dataset_id}}/disp/{route}" in spec["paths"]
    assert "TimeseriesResponse" in spec["components"]["schemas"]


def test_point_series_latency(store, frame):
    """A point series on a time-series-chunked store answers well under 1 s."""
    client = TestClient(serve.make_app({"demo": store}, tiles=False).app)
    lon, lat = lonlat(frame, 40, 25)
    t0 = time.perf_counter()
    r = client.get("/datasets/demo/disp/timeseries", params={"lon": lon, "lat": lat})
    first = time.perf_counter() - t0
    assert r.status_code == 200
    assert first < 1.0  # including the first request of the process


def test_virtual_frame(disp_stacks, tmp_path):
    """A virtual frame is chained on open and served like a store."""
    pytest.importorskip("virtualizarr")
    granules = sorted(disp_stacks["two_ministacks"].paths)
    repo = tmp_path / "virtual"
    dxr.build_virtual_frame(dxr.find_products(granules), repo, max_workers=1)
    expected = dxr.chain(dxr.open_disp(granules))
    client = TestClient(serve.make_app({"virtual": repo}, tiles=False).app)
    lon, lat = lonlat(expected, 30, 20)
    r = client.get("/datasets/virtual/disp/timeseries", params={"lon": lon, "lat": lat})
    assert r.status_code == 200
    got = np.array(r.json()["displacement"], float)
    np.testing.assert_allclose(got, expected.displacement[:, 30, 20], rtol=1e-6)
