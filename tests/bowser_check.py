"""Open a Zarr store with bowser's own server and exercise what its UI calls.

Runs inside an environment where bowser is installed (not disp-xr's), for
example::

    cd 00_tools/src/bowser && pixi run --frozen -e test python \\
        ../disp-xr/tests/bowser_check.py /path/to/store.zarr

Prints one JSON report (after a marker line) and exits non-zero if any
request fails. Used by ``tests/test_geozarr.py`` when
``DISP_XR_BOWSER_PYTHON`` names that python.
"""

import json
import math
import os
import sys
import warnings

MARKER = "--- bowser_check report ---"


def _tile(lon: float, lat: float, z: int) -> tuple[int, int]:
    n = 1 << z
    x = int((lon + 180.0) / 360.0 * n)
    lat_r = math.radians(lat)
    y = int((1.0 - math.asinh(math.tan(lat_r)) / math.pi) / 2.0 * n)
    return x, y


def main(store: str) -> int:
    """Run the checks on ``store`` and print the report."""
    os.environ["BOWSER_STACK_DATA_FILE"] = store
    warnings.filterwarnings("ignore")
    import bowser.main as bm
    from fastapi.testclient import TestClient
    from pyproj import Transformer

    client = TestClient(bm.app)
    state = bm.state
    report: dict = {"levels": [], "requests": {}, "errors": []}
    for level in state.levels:
        report["levels"].append([level.asset, round(level.pixel_size_m, 3)])

    def get(name: str, url: str, **params):
        r = client.get(url, params=params)
        ok = r.status_code == 200
        report["requests"][name] = r.status_code
        if not ok:
            report["errors"].append(f"{name}: {r.status_code} {r.text[:300]}")
        return r if ok else None

    datasets = get("datasets", "/datasets")
    info = datasets.json() if datasets else {}
    report["variables"] = {
        k: {
            "uses_spatial_ref": v["uses_spatial_ref"],
            "unit": v["unit"],
            "label": v["label"],
            "n_times": len(v["x_values"]) or 1,
            "masks": v["available_mask_vars"],
        }
        for k, v in info.items()
    }
    first: dict = next(iter(info.values()), {})
    report["reference_lonlat"] = first.get("reference_lonlat")
    report["los_metadata"] = first.get("los_metadata")

    ds = state.dataset
    to_ll = Transformer.from_crs(ds.rio.crs, 4326, always_xy=True)
    xc, yc = float(ds.x[ds.sizes["x"] // 2]), float(ds.y[ds.sizes["y"] // 2])
    lon, lat = to_ll.transform(xc, yc)
    report["center_lonlat"] = [lon, lat]
    var = "displacement" if "displacement" in info else next(iter(info))
    last = report["variables"][var]["n_times"] - 1
    get(
        "tilejson",
        "/md/WebMercatorQuad/tilejson.json",
        variable=var,
        time_idx=last,
    )
    # One tile per pyramid level: the zoom whose tile pixel matches the level.
    for level in state.levels:
        z = max(0, round(math.log2(40075016.686 / (256 * level.pixel_size_m))))
        picked = state.dataset_for_tile_zoom(z)
        tx, ty = _tile(lon, lat, z)
        r = get(
            f"tile_z{z}",
            f"/md/tiles/WebMercatorQuad/{z}/{tx}/{ty}.png",
            variable=var,
            time_idx=last,
            rescale="-0.05,0.05",
        )
        report["requests"][f"tile_z{z}_level"] = (
            float(abs(picked.x[1] - picked.x[0])) if r is not None else None
        )
        masked = get(
            f"tile_z{z}_masked",
            f"/md/tiles/WebMercatorQuad/{z}/{tx}/{ty}.png",
            variable=var,
            time_idx=last,
            mask_variable="recommended_mask",
            mask_min_value=0.1,
            rescale="-0.05,0.05",
        )
        if r is not None and masked is not None and len(r.content) < 100:
            report["errors"].append(f"tile_z{z}: empty png")
    get("point", "/point", dataset_name=var, lon=lon, lat=lat)
    get("chart_point", "/chart_point", dataset_name=var, lon=lon, lat=lat)
    hist = get("histogram", f"/histogram/{var}", time_index=last)
    if hist is not None:
        report["histogram_keys"] = sorted(hist.json())
    get("dataset_range", f"/dataset_range/{var}")
    # bowser logs to stdout too; the report follows this marker line.
    print(MARKER)
    print(json.dumps(report, default=str))
    return 1 if report["errors"] else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
