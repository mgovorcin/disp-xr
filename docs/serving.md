# Serving stores over HTTP

`disp_xr serve` exposes stores read-only to web clients: point series,
velocities, grid or quadtree cells, GNSS residuals, provenance and map
tiles. The [REST service gallery page](gallery/rest_service.py) shows every
route on a synthetic store. This page covers running it for other people.

## Prepare the stores

The service reads; it does not process. Give each store what its clients
will ask for:

| Clients ask for | Write into the store |
| --- | --- |
| Point series (`timeseries`) | The chained cube (`disp_xr stack`, or `to_zarr`) |
| Velocities (`velocity`, cells of velocity) | `velocity` and `velocity_sigma` from `fit_velocity`, in the same store |
| GNSS residuals (`gnss`) | `residuals.compare(...).to_zarr(store)`, a `gnss` group next to the cube |
| Fast point series | The `timeseries` preset (one chunk holds every epoch of a block of pixels) |
| Fast map tiles | A GeoZarr pyramid (`to_geozarr`), served as its own dataset |

Full frames are built by region jobs (`disp_xr.batch`), not by the service.

## Run it

```bash
pip install "disp_xr[serve]"
disp_xr serve /data/F08882.zarr /data/F11116.zarr --host 127.0.0.1 --port 9000
```

Each store becomes a dataset named after it (`F08882`, `F11116`), served
under `/datasets/<name>/`. The API page is at `/docs`, the OpenAPI schema
at `/openapi.json`. Stores open lazily: the service reads only the chunks a
request needs. Zarr stores and Icechunk repositories are supported; virtual
frames are not yet.

For more than one worker process, run the application with uvicorn
directly, with a small launcher module:

```python
# app.py
from disp_xr.serve import make_app

app = make_app({"F08882": "/data/F08882.zarr"}).app
```

```bash
uvicorn app:app --host 127.0.0.1 --port 9000 --workers 4
```

Each worker opens the stores itself. The first request of a worker takes
about 0.2 s more than the next ones (pyproj and Zarr warm-up).

## Behind a reverse proxy

Keep the service on `127.0.0.1` and let a reverse proxy handle TLS, the
public path and caching. With nginx, serving the API under `/disp-api/`:

```nginx
location /disp-api/ {
    proxy_pass http://127.0.0.1:9000/;
    proxy_set_header Host $host;
    proxy_set_header X-Forwarded-Proto $scheme;
    proxy_read_timeout 120s;          # cells on large boxes can take seconds

    # Responses depend only on the URL and the store: cache them.
    proxy_cache disp_api;
    proxy_cache_valid 200 1h;
    add_header X-Cache-Status $upstream_cache_status;

    # Web maps on other origins (the DISP portal) call the API directly.
    add_header Access-Control-Allow-Origin "*" always;
}
```

with `proxy_cache_path /var/cache/nginx/disp_api keys_zone=disp_api:10m
max_size=2g;` in the `http` block. When a store is rebuilt, clear the
cache, or put a version in the dataset name (`F08882-v2`).

The OpenAPI page builds its links from the request, so behind a prefix,
start uvicorn with `--root-path /disp-api` (or pass `root_path` to the
FastAPI application) so `/docs` works there too.

## Limits

- `cells` downsamples at most 4 M pixels per request (413 above), about a
  60 x 60 km box of a 30 m frame. Clients should ask for the current map
  view, not the whole frame.
- `timeseries` and `velocity` take a `buffer_m` of up to 5 km.
- There is no authentication: put the proxy's authentication in front if
  the stores are not public.
