# Finding and reading granules remotely

DISP-S1 granules are in NASA's archive at the ASF DAAC. disp-xr finds them
through the CMR search API and reads them locally, over HTTPS or directly
from S3. The examples on this page need an Earthdata Login and network
access, so unlike the gallery they are not run when the docs build.

## Credentials

Put your [Earthdata Login](https://urs.earthdata.nasa.gov) in `~/.netrc`:

```text
machine urs.earthdata.nasa.gov login YOUR_USER password YOUR_PASSWORD
```

disp-xr reads it for HTTPS downloads and to request temporary S3
credentials from ASF. The `aws` extra adds the libraries for remote reads
(`pip install 'disp_xr[aws]'`).

## Search

```bash
disp_xr search --frame-id 11116 --start 2024-01-01 --out granules.csv
```

```python
import disp_xr as dxr

table = dxr.search_granules(frame_id=11116)  # one row per granule, version 1.0
table[["reference_datetime", "secondary_datetime", "generation_datetime", "filename"]]
```

`filename` holds the HTTPS URL; `url_type="s3"` (`--url-type s3`) gives S3
URLs instead. A frame can have granules from several productions. disp-xr
reads productions that differ only in float precision together, with a
warning (see the [product notes](product_notes.md)).

## Download

disp-xr has no download command: any HTTPS client that sends the Earthdata
login works. Downloads resume, and six run in parallel:

```bash
cut -d, -f9 granules.csv | tail -n +2 > urls.txt
xargs -P 6 -n 1 curl -s -n -L -c .cookies -b .cookies -f -C - -O < urls.txt
```

## Read without downloading

`open_disp` takes a remote prefix or a list of URLs as well as a folder:

```python
import disp_xr as dxr

ds = dxr.open_disp(list(table["filename"]))       # HTTPS, Earthdata login

s3_urls = search(frame_id=11116, url_type="s3")["filename"]
ds = dxr.open_disp(list(s3_urls))                 # S3, in us-west-2
```

S3 reads use the `AWS_*` environment variables if set, otherwise
temporary ASF credentials (valid one hour, renewed automatically). Direct
S3 access works only from AWS `us-west-2`; elsewhere use HTTPS.

Reading a whole frame over the network is slow: every epoch is a separate
file, and opening it reads its metadata. For repeated work, build a virtual
frame once. It references the chunks in place, and every later open reads
only the chunk index:

```python
dxr.build_virtual_frame(list(s3_urls), "s3://my-bucket/F11116-virtual")
ds = dxr.open_virtual("s3://my-bucket/F11116-virtual")
```

See the [virtual frames example](gallery/virtual_frames.py) for building,
validating and extending one.

## Static layers

The line-of-sight vectors, DEM and layover/shadow mask come from the
frame's DISP-S1-STATIC granule. They are downloaded once and cached:

```bash
disp_xr geometry F11116/ --out F11116_static.zarr --static-dir static/
```

```python
static = dxr.open_static(ds)    # downloads to ~/.cache/disp_xr/static/F11116
```

Frames outside North America have no DISP-S1-STATIC. Pass your own LOS
layers with `open_static(ds, files=...)`.
