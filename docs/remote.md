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
only the chunk index.

### From a CMR search to a lazy data cube

The links CMR returns go straight into a virtual frame, and the rebasing
across ministacks (`chain`) and the re-reference run lazily on top of it:
nothing is read until values are asked for, and then only the chunks needed.

```python
import disp_xr as dxr
from disp_xr.reference import find_stable_point


def main():
    # 1. CMR: the frame's granules, with S3 links in the "filename" column
    table = dxr.search_granules(frame_id=11116, url_type="s3")

    # 2. One production per frame (see below), as a list of links
    granules = dxr.find_products(list(table["filename"]))
    keep = granules.production_time.dt.strftime("%Y-%m") == "2025-04"

    # 3. Virtual frame: one commit per ministack, no pixels copied
    dxr.build_virtual_frame(granules[keep], "s3://my-bucket/F11116-virtual")
    dxr.validate_virtual("s3://my-bucket/F11116-virtual", granules[keep], n_samples=3)

    # 4. Lazy cube: rebase, mask, re-reference to a stable pixel
    frame = dxr.open_virtual("s3://my-bucket/F11116-virtual")
    row, col = find_stable_point(frame)
    cube = dxr.chain(dxr.mask(frame, "recommended")).disp.rereference(
        "point", row=row, col=col
    )
    series = cube.displacement.isel(y=3500, x=4800).values  # reads one pixel's chunks


if __name__ == "__main__":  # build_virtual_frame starts worker processes
    main()
```

New granules later: search again and pass only the new ones to
`build_virtual_frame`; they become new commits on the same frame, and
earlier snapshots stay readable.

Things to know:

- **Run it in AWS `us-west-2`.** Direct S3 reads of the ASF bucket work only
  there (`AWS_*` variables, or temporary ASF credentials renewed
  automatically). Elsewhere, download the granules and build the frame from
  the local files; the repository itself can be a local folder or your own
  bucket.
- **One production per frame.** A frame's granules can come from several
  productions that store displacement in different dtypes (F11116: float32
  for 314 granules produced in 2025-04, float64 for 75 produced later), and a
  virtual frame refuses to mix them (`MixedProductionError`). Select one
  production with `production_time`, or build one frame per production.
- **Pick the reference pixel with `find_stable_point`.** A `"point"`
  reference must be valid at every epoch: where it is masked, the whole
  epoch becomes NaN. `find_stable_point` (what `rereference("auto")` uses)
  picks a pixel never masked, coherent and free of phase jumps; it reads the
  quality layers of every epoch once (33 min on F11116 with dask threads; its
  docstring recommends a `distributed.LocalCluster`), so find it once per
  frame and keep the pixel. Avoid `"median"` for point queries: a
  spatial median needs every pixel of each epoch, so one pixel's series
  reads the whole frame.
- **Guard the script.** `build_virtual_frame(max_workers>1)` parses
  granules in worker processes, which re-import the calling script; without
  `if __name__ == "__main__":` each worker runs it again and the pool fails
  (`BrokenProcessPool`).
- **Heavy work goes to a native store.** Reads follow the granules' own small
  HDF5 chunks, which suits exploring, point series and `disp_xr serve`. For
  whole-frame fits, write the lazy cube once (`dxr.to_zarr(cube, ...)`, or
  `dxr.to_icechunk(cube, repo, tag=...)` to version it) and process it with
  `dxr.open_zarr` / `dxr.open_icechunk`.

Measured on F11116 (314 granules, 118 GB, on local disk of a shared node,
disp-xr 0.5.2):

| Step | Time |
|---|---:|
| CMR search (389 granules) | 6–15 s |
| Build the virtual frame (21 commits; repository 0.5 GB) | 97–110 s |
| Validate 3 pixels in 11 layers | 3 min |
| Lazy cube: chain, mask, point re-reference (1.1 M tasks, nothing read) | 18–20 s |
| One pixel's series, 315 epochs | 13.5 s (`open_disp`: 65 s, identical values) |
| One full epoch map | 47 s |
| Open an earlier snapshot | 0.1 s |

See the [Icechunk example](gallery/virtual_frames.py) for building,
validating, extending and versioning a frame on synthetic granules.

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
