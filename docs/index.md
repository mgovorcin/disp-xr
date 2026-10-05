---
hide:
  - navigation
  - toc
---

<div class="dxr-hero" markdown>

# disp-xr

**Xarray tools for OPERA surface displacement.** Open a frame of DISP-S1 or
DISP-NISAR granules as one lazy dataset, chain the ministacks, check quality,
estimate velocities with honest uncertainties, compare with GNSS, downsample
and export, from a laptop to an AWS Batch cluster.

[Get started](getting-started.md){ .md-button .md-button--primary }
[Tutorials](gallery/index.md){ .md-button }
[API reference](api/index.md){ .md-button }

</div>

```python
import disp_xr as dxr

ds = dxr.open_disp("F11116/")                      # lazy (time, y, x), CRS attached
ds = dxr.stack_disp(dxr.mask(ds, "recommended"))   # displacement from the first date
ds = dxr.rereference(ds, "median")
velocity = dxr.fit_velocity(ds, method="midas")    # robust, with a sigma per pixel
```

## What you can do

<div class="grid cards" markdown>

-   :material-folder-open-outline:{ .lg .middle } **Open and chain frames**

    ---

    Local folders, S3 or HTTPS; ministacks chained across reference dates and
    across product changes; virtual frames that open in seconds.

    [:octicons-arrow-right-24: Getting started](getting-started.md)

-   :material-shield-check-outline:{ .lg .middle } **Quality and references**

    ---

    Product masks, quality weights from similarity, coherence and inversion
    residuals, six reference methods, ionosphere and plate-motion corrections.

    [:octicons-arrow-right-24: Quality tutorials](gallery/index.md#quality-and-references)

-   :material-chart-timeline-variant:{ .lg .middle } **Velocities and time series**

    ---

    Trajectory fits, OLS, Whittle, MIDAS and GLS velocities with colored-noise
    sigmas, model diagnostics, Kalman and LOWESS denoising.

    [:octicons-arrow-right-24: Choosing a velocity method](velocity-methods.md)

-   :material-satellite-uplink:{ .lg .middle } **GNSS validation**

    ---

    Stations next to DISP through geepers, residuals against OPERA
    requirements, a GNSS tie that removes long-wavelength error.

    [:octicons-arrow-right-24: Comparing with GNSS](gallery/gnss_compare.py)

-   :material-grid:{ .lg .middle } **Downsample**

    ---

    Quality-weighted grids, hexagons (planar and H3) and quadtrees, with cell
    uncertainties that account for spatial correlation.

    [:octicons-arrow-right-24: Downsampling](gallery/downsample.py)

-   :material-cloud-upload-outline:{ .lg .middle } **Export and serve**

    ---

    Zarr v3, NetCDF, Cloud Optimized GeoTIFF, MintPy, Icechunk, GeoZarr for
    bowser, web map tiles, and region jobs for AWS Batch.

    [:octicons-arrow-right-24: Large frames and batch jobs](performance.md)

</div>

## Where to go next

| If you want to | Read |
| --- | --- |
| install disp-xr | [Install](install.md) |
| run it on your own frame | [Try it on your frame](try-it.md) |
| understand ministacks, references and quality layers | [Concepts](concepts.md) |
| find and read granules from NASA's archive | [Finding and reading granules remotely](remote.md) |
| process full frames or run on a cluster | [Large frames and batch jobs](performance.md) |
| see what changed between DISP productions | [Product notes](product_notes.md) |
