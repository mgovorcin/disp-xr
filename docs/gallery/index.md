# Tutorials

Each tutorial is a short, executed example. Most run on a synthetic frame
from `disp_xr.tutorial`, so they work anywhere and are re-run every time the
documentation builds; a broken example fails the build. Replace the
synthetic folder with your own granules to run them on real data.

## Basics

<div class="grid cards" markdown>

-   :material-map-search-outline: **[Quick look at a frame](quicklook.py)**

    Open, chain, re-reference, plot a map and a time series.

-   :material-console: **[The command line, end to end](command_line.py)**

    `info`, `stack` and `export` without writing Python.

-   :material-export: **[Exporting results](export.py)**

    Zarr, NetCDF, GeoTIFF, MintPy and pyramids.

-   :material-view-grid-outline: **[Chunks for maps and time series](chunking.py)**

    Pick the layout for the reads you do most.

</div>

## Quality and references

<div class="grid cards" markdown>

-   :material-filter-outline: **[Quality layers and masking](quality_masks.py)**

    The recommended mask, presets and per-criterion masks.

-   :material-scale-balance: **[Quality weights](quality_weights.py)**

    Down-weight instead of mask: similarity, coherence, residual cycles.

-   :material-crosshairs-gps: **[Corrections and reference methods](reference.py)**

    The processing order and all six reference methods compared.

-   :material-angle-acute: **[Line-of-sight geometry and plate motion](geometry.py)**

    Static layers, ENU to LOS, ITRF2014 plate-motion removal.

</div>

## Time series

<div class="grid cards" markdown>

-   :material-chart-line: **[Velocity from a trajectory fit](velocity.py)**

    Offset, velocity, seasonal terms and steps for every pixel.

-   :material-sine-wave: **[Velocity with colored-noise uncertainty](colored_noise.py)**

    OLS, Whittle, MIDAS and GLS sigmas side by side.

-   :material-stethoscope: **[Checking a velocity fit](diagnostics.py)**

    Residual autocorrelation, normality and significance.

-   :material-chart-bell-curve-cumulative: **[Denoising and regular time grids](denoise.py)**

    Kalman and LOWESS, regular grids, refused gaps, declared breaks.

</div>

## Downsampling

<div class="grid cards" markdown>

-   :material-grid: **[Downsampling with quality weights](downsample.py)**

    Regular grids, hexagons and quadtrees with correlated-error sigmas.

</div>

## GNSS

<div class="grid cards" markdown>

-   :material-satellite-uplink: **[Comparing DISP with GNSS](gnss_compare.py)**

    Station velocities, pairwise RMSE and the OPERA requirements.

-   :material-vector-triangle: **[Tying DISP to GNSS](gnss_tie.py)**

    Remove long-wavelength error with a fitted plane, checked by cross-validation.

</div>

## Real data

Run on production granules; the saved outputs are shown and the build does
not run them again.

<div class="grid cards" markdown>

-   :material-home-city-outline: **[Houston subsidence, end to end](../real_data/houston.ipynb)**

    Frame F08882: quality, GNSS reference, MIDAS velocity and the requirements.

-   :material-swap-horizontal: **[Reading a frame across a production change](../real_data/production_change.ipynb)**

    Frame F11116: float64 granules from November 2025 chained with older ones.

-   :material-map-marker-path: **[Creep on the San Andreas Fault](../real_data/san_andreas.ipynb)**

    Frame F11116: the velocity step across the creeping section, against the LOS projection of 26 mm/yr.

-   :material-weather-cloudy: **[Troposphere that follows the terrain](../real_data/troposphere.ipynb)**

    Frame F39362 (Canberra): seasonal displacement-elevation slope per epoch, and what it does to velocity.

</div>

## Cloud and serving

<div class="grid cards" markdown>

-   :material-database-clock-outline: **[Virtual frames and versioned stores](virtual_frames.py)**

    Icechunk: reference granules in place, commit, time travel, tag.

-   :material-map-outline: **[Serving results as web map tiles](web_tiles.py)**

    xpublish-tiles behind MapLibre or Leaflet.

-   :material-cursor-default-click-outline: **[Interactive browsing of a full frame](interactive.py)**

    hvplot and datashader on a lazy stack.

</div>
