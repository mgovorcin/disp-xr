# Export

Every writer records the disp-xr version, package versions, input granules and
the history of operations in the output's attributes. `to_geotiff` writes
Cloud Optimized GeoTIFFs; pass it the Dataset and a `layer` to keep that
provenance.

::: disp_xr.to_zarr

::: disp_xr.to_netcdf

::: disp_xr.to_geotiff

::: disp_xr.to_mintpy

## GeoZarr for bowser

::: disp_xr.export.geozarr
    options:
      members: false
      show_root_heading: false

::: disp_xr.to_geozarr

::: disp_xr.export.geozarr
    options:
      show_root_heading: false
      members:
        - display_factors
        - bowser_attrs
        - reference_lonlat
        - los_attrs
        - histogram_stats

## Multiscale pyramids

::: disp_xr.export.multiscales
    options:
      members: false
      show_root_heading: false

::: disp_xr.build_overviews

::: disp_xr.to_pyramid

::: disp_xr.open_pyramid
