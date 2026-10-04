# GNSS

GNSS stations next to DISP, with [geepers](https://github.com/opera-adt/geepers)
as the engine (install the `gnss` extra). geepers loads, caches and reads the
station series; `disp_xr.gnss` returns them as xarray and aligns them with
DISP.

```python
gps = dxr.gnss.load(ds, start="2017-01-01")            # (station, time) E/N/U
gps = dxr.gnss.to_los(gps, dxr.open_static(ds))        # + los, sigma_los
insar = dxr.gnss.sample(ds, gps, buffer_m=100)         # DISP at the stations
gps_on_disp = dxr.gnss.align(gps["los"], insar.time)   # DISP epochs and reference
residual = insar - gps_on_disp
```

::: disp_xr.gnss
    options:
      members: false
      show_root_heading: false

::: disp_xr.gnss.load

::: disp_xr.gnss.stations

::: disp_xr.gnss.get_source

::: disp_xr.gnss.to_los

::: disp_xr.gnss.sample

::: disp_xr.gnss.align
