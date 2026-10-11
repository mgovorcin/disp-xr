# Accessor

Importing `disp_xr` registers a `.disp` accessor on every `xarray.Dataset` and
`xarray.DataArray`. Each method delegates to the function of the same name,
so `ds.disp.chain()` and `dxr.chain(ds)` are the same operation.

::: disp_xr.accessor.DispDatasetAccessor
    options:
      inherited_members: true

::: disp_xr.accessor.DispArrayAccessor
    options:
      inherited_members: true
