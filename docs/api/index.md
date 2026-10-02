# API reference

Import the package as `import disp_xr as dxr`. Every function below is also
reachable from the top-level namespace unless its module is named, and most
are available as methods of the [`.disp` accessor](accessor.md).

| Page | Functions |
|---|---|
| [Find and open](open.md) | `find_products`, `open_disp`, `open_disp_tree`, `open_static`, `search` |
| [Process](process.md) | `correct`, `remove_plate_motion`, `mask`, `valid_mask`, `quality_summary`, `stack_disp`, `rereference`, `to_units`, `enu_to_los`, `validate_disp` |
| [Export](export.md) | `to_zarr`, `to_netcdf`, `to_geotiff`, `to_mintpy` |
| [Icechunk and virtual stores](cloud.md) | `to_icechunk`, `open_icechunk`, `build_virtual_frame`, `open_virtual`, `validate_virtual` |
| [Plot](plot.md) | `plot_map`, `plot_point`, `diverging_cmap` |
| [Settings and chunks](config.md) | `config`, `rechunk` |
| [Synthetic data](tutorial.md) | `make_disp_stack` |
