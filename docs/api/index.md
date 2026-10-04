# API reference

Import the package as `import disp_xr as dxr`. Every function below is also
reachable from the top-level namespace unless its module is named, and most
are available as methods of the [`.disp` accessor](accessor.md).

| Page | Functions |
|---|---|
| [Find and open](open.md) | `find_products`, `open_disp`, `open_disp_tree`, `open_static`, `search` |
| [Process](process.md) | `correct`, `remove_plate_motion`, `mask`, `valid_mask`, `quality_summary`, `quality_weights`, `stack_disp`, `rereference`, `fit_trajectory`, `Trajectory`, `predict_trajectory`, `fit_velocity`, `fit_noise`, `NoiseResult`, `denoise`, `diagnostics.model_diagnostics`, `diagnostics.significance`, `downsample`, `to_points`, `downsample_hex`, `to_geodataframe`, `quadtree`, `Quadtree`, `empirical_variogram`, `fit_variogram`, `VariogramModel`, `to_units`, `enu_to_los`, `validate_disp` |
| [Spatial models](models.md) | `models.BaseEstimator`, `models.Chain`, `models.Mean`, `models.Trend`, `models.deramp`, `models.Spline`, `models.Nearest`, `models.coordinates`, `models.cross_val_score`, `models.clone` |
| [GNSS](gnss.md) | `gnss.load`, `gnss.stations`, `gnss.get_source`, `gnss.to_los`, `gnss.sample`, `gnss.align`, `residuals.compare`, `residuals.Comparison`, `residuals.residual_spectra`, `residuals.common_mode` |
| [Export](export.md) | `to_zarr`, `to_netcdf`, `to_geotiff`, `to_mintpy`, `to_geozarr`, `build_overviews`, `to_pyramid`, `open_pyramid` |
| [Icechunk and virtual stores](cloud.md) | `to_icechunk`, `open_icechunk`, `build_virtual_frame`, `open_virtual`, `validate_virtual` |
| [Plot](plot.md) | `plot_map`, `plot_point`, `diverging_cmap` |
| [Settings and chunks](config.md) | `config`, `rechunk` |
| [Synthetic data](tutorial.md) | `make_disp_stack` |
