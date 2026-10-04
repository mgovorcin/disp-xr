# Process

Corrections and masks apply per granule, before chaining; re-referencing comes
after. See [Concepts](../concepts.md).

## Corrections

::: disp_xr.correct

::: disp_xr.remove_plate_motion

## Quality

::: disp_xr.mask

::: disp_xr.valid_mask

::: disp_xr.quality_summary

::: disp_xr.quality_weights

::: disp_xr.weights
    options:
      members:
        - residual_cycles
        - network_degree
        - max_bandwidth_from_granule
        - residual_cycle_histogram

## Chaining

::: disp_xr.stack_disp

::: disp_xr.NaNPolicy

## Spatial reference

::: disp_xr.rereference

## Time series

::: disp_xr.fit_trajectory

::: disp_xr.Trajectory

::: disp_xr.predict_trajectory

## Velocity for every pixel

::: disp_xr.timeseries.velocity
    options:
      members: false
      show_root_heading: false

::: disp_xr.fit_velocity

::: disp_xr.timeseries.gls
    options:
      members:
        - coherence_groups
        - estimate_group_noise
        - NoiseShape

## Colored noise (one series)

::: disp_xr.timeseries.noise
    options:
      members: false
      show_root_heading: false

::: disp_xr.fit_noise

::: disp_xr.NoiseResult

## Downsampling

::: disp_xr.downsample

::: disp_xr.to_points

::: disp_xr.downsample_hex

::: disp_xr.quadtree

::: disp_xr.Quadtree

::: disp_xr.hexagons
    options:
      show_root_heading: false
      members:
        - hex_axial
        - hex_centers
        - hexagon_vertices
        - cell_polygons

::: disp_xr.to_geodataframe

### Correlated-error sigma

::: disp_xr.empirical_variogram

::: disp_xr.fit_variogram

::: disp_xr.VariogramModel

::: disp_xr.residuals
    options:
      show_root_heading: false
      members:
        - mean_block_correlation
        - mean_mask_correlation
        - n_eff_correlated

## Units, geometry and checks

::: disp_xr.to_units

::: disp_xr.enu_to_los

::: disp_xr.validate_disp
