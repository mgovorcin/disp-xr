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

## Units, geometry and checks

::: disp_xr.to_units

::: disp_xr.enu_to_los

::: disp_xr.validate_disp
