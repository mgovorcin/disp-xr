# Concepts

## Granules and frames

A DISP granule is one NetCDF file holding the displacement between two
dates, a reference date and a secondary date, on a fixed grid called a
frame. A DISP-S1 frame is about 7,000 x 9,000 pixels at 30 m in a UTM zone. A
frame's history is a few hundred granules.

## Ministacks and reference dates

The processor (dolphin) works on ministacks: blocks of about 15 acquisitions
that share one reference date. Every granule in a ministack measures
displacement from that ministack's reference date, so the raw series restarts
at zero at each new ministack.

The next ministack's reference date is the last secondary date of the
previous one. That shared epoch, the crossover, links them:
[`stack_disp`](api/process.md#disp_xr.stack_disp) adds to each ministack the
displacement accumulated at its crossover, so the whole series becomes
relative to the first reference date. A missing crossover granule breaks the
chain, and `stack_disp` raises `MissingCrossoverError`. A NaN at a crossover
makes later epochs of that pixel NaN (`nan_policy="propagate"`, the default).

`open_disp` returns the series as stored, relative to each ministack
reference; `reference_time` says which one. `stack_disp` returns it chained,
with `attrs["reference_date"]` set and a zero epoch prepended at the first
reference date.

## Spatial reference

DISP displacement is relative to a reference point chosen by the processor.
[`rereference`](api/process.md#disp_xr.rereference) subtracts, at every epoch,
the value of a pixel you choose or the median of an area, so the series is
relative to ground you trust.

## Units and sign

Products store meters, positive toward the satellite (uplift or motion along
the look direction toward it). [`to_units`](api/process.md#disp_xr.to_units)
converts length layers to `cm` or `mm`; plots use `mm` by default.

## Quality layers

Each granule carries quality layers alongside displacement:

| Layer | Meaning |
|---|---|
| `recommended_mask` | 1 where the processor recommends the pixel |
| `temporal_coherence`, `phase_similarity` | 0 to 1; higher is better |
| `connected_component_labels` | unwrapping regions; 0 is unreliable |
| `water_mask` | 0 over water |
| `persistent_scatterer_mask` | 1 for persistent scatterers |
| `shp_counts`, `estimated_phase_quality`, `timeseries_inversion_residuals` | diagnostics |

[`mask`](api/process.md#disp_xr.mask) sets displacement to NaN where pixels
fail: the `"recommended"` preset uses `recommended_mask`, `"strict"` adds
connected components, water and a temporal-coherence threshold.
[`quality_summary`](api/process.md#disp_xr.quality_summary) maps how often
each pixel passes.

## Corrections

The `/corrections` group holds the ionospheric delay and solid-earth tide that
the processor estimated but did not subtract.
[`correct`](api/process.md#disp_xr.correct) subtracts them once, before
chaining, and records what it applied; it refuses a correction the product
already applied (DISP-NISAR ionosphere).

## Chunks

Data is opened lazily in the layout of the files: one epoch and one block per
dask chunk (`chunks="native"`). Each operation then reads what it needs:

| Question | Reads |
|---|---|
| a map of one epoch | that epoch, plus one crossover epoch per earlier ministack |
| a pixel's time series | one block from every granule |
| a per-pixel fit | rechunks to whole time series itself |

[`rechunk`](api/config.md#disp_xr.rechunk) switches layouts when one kind of
access repeats. Stores written with `to_zarr` choose their layout with
`preset=`: `"timeseries"` for per-pixel work, `"maps"` for maps.
