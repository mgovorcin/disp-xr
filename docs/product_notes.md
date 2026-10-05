# Product notes

Differences between DISP productions that affect reading a frame, found
while using real frames. Each note says what changed, how it was found and
what disp-xr does about it.

## DISP-S1 displacement precision: float32 before November 2025, float64 after

**What changed.** DISP-S1 granules produced in November 2025 and later store
the `displacement` layer as float64. Earlier productions store it as
float32. No other layer changes: the corrections, `short_wavelength_displacement`,
the quality layers and the masks have the same dtype, chunks and
compression in every production.

**Not a reprocessing.** The later granules continue the time series rather
than replace it. In frame F11116 (central California), the April 2025
production ends at 2024-10-03. The first November 2025 granule is a
ministack referenced on that date (`20241003T140837Z_20241015T140837Z`). A
time series that runs past October 2024 therefore needs both productions.

**How it was found.** The split-metric study (T29, 2026-10-04) downloaded
all 389 F11116 granules from CMR. `open_disp` refused them with
`MixedProductionError`:

| Production | Granules | `displacement` | Secondary dates |
| --- | --- | --- | --- |
| April 2025 | 314 | float32 | 2016-07-05 to 2024-10-03 |
| November 2025 | 45 | float64 | from 2024-10-15 |
| August 2026 | 30 | float64 | to 2026-04-26 |

Every 2D layer of one granule from each production was compared with h5py:

```text
displacement  float32  float64  float64
same: connected_component_labels, corrections/ionospheric_delay,
      corrections/perpendicular_baseline, corrections/solid_earth_tide,
      estimated_phase_quality, persistent_scatterer_mask, phase_similarity,
      recommended_mask, short_wavelength_displacement, shp_counts,
      temporal_coherence, timeseries_inversion_residuals, water_mask
```

**What disp-xr does.** `open_disp` reads granules that differ only in the
precision of a float layer:

- it raises a `MixedPrecisionWarning` that lists each production;
- it casts the layer to float32 (float32 resolves about 10 nm on a 10 cm
  displacement);
- it records the layer in `attrs["mixed_precision"]`.

Any other difference (an integer layer becoming float, a new fill value,
other compression or chunks) still raises `MixedProductionError`. So does
`build_virtual_frame`, because a virtual store references the stored
chunks and needs one dtype per array.

```python
import warnings
import disp_xr as dxr
from disp_xr.product import MixedPrecisionWarning

with warnings.catch_warnings():
    warnings.simplefilter("ignore", MixedPrecisionWarning)  # once you know
    ds = dxr.open_disp("F11116/")
ds.attrs["mixed_precision"]  # ['displacement']
```

## DISP-S1 time encoding changed in November 2025

**What changed.** The same productions also encode `time` and
`reference_time` differently. The earlier granules store 0 seconds since the
acquisition time itself (`seconds since 2024-10-03 14:08:37.435763`), which
decodes exactly. The later granules store float seconds since 2010
(`seconds since 2010-01-01 00:00:00.000000`), which decodes a few
nanoseconds off: `2024-10-03 14:08:37.435763008`.

**Why it matters.** Ministacks are chained through exact dates: each
ministack's reference date must be an epoch of the previous one. In F11116
the November 2025 ministack references 2024-10-03, but that date read from
the April 2025 granule differs by 8 ns, so `stack_disp` raised
`MissingCrossoverError`.

**What disp-xr does.** `open_disp` reads `time` and `reference_time` to the
whole second (floored), as granule names give them
(`20241003T140837Z`). Epochs of one frame are days apart, so nothing below a
second is used.

## Open questions for OPERA

- Is the float64 displacement intended? Its granules are larger, and float32
  already holds more precision than the measurement has.
- Could the time encoding stay one style across productions?
