# Split metric and weight defaults (T29)

Status: frames chosen (T29.1). The runner, criteria, sweep and decision
(T29.2 to T29.6) are not written yet.

This study picks the default quadtree split metric (`displacement`,
`velocity` or `structure`, T28) and the T26 weight parameters (`penalty`,
`f0`, floors, exponents, and the minimum residual weight of T26.6). It
compares them at 1, 2 and 10 km cells on four criteria:

1. reconstruction error against the 30 m field (velocity and displacement),
2. GNSS agreement per cell (T32),
3. gradient preservation across known features,
4. cell count and runtime.

Scripts and inputs live in `studies/split_metric/` at the repository root.

## Frames

Three DISP-S1 frames, one per kind of signal. Details (paths, bounds,
dates, product version and why each was chosen) are in
`studies/split_metric/frames.yaml`.

| Frame | Role | Site | Granules | Dates | GNSS used / inside |
| --- | --- | --- | --- | --- | --- |
| F08882 | subsidence | Houston, upper Texas Gulf Coast | 269 | 2016-09 to 2026-02 | 97 / 144 |
| F11116 | tectonic, topography-correlated troposphere | Central California: creeping San Andreas, San Joaquin Valley | 314 | 2016-07 to 2024-10 | 130 / 149 |
| F39362 | quiet (troposphere control) | Canberra and the Snowy Mountains | 247 | 2016-07 to 2025-04 | 25 / 34 |

- F11116 has 389 granules on disk from three productions. The 75 granules
  produced in November 2025 and August 2026 store displacement as float64,
  and `open_disp` refuses to mix productions, so the study uses the 314
  granules of April 2025 (up to 2024-10-03).
- F08882 is also the cal-disp frame, so GNSS results can be compared with
  that work.
- F11116 has a sharp velocity step at the creeping fault trace, the
  gradient-preservation test case for a tectonic signal. With the Coast
  Ranges and Sierra foothills it is also the deforming site with
  topography-correlated troposphere that T31.7 asked for.
- F39362 is the control: little deformation but about 1500 m of relief from
  the coast to the Snowy Mountains, so troposphere that follows elevation
  shows up without a deformation signal under it. Its granules are product
  version 0.10 (a stakeholder run outside North America), not 1.0.

New Orleans (F44055) was considered and dropped: it is a second subsidence
site and has no GNSS prepared.

### GNSS stations

`studies/split_metric/stations.py` writes `stations/<frame>.csv`: the UNR
stations inside each frame's grid over the frame's dates, with first and
last day, number of daily solutions and the share of the DISP span covered.
A station is used (`use = True`) when it covers at least half of the span
and has at least 2.5 years of data.

### DISP-NISAR frames for T35.5

Frames of the Whirlwind NISAR campaign, which already has GNSS for each:

| NISAR frame | Pairs with | GNSS stations |
| --- | --- | --- |
| 042_D_070 | F11116 (same area) | 232 |
| 062_A_016 | F08882 (same area) | 100 |
| 133_A_022 | F39362 (no campaign frame covers Australia; a quiet site with moderate relief in the US Northeast instead) | 77 |
