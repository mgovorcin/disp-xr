# GNSS validation against geepers (T34)

Status (2026-10-09): T34.1 and T34.2 are done; T34.3, the v0.3.0 tag, is
not started. The validation found five problems in disp-xr's GNSS
handling and two in geepers. All are fixed on unmerged feature branches,
and the results below are from runs with every fix. With them, disp-xr
agrees with geepers to rounding, and the GNSS tie improves the
leave-one-out score on all three frames.

This is the M2 gate "GNSS tie validated". disp-xr's GNSS comparison
(`residuals.compare`, T32) and GNSS tie (`models.fit_tie`, T33) are run on
the three T29 frames. The comparison is checked against the geepers
workflow (`geepers.workflows.main`, the geepers command line) on the same
stack, stations and sampling radius (100 m). The tie has no geepers
counterpart, so it is checked against an independent weighted
least-squares plane and by leave-one-station-out cross-validation.

Scripts are in `studies/gnss_validation/` at the repository root:
`extract.py` writes the station blocks and `validate.py` runs both tools
and writes `results/`. These results come from disp-xr
`feature/tie-loso-median` (924fbf2, the top of a stack of five branches,
listed under "Discrepancies and fixes") and geepers
`feature/gps-start-date` (607a271). The pre-fix results are in commit
75cbb62.

## Frames

These are the T29 frames (`studies/split_metric/frames.yaml`), unmasked
study stacks. Each stack is referenced spatially to the median of a
65 x 65 pixel box around one GNSS station, and temporally to its first
epoch.

| Frame | Site | Epochs | Acquisition time (UTC) | Reference station | Stations matched by both tools |
| --- | --- | --- | --- | --- | --- |
| F08882 | Houston | 91 | 00:26 | UHEB | 133 |
| F11116 | Central California | 131 | 14:07 | P180 | 124 |
| F39362 | Canberra | 84 | 19:16 | TURO | 26 |

## Station blocks

Reading a station's series from a study stack reads one chunk per epoch.
The chunks are 1024 x 1024 pixels with one epoch each. `extract.py`
copies the 64 x 64 pixel block around every station into
`stations.zarr` on the same grid, chunked as whole series. Each source
chunk is read once, and each worker keeps only the blocks it needs. The
first version kept the whole source chunk, all epochs and layers (about
2.3 GB a task). On the shared node it ran out of memory and stalled:
75 min for F08882. Keeping only the blocks took F11116 to 4 min and
F39362 to under 2 min. Every station has data at its pixel; 90%, 80% and
77% of epochs are valid per frame.

## Results

### Like-for-like: double difference

The two tools' own outputs are not directly comparable, for the reasons
in "The station tables do not compare as they stand" below. The
like-for-like check works on each tool's sampled series at the stations:

1. InSAR and GPS are both referenced to the stack's reference station, so
   both carry the same spatial reference.
2. Only the epochs both tools hold for a station are kept, each keyed by
   its acquisition's UTC day.
3. Each station's mean over those epochs is removed.
4. One least-squares rate is fitted to each series.

A station is usable with at least 10 epochs over at least 2 years.

| Frame | Usable stations | InSAR velocity, max difference | GPS velocity, max difference | GPS series, p95 difference |
| --- | --- | --- | --- | --- |
| F08882 | 108 (86 before the station fix) | 5e-14 mm/yr | 0.017 mm/yr | 2e-13 mm |
| F11116 | 115 (103) | 2e-14 mm/yr | 0.040 mm/yr | 2e-13 mm |
| F39362 | 20 (15) | 2e-14 mm/yr | 2e-14 mm/yr | 1e-13 mm |

- The InSAR sampling is identical to rounding on all frames.
- With the fixes, the GPS series are identical on all three frames.
- F11116's GPS velocities keep a residue of 0.04 mm/yr: a few days that
  one tool fills from a neighbouring day and the other doesn't.
- Before the fixes, the GPS series differed by up to 0.18 mm (95th
  percentile) and GPS velocities by up to 0.16 mm/yr. Both tools were
  wrong in the same way on afternoon frames, so they agreed with each
  other.

### Pairwise and epoch statistics

| Frame | Pairs | Pair RMSE, p95 / max difference | Epochs | Epoch RMSE, median difference |
| --- | --- | --- | --- | --- |
| F08882 | 7404 | 0.001 / 23 mm | 91 | 1.0 mm |
| F11116 | 7249 | 0.007 / 0.38 mm | 131 | 0.06 mm |
| F39362 | 287 | 0.15 / 0.68 mm | 84 | 0.08 mm |

- Pair RMSE agrees closely, because both tools call geepers'
  `pairwise_differential_rmse`. Before the fixes the differences were
  0.06–0.40 mm (95th percentile).
- With the station fix, both tools use the same stations at every epoch
  except one F39362 epoch. Before the fix, they differed at almost every
  epoch, and so did the epoch RMSE (by about 1 mm, with outliers up to
  46 m).
- On the station-days both tools hold, the epoch RMSE is identical
  (0.000 mm on F08882). The remaining difference, 1 mm on F08882, comes
  from 89 of about 7,900 rows that only geepers keeps:
  - DEN4's segment before its relocation, which disp-xr drops.
  - Days whose own GPS solution is missing. geepers then takes the
    neighbouring day on either side; disp-xr takes a solution only within
    24 h of the acquisition.

  The second is a difference in gap-filling policy, not an error. The F08882
  pair maximum (23 mm) is the same rows.
- geepers keeps relocated stations, so its epoch RMSE still carries GINN's
  1 km jump on two F39362 epochs (46 m).

### The station tables do not compare as they stand

51 of 133 F08882 stations differ by more than 1 mm/yr in InSAR velocity
between the two tools' station tables, up to 385 mm/yr. The two tools
estimate different things:

- geepers fits GPS with MIDAS over the whole GPS record. It fits InSAR
  only on the dates that also have a GPS solution: 2 epochs for TXS9
  (388 mm/yr), and 10 for UHC8.
- disp-xr fits both quantities over the whole InSAR record.

Once both are restricted to the same epochs and estimator (the double
difference above), they agree. disp-xr's choice is the right one for
Cal/Val. geepers' station table should not be read as an InSAR velocity
for stations with short GPS records.

### GNSS tie

Leave-one-station-out velocity residual, mm/yr, degree-1 plane, with all
the fixes. In brackets: the first run, before any fix.

| Frame | Tie stations | Plane against independent weighted fit | RMSE before / after | Median per station before / after |
| --- | --- | --- | --- | --- |
| F08882 | 109 (92) | 2e-14 mm/yr | 11.9 / 11.1 (12.6 / 11.9) | 4.7 / 0.95 (4.7 / 0.9) |
| F11116 | 116 (106) | 2e-14 mm/yr | 28.0 / 3.8 (27.9 / 4.2) | 28.7 / 1.1 (28.6 / 1.1) |
| F39362 | 21 (17) | 1e-13 mm/yr | 4.0 / 2.8 (5.1 / 5.7) | 4.0 / 1.6 (4.5 / 1.6) |

- `fit_tie` computes the plane it should.
- The tie improves both scores on every frame. On F39362 the RMSE used to
  get worse; GINN's relocation and stations with short records were behind
  it.
- On F11116 the tie removes a large offset: before the tie, the residual
  is the datum offset between the DISP reference and absolute GNSS.
- On F08882 the RMSE stays high because of UH01 (105 mm/yr): local
  subsidence that a plane can't hold, and that the tie should not absorb.
  Such stations are what `exclude=` is for. The median, now reported by
  `cross_validate_tie`, describes the typical station: about 1 mm/yr after
  the tie on every frame.

## Discrepancies and fixes

1. **The first epoch was paired with the next day's GPS (geepers).**
   `workflows.main` bounds the GPS download with the first acquisition
   *time*. Daily solutions are stamped at midnight, so that day's solution
   was filtered out. Fixed on geepers `feature/gps-start-date` (6ff4264):
   the start bound is floored to its day.
2. **Afternoon acquisitions were paired with the next day's GPS (both
   tools).** Both matched the GPS row with the nearest timestamp. A daily
   solution stamped at midnight is nearest to the *next* day for any
   acquisition after 12:00 UTC, but a UNR daily solution covers the whole
   UTC day. Effect of the fix:

   | Frame | Acquisition (UTC) | Epoch GPS change, median / p95 | Velocity change, p95 / max |
   | --- | --- | --- | --- |
   | F08882 | 00:26 | 0 / 0 mm | 0 / 0 mm/yr |
   | F11116 | 14:07 | 2.6 / 8.3 mm | 0.41 / 4.5 mm/yr |
   | F39362 | 19:16 | 2.2 / 11.5 mm | 0.37 / 0.40 mm/yr |

   - disp-xr, `feature/gnss-daily-day` (93c28ff, 9062c72): the new
     `gnss.centred` stamps all-midnight series at noon before `align` and
     the sigma alignment in `compare`. This changes values; it is in the
     changelog.
   - geepers, `feature/gps-start-date` (607a271): `merge_gps_insar`
     matches daily GPS on the acquisition's floored day.
3. **`compare` dropped stations without GNSS on the DISP reference date.**
   `gnss.align` zeroes GNSS at that date, so a station installed later, or
   with a gap that day, was NaN everywhere. Lost before the fix (stations
   with at least 10 GNSS epochs):

   | Frame | Lost stations | Lost GNSS epochs |
   | --- | --- | --- |
   | F08882 | 23 of 112 | 1132 |
   | F11116 | 15 of 121 | 1085 |
   | F39362 | 7 of 30 | 372 |

   Fixed on disp-xr `feature/gnss-station-reference` (c282c1d). `compare`
   now references both InSAR and GNSS per station to the first epoch they
   share (`reference_epoch` in `stations`). This leaves stations with GNSS
   on the reference date unchanged, and gives the others a double
   difference in time with no offset. It changes values; it is in the
   changelog.
4. **Velocities from a few epochs entered the tie.** Keeping late
   stations (item 3) let in rates from 2 shared epochs: 94 mm/yr for TXH3.
   `compare(min_epochs=10, min_span_years=2.0)` now leaves such velocities
   NaN (`velocity_usable`). The station's displacement series stays in the
   pair and epoch statistics. On `feature/gnss-velocity-gate` (f8ff61d).
5. **Monument relocations under one station id.** GINN (UNR) jumps about
   1 km between 2019 and 2020, and so does DEN4 on F08882. Before, GINN's
   two segments became a 7e5 mm/yr velocity. `gnss.load(max_step_m=10.0)`
   now splits a record at any 3-D step above 10 m and keeps the longest
   segment (`relocated` coordinate). Near-field coseismic steps reach about
   5 m and stay. On `feature/gnss-relocation-screen` (03faba3). geepers has
   no such screen yet.
6. **Leave-one-out summary.** `cross_validate_tie` also reports the median
   per-station score (`median_before`, `median_after`). On
   `feature/tie-loso-median` (924fbf2).

The disp-xr branches are stacked, each on the one before:
`feature/gnss-daily-day`, `feature/gnss-station-reference`,
`feature/gnss-velocity-gate`, `feature/gnss-relocation-screen`,
`feature/tie-loso-median`. Each changes values and has a changelog entry.

## Open items

1. Push the branches, open the pull requests and merge them in order;
   then merge this study branch.
2. geepers: add a relocation screen like `drop_relocations`. Its epoch
   RMSE still carries GINN's jump.
3. Optional: align the gap-filling policy of the two tools (a neighbouring
   day on either side, or within 24 h), if exact agreement of epoch RMSE
   is wanted. It changes about 1% of station-days.
4. T34.3: tag v0.3.0, update the changelog and publish the docs.
