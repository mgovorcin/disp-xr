# GNSS validation against geepers (T34)

Status (2026-10-09): T34.1 is done. For T34.2 the report is written, and
two GNSS date-matching bugs are fixed on feature branches. The results
below are from runs with both fixes. Three items remain open (see "Open
items"). T34.3, the v0.3.0 tag, is not started.

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
`feature/gnss-daily-day` (9062c72) and geepers `feature/gps-start-date`
(607a271). The pre-fix results are in commit 75cbb62.

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
| F08882 | 86 | 2e-14 mm/yr | 0.017 mm/yr | 2e-13 mm |
| F11116 | 103 | 2e-14 mm/yr | 0.040 mm/yr | 0.001 mm |
| F39362 | 15 | 2e-14 mm/yr | 1e-14 mm/yr | 1e-13 mm |

- The InSAR sampling is identical to rounding on all frames.
- With both fixes, the GPS series are identical on F08882 and F39362.
- F11116 keeps a residue of a thousandth of a millimetre: a few days that
  one tool fills from a neighbouring day and the other doesn't.
- Before the fixes, the GPS series differed by up to 0.18 mm (95th
  percentile) and GPS velocities by up to 0.16 mm/yr. Both tools were
  wrong in the same way on afternoon frames, so they agreed with each
  other.

### Pairwise and epoch statistics

| Frame | Pairs | Pair RMSE, p95 / max difference | Epochs | Epoch RMSE, median difference |
| --- | --- | --- | --- | --- |
| F08882 | 4184 | 0.002 / 2.6 mm | 91 | 1.8 mm |
| F11116 | 5563 | 0.009 / 0.38 mm | 131 | 1.1 mm |
| F39362 | 136 | 0.09 / 0.45 mm | 84 | 1.0 mm |

- Pair RMSE agrees closely, because both tools call geepers'
  `pairwise_differential_rmse`. Before the fixes the differences were
  0.06–0.40 mm (95th percentile).
- Epoch RMSE differs at almost every epoch because the station sets
  differ. disp-xr drops every station without a GNSS solution on the stack's
  reference date; geepers keeps them. See open item 1.
- The epoch outliers, up to 46 m on F39362, come from stations geepers
  keeps that carry gross GPS errors. The worst is GINN, with a 195 m jump
  in the UNR series on 2020-03-07. Neither tool screens gross GPS errors.

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

Leave-one-station-out velocity residual, mm/yr:

| Frame | Tie stations | Plane against independent weighted fit | RMSE before / after (degree 1) | RMSE after (degree 0) | Median per station before / after (degree 1) |
| --- | --- | --- | --- | --- | --- |
| F08882 | 92 | 6e-14 mm/yr | 12.6 / 11.9 | 12.1 | 4.7 / 0.9 |
| F11116 | 106 | 7e-14 mm/yr | 27.9 / 3.9 | 4.0 | 28.6 / 1.1 |
| F39362 | 17 | 9e-14 mm/yr | 6.5 / 7.2 | 6.6 | 4.5 / 1.6 |

- `fit_tie` computes the plane it should.
- The median per-station error improves on every frame. The RMSE is driven
  by single stations: UH01 on F08882 (105 mm/yr before the tie, local
  subsidence that a plane can't hold) and GCA2 on F39362 (21 mm/yr).
- On F11116 the tie removes a large offset: before the tie, the residual
  is the datum offset between the DISP reference and absolute GNSS.
- On F39362 the network is clustered. The 12 stations within about 25 km
  of Canberra improve from about 5 to about 1–2 mm/yr each. The four
  distant stations, 65–160 km away (BEG1, TURO, PER5, BRDW), get worse,
  because leaving one out makes the plane extrapolate to it. The east-west
  gradient is real (0.072 ± 0.017 mm/yr/km, measured before the fixes).
- So the median is the better summary, and on a clustered network a score
  per distance from the network centre would say more.

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

## Open items

1. **`compare` drops stations without a GNSS solution on the reference
   date.** `gnss.align` subtracts each station's GNSS at the DISP
   reference date. A station that was installed later, or has a gap on
   that day, becomes NaN everywhere: no GNSS velocity, no pairs, no
   epochs. Stations with at least 10 GNSS epochs that are lost:

   | Frame | Lost stations | Lost GNSS epochs |
   | --- | --- | --- |
   | F08882 | 23 of 112 | 1132 |
   | F11116 | 15 of 121 | 1085 |
   | F39362 | 7 of 30 | 372 |

   Every comparison `compare` makes either fits a rate, which doesn't
   depend on a constant offset, or removes means. So the GNSS could
   instead be referenced per station to its first common epoch, or have
   its mean removed. This changes values (it adds stations), so it waits
   on a maintainer decision.
2. **No gross-error screen on GNSS series** (GINN's 195 m jump): a jump
   or median absolute deviation test in `gnss.load`.
3. **Leave-one-out summary:** a median, or a score per distance from the
   network centre, from `cross_validate_tie`.
