# GNSS validation against geepers (T34)

Status (2026-10-09): T34.1 done, and T34.2 written with the discrepancies
found. Two of them change values and wait on a maintainer decision (see
"Open items"). T34.3, the v0.3.0 tag, is not started.

This is the M2 gate "GNSS tie validated". disp-xr's GNSS comparison
(`residuals.compare`, T32) and GNSS tie (`models.fit_tie`, T33) are run on
the three T29 frames. The comparison is checked against the geepers
workflow (`geepers.workflows.main`, the geepers command line) on the same
stack, stations and sampling radius (100 m). The tie has no geepers
counterpart, so it is checked against an independent weighted
least-squares plane and by leave-one-station-out cross-validation.

Scripts are in `studies/gnss_validation/` at the repository root:
`extract.py` writes the station blocks and `validate.py` runs both tools
and writes `results/`.

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
in the next section. The like-for-like check works on each tool's sampled
series at the stations:

1. InSAR and GPS are both referenced to the stack's reference station, so
   both carry the same spatial reference.
2. Only the epochs both tools hold for a station are kept.
3. Each station's mean over those epochs is removed. This is done instead
   of zeroing at the first epoch, because geepers mis-dates that epoch
   (see "Discrepancies" below).
4. One least-squares rate is fitted to each series.

A station is usable with at least 10 epochs over at least 2 years.

| Frame | Usable stations | InSAR velocity, max difference | GPS velocity, max difference | InSAR series, p95 difference | GPS series, p95 difference |
| --- | --- | --- | --- | --- | --- |
| F08882 | 86 | 2e-14 mm/yr | 0.15 mm/yr | 1e-13 mm | 0.08 mm |
| F11116 | 103 | 2e-14 mm/yr | 0.06 mm/yr | 1e-13 mm | 0.07 mm |
| F39362 | 15 | 2e-14 mm/yr | 0.10 mm/yr | 1e-13 mm | 0.18 mm |

The InSAR sampling is identical to rounding. The GPS series agree to a
tenth of a millimetre. The small remainder is GPS days that one tool
pairs with a neighbouring date that the other doesn't.

### Pairwise and epoch statistics

| Frame | Pairs | Pair RMSE, p95 / max difference | Epochs | Epoch RMSE, median difference |
| --- | --- | --- | --- | --- |
| F08882 | 4184 | 0.35 / 10.9 mm | 91 | 1.8 mm |
| F11116 | 5563 | 0.06 / 0.38 mm | 131 | 1.0 mm |
| F39362 | 136 | 0.40 / 0.59 mm | 84 | 0.9 mm |

Pair RMSE agrees closely, because both tools call geepers'
`pairwise_differential_rmse`. Epoch RMSE differs by about 1 mm, with
outliers up to 46 m on F39362. These come from the station sets: geepers
pairs a few stations with isolated GPS days that disp-xr doesn't match.
The worst case is GINN, whose two matched GPS days carry a 195 m gross
error in the UNR series (a jump on 2020-03-07). Neither tool screens
gross GPS errors.

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

| Frame | Tie stations | Plane against independent weighted fit | Leave-one-out RMSE before / after (degree 1) | Degree 0 |
| --- | --- | --- | --- | --- |
| F08882 | 92 | 6e-14 mm/yr | 12.6 / 11.9 mm/yr | 12.1 |
| F11116 | 106 | 5e-14 mm/yr | 27.9 / 4.2 mm/yr | 4.3 |
| F39362 | 17 | 2e-13 mm/yr | 5.1 / 5.7 mm/yr | 5.2 |

- `fit_tie` computes the plane it should.
- On F11116 the tie removes a large offset: before the tie, the residual
  is the datum offset between the DISP reference and absolute GNSS.
- On F08882 the gain is small. The residual there is dominated by local
  subsidence that a plane can't hold (scatter 11.8 mm/yr).
- On F39362 the leave-one-out RMSE gets worse even though the east-west
  gradient is real: 0.072 ± 0.017 mm/yr/km, about 16 mm/yr across the
  stations. The network is clustered. The 12 stations within about 25 km
  of Canberra improve from about 5 to about 1 mm/yr each. The four
  distant stations, 65–160 km away (BEG1, TURO, PER5, BRDW), get worse,
  because leaving one out makes the plane extrapolate to it. GCA2 is an
  outlier before the tie (12.6 mm/yr).
- So on a clustered network the leave-one-out RMSE measures the network's
  geometry as much as the tie. A median, or a score per distance from the
  network centre, would describe it better.

## Discrepancies

1. **The first epoch is paired with the next day's GPS (geepers only).**
   `workflows.main` bounds the GPS download with the first acquisition
   *time*. Daily solutions are stamped at midnight, so that day's solution
   was filtered out, and the first epoch matched the next day's. That
   epoch is the DISP temporal reference. Fixed in geepers on branch
   `feature/gps-start-date` (6ff4264, with regression tests; not pushed),
   and written up in that repository's
   `dev_notes/gps-start-date-off-by-one.md`.
2. **Afternoon acquisitions are paired with the next day's GPS (both
   tools).** disp-xr's `gnss.align` and geepers' `merge_gps_insar` take the
   GPS row with the nearest timestamp. A daily solution stamped at
   midnight is nearest to the *next* day for any acquisition after
   12:00 UTC, but a UNR daily solution covers that whole UTC day. Effect,
   measured against matching on the acquisition's own day:

   | Frame | Acquisition (UTC) | Epoch GPS difference, median / p95 | Velocity difference, p95 / max |
   | --- | --- | --- | --- |
   | F08882 | 00:26 | 0 / 0 mm | 0 / 0 mm/yr |
   | F11116 | 14:07 | 2.6 / 8.3 mm | 0.41 / 4.5 mm/yr |
   | F39362 | 19:16 | 2.2 / 11.5 mm | 0.37 / 0.40 mm/yr |

   The two tools agree with each other only because they share the
   problem. Fix: match each acquisition to the solution of its own UTC
   day (floor the epoch to the day for daily solutions), in both tools.
3. **Gross GPS errors are not screened (both tools).** For example,
   GINN's 195 m jump. It inflates geepers' epoch RMSE to tens of metres on
   two F39362 epochs.
4. **Leave-one-out RMSE on clustered networks.** See "GNSS tie" above.
   This is a reporting choice, not a defect.

## Open items

- Item 2 changes values in disp-xr (the GNSS series at every afternoon
  epoch). Waiting on a maintainer decision. Then the same fix goes into
  geepers' `merge_gps_insar` as a follow-up on `feature/gps-start-date`.
- Item 3: a gross-error screen for GNSS series (a jump or median absolute
  deviation test in `gnss.load`).
- Item 4: a median or per-distance score from `cross_validate_tie`.
- Reruns of this report after items 1 and 2 land: the double difference
  must still agree, and the epoch statistics should move closer together.
