# V&V of the linearity test and change points

Status (2026-11-02): done for the settings below, on central California and Houston. Step 1
of the recommendation is in `fit_series`; whether the draft label comes off is the owner's
decision.

This study asks three things about per-series models on DISP-S1 at 5 x 5 pixels and 30-day
medians (`fit_series` defaults): can the draft label come off `model="linearity"`, which
settings should it use, and are geepers' change points worth adding to disp-xr?

## Answers

1. **The draft label stays until the owner decides.** The gate's false alarms hold on both
   regions (none in 540 Houston pairs). Detection rests on 13 station pairs in California,
   because Houston has no real departure large enough to test it.
2. **Read a series as non-linear only if `min(p_quadratic, p_piecewise) < 0.01` and
   `departure > 10 mm`.** `linearity_model` alone calls 59 % of straight lines non-linear at
   typical noise; the gate calls 2 %.
3. **Use the verdict and the departure, not the break date or `rate_after`.** Departures
   repeat between tracks (r = 0.99). Break dates for the same pair differ by a median 1.6
   years (3 pairs), and `rate_after` predicts the next three years worse than the plain line.
4. **Take rate uncertainty and the validity horizon from a noise model with a random walk
   (`RWFNWN`), and the verdict from `FNWN`.** With `FNWN`, rate errors are about 5 times
   the reported sigma on synthetic lines at typical noise and 3.6 times against GNSS in
   California. With `RWFNWN` they are 1.7 times in California and 0.84 in Houston. A
   random walk in the verdict's noise model absorbs real signals: it found 1 of the 13
   strong GNSS pairs.
5. **Do not add change points as a ground-signal product.** At typical noise they appear
   on 88 % of straight lines, and rate changes on 60 %. **Offsets above 15 mm are worth
   adding later as a data-quality flag:** no false alarms on synthetic lines, every step
   of 28 mm or more found at the right size, and in Houston they found a bad ministack. `detect_changepoints` is enough;
   `fit_sparse_changepoints` gives the same answer on 99.5 % of pairs.

## Data and method

Scripts and the full tables (`results/summary.txt`) are in `studies/series_vnv/`. Series are
binned to 30 days; all methods run on the same series, with geepers at the commit disp-xr pins.

- **Synthetic** (`synthetic.py`): 10,400 series with known answers, seasonal terms and
  noise from real data. `noise_levels.py` measures windows 500 m apart: a median random walk
  of 2.4 to 3.9 mm/sqrt(yr) per frame, so 3.5 is "typical" here and 1.0 "quiet".
- **Real** (`real.py`): the windows and station pairs of the smooth-trend validation in
  geepers, F11116 and F38502. 445 pairs within 25 km are differenced in pixels and in GNSS,
  from station sets of 53, 42 and 29, plus 125 differences of windows 500 m apart. Of these
  pairs, 43 come from stations screened for signal (the `f38502v` set).
- **Second track** (`fetch_windows.py`): the screened stations inside F11116, read from
  local granules, so strong signals are also seen on another track.
- **Houston** (`houston_stations.py`, `fetch_windows.py`, `real.py`): F08882 with its
  descending partner F38238, 95 stations with six years or more. See "Houston".
- **Holdout** (`real.py`): fit all but the last three years, then predict the last year's
  mean, against GNSS and against the series itself.
- **`fit_series` itself** (`end_to_end.py`): same model as the direct calls on 93 % of 442
  pairs, same `significant` on 98 %, median `velocity_sigma` 0.89 mm/yr against 0.85 and
  `validity_years` 10.6 against 11.4. The two differ in where the bins start and in
  differencing before or after binning, so a plain verdict can turn on such details; the gated
  one rarely does.

## False alarms

Share of 200 synthetic straight lines flagged, 10 years:

| Random walk, mm/sqrt(yr) | 0 | 1 | 2 | 3.5 | 5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `linearity_model` not linear | 0.04 | 0.22 | 0.43 | 0.59 | 0.67 |
| **p < 0.01 and departure > 10 mm** | 0 | 0 | 0 | **0.02** | 0.14 |
| Same test, `RWFNWN` noise | 0.04 | 0.06 | 0.08 | 0.05 | 0.06 |
| Any change point | 0.13 | 0.39 | 0.65 | 0.88 | 0.95 |
| Rate change point | 0.02 | 0.14 | 0.36 | 0.60 | 0.65 |
| Offset above 15 mm | 0 | 0 | 0 | 0 | 0 |
| Smooth trend chosen (limit 5) | 0 | 0 | 0.01 | 0 | 0 |

Shorter records do not raise the gated rate: 0 at 5 years, 0.01 at 7, at 3.5 mm/sqrt(yr).

The geepers docstring says about one linear series in five is flagged when the noise is a
random walk. That holds near 1 mm/sqrt(yr); at DISP-S1 levels it is more like three in five.

This is why the methods seemed to disagree. On the F11116 block tried earlier, 23 of 40
windows had a change point, 9 were non-linear and 4 smooth; the table gives the same order on
series known to be straight. They do not see different signals. They differ in how much
random walk they mistake for change.

Differences of windows 500 m apart are real data, and should be close to straight:

| Share flagged | F11116 | F38502 | F38502, screened | All |
| --- | ---: | ---: | ---: | ---: |
| `linearity_model` not linear | 0.39 | 0.42 | 0.55 | 0.44 |
| p < 0.01 and departure > 10 mm | 0.14 | 0.05 | 0.06 | 0.09 |
| Any change point | 0.73 | 0.51 | 0.84 | 0.68 |
| Offset above 15 mm | 0.10 | 0.02 | 0.10 | 0.07 |

These are upper bounds, since ground can differ over 500 m. The gate's 9 % is still higher
than the synthetic 2 %, so expect more than 2 % in practice.

## Detection

Share flagged by the gate, quiet / typical noise:

| Signal, 10 years | Gate | `linearity_model` |
| --- | --- | --- |
| Hinge 4 mm/yr, mid-record | 0.03 / 0.30 | 1.00 / 0.92 |
| Hinge 8 mm/yr, at 1/4, 1/2, 3/4 of record | 0.68, 0.98, 0.16 / 0.52, 0.81, 0.40 | 1.00 / 0.96 to 1.00 |
| Acceleration 1 mm/yr² | 0.08 / 0.43 | 1.00 / 0.94 |
| Acceleration 2 mm/yr² | 1.00 / 0.97 | 1.00 / 1.00 |
| Decay 20 mm, one-year time constant | 0.01 / 0.26 | 0.98 / 0.87 |
| Decay 60 mm | 1.00 / 1.00 | 1.00 / 1.00 |
| **Step 28 mm** | **0 / 0.32** | 0.02 / 0.46 |
| **Step 56 mm** | **0 / 0.10** | 0 / 0.12 |

The gate misses departures under 10 mm by design, so on quiet series it gives up power that
`linearity_model` has. That is the price of the false-alarm rate at typical noise.

**The test does not find steps**, as the geepers docstring warns: a mid-record step barely
bends a line. An unwrapping error of one or two cycles passes through it and goes into the
rate. `detect_changepoints` finds these every time at the right size (27.8 and 56.0 mm). A
step test is a separate job from a linearity test, and neither replaces the other.

Break dates on synthetic hinges flagged at more than 5 mm: median error 0.12 and 0.27 years
for 8 and 4 mm/yr at quiet noise, 0.35 and 0.72 at typical noise, up to 2 years for smaller
hinges. Rate changes under 4 mm/yr are overestimated (a true 1 mm/yr reads about 4), because
only the large noise excursions get through.

## Against GNSS

GNSS is called non-linear at the same gate, with catalogued steps removed. That makes 13 of
445 pairs, all in the screened set:

| | Pixels flagged where GNSS is | Where GNSS is not |
| --- | ---: | ---: |
| `linearity_model` not linear | 0.69 | 0.10 |
| **p < 0.01 and departure > 10 mm** | **0.69** | **0.03** |
| Same, departure over 20 mm in GNSS (10 pairs) | 0.90 | 0.03 |
| `RWFNWN` model not linear | 0.08 | 0.04 |
| Any change point | 0.85 | 0.25 |
| Smooth trend chosen | 0.69 | 0.01 |

Resampling stations, because pairs share them, gives 90 % intervals of 0.21 to 1.00 for the
gate's 0.69 and 0 to 0.07 for its 0.03. Detection is therefore not well constrained.

Where both flag, the size agrees: GNSS / pixels 111 / 112, 73 / 61, 59 / 67, 50 / 51, 47 / 44,
33 / 29, 30 / 40, 25 / 23, 23 / 28 mm. Smooth matches the gate on these pairs with fewer false
alarms, but on synthetic hinges at typical noise it finds fewer (0.30 against 0.81 for 8 mm/yr
at mid-record). The linearity test also gives a size for the departure.

Rates: pixels and GNSS differ by a median of 1.27 mm/yr. The misfit is 3.6 times the combined
sigma with `FNWN` and 1.7 times with `RWFNWN`. Some of that is not pixel noise (one LOS vector
per frame, monuments), so the true figure is lower for both.

## Two tracks

The same pairs on two tracks have different dates and atmosphere but the same ground.

| Flagged | First track | Second | Both | By chance |
| --- | ---: | ---: | ---: | ---: |
| 121 quiet pairs: `linearity_model` | 14 | 6 | 0 | 0.7 |
| 121 quiet pairs: gate | 3 | 1 | 0 | 0 |
| 121 quiet pairs: any change point | 36 | 19 | 16 | 5.7 |
| 121 quiet pairs: offset above 15 mm | 27 | 11 | 11 | 2.5 |
| 14 screened pairs: `linearity_model` | 3 | 5 | 3 | 1.1 |
| 14 screened pairs: gate | 3 | 3 | 3 | 0.6 |

On quiet ground the verdict never repeats, so those flags are noise. On screened ground every
gated flag repeats, and departures correlate at 0.99. Of the 5 GNSS-confirmed pairs among the
14, 3 are flagged on both tracks. The other 2 (CAND-MIDA, P304-TRAN; 18 to 22 mm in GNSS) are
missed on both.

The break dates do not repeat: CAFP-P300 is 2019-08 on one track and 2023-03 on the other
(2023-02 in GNSS), and CAFP-MULN is 2018-02 and 2019-09.

## Offsets in the pixels

Offsets found in pixel pairs are large (median 20 mm, quartiles 14 to 25), and in GNSS small
(median 4.7 mm). Offsets over 15 mm are 17 % of pixel pairs and 2 % of GNSS pairs.

- They do not fall on ministack boundaries: 27 % within 30 days, against 45 % by chance. So
  this is not a chaining error.
- They repeat. All 11 flagged on the second track are also flagged on the first, and matched
  offsets have the same sign (r = 0.95 for those within 90 days on the F11116 and F38502 pairs).
- They concentrate. 74 of 143 come from pairs with one of two station windows: P533 in April
  2019 and P552 in January 2023, on both tracks, and also in the window's 500 m difference.
  Pairs sharing a station are not independent, which inflates these counts.
- GNSS at those stations shows 3 to 5 mm, months later. What causes them **was not
  determined**. They could be local ground or a weather-related phase error; about a quarter
  of pixel offsets are near one 28 mm cycle.

A repeatable 20 mm offset GNSS does not see is what a data-quality flag is for. It should not
be reported as ground motion.

## Three years ahead

Share of 419 pairs within 10 mm of the target, extrapolating the line fitted to the earlier
record:

| Group at the end of the fit | Pairs | Against GNSS | Against its own future |
| --- | ---: | ---: | ---: |
| `linearity_model` linear | 370 | 0.74 | 0.80 |
| Not linear, the line | 49 | 0.73 | 0.57 |
| Not linear, `rate_after` | 49 | **0.35** | **0.39** |
| `FNWN` horizon at least 3 years | 396 | 0.73 | 0.80 |
| `FNWN` horizon under 3 years | 23 | 0.83 | 0.35 |
| `RWFNWN` horizon at least 3 years | 346 | 0.79 | 0.79 |
| `RWFNWN` horizon under 3 years | 73 | **0.47** | 0.63 |

- Do not extrapolate `rate_after`: against GNSS it lands within 10 mm about half as often as
  the line.
- The `FNWN` horizon does not predict error against GNSS: its 23 short horizons do better than
  the long ones. The `RWFNWN` horizon does separate them (0.79 against 0.47).
- A non-linear verdict warns about the pixel's own future, not about the ground's. This set
  has little real signal, so this does not show how the verdict behaves where the ground does
  change.

## Houston

F08882 (ascending, 540 pairs, 2016-10 to 2026-02) and F38238 (descending, to 2023-11; F08882 is
cut there for the comparison). Scored as above, then without UH01 and TSFT.

**False alarms hold.** The gate flags none of the 540 pairs and none of the near-null windows
(`linearity_model` 19 %), and neither track flags any of the 453 pairs they share.

**UH01 and TSFT are GNSS problems.** At first the pixels seem to miss all 31 pairs that GNSS
calls strongly non-linear (up to 229 mm), but 24 of those involve these two stations. UH01's
LOS series jumps about 650 mm around 2021, more than its one catalogued step removes. TSFT's
vertical moves about 170 mm in 2022-23 and partly returns. Neither track shows either one.

**Detection is not tested.** Without those stations, the largest GNSS departure is 20 mm and the
largest in the pixels is 12 mm. The pixels flag none of the 7 pairs above 10 mm in GNSS, which
the synthetic results lead one to expect at this size. Houston's pairs are close to straight
over 25 km at this level, so this region cannot confirm or refute the 0.69.

**`RWFNWN` uncertainties hold.** Without those stations, pixels and GNSS differ by a median of
0.70 mm/yr, and the misfit is 1.45 times the combined sigma with `FNWN` and 0.84 with `RWFNWN`.

**The offset flag found a bad ministack.** 55 of F08882's 61 offsets above 15 mm fall in 2019-08
to 2019-12, against 3 in all of F38238, so unlike California these do not repeat. In the
ministack from 2019-06-20 to 2020-01-10, the median at the stations swings by up to 89 mm from
one epoch to the next, mostly 35 to 65 (`jump_f08882.py`). At the next ministack that drops to
at most 14 mm.
F38238 does not show it over those months. The cause was not determined. Chaining carries
that ministack's last epoch into all later ones.

## What this does not cover

- **Detection in one region.** All strong pairs are in one part of central California.
  Stations were picked for long records or for signal, so these are not random pixels.
- **Station windows only**, not whole frames. The
  500 m differences are the only check of other pixels.
- **Few independent strong pairs.** 13 pairs share 18 stations; the two-track check has 5.
- **Not tested:** other multilook or bin sizes, the calibration of the smooth model's
  `rate_sigma`, records with long gaps, and detection on shorter records.

## Recommendation

Owner decision, in this order:

1. Done: `fit_series(model="linearity")` has `significant` and takes `velocity`,
   `velocity_sigma` and `validity_years` from a second `RWFNWN` fit, about 15 % more time per
   series. `breakpoint` and `rate_after` stay, documented as not fit for dating or
   extrapolation.
2. Done on Houston: false alarms and uncertainties hold; detection could not be tested there.
   Move `"linearity"` from draft to provisional, documented as checked for departures above
   about 20 mm in one region. Otherwise, find a second region with strong non-linear signals.
3. Add an offsets-above-15-mm flag from `detect_changepoints` as a data-quality layer, after
   checking P533, P552 and F08882's 2019 ministack against coherence. This is separate work.
