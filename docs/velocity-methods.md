# Choosing a velocity method

disp-xr has four ways to estimate a velocity. They differ in how they treat
noise, how fast they are, and how long a record they need. The short answer:

| Your case | Use |
|---|---|
| Exact colored-noise sigmas for every pixel, any length from about 60 epochs | `fit_velocity(ds, "gls")` |
| Full DISP-S1 record, 200 epochs or more (about 6.5 years at 12 days), fastest | `fit_velocity(ds, "whittle")` |
| One pixel or station, to study its noise | `fit_noise` (exact); consider `FNWN` for the shortest records |
| Unmodelled steps, outliers or unwrapping jumps, several years of data | `fit_velocity(ds, "midas")` as a robust check |
| A quick map, or comparing pixels with each other | `fit_velocity(ds, "ols")`, but do not use its sigma |

The rest of this page explains why.

## The methods

| Method | Velocity | Sigma | Noise model | Cost |
|---|---|---|---|---|
| `fit_velocity(ds, "ols")` | least squares with your `Trajectory` | white noise only | none | fast, every pixel |
| `fit_velocity(ds, "whittle")` | the same least-squares velocity | colored noise, per pixel | power law plus white (PLWN), estimated per pixel | moderate, every pixel |
| `fit_velocity(ds, "midas")` | median of one-year slopes, outliers trimmed | MIDAS, deliberately conservative | none (empirical) | fast, every pixel |
| `fit_velocity(ds, "gls")` | generalized least squares | colored noise, exact, per pixel | shape (kappa, white fraction) per group of pixels, by exact maximum likelihood on a sample; amplitude per pixel | exact fits of the samples, then about 1 s per 256 x 256 tile |
| `fit_noise(times, y, model=...)` | generalized least squares | colored noise, exact | WN, FNWN, PLWN or RWFNWN | 0.1 to 1 s per series (n x n covariance) |

## Why not OLS sigmas

InSAR and GNSS noise is correlated in time (flicker-like, kappa near -1).
Least squares treats every epoch as independent, so its sigma is too small.
On the test series in `tests/data/noise_reference.json` the white-noise
sigma is 3 to 30 times smaller than the colored-noise sigma of the same data.
The OLS velocity itself is fine; only its uncertainty is wrong.

## Record length: where the 200 comes from

The limit applies to Whittle only. Whittle estimates the noise from the
periodogram, an approximation that becomes exact as the record grows.
`benchmarks/whittle_bias.py` compared it with the exact estimate on 30
simulated 12-day series per length (trend, flicker with kappa -1, white
noise):

| Epochs | Years | kappa, exact | kappa, Whittle | Velocity error / sigma, exact | Same, Whittle |
|---:|---:|---:|---:|---:|---:|
| 60 | 2.0 | -1.03 | -0.19 | 1.06 | 1.86 |
| 100 | 3.3 | -0.92 | -0.75 | 1.21 | 1.47 |
| 200 | 6.6 | -0.94 | -0.81 | 0.94 | 1.01 |
| 400 | 13.1 | -1.04 | -1.00 | 0.84 | 0.81 |

A ratio near 1 in the last two columns means the sigma is honest; above 1
it is too small. From about 200 epochs Whittle is as good as the exact
estimate; at 100 its sigmas are about 1.5 times too small, at 60 nearly 2
times. `fit_velocity(method="whittle")` logs a warning below 200 epochs.

The exact estimate (`fit_noise`) stays calibrated down to 60 epochs, but it
costs a factorization of an n x n matrix per likelihood evaluation, so it is
for single pixels, GNSS stations or small groups. `fit_velocity(ds, "gls")`
brings it to whole frames: it fits a sample of each group's pixels exactly
to get the group's noise shape, then solves every pixel by exact GLS with
that shape and its own amplitude, so it inherits the exact estimate's
behaviour on short records.

## Groups for `"gls"`

Pixels with similar noise should share a group. By default they are binned
by time-mean `temporal_coherence` (`n_groups` quantile bins); you can pass
any integer map `(y, x)` instead, for example regions you trust or, later,
quadtree cells. More samples per group (`n_samples`, default 16) steady the
shape estimate at the cost of more exact fits. Check `attrs["noise_groups"]`
for each group's model, kappa and white fraction.

## Which noise model (`fit_noise`)

| Model | Noise | When |
|---|---|---|
| `PLWN` | power law with free kappa, plus white | the default |
| `FNWN` | flicker (kappa fixed at -1) plus white | the standard GNSS model; one parameter fewer, so steadier on short records where kappa is poorly constrained (a judgment, not measured here: the study above fitted PLWN only) |
| `RWFNWN` | random walk, flicker and white | long records with suspected random-walk drift (monument motion, slowly accumulating unwrapping errors) |
| `WN` | white only | testing; equivalent to OLS |

When unsure, fit several and compare `res.bic` (lower is better): BIC
penalizes extra noise parameters, so it prefers `FNWN` unless the data
support a free kappa or a random walk. `fit_velocity(method="whittle")`
always uses PLWN; `method="gls"` takes `noise=` with any of these, or
`noise="auto"` to choose FNWN, PLWN or RWFNWN per group by the median BIC of
its samples.

## When MIDAS

MIDAS takes the median of slopes between epochs about one year apart
(Blewitt et al., 2016). Steps, outliers and seasonal signals affect only some
pairs, so it needs no step dates and no seasonal terms. Use it:

- as a check on a trajectory fit, where a large difference points to a step
  or outlier the model misses;
- when steps are not known in advance (unwrapping jumps, equipment or
  processing changes).

It needs a record of several years: every pair spans a year, and a step
biases the median when the pairs that cross it are a large share, about one
year divided by the record length. On a 3-year synthetic record with a 10 mm
step the bias was clear; on 6 years it was small. Its sigma is deliberately
conservative (an empirical factor of 3 makes it realistic for colored noise),
so it is often larger than the Whittle sigma for the same pixel.

## Model what you know

OLS and Whittle fit the `Trajectory` you give them. Include known steps
(earthquakes, reference changes not handled by chaining) and the seasonal
terms; an unmodelled step or seasonal cycle ends up in the residuals and
inflates the estimated noise. Whittle estimates the noise from those
residuals, so a missing term also makes its sigma too large.
