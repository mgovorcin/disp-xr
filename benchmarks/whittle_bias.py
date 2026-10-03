"""How biased is the Whittle noise estimate at InSAR record lengths? (T22)

Simulates 12-day series of a trend plus flicker (kappa -1) and white noise,
and fits each with ``fit_noise(model="PLWN")`` by exact RMLE and by Whittle.
For each record length it reports the median kappa of both, the ratio of the
velocity sigmas (Whittle / exact), and the scatter of the velocity errors
against the median sigma of each method (a calibrated sigma gives about 1).

    pixi run -e test python benchmarks/whittle_bias.py
"""

import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from disp_xr.timeseries.noise import _fractional_diff_coeffs, fit_noise
from disp_xr.timeseries.trajectory import Trajectory

LENGTHS = (60, 100, 200, 400, 800)
REALIZATIONS = 30
VELOCITY = 0.005


def one(args: tuple[int, int]) -> tuple[float, ...]:
    """Fit one realization; return exact and Whittle (kappa, v, sigma) and times."""
    n, seed = args
    rng = np.random.default_rng(seed)
    psi = _fractional_diff_coeffs(0.5, n)  # flicker: d = -kappa / 2
    noise = 2e-3 * np.convolve(rng.standard_normal(n), psi)[:n]
    y = VELOCITY * np.arange(n) * 12 / 365.25 + noise + 2e-3 * rng.standard_normal(n)
    times = pd.date_range("2018-01-01", periods=n, freq="12D").values
    terms = Trajectory(periods=(1.0,))
    t0 = time.perf_counter()
    exact = fit_noise(times, y, terms=terms, model="PLWN")
    t1 = time.perf_counter()
    fast = fit_noise(times, y, terms=terms, model="PLWN", method="whittle")
    t2 = time.perf_counter()
    return (
        exact.kappa, exact.velocity, exact.velocity_sigma,
        fast.kappa, fast.velocity, fast.velocity_sigma, t1 - t0, t2 - t1,
    )


def main() -> None:
    """Print one row per record length."""
    print(
        f"{'epochs':>6} {'years':>5} {'kappa exact':>11} {'kappa whittle':>13} "
        f"{'sigma W/E':>9} {'err/sig exact':>13} {'err/sig whittle':>15} "
        f"{'s exact':>7} {'s whittle':>9}"
    )
    with ProcessPoolExecutor(32) as pool:
        for n in LENGTHS:
            rows = np.array(list(pool.map(one, [(n, s) for s in range(REALIZATIONS)])))
            ke, ve, se, kw, vw, sw, te, tw = rows.T
            print(
                f"{n:6d} {n * 12 / 365.25:5.1f} {np.median(ke):11.2f} "
                f"{np.median(kw):13.2f} {np.median(sw / se):9.2f} "
                f"{np.std(ve - VELOCITY) / np.median(se):13.2f} "
                f"{np.std(vw - VELOCITY) / np.median(sw):15.2f} "
                f"{np.median(te):7.2f} {np.median(tw):9.2f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
