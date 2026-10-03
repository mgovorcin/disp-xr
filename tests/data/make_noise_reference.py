"""Write ``noise_reference.json``: geepers ``estimate_trend`` on five series.

The reference for ``tests/test_noise.py`` (plan task T21.5). geepers is not a
disp-xr dependency, so this script runs once in a geepers environment and its
output is committed::

    cd 00_tools/src/geepers && pixi run python \\
        ../disp-xr/tests/data/make_noise_reference.py

Each series is a trend, an annual cycle and power-law plus white noise built
with the Hosking (1981) recursion, at GNSS (daily) or InSAR (12-day) sampling,
some with gaps. The series and geepers' estimates for each noise model are
stored together, with the geepers commit.
"""

import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from geepers.trend import estimate_trend

OUT = Path(__file__).with_name("noise_reference.json")


def powerlaw_noise(n: int, kappa: float, rng: np.random.Generator) -> np.ndarray:
    """Unit-variance-driven power-law noise: (1 - B)^(kappa / 2) applied to white."""
    d = -kappa / 2.0
    psi = np.empty(n)
    psi[0] = 1.0
    i = np.arange(1, n)
    psi[1:] = np.cumprod((d + i - 1.0) / i)
    return np.convolve(rng.standard_normal(n), psi)[:n]


SERIES = [
    # name, epochs, sampling days, kappa, sigma_pl, sigma_white, gap fraction, seed
    ("gnss_flicker", 1000, 1, -1.0, 1.5e-3, 1.0e-3, 0.0, 1),
    ("gnss_flicker_gaps", 900, 1, -1.0, 1.0e-3, 1.5e-3, 0.15, 2),
    ("gnss_randomwalk", 800, 1, -2.0, 2.0e-4, 1.0e-3, 0.05, 3),
    ("insar_12day", 200, 12, -1.0, 2.0e-3, 3.0e-3, 0.0, 4),
    ("insar_12day_gaps", 220, 12, -0.6, 2.5e-3, 2.0e-3, 0.2, 5),
]
MODELS = ["WN", "FNWN", "PLWN", "RWFNWN"]


def main() -> None:
    """Fit every series with every model and write the JSON reference."""
    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    out: dict = {"geepers_commit": commit, "series": []}
    for name, n, step, kappa, s_pl, s_w, gaps, seed in SERIES:
        rng = np.random.default_rng(seed)
        dates = pd.Timestamp("2018-01-01") + pd.to_timedelta(np.arange(n) * step, "D")
        years = np.arange(n) * step / 365.25
        values = (
            0.005 * years
            + 0.003 * np.sin(2 * np.pi * years)
            + s_pl * powerlaw_noise(n, kappa, rng)
            + s_w * rng.standard_normal(n)
        )
        keep = rng.random(n) >= gaps
        keep[[0, -1]] = True
        dates, values = dates[keep], values[keep]
        fits = {}
        for model in MODELS:
            res = estimate_trend(
                dates,
                values,
                sampling_days=step,
                periods_years=(1.0,),
                noise_model=model,
            )
            fits[model] = {
                "velocity": res.velocity,
                "velocity_sigma": res.velocity_uncertainty,
                "kappa": res.kappa,
                "sigma_powerlaw": res.sigma_powerlaw,
                "sigma_white": res.sigma_white,
                "sigma_flicker": res.sigma_flicker,
                "sigma_randomwalk": res.sigma_randomwalk,
                "log_likelihood": res.log_likelihood,
                "aic": res.aic,
                "bic": res.bic,
            }
            print(
                f"{name:<18} {model:<7} v={res.velocity:.6f} "
                f"sig={res.velocity_uncertainty:.6f} k={res.kappa:.3f}"
            )
        res = estimate_trend(
            dates,
            values,
            sampling_days=step,
            periods_years=(1.0,),
            noise_model="PLWN",
            method="whittle",
        )
        fits["PLWN_whittle"] = {
            "velocity": res.velocity,
            "velocity_sigma": res.velocity_uncertainty,
            "kappa": res.kappa,
        }
        out["series"].append(
            {
                "name": name,
                "sampling_days": step,
                "true_kappa": kappa,
                "dates": [str(d.date()) for d in dates],
                "values": values.tolist(),
                "fits": fits,
            }
        )
    OUT.write_text(json.dumps(out, indent=1, allow_nan=True))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
