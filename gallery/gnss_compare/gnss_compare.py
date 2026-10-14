# %% [markdown]
# # Comparing DISP with GNSS
#
# `disp_xr.residuals.compare` reads DISP at each GNSS station, projects the
# station's east, north and up motion onto the line of sight, and computes the
# metrics with geepers: velocities of both series from the same estimator,
# pairwise relative RMSE against distance, RMSE per epoch and a misfit
# semivariogram. disp-xr adds normalized velocity residuals, outlier and
# bad-epoch flags, and scores the OPERA DISP-S1 requirements.
#
# This page uses a synthetic frame and synthetic stations that follow it, so
# it runs anywhere (`gnss` extra). One station has an extra 3 cm/yr of motion
# that DISP does not see, the kind of disagreement the comparison should flag.

# %%
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import disp_xr as dxr
from disp_xr.residuals import DISP_S1_DISPLACEMENT
from disp_xr.tutorial import make_disp_stack, make_gnss_source, vertical_los

folder = Path(tempfile.mkdtemp())
make_disp_stack(
    folder / "granules", "three_ministacks", shape=(96, 96), epochs_per_ministack=20
)
ds = dxr.open_disp(folder / "granules").disp.chain().disp.rereference("median")

pixels = [(10, 10), (10, 85), (85, 10), (85, 85), (48, 48), (25, 60), (70, 35)]
source = make_gnss_source(ds, pixels, folder / "gnss", extra_rate={3: 0.03})
gps = dxr.gnss.load(ds, source=source)
print(f"{gps.sizes['station']} stations, {gps.sizes['time']} daily solutions")

# %% [markdown]
# ## Compare
#
# `static` holds the line-of-sight vectors (from `dxr.open_static` on a real
# frame; vertical here). With no other argument, the pairwise profile is
# scored against the DISP-S1 displacement requirement and station pairs
# against the velocity requirement.

# %%
result = dxr.residuals.compare(ds, gps, vertical_los(ds))
columns = ["insar_velocity", "gps_velocity", "relative_residual", "outlier"]
print((result.stations[columns] * [1e3, 1e3, 1e3, 1]).round(2))
summary = result.summary
print(
    f"datum offset {1e3 * summary['datum_offset']:+.1f} mm/yr, "
    f"DISP noise per epoch {1e3 * summary['disp_sigma']:.1f} mm"
)
print(f"velocity RMSE {1e3 * summary['rmse']:.1f} mm/yr, r2 {summary['r2']:.2f}")
print(f"station pairs within 2 mm/yr: {summary['velocity_fraction_passing']:.0%}")

# %% [markdown]
# Station `S003` stands out: its velocity residual is close to the 3 cm/yr
# it was given and it is flagged as an outlier. On a real frame such a
# station is either local motion DISP does not resolve (a monument problem,
# or deformation smaller than a pixel) or a DISP problem near the station.

# %% [markdown]
# ## Relative error against distance
#
# Every station pair gives the RMSE of its relative misfit over time. The
# binned profile carries the requirement and the fraction of pairs that pass
# it in each bin.

# %%
fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
left.scatter(result.pairs["distance_km"], 1e3 * result.pairs["rmse"], s=14)
d = np.linspace(0.1, result.pairs["distance_km"].max() * 1.05, 100)
left.plot(d, 1e3 * DISP_S1_DISPLACEMENT(d), "k--", label=DISP_S1_DISPLACEMENT.name)
left.set(xlabel="distance (km)", ylabel="relative RMSE (mm)")
left.legend(loc="upper left")
pairs = result.velocity_pairs
right.scatter(pairs["distance_km"], 1e3 * pairs["difference"].abs(), s=14)
right.axhline(2.0, color="k", ls="--", label="DISP-S1 velocity, 2 mm/yr")
right.set(xlabel="distance (km)", ylabel="|velocity residual difference| (mm/yr)")
right.set_yscale("log")
right.legend(loc="upper left")
plt.show()
profile = result.profile[
    ["distance_km", "rmse_median", "requirement", "fraction_passing"]
]
print(profile.round(4))

# %% [markdown]
# Pairs that include `S003` fail the velocity requirement; the others are
# well under it.
#
# ## Epochs and the misfit semivariogram
#
# The RMSE per epoch flags acquisitions where DISP disagrees with the whole
# network (troposphere, unwrapping); the semivariogram shows the spatial
# scale of the misfit.

# %%
fig, (left, right) = plt.subplots(1, 2, figsize=(11, 3.6), layout="constrained")
epochs = result.epochs
left.plot(epochs.index, 1e3 * epochs["rmse"], ".-")
flagged = epochs[epochs["flag"]]
left.plot(flagged.index, 1e3 * flagged["rmse"], "o", color="C3", label="flagged")
left.set(ylabel="network RMSE (mm)")
left.legend()
result.plot_semivariogram(ax=right)
plt.show()

# %% [markdown]
# On a real frame, pass the output of `dxr.gnss.load(ds)` (UNR stations by
# default) and `dxr.open_static(ds)` instead of the synthetic source and
# vertical line of sight. `compare` also takes downsampled cells (a point
# table or a `Quadtree`) and compares each station with the cell holding it.
