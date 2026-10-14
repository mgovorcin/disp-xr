# %% [markdown]
# # Line-of-sight geometry and plate motion
#
# DISP measures motion along the radar line of sight (LOS), positive toward
# the satellite. The LOS direction changes across a frame: the incidence
# angle grows from near to far range. Converting between east/north/up
# motion and LOS needs the static layers:
#
# - the LOS unit vector (`los_east`, `los_north`, `los_up`);
# - the incidence and azimuth angles derived from it;
# - the DEM;
# - the layover and shadow mask.
#
# `dxr.open_static(ds)` downloads the DISP-S1-STATIC files of the frame once
# and puts them on the DISP grid. `disp_xr geometry` does the same from the
# command line. The docs build cannot download, so this page writes small
# static files in the DISP-S1-STATIC layout and passes them with `files=`.

# %%
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

import disp_xr as dxr
from disp_xr.corrections import plate_velocity_enu
from disp_xr.tutorial import make_disp_stack

folder = Path(tempfile.mkdtemp())
make_disp_stack(folder / "granules", "three_ministacks", shape=(64, 64))
ds = dxr.chain(dxr.open_disp(folder / "granules")).disp.rereference("median")

# An ascending pass: the LOS points west-ish and up, and the incidence angle
# grows from 30 to 46 degrees across the frame.
incidence = np.radians(np.linspace(30, 46, ds.sizes["x"]))[None, :].repeat(
    ds.sizes["y"], axis=0
)
heading = np.radians(-10.0)  # flight direction, clockwise from north
look = heading + np.pi / 2  # right-looking
enu = np.stack(
    [
        -np.sin(incidence) * np.sin(look),
        -np.sin(incidence) * np.cos(look),
        np.cos(incidence),
    ]
).astype("float32")
grid = ds.displacement.isel(time=0, drop=True)
los_file = (
    folder / "OPERA_L3_DISP-S1-STATIC_F11116_20140403_S1A_v1.0_line_of_sight_enu.tif"
)
xr.DataArray(
    enu, dims=("band", "y", "x"), coords={"band": [1, 2, 3], "y": ds.y, "x": ds.x}
).rio.write_crs(ds.rio.crs).rio.to_raster(los_file)
dem_file = los_file.with_name(los_file.name.replace("line_of_sight_enu", "dem"))
yy, xx = np.meshgrid(
    np.linspace(-1, 1, ds.sizes["y"]), np.linspace(-1, 1, ds.sizes["x"])
)
dem = (200 + 1500 * np.exp(-((xx - 0.3) ** 2 + yy**2) / 0.2)).T.astype("float32")
grid.copy(data=dem).rio.write_crs(ds.rio.crs).rio.to_raster(dem_file)

static = dxr.open_static(ds, files={"line_of_sight_enu": los_file, "dem": dem_file})
print(list(static.data_vars))

# %%
fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), layout="constrained")
static.incidence_angle.plot.imshow(ax=axes[0], cbar_kwargs={"label": "degrees"})
axes[0].set_title("incidence angle", loc="left")
static.los_east.plot.imshow(ax=axes[1], cmap="RdBu_r", cbar_kwargs={"label": ""})
axes[1].set_title("LOS east component", loc="left")
static.dem.plot.imshow(ax=axes[2], cmap="terrain", cbar_kwargs={"label": "m"})
axes[2].set_title("DEM", loc="left")
for ax in axes:
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
plt.show()

# %% [markdown]
# ## From east, north, up to LOS
#
# `dxr.enu_to_los` projects any motion onto the LOS. Pure vertical motion
# shows up scaled by the cosine of the incidence angle, so the same
# subsidence looks smaller in far range. Horizontal motion depends on the
# look direction: east-west motion is seen well, north-south hardly at all
# (polar orbits look east or west).

# %%
for name, (e, n, u) in {
    "1 cm down": (0.0, 0.0, -0.01),
    "1 cm east": (0.01, 0.0, 0.0),
    "1 cm north": (0.0, 0.01, 0.0),
}.items():
    los = dxr.enu_to_los(e, n, u, static)
    low, high = 1e3 * float(los.min()), 1e3 * float(los.max())
    print(f"{name:>10}: LOS {low:+.1f} to {high:+.1f} mm")

# %% [markdown]
# ## Plate motion
#
# GNSS velocities in a global frame (ITRF) include the rigid motion of the
# plate: about 2 cm/yr for North America, 5 cm/yr for the Pacific. DISP is
# relative to its reference pixel, so a uniform plate motion would cancel,
# except that the LOS changes across the frame: a uniform motion leaves a
# ramp in DISP. `remove_plate_motion` subtracts the
# ITRF2014 plate-motion model (Altamimi et al. 2017), projected on the LOS of
# each pixel, so the result is in a plate-fixed frame like plate-fixed GNSS.

#
# This synthetic frame is only 2 km across, so the ramp is small; across a
# 250 km DISP-S1 frame it reaches a few mm/yr.

# %%
lon, lat = ds.rio.transform_bounds("EPSG:4326")[:2]
east, north, _ = plate_velocity_enu("NOAM", lat, lon)
print(f"North America at {lat:.1f} N, {lon:.1f} E (mm/yr):")
print(f"  east {1e3 * east:+.1f}, north {1e3 * north:+.1f}")

fixed = dxr.remove_plate_motion(ds, static, "NOAM")
removed = 1e3 * (ds.displacement - fixed.displacement).isel(time=-1)
fig, ax = plt.subplots(figsize=(5, 4), layout="constrained")
removed.plot.imshow(ax=ax, cmap="viridis", cbar_kwargs={"label": "mm, last epoch"})
ax.set_title("plate motion removed (LOS)", loc="left")
ax.set_aspect("equal")
plt.show()
span = float(removed.max() - removed.min())
print(f"range across this 2 km frame: {span:.2f} mm")
