"""An in-memory geepers GNSS source for tests (no network)."""

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from geepers.gps_sources.base import BaseGpsSource
from pyproj import Transformer
from shapely.geometry import Point


class FakeSource(BaseGpsSource):
    """A geepers source serving given stations and series."""

    def __init__(self, table, series, cache_dir):
        super().__init__(cache_dir=cache_dir)
        self.table = table
        self.series = series

    def _read_station_data(self):
        return gpd.GeoDataFrame(
            self.table,
            geometry=[Point(lo, la) for lo, la in zip(self.table.lon, self.table.lat)],
            crs="EPSG:4326",
        )

    def timeseries(
        self,
        station_id,
        /,
        frame="ENU",  # noqa: ARG002
        start_date=None,
        end_date=None,
        zero_by="mean",  # noqa: ARG002
        download_if_missing=True,  # noqa: ARG002
        **kwargs,  # noqa: ARG002
    ):
        df = self.series[station_id].copy()
        if start_date:
            df = df[df.date >= pd.Timestamp(start_date)]
        if end_date:
            df = df[df.date <= pd.Timestamp(end_date)]
        return df.reset_index(drop=True)


def stations_from_disp(ds, pixels, cache_dir, *, noise=0.001, extra_rate=None, seed=0):
    """GNSS whose up motion follows DISP at the given pixels (vertical LOS).

    ``extra_rate`` maps a station index to an added up rate (m/yr), to make
    an outlier.
    """
    rng = np.random.default_rng(seed)
    to_ll = Transformer.from_crs(ds.rio.crs, "EPSG:4326", always_xy=True)
    days = pd.date_range(
        pd.Timestamp(ds.time.values[0]).normalize() - pd.Timedelta("5D"),
        pd.Timestamp(ds.time.values[-1]).normalize() + pd.Timedelta("5D"),
        freq="D",
    )
    rows, series = [], {}
    for k, (r, c) in enumerate(pixels):
        sid = f"S{k:03d}"
        lon, lat = to_ll.transform(float(ds.x[c]), float(ds.y[r]))
        rows.append({"id": sid, "lon": lon, "lat": lat, "alt": 0.0})
        pixel = ds.displacement.isel(y=r, x=c).to_series()
        pixel.index = pd.DatetimeIndex(pixel.index).normalize()
        up = pixel.reindex(days).interpolate(limit_direction="both").to_numpy()
        years = np.asarray((days - days[0]).days) / 365.25
        up = (
            up
            + (extra_rate or {}).get(k, 0.0) * years
            + noise * rng.standard_normal(days.size)
        )
        series[sid] = pd.DataFrame(
            {
                "date": days,
                "east": 0.0,
                "north": 0.0,
                "up": up,
                "sigma_east": noise,
                "sigma_north": noise,
                "sigma_up": noise,
                "corr_en": 0.0,
                "corr_eu": 0.0,
                "corr_nu": 0.0,
            }
        )
    return FakeSource(pd.DataFrame(rows), series, cache_dir)


def vertical_los(ds):
    """Static layers with a vertical line of sight."""
    shape = (ds.sizes["y"], ds.sizes["x"])
    zeros = np.zeros(shape)
    return xr.Dataset(
        {
            "los_east": (("y", "x"), zeros),
            "los_north": (("y", "x"), zeros),
            "los_up": (("y", "x"), np.ones(shape)),
        },
        coords={"y": ds.y, "x": ds.x},
    ).rio.write_crs(ds.rio.crs)
