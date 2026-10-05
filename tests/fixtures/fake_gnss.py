"""Synthetic GNSS for tests (no network); the code lives in disp_xr.tutorial."""

from disp_xr.tutorial import make_gnss_source as stations_from_disp
from disp_xr.tutorial import synthetic_source_class, vertical_los

__all__ = ["FakeSource", "stations_from_disp", "vertical_los"]


def FakeSource(table, series, cache_dir):  # noqa: N802 (was a class)
    """Return a geepers source serving ``table`` stations and their ``series``."""
    return synthetic_source_class()(table, series, cache_dir)
