"""Xarray tools for OPERA DISP products.

Import as ``import disp_xr as dxr``. Functions are added to the top-level
namespace as their plan tasks land (see docs/todo.md).
"""

from . import accessor as _accessor  # noqa: F401  (registers .disp)
from . import diagnostics, gnss, models
from ._chunks import rechunk
from ._version import __version__
from .config import config
from .corrections import correct, remove_plate_motion
from .downsample import downsample, to_points
from .export import (
    build_overviews,
    open_icechunk,
    open_pyramid,
    to_geotiff,
    to_geozarr,
    to_icechunk,
    to_mintpy,
    to_netcdf,
    to_pyramid,
    to_zarr,
)
from .geometry.los import enu_to_los
from .geometry.static import open_static
from .hexagons import downsample_hex, to_geodataframe
from .open import open_disp, open_disp_tree
from .product import find_products, get_disp_info  # noqa: F401  (deprecated)
from .quadtree import Quadtree, quadtree
from .quality import mask, quality_summary, valid_mask
from .reference import rereference
from .residuals import VariogramModel, empirical_variogram, fit_variogram
from .stack import (  # noqa: F401  (combine_disp_product is deprecated)
    NaNPolicy,
    chain,
    combine_disp_product,
    stack_disp,
)
from .timeseries import (
    NoiseResult,
    Trajectory,
    denoise,
    fit_noise,
    fit_trajectory,
    fit_velocity,
    predict_trajectory,
)
from .units import to_units
from .validate import validate_disp
from .virtual import build_virtual_frame, open_virtual, validate_virtual
from .weights import quality_weights

__all__ = [
    "NaNPolicy",
    "NoiseResult",
    "Quadtree",
    "Trajectory",
    "VariogramModel",
    "__version__",
    "build_overviews",
    "build_virtual_frame",
    "config",
    "correct",
    "denoise",
    "diagnostics",
    "downsample",
    "downsample_hex",
    "empirical_variogram",
    "enu_to_los",
    "find_products",
    "fit_noise",
    "fit_trajectory",
    "fit_velocity",
    "fit_variogram",
    "gnss",
    "mask",
    "models",
    "open_disp",
    "open_disp_tree",
    "open_icechunk",
    "open_pyramid",
    "open_static",
    "open_virtual",
    "predict_trajectory",
    "quality_summary",
    "quadtree",
    "quality_weights",
    "rechunk",
    "remove_plate_motion",
    "rereference",
    "chain",
    "stack_disp",
    "to_geodataframe",
    "to_geotiff",
    "to_geozarr",
    "to_icechunk",
    "to_mintpy",
    "to_netcdf",
    "to_points",
    "to_pyramid",
    "to_zarr",
    "to_units",
    "valid_mask",
    "validate_disp",
    "validate_virtual",
]
