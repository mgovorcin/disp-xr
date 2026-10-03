"""Xarray tools for OPERA DISP products.

Import as ``import disp_xr as dxr``. Functions are added to the top-level
namespace as their plan tasks land (see docs/todo.md).
"""

from . import accessor as _accessor  # noqa: F401  (registers .disp)
from ._chunks import rechunk
from ._version import __version__
from .config import config
from .corrections import correct, remove_plate_motion
from .export import (
    open_icechunk,
    to_geotiff,
    to_icechunk,
    to_mintpy,
    to_netcdf,
    to_zarr,
)
from .geometry.los import enu_to_los
from .geometry.static import open_static
from .open import open_disp, open_disp_tree
from .product import find_products, get_disp_info  # noqa: F401  (deprecated)
from .quality import mask, quality_summary, valid_mask
from .reference import rereference
from .stack import (  # noqa: F401  (combine_disp_product is deprecated)
    NaNPolicy,
    combine_disp_product,
    stack_disp,
)
from .timeseries import (
    NoiseResult,
    Trajectory,
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
    "Trajectory",
    "__version__",
    "build_virtual_frame",
    "config",
    "correct",
    "enu_to_los",
    "find_products",
    "fit_noise",
    "fit_trajectory",
    "fit_velocity",
    "mask",
    "open_disp",
    "open_disp_tree",
    "open_icechunk",
    "open_static",
    "open_virtual",
    "predict_trajectory",
    "quality_summary",
    "quality_weights",
    "rechunk",
    "remove_plate_motion",
    "rereference",
    "stack_disp",
    "to_geotiff",
    "to_icechunk",
    "to_mintpy",
    "to_netcdf",
    "to_zarr",
    "to_units",
    "valid_mask",
    "validate_disp",
    "validate_virtual",
]
