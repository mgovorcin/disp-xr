"""Xarray tools for OPERA DISP products.

Import as ``import disp_xr as dxr``. Functions are added to the top-level
namespace as their plan tasks land (see docs/todo.md).
"""

from . import accessor as _accessor  # noqa: F401  (registers .disp)
from ._version import __version__
from .config import config
from .open import open_disp, open_disp_tree
from .product import find_products, get_disp_info
from .quality import mask, quality_summary, valid_mask
from .reference import rereference
from .stack import NaNPolicy, combine_disp_product, stack_disp
from .units import to_units
from .validate import validate_disp

__all__ = [
    "NaNPolicy",
    "__version__",
    "combine_disp_product",
    "config",
    "find_products",
    "get_disp_info",
    "mask",
    "open_disp",
    "open_disp_tree",
    "quality_summary",
    "rereference",
    "stack_disp",
    "to_units",
    "valid_mask",
    "validate_disp",
]
