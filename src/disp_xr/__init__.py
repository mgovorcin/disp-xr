"""Xarray tools for OPERA DISP products.

Import as ``import disp_xr as dxr``. Functions are added to the top-level
namespace as their plan tasks land (see docs/todo.md).
"""

from ._version import __version__
from .config import config
from .open import open_disp
from .product import find_products, get_disp_info
from .stack import combine_disp_product

__all__ = [
    "__version__",
    "combine_disp_product",
    "config",
    "find_products",
    "get_disp_info",
    "open_disp",
]
