"""get_disp_info is deprecated in favor of find_products (plan task T7)."""

import pytest

from disp_xr import product


def test_get_disp_info_warns(stack_dir):
    with pytest.warns(DeprecationWarning, match="use find_products"):
        product.get_disp_info(stack_dir)
