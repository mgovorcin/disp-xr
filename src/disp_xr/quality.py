"""Quality-layer statistics and masks for DISP stacks."""

import logging
import warnings
from typing import Callable

import xarray as xr

logger = logging.getLogger(__name__)


def func_stat(mode: str) -> Callable:
    """Return a statistical operation based on the specified mode.

    Parameters
    ----------
    mode : str
        The statistical operation to perform. Example values include 'mean',
        'median', 'max', 'min', 'std', 'var', and 'sum'.

    Returns
    -------
    callable
        A lambda function that performs the corresponding statistical operation
        on an xarray object.

    """
    kwargs = {"skipna": True, "keepdims": True}
    # Mapping mode to corresponding statistical operation
    operations = {
        "mean": lambda x: x.mean(dim="time", **kwargs),
        "median": lambda x: x.median(dim="time", **kwargs),
        "max": lambda x: x.max(dim="time", **kwargs),
        "min": lambda x: x.min(dim="time", **kwargs),
        "std": lambda x: x.std(dim="time", **kwargs),
        "var": lambda x: x.var(dim="time", **kwargs),
        "sum": lambda x: x.sum(dim="time", **kwargs),
    }

    # Check if the mode is valid and return the corresponding function
    try:
        return operations[mode]
    except KeyError:
        raise ValueError(
            (
                f"Unsupported mode: '{mode}'."
                " Valid options are 'mean', 'median', 'max',"
                " 'min', 'std', 'var', 'sum'"
            )
        )


def get_stack_stat(stack_xr: xr.DataArray, mode: str = "mean") -> xr.DataArray:
    """Compute a statistical operation (mean, median, etc.).

    Parameters
    ----------
    stack_xr : xr.DataArray
        The xarray DataArray on which the statistical operation..
    mode : str, optional
        The statistical operation to perform. Default is 'mean'.
        Options: 'median', 'max', 'min', 'std', 'var', and 'sum'.

    Returns
    -------
    xr.DataArray
        The result of the statistical operation, with the operation applied along the
        'time' dimension and any NaN values handled.

    """
    if not isinstance(stack_xr, xr.DataArray):
        raise TypeError("Input must be an xarray Dataset.")

    logger.info(f"Get {mode} of {stack_xr.name}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        warnings.filterwarnings("ignore", message="All-NaN slice encountered")
        data = func_stat(mode)(stack_xr)
    return data.squeeze()


# Map_block template


def get_value_percentage(
    stack_xr: xr.DataArray, value: int, reverse: bool = False
) -> xr.DataArray:
    """Return, per pixel, the percentage of epochs equal to ``value``.

    Parameters
    ----------
    stack_xr : xr.DataArray
        A ``(time, y, x)`` layer.
    value : int
        The value to count.
    reverse : bool, optional
        Count epochs *not* equal to ``value`` instead.

    Returns
    -------
    xr.DataArray
        Lazy ``(y, x)`` percentage of all epochs.

    """
    if not isinstance(stack_xr, xr.DataArray):
        raise TypeError("Input must be an xarray DataArray.")
    logger.info(f"Get percentage of {stack_xr.name}")
    hits = (stack_xr != value) if reverse else (stack_xr == value)
    return hits.sum("time") / stack_xr.sizes["time"] * 100
