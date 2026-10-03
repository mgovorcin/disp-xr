from typing import Any

from ._design_matrix import (
    create_design_matrix,
    get_coefficient_names,
    infer_parameters_from_names,
)
from ._numpy_solver import lscov_2d
from ._stats_analysis import get_residuals_analysis
from .estimate_fit import (
    fit_timeseries,
    fit_timeseries_block,
    get_predicted,
    sincos_to_amplitude_phase,
)
from .trajectory import Trajectory, fit_trajectory, predict_trajectory

__all__ = [
    "Trajectory",
    "create_design_matrix",
    "fit_timeseries",
    "fit_timeseries_block",
    "fit_trajectory",
    "get_coefficient_names",
    "get_predicted",
    "get_residuals_analysis",
    "infer_parameters_from_names",
    "jax_solve_linear_system",
    "lscov_2d",
    "predict_trajectory",
    "sincos_to_amplitude_phase",
]


def __getattr__(name: str) -> Any:
    # JAX is an optional backend, so its solver is imported on first use.
    if name == "jax_solve_linear_system":
        try:
            from ._jax_solver import jax_solve_linear_system
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "jax_solve_linear_system needs JAX: pip install 'disp_xr[jax]'"
            ) from e
        return jax_solve_linear_system
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
