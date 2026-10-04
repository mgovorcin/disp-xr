from typing import Any

from ._design_matrix import (
    create_design_matrix,
    get_coefficient_names,
    infer_parameters_from_names,
)
from ._numpy_solver import lscov_2d
from ._stats_analysis import get_residuals_analysis
from .denoise import denoise
from .estimate_fit import (
    fit_timeseries,
    fit_timeseries_block,
    get_predicted,
    sincos_to_amplitude_phase,
)
from .noise import NoiseResult, fit_noise
from .temporal import (
    apply_segmented,
    gap_mask,
    grid_step_days,
    regular_time_grid,
    split_at_breaks,
)
from .trajectory import Trajectory, fit_trajectory, predict_trajectory
from .velocity import fit_velocity

__all__ = [
    "NoiseResult",
    "Trajectory",
    "apply_segmented",
    "create_design_matrix",
    "denoise",
    "fit_timeseries",
    "fit_noise",
    "fit_timeseries_block",
    "fit_trajectory",
    "fit_velocity",
    "gap_mask",
    "grid_step_days",
    "get_coefficient_names",
    "get_predicted",
    "get_residuals_analysis",
    "infer_parameters_from_names",
    "jax_solve_linear_system",
    "lscov_2d",
    "predict_trajectory",
    "regular_time_grid",
    "split_at_breaks",
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
