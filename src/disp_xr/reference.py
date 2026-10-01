"""Reference-point selection and re-referencing."""

import logging
import random
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["find_reference_point"]


def find_reference_point(
    pct_mask: np.ndarray,
    pct_ps: np.ndarray,
    phase_jumps: Optional[np.ndarray] = None,
    percentile: int = 99,
    max_jumps: int = 1,
) -> Tuple[int, int]:
    """Select a reference point based on PS stability and mask constraints.

    The function identifies a reference point where the PS (Persistent Scatterer)
    percentage is above the specified percentile and the percentage mask is nonzero.
    If phase jump data is provided, it is used to further refine the selection by
    excluding PS pixels with excessive phase jumps.

    Parameters
    ----------
    pct_mask : np.ndarray
        Percentage mask where nonzero values indicate valid regions.
    pct_ps : np.ndarray
        Percentage of PS stability in the stack (100% remains PS).
    phase_jumps : Optional[np.ndarray], optional
        Number of 2π phase jumps in the stack. Used for refinement. Defaults to None.
    percentile : int, optional
        Percentile threshold for PS stability selection. Defaults to 99.
    max_jumps : int, optional
        Maximum allowable phase jumps for valid reference points. Defaults to 1.

    Returns
    -------
    Tuple[int, int]
        The (y, x) coordinates of the selected reference point.

    """
    # Get valid pixes by masking out 100% recommended mask and 0% of pixels in the stack
    valid_pixels = np.ma.masked_where((pct_ps == 0) | (pct_mask == 100), pct_ps)

    # Compute 90th percentile threshold over valid pixels
    threshold = np.nanpercentile(valid_pixels.filled(np.nan), percentile)

    # Create a mask for high-coherence pixels
    valid = valid_pixels >= threshold

    # if timeseries inversion residualt metric is specified
    # remove pixels with > 1 phase jumps
    if phase_jumps is not None:
        valid[np.where((valid == 1) & (phase_jumps > max_jumps))] = 0

    # Get indices of valid pixels
    valid_indices = np.argwhere(valid.filled(0))

    if valid_indices.size == 0:
        raise ValueError(
            ("No valid reference point found.Adjust threshold or check input data.")
        )

    # Randomly select one valid pixel
    y, x = np.squeeze(random.choice(valid_indices))

    # Flip y-axis if needed (assuming `y` is inverted)
    # y = mean_tcoh.shape[0] - 1 - y

    logger.info(f"Selected reference pixel (y/x): {(y, x)}")

    return y, x
