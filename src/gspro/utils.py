"""
Utility functions for LUT operations (NumPy/CPU implementation)

Provides helper functions for linear interpolation and nearest neighbor lookup.
"""

import numpy as np


def linear_interp_1d(x: np.ndarray, centers: np.ndarray, values: np.ndarray) -> np.ndarray:
    """
    Perform 1D linear interpolation using sorted cluster centers.

    Args:
        x: Input values to interpolate [N]
        centers: Sorted cluster centers [K]
        values: Precomputed output values at cluster centers [K]

    Returns:
        Interpolated values [N]

    Example:
        >>> centers = np.array([0.0, 0.5, 1.0])
        >>> values = np.array([0.0, 0.25, 1.0])
        >>> x = np.array([0.25, 0.75])
        >>> result = linear_interp_1d(x, centers, values)
        >>> print(result)
        [0.125 0.625]
    """
    indices = np.searchsorted(centers, x)
    indices = np.clip(indices, 1, len(centers) - 1)

    left_idx = indices - 1
    right_idx = indices

    left_centers = centers[left_idx]
    right_centers = centers[right_idx]
    left_values = values[left_idx]
    right_values = values[right_idx]

    alpha = (x - left_centers) / (right_centers - left_centers + 1e-8)
    result = left_values + alpha * (right_values - left_values)

    return result


def nearest_neighbor_1d(
    x: np.ndarray, centers: np.ndarray, values: np.ndarray
) -> np.ndarray:
    """
    Perform 1D nearest neighbor lookup.

    Args:
        x: Input values [N]
        centers: Cluster centers [K] (need not be sorted)
        values: Precomputed output values at cluster centers [K]

    Returns:
        Values at nearest neighbors [N]

    Example:
        >>> centers = np.array([0.0, 0.5, 1.0])
        >>> values = np.array([0.0, 0.25, 1.0])
        >>> x = np.array([0.1, 0.7])
        >>> result = nearest_neighbor_1d(x, centers, values)
        >>> print(result)
        [0.   0.25]
    """
    # Compute distances from each x to each center
    x_expanded = x[:, np.newaxis]  # [N, 1]
    centers_expanded = centers[np.newaxis, :]  # [1, K]
    distances = np.abs(x_expanded - centers_expanded)  # [N, K]

    nearest_idx = np.argmin(distances, axis=1)
    return values[nearest_idx]
