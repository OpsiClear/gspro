"""
Scene bounds calculation for filtering.

Provides utilities for calculating spatial bounds of Gaussian splat scenes.
Based on Universal 4D Viewer TransformService.calculate_scene_bounds().
"""

from typing import Any

import numpy as np


class SceneBounds:
    """
    Container for scene spatial bounds.

    Attributes:
        min: Minimum point [x, y, z]
        max: Maximum point [x, y, z]
        sizes: Scene dimensions [width, height, depth]
        max_size: Largest dimension
        center: Scene center point [x, y, z]
    """

    def __init__(
        self,
        min_point: np.ndarray,
        max_point: np.ndarray,
    ):
        """
        Initialize scene bounds.

        Args:
            min_point: Minimum corner [x, y, z]
            max_point: Maximum corner [x, y, z]
        """
        self.min = np.asarray(min_point, dtype=np.float32)
        self.max = np.asarray(max_point, dtype=np.float32)
        self.sizes = self.max - self.min
        self.max_size = float(np.max(self.sizes))
        self.center = (self.min + self.max) * 0.5

    def __repr__(self) -> str:
        return (
            f"SceneBounds(min={self.min}, max={self.max}, "
            f"sizes={self.sizes}, max_size={self.max_size:.3f})"
        )


def calculate_scene_bounds(positions: np.ndarray) -> SceneBounds:
    """
    Calculate spatial bounds of a Gaussian splat scene.

    Uses NumPy's optimized min/max which is already highly tuned.
    Testing showed NumPy is competitive with custom Numba for this operation.

    Args:
        positions: Gaussian positions [N, 3] in format [x, y, z]

    Returns:
        SceneBounds object containing min, max, sizes, max_size, center

    Example:
        >>> positions = np.random.randn(1000, 3)
        >>> bounds = calculate_scene_bounds(positions)
        >>> print(f"Scene size: {bounds.sizes}")
        >>> print(f"Center: {bounds.center}")

    Note:
        - Input should be in world coordinates
        - Performance: ~1.5ms for 1M Gaussians
        - Returns float32 for memory efficiency
    """
    # Validate input
    positions = np.asarray(positions, dtype=np.float32)

    if len(positions) == 0:
        raise ValueError("positions cannot be empty")

    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(f"positions must be [N, 3], got shape {positions.shape}")

    # Use NumPy's optimized min/max (competitive with custom Numba)
    min_point = positions.min(axis=0)
    max_point = positions.max(axis=0)

    return SceneBounds(min_point, max_point)


def calculate_recommended_max_scale(scales: np.ndarray, percentile: float = 99.5) -> float:
    """
    Calculate recommended maximum scale threshold using percentiles.

    Uses Numba-optimized max calculation for 2x speedup.

    Args:
        scales: Gaussian scales [N, 3] in format [scale_x, scale_y, scale_z]
        percentile: Percentile to use (default: 99.5)

    Returns:
        Recommended max_scale threshold

    Example:
        >>> scales = np.random.rand(1000, 3) * 2.0
        >>> threshold = calculate_recommended_max_scale(scales)
        >>> print(f"Recommended max_scale: {threshold:.4f}")

    Note:
        - Run on first frame for good defaults
        - 99.5th percentile filters top 0.5% of scales
        - Adjust percentile based on data quality
        - Performance: ~12ms for 1M Gaussians (optimized with Numba)
    """
    from gspro.filter.kernels import calculate_max_scales_numba

    # Validate input
    scales = np.asarray(scales, dtype=np.float32)

    if len(scales) == 0:
        raise ValueError("scales cannot be empty")

    if scales.ndim != 2 or scales.shape[1] != 3:
        raise ValueError(f"scales must be [N, 3], got shape {scales.shape}")

    if not 0.0 <= percentile <= 100.0:
        raise ValueError(f"percentile must be between 0 and 100, got {percentile}")

    # Get maximum scale per Gaussian with Numba optimization
    max_scales = np.empty(len(scales), dtype=np.float32)
    calculate_max_scales_numba(scales, max_scales)

    # Calculate percentile (NumPy is fast enough for percentile)
    threshold = float(np.percentile(max_scales, percentile))

    return threshold
