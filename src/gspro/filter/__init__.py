"""
Gaussian splat filtering module.

Provides volume, opacity, and scale filtering for Gaussian splats.
Based on Universal 4D Viewer filtering system.

Features:
- Volume filtering (sphere or cuboid)
- Opacity filtering (remove low-opacity Gaussians)
- Scale filtering (remove outlier large Gaussians)
- Combined filtering with AND logic
- Automatic scene bounds calculation
- Recommended threshold calculation

Example:
    >>> from gspro.filter import FilterConfig, apply_filter, calculate_scene_bounds
    >>>
    >>> # Create filter configuration
    >>> config = FilterConfig(
    ...     filter_type="sphere",
    ...     sphere_radius_factor=0.8,
    ...     opacity_threshold=0.05
    ... )
    >>>
    >>> # Apply filtering
    >>> mask = apply_filter(positions, opacities, config=config)
    >>> filtered_positions = positions[mask]
"""

from gspro.filter.api import apply_filter, filter_gaussians
from gspro.filter.bounds import (
    SceneBounds,
    calculate_recommended_max_scale,
    calculate_scene_bounds,
)
from gspro.filter.config import FilterConfig, UI_RANGES

__all__ = [
    # Configuration
    "FilterConfig",
    "UI_RANGES",
    # Filtering functions
    "apply_filter",
    "filter_gaussians",
    # Utilities
    "SceneBounds",
    "calculate_scene_bounds",
    "calculate_recommended_max_scale",
]
