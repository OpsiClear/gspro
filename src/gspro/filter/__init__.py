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
- Chainable pipeline interface for composing filters

Example:
    >>> from gspro import Filter, calculate_scene_bounds
    >>> from gsply import GSData
    >>>
    >>> # Pipeline interface with GSData
    >>> filtered = (Filter()
    ...     .within_sphere(radius=0.8)
    ...     .min_opacity(0.05)
    ...     .max_scale(2.5)
    ...     (data))
"""

from gspro.filter.bounds import (
    SceneBounds,
    calculate_recommended_max_scale,
    calculate_scene_bounds,
)
from gspro.filter.config import UI_RANGES, FilterConfig
from gspro.filter.masks import FilterMasks
from gspro.filter.pipeline import Filter

__all__ = [
    # Pipeline interface
    "Filter",
    "FilterMasks",
    # Configuration
    "FilterConfig",
    "UI_RANGES",
    # Utilities
    "SceneBounds",
    "calculate_scene_bounds",
    "calculate_recommended_max_scale",
]
