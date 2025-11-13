"""
Filter configuration for Gaussian splat filtering.

Provides configuration structure for volume, opacity, and scale filtering.
Based on Universal 4D Viewer filtering system.
"""

from dataclasses import dataclass


@dataclass
class FilterConfig:
    """
    Configuration for Gaussian splat filtering.

    Supports three independent filtering methods that can be combined:
    1. Volume filtering (spatial) - sphere or cuboid
    2. Opacity filtering - remove low-opacity Gaussians
    3. Scale filtering - remove outlier large Gaussians

    All filters use AND logic (all conditions must be met).

    Attributes:
        filter_type: Spatial filter type ("none", "sphere", "cuboid")
        sphere_center: Center point for sphere filtering [x, y, z]
        sphere_radius_factor: Radius multiplier (0.0 to 1.0 of max scene dimension)
        cuboid_center: Center point for cuboid filtering [x, y, z]
        cuboid_size_factor_x: Width multiplier (0.0 to 1.0)
        cuboid_size_factor_y: Height multiplier (0.0 to 1.0)
        cuboid_size_factor_z: Depth multiplier (0.0 to 1.0)
        opacity_threshold: Minimum opacity to keep (0.0 to 1.0)
        max_scale: Maximum scale threshold (large value = disabled)
    """

    # Spatial filtering
    filter_type: str = "none"  # Options: "none", "sphere", "cuboid"

    # Sphere parameters
    sphere_center: tuple[float, float, float] = (0.0, 0.0, 0.0)
    sphere_radius_factor: float = 1.0  # Multiplier (0.0 to 1.0)

    # Cuboid parameters
    cuboid_center: tuple[float, float, float] = (0.0, 0.0, 0.0)
    cuboid_size_factor_x: float = 1.0  # Multipliers (0.0 to 1.0)
    cuboid_size_factor_y: float = 1.0
    cuboid_size_factor_z: float = 1.0

    # Quality filters
    opacity_threshold: float = 0.05  # Range: 0.0 to 1.0
    max_scale: float = 10.0  # Large value = disabled

    def __post_init__(self):
        """Validate configuration parameters."""
        # Validate filter type
        valid_types = {"none", "sphere", "cuboid"}
        if self.filter_type not in valid_types:
            raise ValueError(
                f"Invalid filter_type: {self.filter_type}. "
                f"Must be one of {valid_types}"
            )

        # Validate ranges
        if not 0.0 <= self.sphere_radius_factor <= 1.0:
            raise ValueError("sphere_radius_factor must be between 0.0 and 1.0")

        if not 0.0 <= self.cuboid_size_factor_x <= 1.0:
            raise ValueError("cuboid_size_factor_x must be between 0.0 and 1.0")
        if not 0.0 <= self.cuboid_size_factor_y <= 1.0:
            raise ValueError("cuboid_size_factor_y must be between 0.0 and 1.0")
        if not 0.0 <= self.cuboid_size_factor_z <= 1.0:
            raise ValueError("cuboid_size_factor_z must be between 0.0 and 1.0")

        if not 0.0 <= self.opacity_threshold <= 1.0:
            raise ValueError("opacity_threshold must be between 0.0 and 1.0")

        if self.max_scale < 0.0:
            raise ValueError("max_scale must be non-negative")


# Default UI slider ranges for building interfaces
UI_RANGES = {
    "sphere_radius_factor": {"min": 0.0, "max": 1.0, "step": 0.01, "default": 1.0},
    "cuboid_size_factor": {"min": 0.0, "max": 1.0, "step": 0.01, "default": 1.0},
    "opacity_threshold": {"min": 0.0, "max": 1.0, "step": 0.01, "default": 0.05},
    "max_scale": {"min": 0.1, "max": 10.0, "step": 0.1, "default": 10.0},
}
