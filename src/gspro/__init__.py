"""
gspro - Gaussian Splatting Processing

Ultra-fast CPU-optimized processing for 3D Gaussian Splatting.

Features:
- Ultra-fast RGB color adjustments via compiled LUT pipelines
- 3D geometric transformations (NumPy/Numba optimized)
- Spatial filtering (within_sphere, within_box, min_opacity, max_scale)
- Unified Pipeline API for composing all operations
- GSData-only interface for clean, intuitive workflows
- Zero-copy in-place processing where possible
- Adjustments: temperature, brightness, contrast, gamma, saturation, vibrance, hue_shift, shadows, highlights
- Transforms: translate, rotate_quat/euler/axis_angle/matrix, scale
- Filtering: spherical/cuboid volumes, opacity, scale thresholds
- Parameterized templates for efficient parameter variation

Performance (with inplace=True, recommended):
  - Color: 1,015M/sec (0.10ms for 100K) | Kernel: 1,091M/sec (0.092ms)
  - Transform: 698M Gaussians/sec (1.43ms for 1M)
  - Filter: 46M/sec (2.2ms for 100K)

Example - Parameterized Templates:
    >>> from gspro import Color, Param
    >>>
    >>> # Create template with parameters
    >>> template = Color.template(
    ...     brightness=Param("b", default=1.2, range=(0.5, 2.0)),
    ...     contrast=Param("c", default=1.1, range=(0.5, 2.0))
    ... )
    >>>
    >>> # Use with different parameter values (cached for performance)
    >>> result = template(data, params={"b": 1.5, "c": 1.2})

Example - Unified Pipeline:
    >>> from gspro import Pipeline
    >>>
    >>> pipeline = (
    ...     Pipeline()
    ...     .within_sphere(radius=0.8)
    ...     .min_opacity(0.1)
    ...     .rotate_quat(quaternion)
    ...     .translate([1, 0, 0])
    ...     .brightness(1.2)
    ...     .saturation(1.3)
    ... )
    >>>
    >>> result = pipeline(data, inplace=True)

Example - Individual Pipelines:
    >>> from gspro import Color, Transform, Filter
    >>>
    >>> data = Filter().within_sphere(radius=0.8).min_opacity(0.1)(data)
    >>> data = Transform().rotate_quat(quat).translate([1, 0, 0])(data)
    >>> data = Color().brightness(1.2).saturation(1.3)(data)
"""

__version__ = "0.2.0"

# Import GSData from gsply
from gsply import GSData

# Color processing pipeline
from gspro.activations import apply_pre_activations
from gspro.color.pipeline import Color
from gspro.color.presets import ColorPreset

# Scene composition utilities
from gspro.compose import (
    compose_with_transforms,
    concatenate,
    deduplicate,
    merge_scenes,
    split_by_region,
)

# Filtering utilities
from gspro.filter.bounds import (
    SceneBounds,
    calculate_recommended_max_scale,
    calculate_scene_bounds,
)
from gspro.filter.masks import FilterMasks

# Filter pipeline
from gspro.filter.pipeline import Filter

# Parameterized pipelines
from gspro.params import Param

# Unified pipeline
from gspro.pipeline import Pipeline

# Protocols
from gspro.protocols import PipelineStage

# Quaternion utilities
from gspro.transform.api import (
    axis_angle_to_quaternion,
    euler_to_quaternion,
    quaternion_multiply,
    quaternion_to_euler,
    quaternion_to_rotation_matrix,
    rotation_matrix_to_quaternion,
)

# Transform pipeline
from gspro.transform.pipeline import Transform

# Utility functions
from gspro.utils import linear_interp_1d, multiply_opacity, nearest_neighbor_1d

__all__ = [
    # Version
    "__version__",
    # Data structures
    "GSData",
    # Core pipeline classes
    "Pipeline",
    "Color",
    "ColorPreset",
    "Transform",
    "Filter",
    "FilterMasks",
    # Parameterization
    "Param",
    # Protocols
    "PipelineStage",
    # Scene composition
    "concatenate",
    "compose_with_transforms",
    "deduplicate",
    "merge_scenes",
    "split_by_region",
    # Quaternion utilities
    "quaternion_multiply",
    "quaternion_to_rotation_matrix",
    "rotation_matrix_to_quaternion",
    "axis_angle_to_quaternion",
    "euler_to_quaternion",
    "quaternion_to_euler",
    # Filtering utilities
    "calculate_scene_bounds",
    "calculate_recommended_max_scale",
    "SceneBounds",
    # Utils
    "linear_interp_1d",
    "nearest_neighbor_1d",
    "multiply_opacity",
    "apply_pre_activations",
]
