"""
gspro - Gaussian Splatting Processing

Ultra-fast CPU-optimized processing for 3D Gaussian Splatting.

Features:
- Ultra-fast RGB color adjustments via separated 1D LUTs
- 3D geometric transformations (NumPy/Numba optimized)
- High-level pipeline API for composing operations
- Zero-copy in-place processing API
- Temperature, brightness, contrast, gamma, saturation, shadows, highlights

Performance: 1,851 M colors/sec (0.054ms for 100K colors)
"""

__version__ = "0.1.0"

# Core color processing (CPU-only, NumPy/Numba)
from gspro.color import ColorLUT

# 3D transformations (CPU-only, NumPy/Numba)
from gspro.transform import (
    axis_angle_to_quaternion,
    euler_to_quaternion,
    quaternion_multiply,
    quaternion_to_euler,
    quaternion_to_rotation_matrix,
    rotate,
    rotation_matrix_to_quaternion,
    scale,
    transform,
    translate,
)

# High-level pipeline API
from gspro.pipeline import ColorPreset, Pipeline, adjust_colors, apply_preset

# Filtering API
from gspro.filter import (
    FilterConfig,
    SceneBounds,
    apply_filter,
    calculate_recommended_max_scale,
    calculate_scene_bounds,
    filter_gaussians,
)

# Utility functions
from gspro.utils import linear_interp_1d, nearest_neighbor_1d

__all__ = [
    # Version
    "__version__",
    # Core classes
    "ColorLUT",
    # Transforms
    "translate",
    "rotate",
    "scale",
    "transform",
    "quaternion_multiply",
    "quaternion_to_rotation_matrix",
    "rotation_matrix_to_quaternion",
    "axis_angle_to_quaternion",
    "euler_to_quaternion",
    "quaternion_to_euler",
    # High-level API
    "Pipeline",
    "ColorPreset",
    "adjust_colors",
    "apply_preset",
    # Filtering API
    "FilterConfig",
    "SceneBounds",
    "apply_filter",
    "filter_gaussians",
    "calculate_scene_bounds",
    "calculate_recommended_max_scale",
    # Utils
    "linear_interp_1d",
    "nearest_neighbor_1d",
]
