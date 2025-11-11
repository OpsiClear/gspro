"""
gslut - Gaussian Splatting Look-Up Tables

Fast LUT-based operations for 3D Gaussian Splatting, including:
- Activation functions (exp, sigmoid, normalize) via clustered LUTs
- RGB color adjustments via separated 1D LUTs
- Spherical harmonics and RGB conversions

Performance optimized for both CPU and GPU.
"""

__version__ = "0.1.0"

# Core LUT classes
from gslut.activation import ActivationLUT
from gslut.color import ColorLUT

# Conversion functions
from gslut.conversions import SH_C0, get_sh_c0_constant, rgb2sh, sh2rgb

# Utility functions
from gslut.utils import (
    build_kmeans_clusters,
    linear_interp_1d,
    nearest_neighbor_1d,
)

__all__ = [
    # Version
    "__version__",
    # Core classes
    "ActivationLUT",
    "ColorLUT",
    # Conversions
    "sh2rgb",
    "rgb2sh",
    "get_sh_c0_constant",
    "SH_C0",
    # Utils
    "linear_interp_1d",
    "nearest_neighbor_1d",
    "build_kmeans_clusters",
]
