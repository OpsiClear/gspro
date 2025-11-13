"""
gspro - Gaussian Splatting Processing

Ultra-fast CPU-optimized color processing for 3D Gaussian Splatting.

Features:
- Ultra-fast RGB color adjustments via separated 1D LUTs
- Single fused NumPy/Numba kernel (33x speedup)
- Zero-copy in-place processing API
- Temperature, brightness, contrast, gamma, saturation, shadows, highlights

Performance: 1,851 M colors/sec (0.054ms for 100K colors)
"""

__version__ = "0.1.0"

# Core LUT class (CPU-only, NumPy/Numba)
from gspro.color import ColorLUT

__all__ = [
    "__version__",
    "ColorLUT",
]
