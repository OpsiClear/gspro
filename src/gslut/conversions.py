"""
Spherical Harmonics and RGB Color Conversions

Provides conversion functions between RGB color space and Spherical Harmonics (SH)
coefficients, specifically the DC component (SH0) used in Gaussian Splatting.

The SH0 coefficient represents the constant (zero-order) term of the spherical
harmonics expansion, which encodes a uniform color contribution.
"""

import torch

# Spherical Harmonics DC component constant
# This is 1/(2*sqrt(pi)) and represents the normalization factor for SH0
SH_C0 = 0.28209479177387814


def sh2rgb(sh: torch.Tensor) -> torch.Tensor:
    """
    Convert Spherical Harmonics DC component (SH0) to RGB colors.

    The inverse operation to rgb2sh, this converts SH0 coefficients back to
    RGB color space using the formula: RGB = SH0 * C0 + 0.5

    Args:
        sh: SH0 coefficients tensor, any shape [..., 3]
            Expected range: approximately [-1.77, 1.77] for RGB in [0, 1]

    Returns:
        RGB colors tensor [..., 3] in range [0, 1]

    Example:
        >>> sh = torch.tensor([[-1.0, 0.0, 1.0]])
        >>> rgb = sh2rgb(sh)
        >>> print(rgb)
        tensor([[0.2179, 0.5000, 0.7821]])
    """
    return sh * SH_C0 + 0.5


def rgb2sh(rgb: torch.Tensor) -> torch.Tensor:
    """
    Convert RGB colors to Spherical Harmonics DC component (SH0).

    This converts RGB colors to the 0th-order spherical harmonics coefficient,
    which represents a uniform color contribution. The conversion formula is:
    SH0 = (RGB - 0.5) / C0

    Args:
        rgb: RGB colors tensor, any shape [..., 3]
            Expected range: [0, 1]

    Returns:
        SH0 coefficients tensor [..., 3]
            Range: approximately [-1.77, 1.77] for RGB in [0, 1]

    Example:
        >>> rgb = torch.tensor([[1.0, 0.5, 0.0]])
        >>> sh = rgb2sh(rgb)
        >>> print(sh)
        tensor([[ 1.7725,  0.0000, -1.7725]])
    """
    return (rgb - 0.5) / SH_C0


def get_sh_c0_constant() -> float:
    """
    Get the SH0 normalization constant.

    Returns:
        The C0 constant (0.28209479177387814) used in SH0 conversions
    """
    return SH_C0
