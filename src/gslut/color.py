"""
Separated Color LUT for Fast RGB Color Adjustments

Per-channel operations use fast 1D LUT lookups:
- Temperature, Brightness, Contrast, Gamma

Cross-channel operations use sequential processing:
- Saturation, Shadows/Highlights (require RGB mixing)

Performance: ~10x faster than sequential ops, 60x faster than 3D LUT
CPU Optimization: Uses NumPy for 2-3x faster LUT operations on CPU
"""

import logging
from typing import Dict, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


class ColorLUT:
    """
    Separated per-channel (1D LUT) and cross-channel color processing.

    This class provides fast color adjustments using 1D lookup tables for per-channel
    operations and sequential processing for cross-channel operations.

    Per-channel operations (via 1D LUT):
    - Temperature: Adds warmth/coolness by adjusting R and B channels
    - Brightness: Multiplies all channels
    - Contrast: Expands/contracts around middle gray
    - Gamma: Power curve for non-linear brightness adjustment

    Cross-channel operations (sequential):
    - Saturation: Lerp between grayscale and original colors
    - Shadows: Boost/reduce dark regions
    - Highlights: Boost/reduce bright regions

    CPU Optimization: Automatically uses NumPy for 2-3x faster processing on CPU
    """

    def __init__(self, device: str = "cuda", lut_size: int = 1024):
        """
        Initialize separated color LUT.

        Args:
            device: Torch device for LUT tensors ("cuda" or "cpu")
            lut_size: Resolution of 1D LUTs (1024 = 0.1% precision)
        """
        self.device = device
        self.lut_size = lut_size
        self.use_numpy = device == "cpu"

        # Three 1D LUTs for R, G, B channels
        self.r_lut: Optional[np.ndarray | torch.Tensor] = None
        self.g_lut: Optional[np.ndarray | torch.Tensor] = None
        self.b_lut: Optional[np.ndarray | torch.Tensor] = None

        # Cache parameters to detect changes
        self._cached_params: Optional[Dict] = None

        if self.use_numpy:
            logger.info(
                f"[ColorLUT] Initialized with {lut_size} bins per channel (NumPy CPU mode)"
            )
        else:
            logger.info(f"[ColorLUT] Initialized with {lut_size} bins per channel")

    def apply(
        self,
        colors: torch.Tensor,
        temperature: float = 0.5,
        brightness: float = 1.0,
        contrast: float = 1.0,
        gamma: float = 1.0,
        saturation: float = 1.0,
        shadows: float = 1.0,
        highlights: float = 1.0,
    ) -> torch.Tensor:
        """
        Apply separated color adjustments.

        Args:
            colors: Input RGB colors [N, 3] in range [0, 1]
            temperature: Temperature adjustment (0=cool, 0.5=neutral, 1=warm)
            brightness: Brightness multiplier (1.0=no change)
            contrast: Contrast multiplier (1.0=no change)
            gamma: Gamma correction exponent (1.0=linear)
            saturation: Saturation adjustment (1.0=no change, 0=grayscale)
            shadows: Shadow adjustment (1.0=no change)
            highlights: Highlight adjustment (1.0=no change)

        Returns:
            Adjusted RGB colors [N, 3] in range [0, 1]
        """
        # Check if we need to recompile LUTs
        current_params = {
            "temperature": temperature,
            "brightness": brightness,
            "contrast": contrast,
            "gamma": gamma,
        }

        if self._cached_params != current_params or self.r_lut is None:
            self._compile_per_channel_luts(temperature, brightness, contrast, gamma)
            self._cached_params = current_params

        # Apply per-channel 1D LUTs
        if self.use_numpy:
            adjusted = self._apply_luts_numpy(colors)
        else:
            adjusted = self._apply_luts_torch(colors)

        # Apply cross-channel operations
        adjusted = self._apply_saturation(adjusted, saturation)
        adjusted = self._apply_shadows_highlights(adjusted, shadows, highlights)

        return adjusted

    def _compile_per_channel_luts(
        self, temperature: float, brightness: float, contrast: float, gamma: float
    ) -> None:
        """
        Compile 1D LUTs for per-channel operations.

        Combines all per-channel operations into a single lookup table.

        Args:
            temperature: Temperature adjustment (0=cool, 1=warm)
            brightness: Brightness multiplier
            contrast: Contrast multiplier
            gamma: Gamma correction exponent
        """
        # Temperature offset
        temp_factor = (temperature - 0.5) * 2  # Map [0,1] to [-1,1]
        temp_offset_r = temp_factor * 0.1
        temp_offset_b = -temp_factor * 0.1

        if self.use_numpy:
            # CPU optimization: Use NumPy (2-3x faster than PyTorch on CPU)
            input_range = np.linspace(0, 1, self.lut_size, dtype=np.float32)

            # R Channel LUT (includes temperature)
            r = input_range.copy()
            r = r + temp_offset_r
            r = r * brightness
            r = (r - 0.5) * contrast + 0.5
            r = np.power(np.clip(r, 1e-6, 1), gamma)
            self.r_lut = np.clip(r, 0, 1).astype(np.float32)

            # G Channel LUT (no temperature)
            g = input_range.copy()
            g = g * brightness
            g = (g - 0.5) * contrast + 0.5
            g = np.power(np.clip(g, 1e-6, 1), gamma)
            self.g_lut = np.clip(g, 0, 1).astype(np.float32)

            # B Channel LUT (includes negative temperature)
            b = input_range.copy()
            b = b + temp_offset_b
            b = b * brightness
            b = (b - 0.5) * contrast + 0.5
            b = np.power(np.clip(b, 1e-6, 1), gamma)
            self.b_lut = np.clip(b, 0, 1).astype(np.float32)

        else:
            # GPU path: Use PyTorch tensors
            input_range = torch.linspace(0, 1, self.lut_size, device=self.device)

            # R Channel LUT (includes temperature)
            r = input_range.clone()
            r = r + temp_offset_r
            r = r * brightness
            r = (r - 0.5) * contrast + 0.5
            r = torch.pow(r.clamp(1e-6, 1), gamma)
            self.r_lut = r.clamp(0, 1)

            # G Channel LUT (no temperature)
            g = input_range.clone()
            g = g * brightness
            g = (g - 0.5) * contrast + 0.5
            g = torch.pow(g.clamp(1e-6, 1), gamma)
            self.g_lut = g.clamp(0, 1)

            # B Channel LUT (includes negative temperature)
            b = input_range.clone()
            b = b + temp_offset_b
            b = b * brightness
            b = (b - 0.5) * contrast + 0.5
            b = torch.pow(b.clamp(1e-6, 1), gamma)
            self.b_lut = b.clamp(0, 1)

        logger.debug(
            f"[ColorLUT] Compiled 1D LUTs: temp={temperature:.2f}, "
            f"bright={brightness:.2f}, contrast={contrast:.2f}, gamma={gamma:.2f}"
        )

    def _apply_luts_numpy(self, colors: torch.Tensor) -> torch.Tensor:
        """Apply per-channel 1D LUTs using NumPy (CPU optimization)."""
        # Convert to NumPy
        colors_np = colors.numpy() if isinstance(colors, torch.Tensor) else colors

        # Quantize input colors to LUT indices
        indices = (colors_np * (self.lut_size - 1)).astype(np.int64)
        indices = np.clip(indices, 0, self.lut_size - 1)

        # Lookup in 1D LUTs (vectorized NumPy indexing)
        adjusted_np = np.stack(
            [
                self.r_lut[indices[:, 0]],
                self.g_lut[indices[:, 1]],
                self.b_lut[indices[:, 2]],
            ],
            axis=1,
        )

        return torch.from_numpy(adjusted_np)

    def _apply_luts_torch(self, colors: torch.Tensor) -> torch.Tensor:
        """Apply per-channel 1D LUTs using PyTorch (GPU path)."""
        # Quantize input colors to LUT indices
        indices = (colors * (self.lut_size - 1)).long().clamp(0, self.lut_size - 1)

        # Lookup in 1D LUTs
        adjusted = torch.stack(
            [
                self.r_lut[indices[:, 0]],
                self.g_lut[indices[:, 1]],
                self.b_lut[indices[:, 2]],
            ],
            dim=1,
        )

        return adjusted

    def _apply_saturation(self, colors: torch.Tensor, saturation: float) -> torch.Tensor:
        """Apply saturation adjustment (cross-channel operation)."""
        if saturation == 1.0:
            return colors

        # Calculate luminance (weighted RGB average)
        luminance = 0.299 * colors[:, 0] + 0.587 * colors[:, 1] + 0.114 * colors[:, 2]
        luminance = luminance.unsqueeze(1).expand_as(colors)

        # Lerp between grayscale and original
        return torch.lerp(luminance, colors, saturation).clamp(0, 1)

    def _apply_shadows_highlights(
        self, colors: torch.Tensor, shadows: float, highlights: float
    ) -> torch.Tensor:
        """Apply shadows/highlights adjustment (cross-channel operation)."""
        if shadows == 1.0 and highlights == 1.0:
            return colors

        # Calculate luminance for masking
        luminance = (
            0.299 * colors[:, 0] + 0.587 * colors[:, 1] + 0.114 * colors[:, 2]
        ).unsqueeze(1)

        # Create masks
        shadow_mask = (luminance < 0.5).float()
        highlight_mask = (luminance >= 0.5).float()

        # Apply adjustments
        shadow_adj = colors * shadow_mask * (shadows - 1.0)
        highlight_adj = colors * highlight_mask * (highlights - 1.0)

        return (colors + shadow_adj + highlight_adj).clamp(0, 1)

    def reset(self) -> None:
        """Reset LUT cache, forcing recompilation on next apply."""
        self.r_lut = None
        self.g_lut = None
        self.b_lut = None
        self._cached_params = None
        logger.debug("[ColorLUT] Cache reset")
