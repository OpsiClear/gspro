"""
High-level pipeline API for composing gslut operations.

Provides an elegant, chainable interface for common workflows:
- Color adjustments with presets
- Geometric transformations
- Operation composition
"""

from collections.abc import Callable
from typing import Any

import numpy as np
import torch

from gslut.color import ColorLUT
from gslut.transforms import transform


class Pipeline:
    """
    Composable pipeline for chaining gslut operations.

    Provides optimized pipeline for common workflows:
        pipeline = (
            Pipeline()
            .adjust_colors(brightness=1.2, contrast=1.1)
            .transform(scale_factor=2.0, rotation=quat, translation=[1, 0, 0])
        )
        result = pipeline(data)

    All operations use lazy execution - they only run when you call the pipeline.
    Transform operations use fused 4x4 matrix composition for optimal performance.
    """

    def __init__(self, device: str = "cpu"):
        """
        Initialize pipeline.

        Args:
            device: Device for color LUT operations ("cpu" or "cuda")
        """
        self.device = device
        self._operations: list[tuple[str, Callable, dict]] = []
        self._color_lut: ColorLUT | None = None

    def adjust_colors(
        self,
        temperature: float = 0.5,
        brightness: float = 1.0,
        contrast: float = 1.0,
        gamma: float = 1.0,
        saturation: float = 1.0,
        shadows: float = 1.0,
        highlights: float = 1.0,
    ) -> "Pipeline":
        """
        Add color adjustment step.

        Args:
            temperature: Color temperature (0=cool, 0.5=neutral, 1=warm)
            brightness: Brightness multiplier
            contrast: Contrast multiplier
            gamma: Gamma correction
            saturation: Saturation adjustment
            shadows: Shadow boost/reduce
            highlights: Highlight boost/reduce

        Returns:
            Self for chaining
        """

        def apply_color_lut(data):
            if self._color_lut is None:
                self._color_lut = ColorLUT(device=self.device)
            return self._color_lut.apply(
                data,
                temperature=temperature,
                brightness=brightness,
                contrast=contrast,
                gamma=gamma,
                saturation=saturation,
                shadows=shadows,
                highlights=highlights,
            )

        self._operations.append(("adjust_colors", apply_color_lut, {}))
        return self

    def transform(
        self,
        translation: list | tuple | None = None,
        rotation: Any = None,
        rotation_format: str = "quaternion",
        scale_factor: float | list | tuple | None = None,
        center: list | tuple | None = None,
    ) -> "Pipeline":
        """
        Add optimized transformation step using fused 4x4 matrix composition.

        This method applies scale, rotation, and translation in a single operation
        using pre-computed 4x4 transformation matrices, providing 2-3x speedup
        compared to sequential operations for large point clouds (>100K points).

        Args:
            translation: Translation vector [x, y, z]
            rotation: Rotation in specified format
            rotation_format: "quaternion", "matrix", "axis_angle", or "euler"
            scale_factor: Uniform scale or per-axis [x, y, z]
            center: Center for rotation and scaling [x, y, z]

        Returns:
            Self for chaining

        Note:
            - Requires data to be dict with 'means' key
            - 'quaternions' required if rotation is specified
            - 'scales' required if scale_factor is specified
            - Transform order: scale -> rotate -> translate

        Example:
            >>> pipeline = Pipeline().transform(
            ...     scale_factor=2.0,
            ...     rotation=np.array([0.9239, 0, 0, 0.3827]),
            ...     translation=[1.0, 0.0, 0.0]
            ... )
        """
        self._operations.append(
            (
                "transform",
                transform,
                {
                    "translation": translation,
                    "rotation": rotation,
                    "rotation_format": rotation_format,
                    "scale_factor": scale_factor,
                    "center": center,
                },
            )
        )
        return self

    def custom(self, func: Callable, **kwargs) -> "Pipeline":
        """
        Add custom operation.

        Args:
            func: Custom function to apply
            **kwargs: Keyword arguments for the function

        Returns:
            Self for chaining

        Example:
            >>> pipeline.custom(lambda x: x * 2)
            >>> pipeline.custom(my_function, param1=value1)
        """
        self._operations.append(("custom", func, kwargs))
        return self

    def __call__(self, data: Any) -> Any:
        """
        Execute pipeline on data.

        Args:
            data: Input data (tensor, array, or dict)

        Returns:
            Transformed data (preserves input type)

        Note: For transform operations, data should be a dict with keys:
            - 'means': positions [N, 3]
            - 'quaternions': orientations [N, 4] (optional)
            - 'scales': sizes [N, 3] (optional)
        """
        # Track original type for preservation
        input_was_numpy = isinstance(data, np.ndarray)

        # Copy dict to avoid modifying input
        if isinstance(data, dict):
            result = data.copy()
        else:
            result = data

        for name, func, kwargs in self._operations:
            if name == "transform":
                # Transform operation (fused matrix-based)
                if isinstance(result, dict):
                    means, quats, scales = func(
                        result["means"],
                        result.get("quaternions"),
                        result.get("scales"),
                        **kwargs,
                    )
                    result["means"] = means
                    if quats is not None:
                        result["quaternions"] = quats
                    if scales is not None:
                        result["scales"] = scales
                else:
                    raise ValueError(
                        "transform requires dict input with means, quaternions, scales"
                    )
            else:
                # Simple functions (color adjustments, custom)
                result = func(result, **kwargs) if kwargs else func(result)

        # Preserve input type (ColorLUT may convert numpy -> torch)
        if input_was_numpy and isinstance(result, torch.Tensor):
            result = result.cpu().numpy()

        return result

    def reset(self) -> "Pipeline":
        """
        Clear all operations from pipeline.

        Returns:
            Self for chaining
        """
        self._operations.clear()
        self._color_lut = None
        return self


# ============================================================================
# Preset System
# ============================================================================


class ColorPreset:
    """
    Pre-configured color adjustment presets.

    Provides commonly used color grading looks:
        preset = ColorPreset.cinematic()
        colors = preset.apply(colors)
    """

    def __init__(
        self,
        temperature: float = 0.5,
        brightness: float = 1.0,
        contrast: float = 1.0,
        gamma: float = 1.0,
        saturation: float = 1.0,
        shadows: float = 1.0,
        highlights: float = 1.0,
        device: str = "cpu",
    ):
        """
        Create custom preset.

        Args:
            temperature: Color temperature (0=cool, 0.5=neutral, 1=warm)
            brightness: Brightness multiplier
            contrast: Contrast multiplier
            gamma: Gamma correction
            saturation: Saturation adjustment
            shadows: Shadow boost/reduce
            highlights: Highlight boost/reduce
            device: Device for LUT operations
        """
        self.params = {
            "temperature": temperature,
            "brightness": brightness,
            "contrast": contrast,
            "gamma": gamma,
            "saturation": saturation,
            "shadows": shadows,
            "highlights": highlights,
        }
        self.device = device
        self._lut = ColorLUT(device=device)

    def apply(self, colors: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        """
        Apply preset to colors.

        Args:
            colors: RGB colors [N, 3] in range [0, 1]

        Returns:
            Adjusted colors [N, 3] (same type as input)
        """
        input_was_numpy = isinstance(colors, np.ndarray)
        result = self._lut.apply(colors, **self.params)

        # Preserve input type
        if input_was_numpy and isinstance(result, torch.Tensor):
            result = result.cpu().numpy()

        return result

    def to_pipeline(self) -> Pipeline:
        """
        Convert preset to pipeline for further composition.

        Returns:
            Pipeline with this preset's color adjustments
        """
        return Pipeline(device=self.device).adjust_colors(**self.params)

    @classmethod
    def neutral(cls, device: str = "cpu") -> "ColorPreset":
        """Identity transformation (no changes)."""
        return cls(device=device)

    @classmethod
    def cinematic(cls, device: str = "cpu") -> "ColorPreset":
        """
        Cinematic look: desaturated, high contrast, teal shadows, orange highlights.
        """
        return cls(
            temperature=0.55,  # Slightly warm
            brightness=1.0,
            contrast=1.2,  # Increased contrast
            gamma=0.95,
            saturation=0.85,  # Desaturated
            shadows=1.1,  # Lift shadows (teal look)
            highlights=0.95,  # Compress highlights
            device=device,
        )

    @classmethod
    def warm(cls, device: str = "cpu") -> "ColorPreset":
        """
        Warm sunset look: orange tones, boosted brightness.
        """
        return cls(
            temperature=0.75,  # Very warm
            brightness=1.15,
            contrast=1.1,
            gamma=0.9,
            saturation=1.2,  # Boosted saturation
            shadows=1.15,
            highlights=0.9,
            device=device,
        )

    @classmethod
    def cool(cls, device: str = "cpu") -> "ColorPreset":
        """
        Cool blue look: blue tones, crisp contrast.
        """
        return cls(
            temperature=0.25,  # Very cool
            brightness=1.0,
            contrast=1.15,
            gamma=1.0,
            saturation=1.1,
            shadows=1.0,
            highlights=0.95,
            device=device,
        )

    @classmethod
    def vibrant(cls, device: str = "cpu") -> "ColorPreset":
        """
        Vibrant look: boosted saturation and contrast.
        """
        return cls(
            temperature=0.5,
            brightness=1.1,
            contrast=1.2,
            gamma=0.95,
            saturation=1.4,  # High saturation
            shadows=1.1,
            highlights=0.9,
            device=device,
        )

    @classmethod
    def muted(cls, device: str = "cpu") -> "ColorPreset":
        """
        Muted/pastel look: low saturation, lifted shadows.
        """
        return cls(
            temperature=0.5,
            brightness=1.05,
            contrast=0.95,  # Reduced contrast
            gamma=1.0,
            saturation=0.7,  # Low saturation
            shadows=1.2,  # Lifted shadows
            highlights=0.95,
            device=device,
        )

    @classmethod
    def dramatic(cls, device: str = "cpu") -> "ColorPreset":
        """
        Dramatic look: high contrast, crushed shadows, blown highlights.
        """
        return cls(
            temperature=0.5,
            brightness=1.0,
            contrast=1.4,  # Very high contrast
            gamma=0.85,
            saturation=1.1,
            shadows=0.8,  # Crushed shadows
            highlights=1.1,  # Blown highlights
            device=device,
        )


# ============================================================================
# High-level Functional API
# ============================================================================


def adjust_colors(
    data: np.ndarray | torch.Tensor,
    temperature: float = 0.5,
    brightness: float = 1.0,
    contrast: float = 1.0,
    gamma: float = 1.0,
    saturation: float = 1.0,
    shadows: float = 1.0,
    highlights: float = 1.0,
    device: str = "cpu",
) -> np.ndarray | torch.Tensor:
    """
    High-level function for color adjustments.

    Args:
        data: Input RGB colors [N, 3] in range [0, 1]
        temperature: Color temperature (0=cool, 0.5=neutral, 1=warm)
        brightness: Brightness multiplier
        contrast: Contrast multiplier
        gamma: Gamma correction
        saturation: Saturation adjustment
        shadows: Shadow boost/reduce
        highlights: Highlight boost/reduce
        device: Device for LUT operations

    Returns:
        Adjusted RGB colors [N, 3]

    Example:
        >>> # Apply color adjustments
        >>> adjusted = adjust_colors(rgb_colors, brightness=1.2, contrast=1.1)
    """
    input_was_numpy = isinstance(data, np.ndarray)

    # Apply color adjustments
    lut = ColorLUT(device=device)
    result = lut.apply(
        data,
        temperature=temperature,
        brightness=brightness,
        contrast=contrast,
        gamma=gamma,
        saturation=saturation,
        shadows=shadows,
        highlights=highlights,
    )

    # Preserve input type
    if input_was_numpy and isinstance(result, torch.Tensor):
        result = result.cpu().numpy()

    return result


def apply_preset(
    data: np.ndarray | torch.Tensor,
    preset: str | ColorPreset,
    device: str = "cpu",
) -> np.ndarray | torch.Tensor:
    """
    Apply color preset to RGB colors.

    Args:
        data: Input RGB colors [N, 3] in range [0, 1]
        preset: Preset name or ColorPreset instance
            Names: "neutral", "cinematic", "warm", "cool", "vibrant", "muted", "dramatic"
        device: Device for LUT operations

    Returns:
        Adjusted RGB colors [N, 3]

    Example:
        >>> adjusted = apply_preset(colors, "cinematic")
        >>> custom = ColorPreset(brightness=1.2, contrast=1.1)
        >>> adjusted = apply_preset(colors, custom)
    """
    # Get preset instance
    if isinstance(preset, str):
        preset_map = {
            "neutral": ColorPreset.neutral,
            "cinematic": ColorPreset.cinematic,
            "warm": ColorPreset.warm,
            "cool": ColorPreset.cool,
            "vibrant": ColorPreset.vibrant,
            "muted": ColorPreset.muted,
            "dramatic": ColorPreset.dramatic,
        }
        if preset not in preset_map:
            raise ValueError(f"Unknown preset: {preset}. Available: {', '.join(preset_map.keys())}")
        preset_obj = preset_map[preset](device=device)
    else:
        preset_obj = preset

    # Apply preset
    return preset_obj.apply(data)
