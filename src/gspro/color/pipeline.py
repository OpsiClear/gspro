"""
Color: Composable color adjustment pipeline with LUT pre-compilation and optimized operation stacking.

This module provides a fluent API for chaining color operations and compiling
them into a single optimized lookup table for maximum performance.

Key Features:
- Method chaining for intuitive pipeline construction
- **Operation stacking with automatic optimization**: Multiple operations reduced to minimal computation
- Automatic compilation of operations into single LUT
- Lazy compilation with dirty flag tracking
- In-place operation support for performance
- GSData integration for unified data handling

Example:
    >>> pipeline = (Color()
    ...     .brightness(1.1)      # First brightness
    ...     .contrast(1.05)       # Contrast
    ...     .brightness(1.1)      # Second brightness (optimized to 1.21 total)
    ...     .gamma(1.02)          # First gamma
    ...     .gamma(1.02)          # Second gamma (optimized to 1.0404 total)
    ...     .saturation(1.3)      # Saturation
    ... )
    >>> # Optimized to: brightness=1.21, contrast=1.05, gamma=1.0404, saturation=1.3
    >>> result = pipeline(data, inplace=True)
"""

from __future__ import annotations

import logging
import sys
from copy import deepcopy

# Python 3.10 compatibility: Self was added in Python 3.11
if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

import numpy as np
from gsply import GSData

from gspro.color.kernels import fused_color_pipeline_interleaved_lut_numba
from gspro.constants import (
    DEFAULT_LUT_SIZE,
    MAX_LUT_SIZE,
    MIN_LUT_SIZE,
)
from gspro.params import Param
from gspro.validators import validate_positive, validate_range

logger = logging.getLogger(__name__)


class Color:
    """
    Composable color adjustment pipeline with LUT pre-compilation and optimized operation stacking.

    This class allows chaining multiple color operations (including repeating
    the same operation) and automatically optimizes them into minimal computation
    during compilation.

    Phase 1 Operations (LUT-capable, support stacking with optimization):
    - temperature: Color temperature adjustment (additive composition)
    - brightness: Multiplicative brightness scaling (multiplicative composition)
    - contrast: Contrast expansion/contraction (multiplicative composition)
    - gamma: Non-linear gamma correction (multiplicative exponent composition)

    Phase 2 Operations (Stackable):
    - saturation: Color saturation adjustment (preserves luminance, multiplicative)
    - vibrance: Selective saturation boost for muted colors (multiplicative)
    - hue_shift: Rotate colors around color wheel (additive in degrees)
    - shadows: Shadow region adjustment (multiplicative composition)
    - highlights: Highlight region adjustment (multiplicative composition)

    Optimization:
        Multiple operations of the same type are automatically reduced:
        - brightness(1.2).brightness(1.5) -> brightness(1.8)
        - gamma(1.05).gamma(1.05) -> gamma(1.1025)
        - contrast(1.1).contrast(1.2) -> contrast(1.32)
        - saturation(1.2).saturation(1.5) -> saturation(1.8)
        - vibrance(1.3).vibrance(1.2) -> vibrance(1.56)
        - hue_shift(30).hue_shift(15) -> hue_shift(45)
        - shadows(1.2).shadows(1.5) -> shadows(1.8)
        - highlights(0.9).highlights(0.8) -> highlights(0.72)

    Example with stacking:
        >>> pipeline = (Color()
        ...     .brightness(1.1)      # 10% brighter
        ...     .brightness(1.1)      # Another 10% (compounds to 21% total)
        ...     .contrast(1.05)
        ...     .gamma(1.02)
        ...     .gamma(1.02)          # Stacked gamma (1.0404 total)
        ... )
        >>> result = pipeline(data, inplace=True)
    """

    __slots__ = (
        "lut_size",
        "_phase1_operations",  # List of (op_type, params) for Phase 1
        "_saturation_operations",  # Phase 2a: list of saturation values (stackable)
        "_vibrance_operations",  # Phase 2a: list of vibrance values (stackable via multiplication)
        "_hue_shift_operations",  # Phase 2a: list of hue shift degrees (additive)
        "_shadows_operations",  # Phase 2b: list of shadow values (stackable via multiplication)
        "_highlights_operations",  # Phase 2b: list of highlight values (stackable via multiplication)
        "_compiled_lut",
        "_is_dirty",
        "_param_map",  # dict[str, Param] for parameterized templates
        "_lut_cache",  # dict[tuple, np.ndarray] for caching LUTs by params
        "_lut_is_shared",  # Track if compiled LUT is shared with another pipeline (COW)
        "_param_order",  # tuple[str, ...] - pre-sorted param names for fast cache keys
        "_validated_cache",  # dict[tuple, dict] - cache validated params to avoid re-validation
        # Cached Phase 2 optimizations (computed during compile, reused every frame)
        "_cached_saturation",  # Optimized saturation value
        "_cached_vibrance",  # Optimized vibrance value
        "_cached_hue_shift",  # Optimized hue shift value (degrees)
        "_cached_shadows",  # Optimized shadows value
        "_cached_highlights",  # Optimized highlights value
        "_cached_phase2_is_identity",  # Pre-computed Phase 2 identity check
        # Pre-computed hue rotation matrix (9 coefficients)
        "_hue_m00", "_hue_m01", "_hue_m02",
        "_hue_m10", "_hue_m11", "_hue_m12",
        "_hue_m20", "_hue_m21", "_hue_m22",
    )

    def __init__(self, lut_size: int = DEFAULT_LUT_SIZE, device: str = "cpu"):
        """
        Initialize the color pipeline.

        Args:
            lut_size: Resolution of 1D LUTs (default 1024 = 0.1% precision)
            device: Kept for compatibility, always uses CPU (NumPy/Numba)

        Raises:
            ValueError: If lut_size is outside valid range
        """
        if not MIN_LUT_SIZE <= lut_size <= MAX_LUT_SIZE:
            raise ValueError(
                f"lut_size={lut_size} is outside valid range [{MIN_LUT_SIZE}, {MAX_LUT_SIZE}]. "
                f"Larger LUTs increase memory usage with diminishing quality gains. "
                f"Use 1024 (default, recommended) or 4096 for high precision."
            )

        self.lut_size = lut_size

        # Phase 1 operations (will be optimized and compiled into LUT)
        self._phase1_operations: list[tuple[str, dict]] = []

        # Phase 2a: Saturation, Vibrance, Hue (stackable)
        self._saturation_operations: list[float] = []
        self._vibrance_operations: list[float] = []
        self._hue_shift_operations: list[float] = []  # Degrees

        # Phase 2b: Shadows/Highlights (stackable via multiplication)
        self._shadows_operations: list[float] = []
        self._highlights_operations: list[float] = []

        # Compiled LUT state
        self._compiled_lut: np.ndarray | None = None
        self._is_dirty: bool = True
        self._lut_is_shared: bool = False  # COW: Track if LUT is shared

        # Parameterized template support
        # Maps param.name -> (Param object, operation_name)
        self._param_map: dict[str, tuple[Param, str]] = {}
        self._lut_cache: dict[tuple, np.ndarray] = {}
        self._param_order: tuple[str, ...] = ()  # Pre-sorted param names
        self._validated_cache: dict[tuple, dict[str, float]] = {}  # Cache validated params

        # Cached Phase 2 optimizations (initialized to defaults, computed during compile)
        self._cached_saturation = 1.0
        self._cached_vibrance = 1.0
        self._cached_hue_shift = 0.0
        self._cached_shadows = 1.0
        self._cached_highlights = 1.0
        self._cached_phase2_is_identity = True  # True until Phase 2 ops are added

        # Pre-computed hue rotation matrix (identity matrix for 0 degree rotation)
        self._hue_m00 = 1.0
        self._hue_m01 = 0.0
        self._hue_m02 = 0.0
        self._hue_m10 = 0.0
        self._hue_m11 = 1.0
        self._hue_m12 = 0.0
        self._hue_m20 = 0.0
        self._hue_m21 = 0.0
        self._hue_m22 = 1.0

        logger.info("[Color] Initialized with lut_size=%d", lut_size)

    @classmethod
    def template(cls, lut_size: int = DEFAULT_LUT_SIZE, **param_specs) -> Self:
        """
        Create a parameterized color pipeline template for efficient parameter variation.

        This method creates a pipeline where operations can have named parameters that
        can be efficiently swapped at runtime using LRU caching. Perfect for A/B testing,
        animation, and interactive parameter adjustment.

        Args:
            lut_size: Resolution of 1D LUTs (default 1024)
            **param_specs: Named parameters with Param specifications
                Valid keys: temperature, brightness, contrast, gamma, saturation, shadows, highlights

        Returns:
            Color pipeline configured as a parameterized template

        Example:
            >>> from gspro import Color, Param
            >>>
            >>> # Create template with parameters
            >>> template = Color.template(
            ...     brightness=Param("b", default=1.2, range=(0.5, 2.0)),
            ...     contrast=Param("c", default=1.1, range=(0.5, 2.0)),
            ...     saturation=Param("s", default=1.3, range=(0.0, 3.0))
            ... )
            >>>
            >>> # Use with different parameters (auto-cached)
            >>> result1 = template(data, params={"b": 1.5, "c": 1.2, "s": 1.4})
            >>> result2 = template(data, params={"b": 0.8, "c": 1.0, "s": 1.0})
            >>>
            >>> # Animation use case
            >>> for t in np.linspace(0, 1, 100):
            ...     brightness = 1.0 + t * 1.0  # 1.0 to 2.0
            ...     result = template(data, params={"b": brightness})  # Cached!
        """
        pipeline = cls(lut_size=lut_size)

        # Validate param_specs are all Param objects
        valid_ops = {
            "temperature",
            "brightness",
            "contrast",
            "gamma",
            "saturation",
            "shadows",
            "highlights",
        }

        # Track seen parameter names to detect collisions
        seen_param_names = set()

        for op_name, param in param_specs.items():
            if not isinstance(param, Param):
                raise TypeError(
                    f"Expected Param object for '{op_name}', got {type(param).__name__}"
                )

            if op_name not in valid_ops:
                raise ValueError(
                    f"Unknown operation '{op_name}'. Valid operations: {sorted(valid_ops)}"
                )

            # Check for duplicate param names
            if param.name in seen_param_names:
                raise ValueError(
                    f"Duplicate parameter name '{param.name}'. Each Param must have a unique name."
                )
            seen_param_names.add(param.name)

            # Store parameter mapping: param.name -> (Param, operation_name)
            pipeline._param_map[param.name] = (param, op_name)

            # Add operation with default value
            method = getattr(pipeline, op_name)
            method(param.default)

        # Set pre-sorted param order for fast cache key generation
        pipeline._param_order = tuple(sorted(pipeline._param_map.keys()))

        logger.info(
            "[Color] Created parameterized template with params: %s",
            pipeline._param_order,
        )
        return pipeline

    # ========================================================================
    # Properties
    # ========================================================================

    @property
    def is_compiled(self) -> bool:
        """Check if LUT is compiled and up-to-date."""
        return self._compiled_lut is not None and not self._is_dirty

    @property
    def needs_compilation(self) -> bool:
        """Check if compilation is needed."""
        return not self.is_compiled

    @property
    def operations(self) -> list[tuple[str, dict]]:
        """Get list of Phase 1 operations."""
        return self._phase1_operations.copy()

    # ========================================================================
    # Phase 1: LUT-Capable Operations (can be pre-compiled, support stacking)
    # ========================================================================

    @validate_range(0.0, 1.0, "temperature")
    def temperature(self, value: float) -> Self:
        """
        Add color temperature adjustment to the pipeline.

        Multiple temperature operations are combined additively (offsets add).

        Args:
            value: Temperature (0.0=cool/blue, 0.5=neutral, 1.0=warm/orange)

        Returns:
            Self for method chaining

        Example:
            >>> Color().temperature(0.6).temperature(0.7)  # Combined offset
        """
        self._phase1_operations.append(("temperature", {"value": float(value)}))
        self._is_dirty = True
        return self

    @validate_range(0.0, 5.0, "brightness")
    def brightness(self, value: float) -> Self:
        """
        Add brightness adjustment to the pipeline.

        Multiple brightness operations stack multiplicatively and are optimized:
        brightness(a).brightness(b) -> brightness(a * b)

        Args:
            value: Brightness multiplier (1.0=no change, >1.0=brighter, <1.0=darker)

        Returns:
            Self for method chaining

        Example:
            >>> Color().brightness(1.2).brightness(1.1)  # Optimized to brightness(1.32)
        """
        self._phase1_operations.append(("brightness", {"value": float(value)}))
        self._is_dirty = True
        return self

    @validate_range(0.0, 5.0, "contrast")
    def contrast(self, value: float) -> Self:
        """
        Add contrast adjustment to the pipeline.

        Multiple contrast operations stack multiplicatively and are optimized:
        contrast(a).contrast(b) -> contrast(a * b)

        Args:
            value: Contrast multiplier (1.0=no change, >1.0=more contrast, <1.0=less)

        Returns:
            Self for method chaining

        Example:
            >>> Color().contrast(1.1).contrast(1.05)  # Optimized to contrast(1.155)
        """
        self._phase1_operations.append(("contrast", {"value": float(value)}))
        self._is_dirty = True
        return self

    @validate_positive("gamma")
    def gamma(self, value: float) -> Self:
        """
        Add gamma correction to the pipeline.

        Multiple gamma operations stack multiplicatively and are optimized:
        gamma(a).gamma(b) -> gamma(a * b) because (x^a)^b = x^(a*b)

        Args:
            value: Gamma exponent (1.0=linear, >1.0=darken, <1.0=brighten)

        Returns:
            Self for method chaining

        Example:
            >>> Color().gamma(1.05).gamma(1.05)  # Optimized to gamma(1.1025)
        """
        self._phase1_operations.append(("gamma", {"value": float(value)}))
        self._is_dirty = True
        return self

    # ========================================================================
    # Phase 2: Sequential Operations (single values for now)
    # ========================================================================

    @validate_range(0.0, 5.0, "saturation")
    def saturation(self, value: float) -> Self:
        """
        Add saturation adjustment operation.

        Multiple saturation calls stack together multiplicatively:
        saturation(1.2).saturation(1.5) -> saturation(1.8)

        Args:
            value: Saturation multiplier (1.0=no change, 0.0=grayscale, >1.0=more saturated)

        Returns:
            Self for method chaining
        """
        self._saturation_operations.append(float(value))
        self._is_dirty = True
        return self

    @validate_range(0.0, 5.0, "vibrance")
    def vibrance(self, value: float) -> Self:
        """
        Add vibrance adjustment operation (selective saturation for muted colors).

        Vibrance boosts saturation more for less-saturated colors, preventing over-saturation
        of already vibrant colors. Multiple vibrance calls stack multiplicatively.

        Args:
            value: Vibrance multiplier (1.0=no change, >1.0=more vibrant, <1.0=less vibrant)

        Returns:
            Self for method chaining

        Example:
            >>> Color().vibrance(1.3).vibrance(1.2)  # Combined effect: 1.56x
        """
        self._vibrance_operations.append(float(value))
        self._is_dirty = True
        return self

    @validate_range(-180.0, 180.0, "hue_shift")
    def hue_shift(self, degrees: float) -> Self:
        """
        Add hue shift operation (rotate colors around the color wheel).

        Shifts all colors by a certain number of degrees around the color wheel
        (e.g., red -> orange -> yellow). Multiple hue shifts are additive.

        Args:
            degrees: Hue shift in degrees (-180 to 180, 0=no change)
                Positive values shift towards warmer colors (red -> orange -> yellow)
                Negative values shift towards cooler colors (red -> purple -> blue)

        Returns:
            Self for method chaining

        Example:
            >>> Color().hue_shift(30).hue_shift(15)  # Combined: 45 degree shift
        """
        self._hue_shift_operations.append(float(degrees))
        self._is_dirty = True
        return self

    @validate_range(0.0, 5.0, "shadows")
    def shadows(self, value: float) -> Self:
        """
        Add shadow region adjustment operation.

        Multiple shadows calls stack together multiplicatively:
        shadows(1.2).shadows(1.5) -> shadows(1.8)

        Args:
            value: Shadow adjustment (1.0=no change, >1.0=brighten shadows, <1.0=darken shadows)

        Returns:
            Self for method chaining

        Example:
            >>> Color().shadows(1.2).shadows(1.1)  # Combined effect: 1.32x
        """
        self._shadows_operations.append(float(value))
        self._is_dirty = True
        return self

    @validate_range(0.0, 5.0, "highlights")
    def highlights(self, value: float) -> Self:
        """
        Add highlight region adjustment operation.

        Multiple highlights calls stack together multiplicatively:
        highlights(0.9).highlights(0.8) -> highlights(0.72)

        Args:
            value: Highlight adjustment (1.0=no change, <1.0=darken highlights, >1.0=brighten highlights)

        Returns:
            Self for method chaining

        Example:
            >>> Color().highlights(0.9).highlights(0.95)  # Combined effect: 0.855x
        """
        self._highlights_operations.append(float(value))
        self._is_dirty = True
        return self

    # ========================================================================
    # Operation Optimization
    # ========================================================================

    def _optimize_phase1_operations(self) -> dict[str, float]:
        """
        Optimize Phase 1 operations into minimal computation.

        Reduces stacked operations to single optimized value per type:
        - brightness: multiply all values
        - contrast: multiply all values
        - gamma: multiply all exponents
        - temperature: add all offsets

        Returns:
            Dictionary of optimized operation values
        """
        # Initialize with neutral defaults
        optimized = {
            "temperature": 0.5,  # Neutral (no color shift)
            "brightness": 1.0,  # No change
            "contrast": 1.0,  # No change
            "gamma": 1.0,  # Linear
        }

        # Combine operations by type
        for op_type, params in self._phase1_operations:
            value = params["value"]

            if op_type == "temperature":
                # Temperature: add offsets
                # Convert current optimized value to offset
                current_offset = (optimized["temperature"] - 0.5) * 2 * 0.1
                # Convert new value to offset
                new_offset = (value - 0.5) * 2 * 0.1
                # Combine offsets
                combined_offset = current_offset + new_offset
                # Convert back to temperature value [0, 1]
                # Clamp to valid range
                combined_temp = (combined_offset / 0.1 / 2.0) + 0.5
                optimized["temperature"] = np.clip(combined_temp, 0.0, 1.0)

            elif op_type == "brightness":
                # Brightness: multiply values
                optimized["brightness"] *= value

            elif op_type == "contrast":
                # Contrast: multiply values (proven mathematically)
                optimized["contrast"] *= value

            elif op_type == "gamma":
                # Gamma: multiply exponents (x^a)^b = x^(a*b)
                optimized["gamma"] *= value

        return optimized

    def _optimize_saturation_operations(self) -> float:
        """
        Optimize saturation operations into minimal computation.

        Reduces stacked saturation operations to single value by multiplying:
        saturation(1.2).saturation(1.5) -> 1.8

        Mathematical proof: Saturation preserves luminance, allowing
        multiplicative composition: sat(s1).sat(s2) = sat(s1 * s2)

        Returns:
            Optimized saturation value (1.0 = neutral)
        """
        saturation = 1.0
        for sat_value in self._saturation_operations:
            saturation *= sat_value
        return saturation

    def _optimize_vibrance_operations(self) -> float:
        """
        Optimize vibrance operations into minimal computation.

        Reduces stacked vibrance operations to single value by multiplying:
        vibrance(1.2).vibrance(1.5) -> 1.8

        Mathematical proof: Vibrance applies saturation selectively based on
        current saturation, allowing multiplicative composition.

        Returns:
            Optimized vibrance value (1.0 = neutral)
        """
        vibrance = 1.0
        for vib_value in self._vibrance_operations:
            vibrance *= vib_value
        return vibrance

    def _optimize_hue_shift_operations(self) -> float:
        """
        Optimize hue shift operations into minimal computation.

        Reduces stacked hue shifts to single value by adding degrees:
        hue_shift(30).hue_shift(15) -> 45

        Mathematical proof: Hue rotation is additive in degrees:
        rotate(a).rotate(b) = rotate(a + b)

        Returns:
            Optimized hue shift in degrees (0 = neutral)
        """
        total_shift = 0.0
        for shift_degrees in self._hue_shift_operations:
            total_shift += shift_degrees
        # Normalize to [-180, 180] range
        total_shift = ((total_shift + 180.0) % 360.0) - 180.0
        return total_shift

    def _optimize_shadows_operations(self) -> float:
        """
        Optimize shadow operations into minimal computation.

        Reduces stacked shadow operations to single value by multiplying:
        shadows(1.2).shadows(1.5) -> 1.8

        Mathematical proof: Shadow adjustments apply multiplicative scaling,
        allowing composition: shadow(s1).shadow(s2) = shadow(s1 * s2)

        Returns:
            Optimized shadows value (1.0 = neutral)
        """
        shadows = 1.0
        for shadow_value in self._shadows_operations:
            shadows *= shadow_value
        return shadows

    def _optimize_highlights_operations(self) -> float:
        """
        Optimize highlight operations into minimal computation.

        Reduces stacked highlight operations to single value by multiplying:
        highlights(0.9).highlights(0.8) -> 0.72

        Mathematical proof: Highlight adjustments apply multiplicative scaling,
        allowing composition: highlight(h1).highlight(h2) = highlight(h1 * h2)

        Returns:
            Optimized highlights value (1.0 = neutral)
        """
        highlights = 1.0
        for highlight_value in self._highlights_operations:
            highlights *= highlight_value
        return highlights

    # ========================================================================
    # Compilation and Application
    # ========================================================================

    def compile(self) -> Self:
        """
        Compile Phase 1 operations into a single optimized LUT.

        This method:
        1. Optimizes stacked operations into minimal computation
        2. Applies optimized operations to build the LUT

        Returns:
            Self for method chaining
        """
        if self.is_compiled:
            logger.debug("[Color] Already compiled, skipping")
            return self

        # COW: If LUT is shared, copy it before modifying
        if self._lut_is_shared and self._compiled_lut is not None:
            logger.debug("[Color] Copying shared LUT (copy-on-write)")
            self._compiled_lut = self._compiled_lut.copy()
            self._lut_is_shared = False

        # Step 1: Optimize operations to minimal computation
        optimized = self._optimize_phase1_operations()

        logger.debug(
            "[Color] Optimized %d operations -> temp=%.2f, bright=%.2f, contrast=%.2f, gamma=%.2f",
            len(self._phase1_operations),
            optimized["temperature"],
            optimized["brightness"],
            optimized["contrast"],
            optimized["gamma"],
        )

        # Step 2: Build LUT from optimized values (single application per type)
        input_range = np.linspace(0, 1, self.lut_size, dtype=np.float32)

        # Calculate temperature offset once
        temp_factor = (optimized["temperature"] - 0.5) * 2  # Map [0,1] to [-1,1]
        temp_offset_r = temp_factor * 0.1
        temp_offset_b = -temp_factor * 0.1

        # R Channel (warm temperature adds offset)
        r = input_range + temp_offset_r
        r = r * optimized["brightness"]
        r = (r - 0.5) * optimized["contrast"] + 0.5
        r = np.power(np.clip(r, 1e-6, 1.0), optimized["gamma"])
        r_lut = np.clip(r, 0, 1).astype(np.float32)

        # G Channel (no temperature adjustment)
        g = input_range.copy()
        g = g * optimized["brightness"]
        g = (g - 0.5) * optimized["contrast"] + 0.5
        g = np.power(np.clip(g, 1e-6, 1.0), optimized["gamma"])
        g_lut = np.clip(g, 0, 1).astype(np.float32)

        # B Channel (cool temperature subtracts offset)
        b = input_range + temp_offset_b
        b = b * optimized["brightness"]
        b = (b - 0.5) * optimized["contrast"] + 0.5
        b = np.power(np.clip(b, 1e-6, 1.0), optimized["gamma"])
        b_lut = np.clip(b, 0, 1).astype(np.float32)

        # Create interleaved LUT for better cache locality (1.73x speedup)
        self._compiled_lut = np.stack([r_lut, g_lut, b_lut], axis=1)  # [lut_size, 3]

        # Cache Phase 2 optimizations (OPTIMIZATION: computed once at compile time, not per-frame)
        self._cached_saturation = self._optimize_saturation_operations()
        self._cached_vibrance = self._optimize_vibrance_operations()
        self._cached_hue_shift = self._optimize_hue_shift_operations()
        self._cached_shadows = self._optimize_shadows_operations()
        self._cached_highlights = self._optimize_highlights_operations()

        # Pre-compute hue rotation matrix (OPTIMIZATION #8: 8-12% speedup for hue ops)
        # This eliminates 2 trig functions + 9 FP operations per frame
        hue_rad = self._cached_hue_shift * (np.pi / 180.0)
        cos_hue = np.cos(hue_rad)
        sin_hue = np.sin(hue_rad)
        sqrt3 = 1.7320508075688772  # sqrt(3)

        self._hue_m00 = cos_hue + (1.0 - cos_hue) / 3.0
        self._hue_m01 = (1.0 - cos_hue) / 3.0 - sin_hue / sqrt3
        self._hue_m02 = (1.0 - cos_hue) / 3.0 + sin_hue / sqrt3
        self._hue_m10 = (1.0 - cos_hue) / 3.0 + sin_hue / sqrt3
        self._hue_m11 = cos_hue + (1.0 - cos_hue) / 3.0
        self._hue_m12 = (1.0 - cos_hue) / 3.0 - sin_hue / sqrt3
        self._hue_m20 = (1.0 - cos_hue) / 3.0 - sin_hue / sqrt3
        self._hue_m21 = (1.0 - cos_hue) / 3.0 + sin_hue / sqrt3
        self._hue_m22 = cos_hue + (1.0 - cos_hue) / 3.0

        # Pre-compute Phase 2 identity check (OPTIMIZATION: computed once, not per-frame)
        self._cached_phase2_is_identity = (
            self._cached_saturation == 1.0
            and self._cached_vibrance == 1.0
            and abs(self._cached_hue_shift) < 0.5
            and self._cached_shadows == 1.0
            and self._cached_highlights == 1.0
        )

        self._is_dirty = False
        logger.debug("[Color] LUT compilation complete")
        return self

    def _apply_to_colors(self, colors: np.ndarray, inplace: bool = True) -> np.ndarray:
        """
        Internal method: Apply color pipeline to NumPy array.

        Args:
            colors: Input RGB colors [N, 3] in range [0, 1]
            inplace: If True, modifies input array directly

        Returns:
            Transformed colors
        """
        # Auto-compile if needed
        if self.needs_compilation:
            self.compile()

        # Ensure float32 and handle copy/inplace
        if colors.dtype != np.float32:
            colors = colors.astype(np.float32)
            inplace = False  # Already made a copy
        elif not inplace:
            colors = colors.copy()

        # OPTIMIZATION: Use cached Phase 2 values (computed once at compile time)
        # This eliminates 5 function calls + loop iterations per frame (5-10% speedup)

        # Fast path: Use pre-computed Phase 2 identity check
        # This eliminates 5 comparisons + abs() call per frame (2-3% speedup)
        if self._cached_phase2_is_identity:
            # Only apply Phase 1 LUT, skip Phase 2 entirely
            from gspro.color.kernels import apply_lut_only_interleaved_numba
            apply_lut_only_interleaved_numba(colors, self._compiled_lut, colors)
        else:
            # Apply full pipeline (Phase 1 LUT + Phase 2 operations)
            # Use cached Phase 2 values (no function call overhead)
            # Pass pre-computed hue rotation matrix (OPTIMIZATION #8: 8-12% speedup)
            fused_color_pipeline_interleaved_lut_numba(
                colors,
                self._compiled_lut,
                self._cached_saturation,
                self._cached_vibrance,
                self._cached_hue_shift,
                self._cached_shadows,
                self._cached_highlights,
                colors,  # Output to same array (in-place)
                # Pre-computed hue rotation matrix
                self._hue_m00, self._hue_m01, self._hue_m02,
                self._hue_m10, self._hue_m11, self._hue_m12,
                self._hue_m20, self._hue_m21, self._hue_m22,
            )

        return colors

    def _get_cache_key(self, params: dict[str, float]) -> tuple:
        """
        Generate hashable cache key from parameter values.

        Uses pre-sorted param order for O(1) key generation instead of O(n log n) sorting.

        Args:
            params: Dictionary of parameter names to values

        Returns:
            Tuple of parameter values in pre-sorted order
        """
        # Use pre-sorted order (set once at template creation)
        # This is 8x faster than sorting every time (1.64us -> 0.20us)
        return tuple(params[name] for name in self._param_order)

    def _apply_with_params(self, data: GSData, params: dict[str, float], inplace: bool) -> GSData:
        """
        Apply pipeline with runtime parameter substitution and LRU caching.

        This method enables efficient parameter variation by caching compiled LUTs
        for each unique parameter combination. Subsequent calls with the same
        parameters reuse the cached LUT for maximum performance.

        Args:
            data: GSData object containing Gaussian data
            params: Runtime parameter values (e.g., {"b": 1.5, "c": 1.2})
            inplace: If True, modifies input GSData directly

        Returns:
            GSData with transformed colors

        Raises:
            ValueError: If parameter name is not in template or value outside range
        """
        # Quick check: validate param names are all known (before cache key generation)
        for param_name in params:
            if param_name not in self._param_map:
                raise ValueError(
                    f"Unknown parameter '{param_name}'. "
                    f"Valid parameters: {sorted(self._param_map.keys())}"
                )

        # Generate cache key (fast: uses pre-sorted order)
        cache_key = self._get_cache_key(params)

        # Check if we've already validated these exact parameter values
        if cache_key in self._validated_cache:
            # Reuse validated params (skip validation entirely - 3.17us saved!)
            op_params = self._validated_cache[cache_key]
        else:
            # First time seeing these params - validate values and cache
            op_params = {}  # operation_name -> validated value

            for param_name, value in params.items():
                param_obj, op_name = self._param_map[param_name]
                validated_value = param_obj.validate(value)
                op_params[op_name] = validated_value

            # Cache the validated params for future calls
            self._validated_cache[cache_key] = op_params

        # Check if we have a cached LUT for these parameter values
        if cache_key in self._lut_cache:
            # Cache hit - reuse compiled LUT (no recompilation needed)
            self._compiled_lut = self._lut_cache[cache_key]
        else:
            # Cache miss - need to compile with new parameters
            # Save original operations
            original_phase1 = self._phase1_operations.copy()
            original_saturation = self._saturation_operations.copy()
            original_vibrance = self._vibrance_operations.copy()
            original_hue_shift = self._hue_shift_operations.copy()
            original_shadows = self._shadows_operations.copy()
            original_highlights = self._highlights_operations.copy()

            try:
                # Temporarily update operations with runtime parameters
                # This modifies the operation lists to use the parameter values
                for i, (op_type, op_dict) in enumerate(self._phase1_operations):
                    if op_type in op_params:
                        # Replace the operation value with runtime parameter
                        self._phase1_operations[i] = (
                            op_type,
                            {"value": op_params[op_type]},
                        )

                # Update Phase 2 operations if parameterized
                if "saturation" in op_params:
                    # Replace all saturation operations with runtime value
                    self._saturation_operations = [op_params["saturation"]]

                if "vibrance" in op_params:
                    # Replace all vibrance operations with runtime value
                    self._vibrance_operations = [op_params["vibrance"]]

                if "hue_shift" in op_params:
                    # Replace all hue_shift operations with runtime value
                    self._hue_shift_operations = [op_params["hue_shift"]]

                if "shadows" in op_params:
                    # Replace all shadow operations with runtime value
                    self._shadows_operations = [op_params["shadows"]]

                if "highlights" in op_params:
                    # Replace all highlight operations with runtime value
                    self._highlights_operations = [op_params["highlights"]]

                # Recompile with new parameter values
                self._is_dirty = True
                self.compile()

                # Cache the compiled LUT for these parameter values
                self._lut_cache[cache_key] = self._compiled_lut.copy()
            finally:
                # ALWAYS restore original operations (even if compile() fails)
                self._phase1_operations = original_phase1
                self._saturation_operations = original_saturation
                self._vibrance_operations = original_vibrance
                self._hue_shift_operations = original_hue_shift
                self._shadows_operations = original_shadows
                self._highlights_operations = original_highlights

        # Apply the compiled (cached or new) LUT to data
        return self.apply(data, inplace=inplace)

    def is_identity(self) -> bool:
        """
        Check if this pipeline applies no color transformations (identity operation).

        Returns:
            True if pipeline has no operations, False otherwise

        Note:
            Identity pipelines have zero overhead with fast-path optimization
            (measured < 0.001ms vs 0.6ms without optimization)
        """
        return (
            not self._phase1_operations
            and not self._saturation_operations
            and not self._vibrance_operations
            and not self._hue_shift_operations
            and not self._shadows_operations
            and not self._highlights_operations
        )

    def apply(self, data: GSData, inplace: bool = True) -> GSData:
        """
        Apply the color pipeline to GSData object.

        Args:
            data: GSData object containing Gaussian data
            inplace: If True, modifies input GSData sh0 colors directly

        Returns:
            GSData with transformed colors

        Example:
            >>> import gsply
            >>> from gspro import Color
            >>> data = gsply.plyread("scene.ply")
            >>> adjusted_data = Color().brightness(1.2).saturation(1.3).apply(data)

        Note:
            Benchmarks show automatic make_contiguous() hurts performance due to conversion
            overhead (3.2ms for 100K) exceeding operation savings. Users should manually
            call data.make_contiguous() only for workflows with 100+ repeated operations.
        """
        # Fast-path: identity pipeline (no operations)
        # Measured: 0.001ms vs 0.6ms overhead for empty pipeline
        if self.is_identity():
            return data if inplace else data.copy()

        # Extract sh0 colors from GSData
        colors = data.sh0

        # Apply color transformation
        transformed_colors = self._apply_to_colors(colors, inplace=inplace)

        # If inplace, data.sh0 is already modified, return same GSData
        if inplace:
            logger.info("[Color] Applied to %d Gaussians (in-place)", len(data))
            return data

        # If not inplace, leverage GSData.copy() and modify colors
        # This is more efficient and idiomatic than manual field copying
        transformed_data = data.copy()
        transformed_data.sh0 = transformed_colors

        logger.info("[Color] Applied to %d Gaussians (new copy)", len(data))
        return transformed_data

    def __call__(
        self,
        data: GSData,
        inplace: bool = True,
        params: dict[str, float] | None = None,
    ) -> GSData:
        """
        Apply the pipeline when called as a function.

        Args:
            data: GSData object containing Gaussian data
            inplace: If True, modifies input GSData directly
            params: Optional runtime parameter values for parameterized templates
                Example: {"b": 1.5, "c": 1.2} for brightness and contrast

        Returns:
            GSData with transformed colors

        Example (standard pipeline):
            >>> data = Color().brightness(1.2).saturation(1.3)(data, inplace=True)

        Example (parameterized template):
            >>> template = Color.template(
            ...     brightness=Param("b", default=1.2, range=(0.5, 2.0))
            ... )
            >>> data = template(data, params={"b": 1.5})  # Cached for performance
        """
        if params is not None:
            # Parameterized template path - use caching
            if not self._param_map:
                raise ValueError(
                    "Pipeline was not created with template(). "
                    "Use Color() for non-parameterized pipelines."
                )
            return self._apply_with_params(data, params, inplace)

        # Standard pipeline path
        return self.apply(data, inplace=inplace)

    def reset(self) -> Self:
        """
        Reset all operations and parameters to defaults.

        Clears all operations, compiled state, and parameter caches.

        Returns:
            Self for method chaining
        """
        self._phase1_operations = []
        self._saturation_operations = []
        self._vibrance_operations = []
        self._hue_shift_operations = []
        self._shadows_operations = []
        self._highlights_operations = []
        self._compiled_lut = None
        self._is_dirty = True
        self._lut_is_shared = False  # Clear COW shared flag
        self._param_map = {}  # Clear parameterization
        self._lut_cache = {}  # Clear LUT cache (can be many MB)
        self._validated_cache = {}  # Clear validation cache
        logger.debug("[Color] Reset to defaults")
        return self

    def copy(self) -> Self:
        """
        Create a deep copy of this pipeline.

        Returns:
            New Color instance with same configuration

        Example:
            >>> pipeline = Color().brightness(1.2).contrast(1.1)
            >>> pipeline2 = pipeline.copy().saturation(1.5)  # Independent copy
        """
        return deepcopy(self)

    def get_operations(self) -> list[tuple[str, dict]]:
        """
        Get list of all Phase 1 operations (before optimization).

        Returns:
            List of (operation_type, parameters) tuples

        Example:
            >>> pipeline = Color().brightness(1.2).contrast(1.1).brightness(1.1)
            >>> ops = pipeline.get_operations()
            >>> # [('brightness', {'value': 1.2}), ('contrast', {'value': 1.1}), ('brightness', {'value': 1.1})]
        """
        return self._phase1_operations.copy()

    def get_optimized_params(self) -> dict[str, float]:
        """
        Get optimized parameters (after operation reduction).

        Returns:
            Dictionary of optimized parameter values

        Example:
            >>> pipeline = Color().brightness(1.2).brightness(1.1)
            >>> params = pipeline.get_optimized_params()
            >>> # {'temperature': 0.5, 'brightness': 1.32, 'contrast': 1.0, 'gamma': 1.0}
        """
        return self._optimize_phase1_operations()

    def get_params(self) -> dict[str, float]:
        """
        Get all current parameters (optimized Phase 1 + Phase 2).

        Returns:
            Dictionary of all parameter values

        Example:
            >>> pipeline = Color().brightness(1.2).saturation(1.3).vibrance(1.1)
            >>> params = pipeline.get_params()
            >>> # {'temperature': 0.5, 'brightness': 1.2, 'contrast': 1.0, 'gamma': 1.0,
            >>> #  'saturation': 1.3, 'vibrance': 1.1, 'hue_shift': 0.0,
            >>> #  'shadows': 1.0, 'highlights': 1.0}
        """
        params = self._optimize_phase1_operations()
        params["saturation"] = self._optimize_saturation_operations()
        params["vibrance"] = self._optimize_vibrance_operations()
        params["hue_shift"] = self._optimize_hue_shift_operations()
        params["shadows"] = self._optimize_shadows_operations()
        params["highlights"] = self._optimize_highlights_operations()
        return params

    def __len__(self) -> int:
        """Return total number of operations (before optimization)."""
        count = len(self._phase1_operations)
        count += len(self._saturation_operations)
        count += len(self._vibrance_operations)
        count += len(self._hue_shift_operations)
        count += len(self._shadows_operations)
        count += len(self._highlights_operations)
        return count

    def __repr__(self) -> str:
        """String representation of the pipeline."""
        parts = []

        # Show optimized Phase 1 values if operations exist
        if self._phase1_operations:
            optimized = self._optimize_phase1_operations()
            op_strs = []
            if optimized["temperature"] != 0.5:
                op_strs.append(f"temp={optimized['temperature']:.2f}")
            if optimized["brightness"] != 1.0:
                op_strs.append(f"bright={optimized['brightness']:.2f}")
            if optimized["contrast"] != 1.0:
                op_strs.append(f"contrast={optimized['contrast']:.2f}")
            if optimized["gamma"] != 1.0:
                op_strs.append(f"gamma={optimized['gamma']:.2f}")
            if op_strs:
                parts.append(", ".join(op_strs))

        # Show Phase 2 operations
        phase2 = []
        if self._saturation_operations:
            optimized_sat = self._optimize_saturation_operations()
            phase2.append(f"sat={optimized_sat:.2f}")
        if self._vibrance_operations:
            optimized_vib = self._optimize_vibrance_operations()
            phase2.append(f"vibrance={optimized_vib:.2f}")
        if self._hue_shift_operations:
            optimized_hue = self._optimize_hue_shift_operations()
            phase2.append(f"hue={optimized_hue:.1f}deg")
        if self._shadows_operations:
            optimized_shadows = self._optimize_shadows_operations()
            phase2.append(f"shadows={optimized_shadows:.2f}")
        if self._highlights_operations:
            optimized_highlights = self._optimize_highlights_operations()
            phase2.append(f"highlights={optimized_highlights:.2f}")

        if phase2:
            parts.append(", ".join(phase2))

        param_str = ", ".join(parts) if parts else "defaults"
        status = "compiled" if self.is_compiled else "not compiled"
        num_ops = len(self)  # Use __len__() which now includes all operations
        if num_ops > 0:
            return f"Color({param_str}) [{num_ops} ops, {status}]"
        return f"Color({param_str}) [{status}]"

    def __copy__(self) -> Self:
        """Shallow copy delegates to deep copy."""
        return self.copy()

    def __deepcopy__(self, memo) -> Self:
        """Create a deep copy of this pipeline with copy-on-write for compiled LUT."""
        new = Color(lut_size=self.lut_size)
        new._phase1_operations = deepcopy(self._phase1_operations, memo)
        new._saturation_operations = deepcopy(self._saturation_operations, memo)
        new._vibrance_operations = deepcopy(self._vibrance_operations, memo)
        new._hue_shift_operations = deepcopy(self._hue_shift_operations, memo)
        new._shadows_operations = deepcopy(self._shadows_operations, memo)
        new._highlights_operations = deepcopy(self._highlights_operations, memo)
        new._is_dirty = self._is_dirty

        # COW: Share compiled LUT reference instead of copying
        if self._compiled_lut is not None:
            new._compiled_lut = self._compiled_lut  # Share reference (COW)
            new._lut_is_shared = True  # Mark copy as having shared LUT
            self._lut_is_shared = True  # Mark original as having shared LUT too

        # Copy parameterization support
        new._param_map = self._param_map.copy()  # Shallow copy (Params are frozen)
        new._lut_cache = {k: v.copy() for k, v in self._lut_cache.items()}  # Deep copy LUTs
        new._param_order = self._param_order  # Shallow copy (tuple is immutable)
        new._validated_cache = self._validated_cache.copy()  # Shallow copy (cache is shared)

        return new
