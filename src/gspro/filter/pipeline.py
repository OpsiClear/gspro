"""
Filter: Composable filtering pipeline for Gaussian Splatting.

This module provides a fluent API for chaining filter operations into
an optimized filtering pipeline with elegant, self-documenting interface.

Key Features:
- Semantic method names (within_sphere, min_opacity, max_scale)
- Method chaining for intuitive pipeline construction
- Support for volume, opacity, and scale filtering
- Explicit compile() for user control over optimization
- Consistent API with Color and Transform pipelines
- GSData integration for unified data handling
"""

from __future__ import annotations

import logging
import sys
from copy import deepcopy
from typing import Any

# Python 3.10 compatibility: Self was added in Python 3.11
if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

import numpy as np

# Import GSData from gsply
from gsply import GSData

# Import constants and validators
from gspro.constants import (
    DEFAULT_MAX_SCALE,
    DEFAULT_OPACITY_THRESHOLD,
    OPACITY_MAX,
    OPACITY_MIN,
)

# Import existing filter functions
from gspro.filter.api import _apply_filter
from gspro.filter.bounds import SceneBounds, calculate_scene_bounds
from gspro.params import Param
from gspro.validators import validate_range

logger = logging.getLogger(__name__)


class Filter:
    """
    Composable filtering pipeline for Gaussian Splatting data.

    This class allows chaining multiple filter operations for efficient
    data filtering with a clean, consistent interface.

    Supported Operations:
    - within_sphere: Keep Gaussians within spherical volume
    - within_box: Keep Gaussians within box-shaped volume
    - min_opacity: Keep Gaussians with opacity >= threshold
    - max_scale: Keep Gaussians with scale <= threshold
    - bounds: Use pre-calculated scene bounds

    Example:
        >>> pipeline = (Filter()
        ...     .within_sphere(center=(0,0,0), radius=0.8)
        ...     .min_opacity(0.05)
        ...     .max_scale(2.5)
        ...     .compile()
        ... )
        >>> filtered = pipeline(data, inplace=True)
    """

    __slots__ = (
        "_volume_filters",  # List of (type, params) tuples
        "_opacity_thresholds",  # List of opacity thresholds
        "_scale_thresholds",  # List of scale thresholds
        "_scene_bounds",
        "_optimized_opacity",  # Cached optimized value
        "_optimized_scale",  # Cached optimized value
        "_optimized_volumes",  # Cached optimized volume filters
        "_is_compiled",  # Track if compile() was called
        "_param_map",  # dict[str, tuple[Param, str]] for parameterized templates
        "_filter_cache",  # dict[tuple, tuple] for caching optimized values by params
        "_param_order",  # Pre-sorted param names for fast cache keys (O(1) vs O(n log n))
    )

    def __init__(self):
        """Initialize the filter pipeline."""
        # Operation lists (support stacking)
        self._volume_filters: list[tuple[str, dict]] = []
        self._opacity_thresholds: list[float] = []
        self._scale_thresholds: list[float] = []

        # Pre-calculated scene bounds (optional)
        self._scene_bounds = None

        # Optimized values (cached after compile())
        self._optimized_opacity = DEFAULT_OPACITY_THRESHOLD
        self._optimized_scale = DEFAULT_MAX_SCALE
        self._optimized_volumes: list[tuple[str, dict]] = []
        self._is_compiled = False

        # Parameterized template support
        # Maps param.name -> (Param object, operation_name)
        self._param_map: dict[str, tuple[Param, str]] = {}
        self._filter_cache: dict[tuple, tuple] = {}
        self._param_order: tuple[str, ...] = ()  # Pre-sorted param names for fast cache keys

        logger.info("[Filter] Pipeline initialized")

    @classmethod
    def template(cls, **param_specs) -> Self:
        """
        Create a parameterized filter pipeline template for efficient parameter variation.

        Allows runtime parameter substitution with caching for performance.
        Useful for interactive adjustment, animation, A/B testing, etc.

        Supported parameterizable operations:
        - sphere_radius: Radius for within_sphere() filter
        - min_opacity: Threshold for opacity filtering
        - max_scale: Threshold for scale filtering

        Args:
            **param_specs: Keyword arguments mapping operation names to Param objects

        Returns:
            Filter pipeline configured as a template

        Raises:
            TypeError: If any value is not a Param object
            ValueError: If operation name is not supported

        Example:
            >>> from gspro import Filter, Param
            >>> template = Filter.template(
            ...     sphere_radius=Param("r", default=0.8, range=(0.1, 1.0)),
            ...     min_opacity=Param("o", default=0.1, range=(0.0, 1.0))
            ... )
            >>> # Use with different parameter values (cached)
            >>> result = template(data, params={"r": 0.6, "o": 0.05})
        """
        pipeline = cls()

        valid_ops = {"sphere_radius", "min_opacity", "max_scale"}

        # Track seen parameter names to detect collisions
        seen_param_names = set()

        for op_name, param in param_specs.items():
            if not isinstance(param, Param):
                raise TypeError(
                    f"Expected Param object for '{op_name}', got {type(param).__name__}"
                )

            if op_name not in valid_ops:
                raise ValueError(f"Unknown operation '{op_name}'. Valid operations: {valid_ops}")

            # Check for duplicate param names
            if param.name in seen_param_names:
                raise ValueError(
                    f"Duplicate parameter name '{param.name}'. Each Param must have a unique name."
                )
            seen_param_names.add(param.name)

            # Store mapping: param.name -> (Param, operation_name)
            pipeline._param_map[param.name] = (param, op_name)

            # Add operation with default value
            if op_name == "sphere_radius":
                pipeline.within_sphere(radius=param.default)
            elif op_name == "min_opacity":
                pipeline.min_opacity(param.default)
            elif op_name == "max_scale":
                pipeline.max_scale(param.default)

        # Set pre-sorted param order for fast cache key generation (O(1) vs O(n log n))
        pipeline._param_order = tuple(sorted(pipeline._param_map.keys()))

        logger.info("[Filter] Template created with %d parameters", len(param_specs))
        return pipeline

    def within_sphere(
        self, center: tuple | list | np.ndarray | None = None, radius: float = 1.0
    ) -> Self:
        """
        Keep Gaussians within a spherical volume. Multiple calls stack with AND logic (intersection).

        Same-center spheres are optimized to minimum radius.
        Different-center spheres require computing geometric intersection.

        Args:
            center: Sphere center (default: scene center from bounds)
            radius: Radius as fraction of scene bounds (1.0 = full scene, 0.5 = half)

        Returns:
            Self for method chaining

        Example:
            >>> Filter().within_sphere(center=(0,0,0), radius=0.8)  # Keep central 80%
        """
        # Store as (type, params) tuple (internal storage uses "radius_factor")
        params = {
            "center": center,
            "radius_factor": float(radius),
        }
        self._volume_filters.append(("sphere", params))
        self._is_compiled = False  # Mark as needing recompilation

        logger.debug("[Filter] Sphere filter added: center=%s, radius=%s", center, radius)
        return self

    def within_box(
        self,
        center: tuple | list | np.ndarray | None = None,
        size: tuple | list | np.ndarray | None = None,
    ) -> Self:
        """
        Keep Gaussians within a box-shaped volume. Multiple calls stack with AND logic (intersection).

        Same-center boxes are optimized to minimum size per dimension.
        Different-center boxes require computing geometric intersection.

        Args:
            center: Box center (default: scene center from bounds)
            size: Box size [x,y,z] as fraction of scene bounds (1.0 = full scene, 0.5 = half)

        Returns:
            Self for method chaining

        Example:
            >>> Filter().within_box(center=(0,0,0), size=(0.8, 0.8, 0.8))  # Keep central 80% cube
        """
        # Store as (type, params) tuple (internal storage uses "size_factors")
        params = {
            "center": center,
            "size_factors": size,
        }
        self._volume_filters.append(("cuboid", params))
        self._is_compiled = False  # Mark as needing recompilation

        logger.debug("[Filter] Box filter added: center=%s, size=%s", center, size)
        return self

    @validate_range(OPACITY_MIN, OPACITY_MAX, "threshold")
    def min_opacity(self, threshold: float = DEFAULT_OPACITY_THRESHOLD) -> Self:
        """
        Keep Gaussians with opacity >= threshold. Multiple calls stack with AND logic (max threshold).

        min_opacity(0.1).min_opacity(0.2) is equivalent to min_opacity(0.2)

        Args:
            threshold: Minimum opacity to keep (range [0, 1])

        Returns:
            Self for method chaining

        Example:
            >>> Filter().min_opacity(0.05)  # Keep if opacity >= 0.05
        """
        self._opacity_thresholds.append(float(threshold))
        self._is_compiled = False  # Mark as needing recompilation
        logger.debug("[Filter] Min opacity threshold added: %s", threshold)
        return self

    def max_scale(self, threshold: float = DEFAULT_MAX_SCALE) -> Self:
        """
        Keep Gaussians with scale <= threshold. Multiple calls stack with AND logic (min threshold).

        max_scale(10.0).max_scale(5.0) is equivalent to max_scale(5.0)

        Args:
            threshold: Maximum allowed scale value

        Returns:
            Self for method chaining

        Example:
            >>> Filter().max_scale(2.5)  # Keep if scale <= 2.5
        """
        if threshold <= 0:
            raise ValueError(f"Max scale threshold must be positive, got {threshold}")

        self._scale_thresholds.append(float(threshold))
        self._is_compiled = False  # Mark as needing recompilation
        logger.debug("[Filter] Max scale threshold added: %s", threshold)
        return self

    def bounds(self, scene_bounds: SceneBounds) -> Self:
        """
        Set pre-calculated scene bounds for volume filtering.

        Args:
            scene_bounds: Pre-calculated scene bounds

        Returns:
            Self for method chaining
        """
        self._scene_bounds = scene_bounds
        logger.debug("[Filter] Using pre-calculated scene bounds")
        return self

    def _optimize_thresholds(self) -> tuple[float, float]:
        """
        Optimize threshold operations (VALIDATED: always works).

        Opacity: Multiple thresholds combine with AND logic -> max(all)
        Scale: Multiple thresholds combine with AND logic -> min(all)

        Returns:
            (optimized_opacity, optimized_scale)
        """
        # Opacity: take MAXIMUM (most restrictive, mathematically validated)
        if self._opacity_thresholds:
            optimized_opacity = max(self._opacity_thresholds)
        else:
            optimized_opacity = DEFAULT_OPACITY_THRESHOLD

        # Scale: take MINIMUM (most restrictive, mathematically validated)
        if self._scale_thresholds:
            optimized_scale = min(self._scale_thresholds)
        else:
            optimized_scale = DEFAULT_MAX_SCALE

        return optimized_opacity, optimized_scale

    def _optimize_volume_filters(self) -> list[tuple[str, dict]]:
        """
        Optimize volume filters where mathematically validated.

        VALIDATED optimizations:
        - Same-center spheres: min(radius_factors)
        - Same-center cuboids: min per dimension

        NOT optimized (kept as-is):
        - Different-center volumes (complex geometry)
        - Mixed sphere+cuboid (complex geometry)

        Returns:
            Optimized list of volume filters
        """
        if not self._volume_filters:
            return []

        # Group by type and center
        spheres_by_center: dict[tuple, list[float]] = {}
        cuboids_by_center: dict[tuple, list[tuple]] = {}

        for ftype, params in self._volume_filters:
            center = params.get("center")
            # Convert center to hashable tuple (None -> None, arrays -> tuple)
            if center is None:
                center_key = None
            elif isinstance(center, np.ndarray):
                center_key = tuple(center.tolist())
            elif isinstance(center, (list, tuple)):
                center_key = tuple(center)
            else:
                center_key = None

            if ftype == "sphere":
                radius_factor = params["radius_factor"]
                if center_key not in spheres_by_center:
                    spheres_by_center[center_key] = []
                spheres_by_center[center_key].append(radius_factor)

            elif ftype == "cuboid":
                size_factors = params["size_factors"]
                if center_key not in cuboids_by_center:
                    cuboids_by_center[center_key] = []
                cuboids_by_center[center_key].append(size_factors)

        optimized = []

        # Optimize same-center spheres (VALIDATED: min radius)
        for center_key, radius_factors in spheres_by_center.items():
            if len(radius_factors) == 1:
                # Single sphere - no optimization needed
                optimized.append(
                    (
                        "sphere",
                        {
                            "center": center_key,
                            "radius_factor": radius_factors[0],
                        },
                    )
                )
            else:
                # Multiple spheres at same center - take minimum radius (VALIDATED)
                min_radius = min(radius_factors)
                optimized.append(
                    (
                        "sphere",
                        {
                            "center": center_key,
                            "radius_factor": min_radius,
                        },
                    )
                )
                logger.debug(
                    "[Filter] Optimized %d same-center spheres: %s -> %s",
                    len(radius_factors),
                    radius_factors,
                    min_radius,
                )

        # Optimize same-center cuboids (VALIDATED: min per dimension)
        for center_key, size_factors_list in cuboids_by_center.items():
            if len(size_factors_list) == 1:
                # Single cuboid - no optimization needed
                optimized.append(
                    (
                        "cuboid",
                        {
                            "center": center_key,
                            "size_factors": size_factors_list[0],
                        },
                    )
                )
            else:
                # Multiple cuboids at same center - take min per dimension (VALIDATED)
                # Handle both tuple and array size_factors
                min_x = min(
                    sf[0] if isinstance(sf, (tuple, list, np.ndarray)) else sf
                    for sf in size_factors_list
                )
                min_y = min(
                    sf[1] if isinstance(sf, (tuple, list, np.ndarray)) else sf
                    for sf in size_factors_list
                )
                min_z = min(
                    sf[2] if isinstance(sf, (tuple, list, np.ndarray)) else sf
                    for sf in size_factors_list
                )
                optimized.append(
                    (
                        "cuboid",
                        {
                            "center": center_key,
                            "size_factors": (min_x, min_y, min_z),
                        },
                    )
                )
                logger.debug(
                    "[Filter] Optimized %d same-center cuboids: %s -> (%s, %s, %s)",
                    len(size_factors_list),
                    size_factors_list,
                    min_x,
                    min_y,
                    min_z,
                )

        return optimized

    def _get_cache_key(self, params: dict[str, float]) -> tuple:
        """
        Generate hashable cache key from parameter values.

        Uses pre-sorted param order for O(1) key generation instead of O(n log n) sorting.

        Args:
            params: Dictionary mapping param names to values

        Returns:
            Tuple of parameter values in pre-sorted order
        """
        # Use pre-sorted order (set once at template creation)
        # This is 8x faster than sorting every time (1.64us -> 0.20us)
        return tuple(params[name] for name in self._param_order)

    def _apply_with_params(self, data: GSData, params: dict[str, float], inplace: bool) -> GSData:
        """
        Apply filter pipeline with runtime parameter substitution and caching.

        Args:
            data: Input GSData to filter
            params: Dictionary mapping param names to values
            inplace: Whether to modify data in-place

        Returns:
            Filtered GSData

        Raises:
            ValueError: If unknown parameter or parameter out of range
        """
        # Validate and build operation mapping
        validated_params = {}  # param.name -> validated value
        op_params = {}  # operation_name -> validated value

        for param_name, value in params.items():
            if param_name not in self._param_map:
                valid_params = list(self._param_map.keys())
                raise ValueError(
                    f"Unknown parameter '{param_name}'. Valid parameters: {valid_params}"
                )

            param_obj, op_name = self._param_map[param_name]
            validated_value = param_obj.validate(value)
            validated_params[param_name] = validated_value
            op_params[op_name] = validated_value

        # Check cache
        cache_key = self._get_cache_key(validated_params)

        if cache_key in self._filter_cache:
            # Cache HIT - reuse optimized values
            self._optimized_opacity, self._optimized_scale, self._optimized_volumes = (
                self._filter_cache[cache_key]
            )
            self._is_compiled = True
            logger.debug("[Filter] Cache hit for params=%s", validated_params)
        else:
            # Cache MISS - recompile with new parameters
            logger.debug("[Filter] Cache miss for params=%s", validated_params)

            # Save original operations
            original_volume_filters = self._volume_filters.copy()
            original_opacity_thresholds = self._opacity_thresholds.copy()
            original_scale_thresholds = self._scale_thresholds.copy()

            try:
                # Temporarily update operations with runtime parameters
                for i, (ftype, fparams) in enumerate(self._volume_filters):
                    if ftype == "sphere" and "sphere_radius" in op_params:
                        # Update sphere radius
                        new_params = fparams.copy()
                        new_params["radius_factor"] = op_params["sphere_radius"]
                        self._volume_filters[i] = (ftype, new_params)

                if "min_opacity" in op_params:
                    # Replace all opacity thresholds with the parameterized one
                    self._opacity_thresholds = [op_params["min_opacity"]]

                if "max_scale" in op_params:
                    # Replace all scale thresholds with the parameterized one
                    self._scale_thresholds = [op_params["max_scale"]]

                # Recompile
                self._is_compiled = False
                self.compile()

                # Cache the optimized values
                self._filter_cache[cache_key] = (
                    self._optimized_opacity,
                    self._optimized_scale,
                    self._optimized_volumes.copy(),
                )
            finally:
                # ALWAYS restore original operations (even if compile() fails)
                self._volume_filters = original_volume_filters
                self._opacity_thresholds = original_opacity_thresholds
                self._scale_thresholds = original_scale_thresholds

        # Apply with optimized values
        return self.apply(data, inplace=inplace)

    def compile(self) -> Self:
        """
        Compile and optimize filter operations.

        Optimizations (all mathematically VALIDATED):
        - Opacity thresholds: max(all) - O(N) -> O(1)
        - Scale thresholds: min(all) - O(N) -> O(1)
        - Same-center spheres: min(radii) - O(N) -> O(1)
        - Same-center cuboids: min per dimension - O(N) -> O(1)

        Not optimized (different-center volumes):
        - Must compute geometric intersections at runtime

        Returns:
            Self for method chaining

        Example:
            >>> pipeline = (Filter()
            ...     .opacity(0.1).opacity(0.2)  # Will optimize to max(0.1, 0.2) = 0.2
            ...     .compile()
            ... )
        """
        # Optimize thresholds (always works)
        self._optimized_opacity, self._optimized_scale = self._optimize_thresholds()

        # Optimize volume filters (same-center only)
        self._optimized_volumes = self._optimize_volume_filters()

        self._is_compiled = True

        logger.info(
            "[Filter] Compiled: %d opacity thresholds -> 1, %d scale thresholds -> 1, "
            "%d volume filters -> %d",
            len(self._opacity_thresholds),
            len(self._scale_thresholds),
            len(self._volume_filters),
            len(self._optimized_volumes),
        )

        return self

    def _apply_to_arrays(
        self,
        positions: np.ndarray,
        quaternions: np.ndarray | None = None,
        scales: np.ndarray | None = None,
        opacities: np.ndarray | None = None,
        colors: np.ndarray | None = None,
        shN: np.ndarray | None = None,
    ) -> tuple[np.ndarray, ...]:
        """
        Internal method: Apply filter pipeline to NumPy arrays.

        Args:
            positions: Gaussian positions [N, 3]
            quaternions: Optional orientations [N, 4]
            scales: Optional scales [N, 3]
            opacities: Optional opacities [N] or [N, 1]
            colors: Optional colors [N, C] where C is color channels
            shN: Optional higher-order SH coefficients [N, K, 3]

        Returns:
            Tuple of filtered arrays: (positions, quaternions, scales, opacities, colors, shN)
        """
        # Auto-compile if not compiled
        if not self._is_compiled:
            self.compile()

        # Calculate scene bounds if needed and not provided
        if self._scene_bounds is None and self._optimized_volumes:
            self._scene_bounds = calculate_scene_bounds(positions)
            logger.debug("[Filter] Calculated scene bounds automatically")

        # OPTIMIZATION: Use fast path for 0 or 1 volume filters
        # This allows a single apply_filter() call instead of multiple
        if len(self._optimized_volumes) <= 1:
            # Fast path: Call apply_filter() once with all parameters
            if len(self._optimized_volumes) == 1:
                ftype, params = self._optimized_volumes[0]
            else:
                ftype = "none"
                params = {}

            # Build kwargs for single apply_filter() call
            kwargs = {
                "positions": positions,
                "opacities": opacities,
                "scales": scales,
                "filter_type": ftype,
                "opacity_threshold": self._optimized_opacity,
                "max_scale": self._optimized_scale,
                "scene_bounds": self._scene_bounds,
            }

            # Add volume filter parameters if present
            if ftype == "sphere":
                center = params.get("center")
                if center is None and self._scene_bounds:
                    center = self._scene_bounds.center
                if center is not None:
                    kwargs["sphere_center"] = center
                kwargs["sphere_radius_factor"] = params["radius_factor"]

            elif ftype == "cuboid":
                center = params.get("center")
                if center is None and self._scene_bounds:
                    center = self._scene_bounds.center
                if center is not None:
                    kwargs["cuboid_center"] = center
                size_factors = params.get("size_factors")
                if size_factors is not None:
                    if isinstance(size_factors, (tuple, list, np.ndarray)):
                        kwargs["cuboid_size_factor_x"] = size_factors[0]
                        kwargs["cuboid_size_factor_y"] = size_factors[1]
                        kwargs["cuboid_size_factor_z"] = size_factors[2]

            # Single call to apply_filter() - FAST PATH
            mask = _apply_filter(**kwargs)
            logger.debug(
                "[Filter] Fast path: Applied all filters in single call (type=%s, opacity=%s, scale=%s)",
                ftype,
                self._optimized_opacity,
                self._optimized_scale,
            )

        else:
            # Slow path: Multiple volume filters require separate application
            mask = np.ones(len(positions), dtype=bool)

            # Apply threshold filters first
            if self._optimized_opacity > DEFAULT_OPACITY_THRESHOLD and opacities is not None:
                threshold_mask = _apply_filter(
                    positions=positions,
                    opacities=opacities,
                    scales=None,
                    filter_type="none",
                    opacity_threshold=self._optimized_opacity,
                    max_scale=None,
                    scene_bounds=None,
                )
                mask &= threshold_mask

            if self._optimized_scale < DEFAULT_MAX_SCALE and scales is not None:
                scale_mask = _apply_filter(
                    positions=positions,
                    opacities=None,
                    scales=scales,
                    filter_type="none",
                    opacity_threshold=0.0,
                    max_scale=self._optimized_scale,
                    scene_bounds=None,
                )
                mask &= scale_mask

            # Apply each volume filter
            for ftype, params in self._optimized_volumes:
                kwargs = {
                    "positions": positions,
                    "opacities": None,
                    "scales": None,
                    "filter_type": ftype,
                    "opacity_threshold": 0.0,
                    "max_scale": None,
                    "scene_bounds": self._scene_bounds,
                }

                if ftype == "sphere":
                    center = params.get("center")
                    if center is None and self._scene_bounds:
                        center = self._scene_bounds.center
                    if center is not None:
                        kwargs["sphere_center"] = center
                    kwargs["sphere_radius_factor"] = params["radius_factor"]

                elif ftype == "cuboid":
                    center = params.get("center")
                    if center is None and self._scene_bounds:
                        center = self._scene_bounds.center
                    if center is not None:
                        kwargs["cuboid_center"] = center
                    size_factors = params.get("size_factors")
                    if size_factors is not None:
                        if isinstance(size_factors, (tuple, list, np.ndarray)):
                            kwargs["cuboid_size_factor_x"] = size_factors[0]
                            kwargs["cuboid_size_factor_y"] = size_factors[1]
                            kwargs["cuboid_size_factor_z"] = size_factors[2]

                volume_mask = apply_filter(**kwargs)
                mask &= volume_mask

            logger.debug(
                "[Filter] Slow path: Applied %d volume filters separately",
                len(self._optimized_volumes),
            )

        # Apply mask to filter data using optimized Numba kernel
        # Note: This method always creates new filtered arrays since filtering changes array sizes
        # The "inplace" behavior is handled at the GSData level in the apply() method

        # Import the fused parallel kernel
        from gspro.filter.kernels import (
            compute_output_indices_and_count,
            filter_gaussians_fused_parallel,
        )

        # Pre-compute output indices and count in single pass
        out_indices = np.empty(len(mask), dtype=np.int32)
        num_kept = compute_output_indices_and_count(mask, out_indices)

        # Allocate output arrays
        filtered_positions = np.empty((num_kept, 3), dtype=positions.dtype)
        filtered_quaternions = (
            np.empty((num_kept, 4), dtype=quaternions.dtype) if quaternions is not None else None
        )
        filtered_scales = (
            np.empty((num_kept, 3), dtype=scales.dtype) if scales is not None else None
        )
        filtered_opacities = (
            np.empty(num_kept, dtype=opacities.dtype) if opacities is not None else None
        )
        filtered_colors = (
            np.empty((num_kept, colors.shape[1]), dtype=colors.dtype)
            if colors is not None and colors.ndim == 2
            else np.empty(num_kept, dtype=colors.dtype)
            if colors is not None
            else None
        )
        filtered_shN = (
            np.empty((num_kept, shN.shape[1], 3), dtype=shN.dtype) if shN is not None else None
        )

        # Use fused parallel Numba kernel to copy data (5-20x faster than boolean indexing)
        filter_gaussians_fused_parallel(
            mask,
            out_indices,
            positions,
            quaternions,
            scales,
            opacities,
            colors,
            shN,
            filtered_positions,
            filtered_quaternions,
            filtered_scales,
            filtered_opacities,
            filtered_colors,
            filtered_shN,
        )

        num_total = len(mask)
        logger.info(
            "[Filter] Kept %d/%d Gaussians (%.1f%%)",
            num_kept,
            num_total,
            100 * num_kept / num_total,
        )

        return (
            filtered_positions,
            filtered_quaternions,
            filtered_scales,
            filtered_opacities,
            filtered_colors,
            filtered_shN,
        )

    def get_mask(self, data: GSData) -> np.ndarray:
        """
        Compute boolean mask for filtering without copying data.

        This is much faster than apply() when you only need to inspect
        which Gaussians pass the filter, or want to combine multiple masks.

        Args:
            data: GSData object containing Gaussian data

        Returns:
            Boolean array of shape (N,) where True = keep Gaussian

        Example:
            >>> # Create mask only (no data copying)
            >>> mask = Filter().within_sphere(radius=0.8).min_opacity(0.1).get_mask(data)
            >>> print(f"Keeping {mask.sum()}/{len(mask)} Gaussians ({mask.sum()/len(mask)*100:.1f}%)")
            >>>
            >>> # Apply mask using GSData's built-in slicing
            >>> filtered = data[mask]  # or data.copy_slice(mask)
            >>>
            >>> # Combine multiple masks
            >>> mask1 = Filter().within_sphere(radius=0.8).get_mask(data)
            >>> mask2 = Filter().min_opacity(0.1).get_mask(data)
            >>> combined_mask = mask1 & mask2
            >>> filtered = data[combined_mask]
        """
        # Compile filter if needed
        if not self._is_compiled:
            self.compile()

        # Import here to avoid circular dependency
        from gspro.filter.api import _apply_filter  # Internal use only

        # Start with all Gaussians included
        n_gaussians = len(data.means)
        mask = np.ones(n_gaussians, dtype=bool)

        # Apply opacity + scale filters (if any)
        if self._optimized_opacity > 0.0 or self._optimized_scale < 10.0:
            opacity_scale_mask = _apply_filter(
                positions=data.means,
                opacities=data.opacities,
                scales=data.scales,
                filter_type="none",  # No spatial filtering in this pass
                opacity_threshold=self._optimized_opacity,
                max_scale=self._optimized_scale,
                scene_bounds=self._scene_bounds,
            )
            mask = mask & opacity_scale_mask

        # Apply volume filters (if any)
        if self._optimized_volumes:
            for filter_type, params in self._optimized_volumes:
                # Build kwargs with proper parameter mapping
                kwargs = {
                    "positions": data.means,
                    "filter_type": filter_type,
                    "scene_bounds": self._scene_bounds,
                }

                # Map parameters to apply_filter's expected names
                if filter_type == "sphere":
                    center = params.get("center")
                    if center is None and self._scene_bounds:
                        center = self._scene_bounds.center
                    if center is not None:
                        kwargs["sphere_center"] = center
                    kwargs["sphere_radius_factor"] = params["radius_factor"]

                elif filter_type == "cuboid":
                    center = params.get("center")
                    if center is None and self._scene_bounds:
                        center = self._scene_bounds.center
                    if center is not None:
                        kwargs["cuboid_center"] = center
                    size_factors = params.get("size_factors")
                    if size_factors is not None:
                        if isinstance(size_factors, (tuple, list, np.ndarray)):
                            kwargs["cuboid_size_factor_x"] = size_factors[0]
                            kwargs["cuboid_size_factor_y"] = size_factors[1]
                            kwargs["cuboid_size_factor_z"] = size_factors[2]

                volume_mask = _apply_filter(**kwargs)
                mask = mask & volume_mask  # AND logic

        logger.debug(
            "[Filter] Mask computed: %d/%d Gaussians pass (%.1f%%)",
            mask.sum(),
            len(mask),
            mask.sum() / len(mask) * 100 if len(mask) > 0 else 0,
        )

        return mask

    def is_empty(self) -> bool:
        """
        Check if this pipeline applies no filtering (empty filter).

        Returns:
            True if pipeline has no filter operations, False otherwise

        Note:
            Empty pipelines have zero overhead with fast-path optimization
            (measured < 0.001ms vs 1.8ms without optimization)
        """
        return (
            not self._volume_filters
            and not self._opacity_thresholds
            and not self._scale_thresholds
        )

    def apply(self, data: GSData, inplace: bool = True) -> GSData:
        """
        Apply the filter pipeline to GSData object.

        Internally uses get_mask() to create a boolean mask, then applies it
        using GSData's built-in slicing. For mask-only operations, use get_mask()
        directly for better performance.

        Args:
            data: GSData object containing Gaussian data
            inplace: If True, modifies input GSData arrays directly (replaces with filtered data)

        Returns:
            Filtered GSData object

        Example:
            >>> import gsply
            >>> from gspro import Filter
            >>> data = gsply.plyread("scene.ply")
            >>> filtered_data = Filter().within_sphere(radius=0.8).apply(data)
            >>>
            >>> # Or use __call__:
            >>> filtered_data = Filter().within_sphere(radius=0.8)(data)
            >>>
            >>> # For mask-only (no data copying):
            >>> mask = Filter().within_sphere(radius=0.8).get_mask(data)
            >>> print(f"{mask.sum()} Gaussians pass filter")
            >>> filtered_data = data[mask]  # Apply mask when ready
        """
        # Fast-path: empty filter (no operations)
        # Measured: 0.001ms vs 1.8ms overhead for empty filter
        if self.is_empty():
            return data if inplace else data.copy()

        # Get boolean mask (fast - no data copying)
        mask = self.get_mask(data)

        # Apply mask using GSData's built-in slicing (creates filtered copy)
        filtered_data = data.copy_slice(mask)

        # If inplace, replace input data's arrays with filtered ones
        if inplace:
            data.means = filtered_data.means
            data.quats = filtered_data.quats
            data.scales = filtered_data.scales
            data.opacities = filtered_data.opacities
            data.sh0 = filtered_data.sh0
            data.shN = filtered_data.shN

            logger.info("[Filter] Filtered to %d/%d Gaussians (in-place)", len(data), mask.sum())
            return data

        # If not inplace, return the filtered copy

        logger.info(
            "[Filter] Filtered: %d -> %d Gaussians (new copy)", len(data), len(filtered_data)
        )
        return filtered_data

    def __call__(
        self, data: GSData, inplace: bool = True, params: dict[str, float] | None = None
    ) -> GSData:
        """
        Apply the pipeline when called as a function.

        This is the primary way to use the pipeline - clean and Pythonic.

        Args:
            data: GSData object containing Gaussian data
            inplace: If True, modifies input GSData directly
            params: Optional runtime parameters for template pipelines

        Returns:
            Filtered GSData object

        Example (standard):
            >>> data = Filter().within_sphere(radius=0.8).min_opacity(0.1)(data)

        Example (parameterized template):
            >>> template = Filter.template(
            ...     sphere_radius=Param("r", default=0.8, range=(0.1, 1.0))
            ... )
            >>> data = template(data, params={"r": 0.6})  # Cached for performance
        """
        if params is not None:
            # Parameterized template path - use caching
            if not self._param_map:
                raise ValueError(
                    "Pipeline was not created with template(). "
                    "Use Filter() for non-parameterized pipelines."
                )
            return self._apply_with_params(data, params, inplace)

        # Standard pipeline path
        return self.apply(data, inplace=inplace)

    def reset(self) -> Self:
        """
        Reset the filter pipeline to default state.

        Returns:
            Self for method chaining
        """
        # Clear operation lists
        self._volume_filters = []
        self._opacity_thresholds = []
        self._scale_thresholds = []
        self._scene_bounds = None

        # Reset optimized values
        self._optimized_opacity = DEFAULT_OPACITY_THRESHOLD
        self._optimized_scale = DEFAULT_MAX_SCALE
        self._optimized_volumes = []
        self._is_compiled = False

        # Clear parameterization
        self._param_map = {}
        self._filter_cache = {}

        logger.debug("[Filter] Pipeline reset")
        return self

    def copy(self) -> Self:
        """
        Create a deep copy of the pipeline.

        Returns:
            New Filter instance with copied state
        """
        return deepcopy(self)

    def __copy__(self) -> Self:
        """Shallow copy (creates deep copy for safety)."""
        return self.copy()

    def __deepcopy__(self, memo: dict[int, Any]) -> Self:
        """Deep copy implementation."""
        # Create new instance
        new_obj = self.__class__()

        # Copy operation lists (deep copy)
        new_obj._volume_filters = deepcopy(self._volume_filters, memo)
        new_obj._opacity_thresholds = self._opacity_thresholds.copy()
        new_obj._scale_thresholds = self._scale_thresholds.copy()
        new_obj._scene_bounds = deepcopy(self._scene_bounds, memo)

        # Copy optimized values
        new_obj._optimized_opacity = self._optimized_opacity
        new_obj._optimized_scale = self._optimized_scale
        new_obj._optimized_volumes = deepcopy(self._optimized_volumes, memo)
        new_obj._is_compiled = self._is_compiled

        # Copy parameterization support
        new_obj._param_map = self._param_map.copy()  # Shallow copy (Params are frozen)
        new_obj._filter_cache = deepcopy(self._filter_cache, memo)  # Deep copy cached values
        new_obj._param_order = self._param_order  # Shallow copy (tuple is immutable)

        return new_obj

    def get_params(self) -> dict[str, Any]:
        """
        Get current filter parameters (optimized values if compiled).

        Returns:
            Dictionary of parameter names to values
        """
        if self._is_compiled:
            # Return optimized values
            params = {
                "is_compiled": True,
                "opacity_threshold": self._optimized_opacity,
                "max_scale": self._optimized_scale,
                "volume_filters": self._optimized_volumes,
                "num_opacity_ops": len(self._opacity_thresholds),
                "num_scale_ops": len(self._scale_thresholds),
                "num_volume_ops": len(self._volume_filters),
            }
        else:
            # Return raw operation lists
            params = {
                "is_compiled": False,
                "opacity_thresholds": self._opacity_thresholds,
                "scale_thresholds": self._scale_thresholds,
                "volume_filters": self._volume_filters,
            }

        return params

    @property
    def has_volume_filter(self) -> bool:
        """Check if a volume filter is active."""
        return len(self._volume_filters) > 0

    @property
    def has_opacity_filter(self) -> bool:
        """Check if opacity filter is active."""
        return len(self._opacity_thresholds) > 0

    @property
    def has_scale_filter(self) -> bool:
        """Check if scale filter is active."""
        return len(self._scale_thresholds) > 0

    @property
    def num_active_filters(self) -> int:
        """Number of active filter operations (before optimization)."""
        return (
            len(self._volume_filters) + len(self._opacity_thresholds) + len(self._scale_thresholds)
        )

    def __repr__(self) -> str:
        """String representation of the pipeline."""
        if not self._is_compiled:
            # Show operation counts before compilation
            parts = []
            if self._opacity_thresholds:
                parts.append(f"opacity x{len(self._opacity_thresholds)}")
            if self._scale_thresholds:
                parts.append(f"scale x{len(self._scale_thresholds)}")
            if self._volume_filters:
                parts.append(f"volume x{len(self._volume_filters)}")

            if parts:
                return f"Filter({', '.join(parts)}, not compiled)"
            return "Filter(no filters)"
        else:
            # Show optimized values after compilation
            parts = []
            if self._optimized_opacity > DEFAULT_OPACITY_THRESHOLD:
                parts.append(f"opacity>{self._optimized_opacity:.3f}")
            if self._optimized_scale < DEFAULT_MAX_SCALE:
                parts.append(f"scale<{self._optimized_scale:.1f}")
            if self._optimized_volumes:
                parts.append(f"volume x{len(self._optimized_volumes)}")

            if parts:
                return f"Filter({', '.join(parts)}, compiled)"
            return "Filter(compiled, no filters)"

    def __len__(self) -> int:
        """Number of active filters."""
        return self.num_active_filters
