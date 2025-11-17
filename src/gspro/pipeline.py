"""
Unified Pipeline for composing Color, Transform, and Filter operations.

This module provides a single composable pipeline that can chain all gspro
operations together for clean, fluent code.

Example:
    >>> import gsply
    >>> from gspro import Pipeline
    >>>
    >>> data = gsply.plyread("scene.ply")
    >>>
    >>> # Compose all operations in one pipeline
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
    >>> # Execute pipeline
    >>> result = pipeline(data, inplace=True)
    >>> gsply.plywrite("output.ply", result)
"""

from __future__ import annotations

import logging
from copy import deepcopy
from typing import Any, Self

from gsply import GSData

from gspro.color.pipeline import Color
from gspro.filter.pipeline import Filter
from gspro.transform.pipeline import Transform

logger = logging.getLogger(__name__)


class Pipeline:
    """
    Unified pipeline for composing Color, Transform, and Filter operations.

    This class provides a fluent API for chaining multiple operations together
    and applying them to GSData in a single pass.

    All operations work with GSData only - no array-based interface.

    Example:
        >>> pipeline = (
        ...     Pipeline()
        ...     .within_sphere(radius=0.8)
        ...     .min_opacity(0.1)
        ...     .rotate_quat(quat)
        ...     .translate([1, 0, 0])
        ...     .scale(2.0)
        ...     .brightness(1.2)
        ...     .contrast(1.1)
        ...     .saturation(1.3)
        ... )
        >>> result = pipeline(data, inplace=True)
    """

    __slots__ = ("_color_pipeline", "_transform_pipeline", "_filter_pipeline", "_execution_order")

    # Method names for each pipeline type (for __getattr__ delegation)
    _COLOR_METHODS = frozenset(
        {
            "temperature",
            "brightness",
            "contrast",
            "gamma",
            "saturation",
            "vibrance",
            "hue_shift",
            "shadows",
            "highlights",
            "compile",  # Color-specific
        }
    )

    _TRANSFORM_METHODS = frozenset(
        {
            "translate",
            "rotate_quat",
            "rotate_euler",
            "rotate_axis_angle",
            "rotate_matrix",
            "scale",
            "set_center",
        }
    )

    _FILTER_METHODS = frozenset(
        {"within_sphere", "within_box", "min_opacity", "max_scale", "bounds"}
    )  # Note: "max_scale" is for filtering, "scale" in Transform is for scaling

    # Default execution order (optimized for performance)
    _DEFAULT_ORDER = ("filter", "transform", "color")

    def __init__(self, execution_order: tuple[str, str, str] | None = None):
        """
        Initialize the unified pipeline.

        Args:
            execution_order: Custom execution order for operations.
                Default is ("filter", "transform", "color") which is optimized for performance.
                Can be any permutation of these three strings.

        Raises:
            ValueError: If execution_order contains invalid operation names or duplicates

        Performance Note:
            The default order ("filter", "transform", "color") is optimal because:
            1. Filter first reduces data size early, minimizing work for subsequent operations
            2. Transform second modifies geometry before color adjustments
            3. Color last operates on final geometry for accurate appearance

        Example:
            >>> # Default order (recommended)
            >>> pipeline = Pipeline()
            >>>
            >>> # Custom order (advanced use cases)
            >>> pipeline = Pipeline(execution_order=("color", "filter", "transform"))
        """
        self._color_pipeline = Color()
        self._transform_pipeline = Transform()
        self._filter_pipeline = Filter()

        # Validate and set execution order
        if execution_order is None:
            self._execution_order = self._DEFAULT_ORDER
        else:
            # Validate execution order
            valid_ops = {"filter", "transform", "color"}
            if set(execution_order) != valid_ops:
                missing = valid_ops - set(execution_order)
                extra = set(execution_order) - valid_ops
                error_parts = []
                if missing:
                    error_parts.append(f"missing: {', '.join(sorted(missing))}")
                if extra:
                    error_parts.append(f"invalid: {', '.join(sorted(extra))}")

                raise ValueError(
                    f"execution_order must contain exactly 'filter', 'transform', and 'color'. "
                    f"{'; '.join(error_parts)}. "
                    f"Example: Pipeline(execution_order=('filter', 'transform', 'color'))"
                )
            if len(execution_order) != 3:
                raise ValueError(
                    f"execution_order must contain exactly 3 operations, got {len(execution_order)}. "
                    f"Provide a tuple with 'filter', 'transform', and 'color' in your desired order."
                )
            self._execution_order = tuple(execution_order)

        logger.info("[Pipeline] Initialized with order: %s", self._execution_order)

    # ========================================================================
    # Properties (Computed from Sub-Pipelines)
    # ========================================================================

    @property
    def has_color(self) -> bool:
        """Check if color pipeline has operations."""
        return len(self._color_pipeline) > 0

    @property
    def has_transform(self) -> bool:
        """Check if transform pipeline has operations."""
        return len(self._transform_pipeline) > 0

    @property
    def has_filter(self) -> bool:
        """Check if filter pipeline has operations."""
        return len(self._filter_pipeline) > 0

    # ========================================================================
    # Dynamic Method Delegation via __getattr__
    # ========================================================================

    def __getattr__(self, name: str) -> Any:
        """
        Dynamically delegate method calls to appropriate sub-pipeline.

        This eliminates the need for explicit wrapper methods for each operation.
        Methods are delegated based on method name registration.

        Args:
            name: Method name being accessed

        Returns:
            Wrapper function that delegates to sub-pipeline

        Raises:
            AttributeError: If method name is not registered
        """
        # Color methods
        if name in self._COLOR_METHODS:

            def wrapper(*args, **kwargs):
                getattr(self._color_pipeline, name)(*args, **kwargs)
                return self

            return wrapper

        # Transform methods
        if name in self._TRANSFORM_METHODS:

            def wrapper(*args, **kwargs):
                getattr(self._transform_pipeline, name)(*args, **kwargs)
                return self

            return wrapper

        # Filter methods
        if name in self._FILTER_METHODS:

            def wrapper(*args, **kwargs):
                getattr(self._filter_pipeline, name)(*args, **kwargs)
                return self

            return wrapper

        # Not found - raise AttributeError
        raise AttributeError(
            f"'{type(self).__name__}' has no attribute '{name}'. "
            f"Available methods: {sorted(self._COLOR_METHODS | self._TRANSFORM_METHODS | self._FILTER_METHODS)}"
        )

    # ========================================================================
    # Execution
    # ========================================================================

    def apply(self, data: GSData, inplace: bool = True) -> GSData:
        """
        Apply the full pipeline to GSData.

        Operations are applied in the order specified during initialization.
        Default order: Filter -> Transform -> Color (optimized for performance).

        Why the default order is optimal:
        1. Filter first: Reduces Gaussian count early, minimizing work for subsequent operations
           (e.g., filtering 1M -> 100K before transform/color saves 90% of computation)
        2. Transform second: Modifies geometry before appearance adjustments
        3. Color last: Operates on final geometry for accurate visual results

        Args:
            data: GSData object containing Gaussian data
            inplace: If True, modifies input GSData directly

        Returns:
            Processed GSData object

        Example:
            >>> pipeline = Pipeline().brightness(1.2).rotate_quat(quat).within_sphere(radius=0.8)
            >>> result = pipeline.apply(data, inplace=True)
        """
        result = data

        # Track whether we've created a copy yet
        first_operation_inplace = inplace

        # Map operation names to pipeline objects and has_* flags
        pipeline_map = {
            "filter": (self._filter_pipeline, self.has_filter),
            "transform": (self._transform_pipeline, self.has_transform),
            "color": (self._color_pipeline, self.has_color),
        }

        # Apply operations in configured order
        # OPTIMIZATION #4: Removed has_operations check (1-2% speedup)
        # Sub-pipelines have identity fast-paths (<0.001ms), so always calling apply() is faster
        for op_name in self._execution_order:
            pipeline_obj, _ = pipeline_map[op_name]  # Ignore has_operations flag
            logger.debug("[Pipeline] Applying %s operations", op_name)
            result = pipeline_obj.apply(result, inplace=first_operation_inplace)
            first_operation_inplace = True  # Subsequent operations can be inplace

        logger.info("[Pipeline] Completed processing of %d Gaussians", len(result))
        return result

    def __call__(self, data: GSData, inplace: bool = True) -> GSData:
        """
        Apply the pipeline when called as a function.

        This is the primary way to use the pipeline - clean and Pythonic.

        Args:
            data: GSData object containing Gaussian data
            inplace: If True, modifies input GSData directly

        Returns:
            Processed GSData object

        Example:
            >>> pipeline = Pipeline().brightness(1.2).within_sphere(radius=0.8)
            >>> result = pipeline(data, inplace=True)
        """
        return self.apply(data, inplace=inplace)

    def reset(self) -> Self:
        """
        Reset all operations in the pipeline.

        Returns:
            Self for chaining
        """
        self._color_pipeline.reset()
        self._transform_pipeline.reset()
        self._filter_pipeline.reset()

        logger.debug("[Pipeline] Reset")
        return self

    def copy(self) -> Self:
        """
        Create a deep copy of this pipeline.

        Returns:
            New Pipeline instance with same configuration

        Example:
            >>> pipeline = Pipeline().brightness(1.2).translate([1, 0, 0])
            >>> pipeline2 = pipeline.copy().saturation(1.5)  # Independent copy
        """
        return deepcopy(self)

    def __repr__(self) -> str:
        """String representation of the pipeline."""
        ops = []
        if self.has_filter:
            ops.append(f"filter({len(self._filter_pipeline)} ops)")
        if self.has_transform:
            ops.append(f"transform({len(self._transform_pipeline)} ops)")
        if self.has_color:
            ops.append("color")

        op_str = ", ".join(ops) if ops else "empty"
        return f"Pipeline({op_str})"

    def __len__(self) -> int:
        """Total number of operations in the pipeline."""
        return (
            len(self._filter_pipeline) + len(self._transform_pipeline) + len(self._color_pipeline)
        )

    def __copy__(self) -> Self:
        """Shallow copy delegates to deep copy."""
        return self.copy()

    def __deepcopy__(self, memo: dict[int, Any]) -> Self:
        """Create a deep copy of this pipeline."""
        new = Pipeline(execution_order=self._execution_order)
        new._color_pipeline = deepcopy(self._color_pipeline, memo)
        new._transform_pipeline = deepcopy(self._transform_pipeline, memo)
        new._filter_pipeline = deepcopy(self._filter_pipeline, memo)
        return new
