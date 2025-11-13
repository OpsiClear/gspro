"""
Gaussian splat filtering API.

Provides volume, opacity, and scale filtering for Gaussian splats.
Based on Universal 4D Viewer filtering system.

Interface follows the transform module pattern: function-based with kwargs.
CPU-optimized using NumPy and Numba for maximum performance.
"""

import logging

import numpy as np

from gspro.filter.bounds import SceneBounds, calculate_scene_bounds
from gspro.filter.config import FilterConfig

# Import Numba kernels for optimization
from gspro.filter.kernels import (
    combine_masks_numba,
    cuboid_filter_numba,
    opacity_filter_numba,
    scale_filter_numba,
    sphere_filter_numba,
)

logger = logging.getLogger(__name__)


def apply_filter(
    positions: np.ndarray,
    opacities: np.ndarray | None = None,
    scales: np.ndarray | None = None,
    # Filter parameters (kwargs like transform module)
    filter_type: str = "none",
    sphere_center: tuple[float, float, float] = (0.0, 0.0, 0.0),
    sphere_radius_factor: float = 1.0,
    cuboid_center: tuple[float, float, float] = (0.0, 0.0, 0.0),
    cuboid_size_factor_x: float = 1.0,
    cuboid_size_factor_y: float = 1.0,
    cuboid_size_factor_z: float = 1.0,
    opacity_threshold: float = 0.05,
    max_scale: float = 10.0,
    scene_bounds: SceneBounds | None = None,
    config: FilterConfig | None = None,
) -> np.ndarray:
    """
    Apply volume, opacity, and scale filtering to Gaussian splats.

    Creates a boolean mask indicating which Gaussians pass all filter criteria.
    All filters use AND logic (all conditions must be met).

    Interface follows transform module pattern with kwargs.

    Args:
        positions: Gaussian positions [N, 3] in format [x, y, z]
        opacities: Gaussian opacities [N] in range [0, 1] (optional)
        scales: Gaussian scales [N, 3] in format [scale_x, scale_y, scale_z] (optional)
        filter_type: Spatial filter type ("none", "sphere", "cuboid")
        sphere_center: Center point for sphere filtering [x, y, z]
        sphere_radius_factor: Radius multiplier (0.0 to 1.0)
        cuboid_center: Center point for cuboid filtering [x, y, z]
        cuboid_size_factor_x: Width multiplier (0.0 to 1.0)
        cuboid_size_factor_y: Height multiplier (0.0 to 1.0)
        cuboid_size_factor_z: Depth multiplier (0.0 to 1.0)
        opacity_threshold: Minimum opacity to keep (0.0 to 1.0)
        max_scale: Maximum scale threshold
        scene_bounds: Pre-calculated scene bounds (auto-calculated if None)
        config: Optional FilterConfig object (overrides kwargs if provided)

    Returns:
        Boolean mask [N] where True = keep Gaussian

    Example:
        >>> # Simple opacity filtering (kwargs like transform module)
        >>> mask = apply_filter(positions, opacities, opacity_threshold=0.1)
        >>> filtered_positions = positions[mask]
        >>> print(f"Kept {mask.sum()} / {len(mask)} Gaussians")

        >>> # Sphere filtering
        >>> mask = apply_filter(
        ...     positions,
        ...     filter_type="sphere",
        ...     sphere_center=(0.0, 0.0, 0.0),
        ...     sphere_radius_factor=0.5
        ... )

        >>> # Combined filtering
        >>> mask = apply_filter(
        ...     positions, opacities, scales,
        ...     filter_type="cuboid",
        ...     cuboid_center=(0.0, 0.0, 0.0),
        ...     cuboid_size_factor_x=0.8,
        ...     cuboid_size_factor_y=0.8,
        ...     cuboid_size_factor_z=0.8,
        ...     opacity_threshold=0.05,
        ...     max_scale=2.5
        ... )

        >>> # Or use FilterConfig for convenience (backward compat)
        >>> config = FilterConfig(filter_type="sphere", sphere_radius_factor=0.5)
        >>> mask = apply_filter(positions, config=config)

    Note:
        - Performance: ~0.05ms per frame for typical scenes
        - All spatial filtering happens in world coordinates
        - Empty results (all False) are allowed - handle in calling code
        - scene_bounds can be pre-calculated and reused for multiple frames
    """
    # If config provided, use it (backward compatibility)
    if config is not None:
        filter_type = config.filter_type
        sphere_center = config.sphere_center
        sphere_radius_factor = config.sphere_radius_factor
        cuboid_center = config.cuboid_center
        cuboid_size_factor_x = config.cuboid_size_factor_x
        cuboid_size_factor_y = config.cuboid_size_factor_y
        cuboid_size_factor_z = config.cuboid_size_factor_z
        opacity_threshold = config.opacity_threshold
        max_scale = config.max_scale

    # Validate inputs
    positions = np.asarray(positions, dtype=np.float32)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(f"positions must be [N, 3], got shape {positions.shape}")

    n_gaussians = len(positions)
    if n_gaussians == 0:
        return np.array([], dtype=bool)

    # Start with all Gaussians included
    mask = np.ones(n_gaussians, dtype=bool)

    # === SPATIAL FILTERING ===
    if filter_type != "none":
        # Calculate scene bounds if not provided
        if scene_bounds is None:
            scene_bounds = calculate_scene_bounds(positions)

        if filter_type == "sphere":
            mask = mask & _apply_sphere_filter(
                positions,
                sphere_center,
                sphere_radius_factor,
                scene_bounds,
            )

        elif filter_type == "cuboid":
            mask = mask & _apply_cuboid_filter(
                positions,
                cuboid_center,
                cuboid_size_factor_x,
                cuboid_size_factor_y,
                cuboid_size_factor_z,
                scene_bounds,
            )

        else:
            raise ValueError(f"Unknown filter_type: {filter_type}")

    # === OPACITY + SCALE FILTERING (FUSED) ===
    # Use fused kernel to combine opacity and scale filtering in single pass
    # This eliminates multiple memory passes and combine_masks calls (20-30% faster)
    needs_opacity_filter = opacities is not None and opacity_threshold > 0.0
    needs_scale_filter = scales is not None and max_scale < 10.0

    if needs_opacity_filter or needs_scale_filter:
        # Validate inputs
        if needs_opacity_filter:
            opacities = np.asarray(opacities, dtype=np.float32)
            if opacities.shape != (n_gaussians,):
                raise ValueError(
                    f"opacities shape {opacities.shape} doesn't match "
                    f"positions shape {positions.shape}"
                )

        if needs_scale_filter:
            scales = np.asarray(scales, dtype=np.float32)
            if scales.ndim != 2 or scales.shape != (n_gaussians, 3):
                raise ValueError(
                    f"scales shape {scales.shape} doesn't match "
                    f"positions shape {positions.shape}"
                )

        # Import fused kernel
        from gspro.filter.kernels import opacity_scale_filter_fused

        # Apply fused opacity + scale filter in single pass
        out_mask = np.empty(n_gaussians, dtype=np.bool_)
        opacity_scale_filter_fused(
            mask,
            opacities if needs_opacity_filter else None,
            scales if needs_scale_filter else None,
            opacity_threshold,
            max_scale,
            out_mask,
        )
        mask = out_mask

        logger.debug(
            f"Fused filter: opacity_threshold={opacity_threshold}, "
            f"max_scale={max_scale}, kept={mask.sum()}/{n_gaussians}"
        )

    logger.info(f"Total filtered: {n_gaussians} -> {mask.sum()} Gaussians")

    return mask


def _apply_sphere_filter(
    positions: np.ndarray,
    sphere_center: tuple[float, float, float],
    sphere_radius_factor: float,
    scene_bounds: SceneBounds,
) -> np.ndarray:
    """
    Apply sphere volume filter with Numba optimization.

    Args:
        positions: Gaussian positions [N, 3]
        sphere_center: Center point
        sphere_radius_factor: Radius multiplier
        scene_bounds: Scene spatial bounds

    Returns:
        Boolean mask [N] where True = inside sphere
    """
    # Calculate actual radius from scene size
    base_radius = scene_bounds.max_size * 0.5
    actual_radius = base_radius * sphere_radius_factor
    radius_sq = actual_radius**2

    # Convert center to array
    center = np.array(sphere_center, dtype=np.float32)

    # Use Numba kernel for distance calculations
    mask = np.empty(len(positions), dtype=np.bool_)
    sphere_filter_numba(positions, center, radius_sq, mask)

    logger.debug(
        f"Sphere filter: kept {mask.sum()}/{len(mask)} "
        f"(center={center}, radius={actual_radius:.3f})"
    )

    return mask


def _apply_cuboid_filter(
    positions: np.ndarray,
    cuboid_center: tuple[float, float, float],
    cuboid_size_factor_x: float,
    cuboid_size_factor_y: float,
    cuboid_size_factor_z: float,
    scene_bounds: SceneBounds,
) -> np.ndarray:
    """
    Apply cuboid (box) volume filter with Numba optimization.

    Args:
        positions: Gaussian positions [N, 3]
        cuboid_center: Center point
        cuboid_size_factor_x: Width multiplier
        cuboid_size_factor_y: Height multiplier
        cuboid_size_factor_z: Depth multiplier
        scene_bounds: Scene spatial bounds

    Returns:
        Boolean mask [N] where True = inside cuboid
    """
    # Calculate actual box half-sizes from scene dimensions
    actual_half_sizes = np.array(
        [
            scene_bounds.sizes[0] * cuboid_size_factor_x * 0.5,
            scene_bounds.sizes[1] * cuboid_size_factor_y * 0.5,
            scene_bounds.sizes[2] * cuboid_size_factor_z * 0.5,
        ],
        dtype=np.float32,
    )

    # Convert center to array
    center = np.array(cuboid_center, dtype=np.float32)

    # Calculate box bounds
    min_bounds = center - actual_half_sizes
    max_bounds = center + actual_half_sizes

    # Use Numba kernel for bound checking
    mask = np.empty(len(positions), dtype=np.bool_)
    cuboid_filter_numba(positions, min_bounds, max_bounds, mask)

    logger.debug(
        f"Cuboid filter: kept {mask.sum()}/{len(mask)} "
        f"(center={center}, half_sizes={actual_half_sizes})"
    )

    return mask


def filter_gaussians(
    positions: np.ndarray,
    quaternions: np.ndarray | None = None,
    scales: np.ndarray | None = None,
    opacities: np.ndarray | None = None,
    colors: np.ndarray | None = None,
    # Filter parameters (kwargs like transform module)
    filter_type: str = "none",
    sphere_center: tuple[float, float, float] = (0.0, 0.0, 0.0),
    sphere_radius_factor: float = 1.0,
    cuboid_center: tuple[float, float, float] = (0.0, 0.0, 0.0),
    cuboid_size_factor_x: float = 1.0,
    cuboid_size_factor_y: float = 1.0,
    cuboid_size_factor_z: float = 1.0,
    opacity_threshold: float = 0.05,
    max_scale: float = 10.0,
    scene_bounds: SceneBounds | None = None,
    config: FilterConfig | None = None,
) -> tuple[
    np.ndarray,  # positions
    np.ndarray | None,  # quaternions
    np.ndarray | None,  # scales
    np.ndarray | None,  # opacities
    np.ndarray | None,  # colors
]:
    """
    Apply filtering and return filtered Gaussian attributes.

    Convenience function that applies filtering mask to all provided attributes.
    Returns tuple (like transform module) instead of dict.

    Args:
        positions: Gaussian positions [N, 3]
        quaternions: Gaussian rotations [N, 4] (optional)
        scales: Gaussian scales [N, 3] (optional)
        opacities: Gaussian opacities [N] (optional)
        colors: Gaussian colors [N, 3] or [N, C] (optional)
        filter_type: Spatial filter type ("none", "sphere", "cuboid")
        sphere_center: Center point for sphere filtering
        sphere_radius_factor: Radius multiplier (0.0 to 1.0)
        cuboid_center: Center point for cuboid filtering
        cuboid_size_factor_x: Width multiplier (0.0 to 1.0)
        cuboid_size_factor_y: Height multiplier (0.0 to 1.0)
        cuboid_size_factor_z: Depth multiplier (0.0 to 1.0)
        opacity_threshold: Minimum opacity to keep (0.0 to 1.0)
        max_scale: Maximum scale threshold
        scene_bounds: Pre-calculated scene bounds (auto-calculated if None)
        config: Optional FilterConfig object (overrides kwargs if provided)

    Returns:
        Tuple of (positions, quaternions, scales, opacities, colors)
        where each is the filtered array or None if not provided.
        Like transform module, returns tuple not dict.

    Example:
        >>> # Filter with kwargs (like transform module)
        >>> new_pos, new_quats, new_scales, new_opac, new_colors = filter_gaussians(
        ...     positions, quaternions, scales, opacities, colors,
        ...     filter_type="sphere",
        ...     sphere_radius_factor=0.8,
        ...     opacity_threshold=0.05
        ... )

        >>> # Or with FilterConfig for convenience
        >>> config = FilterConfig(filter_type="sphere", sphere_radius_factor=0.8)
        >>> new_pos, new_quats, new_scales, new_opac, new_colors = filter_gaussians(
        ...     positions, quaternions, scales, opacities, colors,
        ...     config=config
        ... )

        >>> # Unpack only what you need
        >>> new_pos, new_quats, *_ = filter_gaussians(positions, quaternions, ...)
    """
    # Apply filtering to get mask
    mask = apply_filter(
        positions=positions,
        opacities=opacities,
        scales=scales,
        filter_type=filter_type,
        sphere_center=sphere_center,
        sphere_radius_factor=sphere_radius_factor,
        cuboid_center=cuboid_center,
        cuboid_size_factor_x=cuboid_size_factor_x,
        cuboid_size_factor_y=cuboid_size_factor_y,
        cuboid_size_factor_z=cuboid_size_factor_z,
        opacity_threshold=opacity_threshold,
        max_scale=max_scale,
        scene_bounds=scene_bounds,
        config=config,
    )

    # Optimize masking: parallel fused kernel with prefix sum
    # Phase 1: Count and compute indices (two simple passes, well-optimized)
    # Phase 2: Parallel scatter - each thread copies independently
    # 5-10x faster than NumPy's separate masking operations
    from gspro.filter.kernels import (
        compute_output_indices_and_count,
        filter_gaussians_fused_parallel,
    )

    # Allocate indices array
    out_indices = np.empty(len(mask), dtype=np.int32)

    # Compute indices and count in single fused operation
    n_kept = compute_output_indices_and_count(mask, out_indices)

    # Pre-allocate all output arrays
    out_positions = np.empty((n_kept, 3), dtype=positions.dtype)

    out_quaternions = None
    if quaternions is not None:
        out_quaternions = np.empty((n_kept, 4), dtype=quaternions.dtype)

    out_scales = None
    if scales is not None:
        out_scales = np.empty((n_kept, 3), dtype=scales.dtype)

    out_opacities = None
    if opacities is not None:
        out_opacities = np.empty(n_kept, dtype=opacities.dtype)

    out_colors = None
    if colors is not None:
        if colors.ndim == 2:
            out_colors = np.empty((n_kept, colors.shape[1]), dtype=colors.dtype)
        else:
            out_colors = np.empty(n_kept, dtype=colors.dtype)

    # Parallel fused kernel: scatter with pre-computed indices
    filter_gaussians_fused_parallel(
        mask,
        out_indices,
        positions,
        quaternions,
        scales,
        opacities,
        colors,
        out_positions,
        out_quaternions,
        out_scales,
        out_opacities,
        out_colors,
    )

    return (
        out_positions,
        out_quaternions,
        out_scales,
        out_opacities,
        out_colors,
    )
