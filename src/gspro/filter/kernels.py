"""
Numba-optimized kernels for filtering operations.

Provides JIT-compiled kernels for performance-critical filtering operations.
"""

import numpy as np
from numba import njit, prange


@njit(parallel=True, fastmath=True, cache=True, nogil=True)
def sphere_filter_numba(
    positions: np.ndarray,
    center: np.ndarray,
    radius_sq: float,
    out: np.ndarray,
) -> None:
    """
    Apply sphere filter with Numba optimization.

    Args:
        positions: Gaussian positions [N, 3]
        center: Sphere center [3]
        radius_sq: Squared radius threshold
        out: Output mask [N] (modified in-place)
    """
    n = positions.shape[0]

    for i in prange(n):
        # Calculate squared distance from center
        dx = positions[i, 0] - center[0]
        dy = positions[i, 1] - center[1]
        dz = positions[i, 2] - center[2]
        dist_sq = dx * dx + dy * dy + dz * dz

        out[i] = dist_sq <= radius_sq


@njit(parallel=True, fastmath=True, cache=True, nogil=True)
def cuboid_filter_numba(
    positions: np.ndarray,
    min_bounds: np.ndarray,
    max_bounds: np.ndarray,
    out: np.ndarray,
) -> None:
    """
    Apply cuboid filter with Numba optimization.

    Args:
        positions: Gaussian positions [N, 3]
        min_bounds: Minimum bounds [3]
        max_bounds: Maximum bounds [3]
        out: Output mask [N] (modified in-place)
    """
    n = positions.shape[0]

    for i in prange(n):
        # Check if point is inside cuboid (all dimensions)
        inside = (
            positions[i, 0] >= min_bounds[0]
            and positions[i, 0] <= max_bounds[0]
            and positions[i, 1] >= min_bounds[1]
            and positions[i, 1] <= max_bounds[1]
            and positions[i, 2] >= min_bounds[2]
            and positions[i, 2] <= max_bounds[2]
        )

        out[i] = inside


@njit(parallel=True, fastmath=True, cache=True, nogil=True)
def scale_filter_numba(
    scales: np.ndarray,
    max_scale: float,
    out: np.ndarray,
) -> None:
    """
    Apply scale filter with Numba optimization.

    Args:
        scales: Gaussian scales [N, 3]
        max_scale: Maximum scale threshold
        out: Output mask [N] (modified in-place)
    """
    n = scales.shape[0]

    for i in prange(n):
        # Get maximum scale across x, y, z
        max_s = scales[i, 0]
        if scales[i, 1] > max_s:
            max_s = scales[i, 1]
        if scales[i, 2] > max_s:
            max_s = scales[i, 2]

        out[i] = max_s <= max_scale


@njit(parallel=True, fastmath=True, cache=True, nogil=True)
def opacity_filter_numba(
    opacities: np.ndarray,
    threshold: float,
    out: np.ndarray,
) -> None:
    """
    Apply opacity filter with Numba optimization.

    Args:
        opacities: Gaussian opacities [N]
        threshold: Minimum opacity threshold
        out: Output mask [N] (modified in-place)
    """
    n = opacities.shape[0]

    for i in prange(n):
        out[i] = opacities[i] >= threshold


@njit(parallel=True, fastmath=True, cache=True, nogil=True)
def combine_masks_numba(
    mask1: np.ndarray,
    mask2: np.ndarray,
    out: np.ndarray,
) -> None:
    """
    Combine two boolean masks with AND operation.

    Args:
        mask1: First mask [N]
        mask2: Second mask [N]
        out: Output mask [N] (modified in-place)
    """
    n = mask1.shape[0]

    for i in prange(n):
        out[i] = mask1[i] and mask2[i]


@njit(parallel=True, fastmath=True, cache=True, nogil=True)
def opacity_scale_filter_fused(
    mask: np.ndarray,
    opacities: np.ndarray | None,
    scales: np.ndarray | None,
    opacity_threshold: float,
    max_scale: float,
    out: np.ndarray,
) -> None:
    """
    Fused opacity and scale filtering in a single pass (20-30% faster).

    Combines opacity_filter + scale_filter + combine_masks into one kernel,
    eliminating multiple passes through data and kernel launches.

    Args:
        mask: Input spatial filter mask [N] (from sphere/cuboid, or all True)
        opacities: Gaussian opacities [N] (optional, None to skip)
        scales: Gaussian scales [N, 3] (optional, None to skip)
        opacity_threshold: Minimum opacity threshold
        max_scale: Maximum scale threshold
        out: Output mask [N] (modified in-place)
    """
    n = mask.shape[0]

    # Check which filters are active
    has_opacities = opacities is not None
    has_scales = scales is not None

    for i in prange(n):
        # Start with spatial filter result
        passed = mask[i]

        # Apply opacity filter if active
        if passed and has_opacities:
            if opacities[i] < opacity_threshold:
                passed = False

        # Apply scale filter if active
        if passed and has_scales:
            max_s = scales[i, 0]
            if scales[i, 1] > max_s:
                max_s = scales[i, 1]
            if scales[i, 2] > max_s:
                max_s = scales[i, 2]

            if max_s > max_scale:
                passed = False

        out[i] = passed


@njit(parallel=True, fastmath=True, cache=True, nogil=True)
def calculate_max_scales_numba(
    scales: np.ndarray,
    out: np.ndarray,
) -> None:
    """
    Calculate maximum scale per Gaussian with Numba optimization.

    Args:
        scales: Gaussian scales [N, 3]
        out: Output max scales [N] (modified in-place)
    """
    n = scales.shape[0]

    for i in prange(n):
        max_s = scales[i, 0]
        if scales[i, 1] > max_s:
            max_s = scales[i, 1]
        if scales[i, 2] > max_s:
            max_s = scales[i, 2]
        out[i] = max_s


@njit(cache=True, nogil=True)
def compute_output_indices_and_count(mask: np.ndarray, out_indices: np.ndarray) -> int:
    """
    Compute output indices and count in single pass (fused operation).

    Combines count_true_values + compute_output_indices to eliminate redundant pass.

    Args:
        mask: Boolean mask [N]
        out_indices: Output indices [N] (modified in-place)

    Returns:
        Number of True values in mask
    """
    n = mask.shape[0]
    count = 0

    # Single pass: compute indices and count simultaneously
    for i in range(n):
        if mask[i]:
            out_indices[i] = count
            count += 1
        else:
            out_indices[i] = -1

    return count


@njit(parallel=True, fastmath=True, cache=True, nogil=True)
def filter_gaussians_fused_parallel(
    mask: np.ndarray,
    out_indices: np.ndarray,
    positions: np.ndarray,
    quaternions: np.ndarray | None,
    scales: np.ndarray | None,
    opacities: np.ndarray | None,
    colors: np.ndarray | None,
    shN: np.ndarray | None,
    out_positions: np.ndarray,
    out_quaternions: np.ndarray | None,
    out_scales: np.ndarray | None,
    out_opacities: np.ndarray | None,
    out_colors: np.ndarray | None,
    out_shN: np.ndarray | None,
) -> None:
    """
    Parallel fused masking kernel for all Gaussian attributes.

    Optimizations:
    - fastmath=True for aggressive float optimizations
    - None checks hoisted outside loop
    - Parallel scatter with prange

    Args:
        mask: Boolean mask [N]
        out_indices: Pre-computed output indices [N]
        positions: Input positions [N, 3]
        quaternions: Input quaternions [N, 4] or None
        scales: Input scales [N, 3] or None
        opacities: Input opacities [N] or None
        colors: Input colors [N, C] or None
        shN: Input higher-order SH [N, K, 3] or None
        out_positions: Output positions [n_kept, 3]
        out_quaternions: Output quaternions [n_kept, 4] or None
        out_scales: Output scales [n_kept, 3] or None
        out_opacities: Output opacities [n_kept] or None
        out_colors: Output colors [n_kept, C] or None
        out_shN: Output higher-order SH [n_kept, K, 3] or None
    """
    n = mask.shape[0]

    # Hoist None checks outside loop
    has_quaternions = quaternions is not None and out_quaternions is not None
    has_scales = scales is not None and out_scales is not None
    has_opacities = opacities is not None and out_opacities is not None
    has_colors = colors is not None and out_colors is not None
    has_shN = shN is not None and out_shN is not None

    # Hoist color shape checks outside loop (avoid Numba typing issues with 1D arrays)
    colors_2d = False
    n_cols = 0
    if has_colors:
        if colors.ndim == 2:
            colors_2d = True
            n_cols = colors.shape[1]

    # Hoist shN shape checks outside loop
    n_sh_bands = 0
    if has_shN:
        n_sh_bands = shN.shape[1]  # K dimension

    # Parallel scatter with hoisted checks
    for i in prange(n):
        if mask[i]:
            out_idx = out_indices[i]

            # Copy positions (always present)
            out_positions[out_idx, 0] = positions[i, 0]
            out_positions[out_idx, 1] = positions[i, 1]
            out_positions[out_idx, 2] = positions[i, 2]

            # Copy quaternions (branch hoisted)
            if has_quaternions:
                out_quaternions[out_idx, 0] = quaternions[i, 0]
                out_quaternions[out_idx, 1] = quaternions[i, 1]
                out_quaternions[out_idx, 2] = quaternions[i, 2]
                out_quaternions[out_idx, 3] = quaternions[i, 3]

            # Copy scales (branch hoisted)
            if has_scales:
                out_scales[out_idx, 0] = scales[i, 0]
                out_scales[out_idx, 1] = scales[i, 1]
                out_scales[out_idx, 2] = scales[i, 2]

            # Copy opacities (branch hoisted)
            if has_opacities:
                out_opacities[out_idx] = opacities[i]

            # Copy colors (branch hoisted, shape access hoisted, unrolled for common cases)
            if has_colors:
                if colors_2d:
                    # Unrolled loop for common RGB case (11% faster than generic loop)
                    if n_cols == 3:
                        out_colors[out_idx, 0] = colors[i, 0]
                        out_colors[out_idx, 1] = colors[i, 1]
                        out_colors[out_idx, 2] = colors[i, 2]
                    elif n_cols == 4:
                        # RGBA case
                        out_colors[out_idx, 0] = colors[i, 0]
                        out_colors[out_idx, 1] = colors[i, 1]
                        out_colors[out_idx, 2] = colors[i, 2]
                        out_colors[out_idx, 3] = colors[i, 3]
                    else:
                        # Generic case for other channel counts
                        for j in range(n_cols):
                            out_colors[out_idx, j] = colors[i, j]
                else:
                    out_colors[out_idx] = colors[i]

            # Copy shN (branch hoisted, unrolled for RGB channels)
            # shN shape: [N, K, 3] where K is number of SH bands
            if has_shN:
                for j in range(n_sh_bands):
                    out_shN[out_idx, j, 0] = shN[i, j, 0]
                    out_shN[out_idx, j, 1] = shN[i, j, 1]
                    out_shN[out_idx, j, 2] = shN[i, j, 2]
