"""
3D Geometric Transformations for Gaussian Splatting

CPU-optimized using NumPy and Numba for maximum performance.

Functions:
- translate(): Translate positions
- scale(): Scale positions and sizes
- rotate(): Rotate positions and orientations
- transform(): Combined transformation (with fused kernel optimization)
- Quaternion utilities: multiply, conversions, etc.
"""

from typing import Union

import numpy as np

# Type aliases for better readability
ArrayLike = Union[np.ndarray, tuple, list]

# Import Numba kernels at module level - Numba is required
from gspro.transform.kernels import (
    elementwise_multiply_scalar_numba,
    elementwise_multiply_vector_numba,
    fused_transform_numba,
    quaternion_multiply_batched_numba,
    quaternion_multiply_single_numba,
)


# ============================================================================
# 4x4 Homogeneous Transformation Matrix Building (NumPy)
# ============================================================================


def _build_translation_matrix_4x4_numpy(translation: np.ndarray) -> np.ndarray:
    """Build 4x4 translation matrix."""
    T = np.eye(4, dtype=translation.dtype)
    T[:3, 3] = translation
    return T


def _build_rotation_matrix_4x4_numpy(rotation_3x3: np.ndarray) -> np.ndarray:
    """Build 4x4 rotation matrix from 3x3 rotation matrix."""
    R = np.eye(4, dtype=rotation_3x3.dtype)
    R[:3, :3] = rotation_3x3
    return R


def _build_scale_matrix_4x4_numpy(scale_factor: np.ndarray) -> np.ndarray:
    """Build 4x4 scale matrix."""
    S = np.eye(4, dtype=scale_factor.dtype)
    S[0, 0] = scale_factor[0]
    S[1, 1] = scale_factor[1]
    S[2, 2] = scale_factor[2]
    return S


def _compose_transform_matrix_numpy(
    translation: ArrayLike | None,
    rotation: ArrayLike | None,
    rotation_format: str,
    scale_factor: float | ArrayLike | None,
    center: ArrayLike | None,
    dtype: np.dtype = np.float32,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """
    Compose transformation matrix and auxiliary data.

    Returns:
        transform_matrix: 4x4 homogeneous transformation matrix
        rotation_quat: Quaternion for orientation updates (or None)
        scale_vec: Scale vector for size updates (or None)
    """
    # Identity matrix
    M = np.eye(4, dtype=dtype)
    rotation_quat = None
    scale_vec = None

    # Convert inputs to numpy arrays with proper dtype
    if scale_factor is not None:
        if isinstance(scale_factor, (int, float)):
            scale_vec = np.array([scale_factor, scale_factor, scale_factor], dtype=dtype)
        elif not isinstance(scale_factor, np.ndarray):
            scale_vec = np.array(scale_factor, dtype=dtype)
        else:
            scale_vec = scale_factor.astype(dtype)

        if scale_vec.ndim == 0:
            scale_vec = np.array([float(scale_vec)] * 3, dtype=dtype)

    if translation is not None and not isinstance(translation, np.ndarray):
        translation = np.array(translation, dtype=dtype)

    if center is not None and not isinstance(center, np.ndarray):
        center = np.array(center, dtype=dtype)

    # Build transformation: T @ R @ S (applied right-to-left)
    # With center: T(center) @ T @ R @ S @ T(-center)

    if center is not None:
        # Translate to origin
        T_neg_center = _build_translation_matrix_4x4_numpy(-center)
        M = T_neg_center @ M

    # Scale
    if scale_vec is not None:
        S = _build_scale_matrix_4x4_numpy(scale_vec)
        M = S @ M

    # Rotate
    if rotation is not None:
        # Convert rotation to quaternion and 3x3 matrix
        if rotation_format == "quaternion":
            rotation_quat = np.array(rotation, dtype=dtype) if not isinstance(rotation, np.ndarray) else rotation.astype(dtype)
        elif rotation_format == "matrix":
            rotation_matrix_3x3 = np.array(rotation, dtype=dtype) if not isinstance(rotation, np.ndarray) else rotation.astype(dtype)
            rotation_quat = _rotation_matrix_to_quaternion_numpy(rotation_matrix_3x3)
        elif rotation_format == "axis_angle":
            rotation_quat = _axis_angle_to_quaternion_numpy(np.array(rotation, dtype=dtype))
        elif rotation_format == "euler":
            rotation_quat = _euler_to_quaternion_numpy(np.array(rotation, dtype=dtype))
        else:
            raise ValueError(f"Unknown rotation format: {rotation_format}")

        # Note: Quaternion normalization is handled in _quaternion_to_rotation_matrix_numpy
        # to avoid redundant normalization

        # Convert to 3x3 rotation matrix
        rotation_matrix_3x3 = _quaternion_to_rotation_matrix_numpy(rotation_quat)
        R = _build_rotation_matrix_4x4_numpy(rotation_matrix_3x3)
        M = R @ M

    # Translate
    if translation is not None:
        T = _build_translation_matrix_4x4_numpy(translation)
        M = T @ M

    if center is not None:
        # Translate back from origin
        T_center = _build_translation_matrix_4x4_numpy(center)
        M = T_center @ M

    return M, rotation_quat, scale_vec


def _apply_homogeneous_transform_numpy(
    points: np.ndarray, matrix: np.ndarray, out: np.ndarray | None = None
) -> np.ndarray:
    """Apply 4x4 homogeneous transformation matrix to Nx3 points.

    Optimized to avoid homogeneous coordinate overhead by extracting
    the 3x3 rotation/scale and 3-vector translation directly.

    Uses NumPy's BLAS-optimized matrix multiplication for best performance.

    Args:
        points: Input points [N, 3]
        matrix: 4x4 transformation matrix
        out: Optional pre-allocated output buffer [N, 3]

    Returns:
        Transformed points (same as `out` if provided)
    """
    # Extract 3x3 combined rotation/scale matrix and translation vector
    R = matrix[:3, :3]  # Upper-left 3x3
    t = matrix[:3, 3]   # Translation column

    if out is not None:
        # Use NumPy's BLAS-optimized matmul with output buffer
        np.matmul(points, R.T, out=out)
        out += t
        return out
    else:
        # Use NumPy's BLAS-optimized matmul - faster than Numba for this operation
        return points @ R.T + t


# ============================================================================
# ============================================================================


def _translate_numpy(
    means: np.ndarray,
    translation: ArrayLike,
) -> np.ndarray:
    """NumPy implementation of translate."""
    if not isinstance(translation, np.ndarray):
        translation = np.array(translation, dtype=means.dtype)

    if translation.ndim == 1:
        translation = translation[np.newaxis, :]

    return means + translation


def _scale_numpy(
    means: np.ndarray,
    scales: np.ndarray,
    scale_factor: float | ArrayLike,
    center: ArrayLike | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """NumPy implementation of scale.

    Automatically uses Numba-optimized version when available (8x faster).
    Falls back to pure NumPy if Numba is not installed.
    """
    # Convert scale_factor to array
    if isinstance(scale_factor, (int, float)):
        scale_factor_arr = np.array([scale_factor, scale_factor, scale_factor], dtype=means.dtype)
        is_uniform_scale = True
        uniform_scale_value = float(scale_factor)
    elif not isinstance(scale_factor, np.ndarray):
        scale_factor_arr = np.array(scale_factor, dtype=means.dtype)
        is_uniform_scale = False
    else:
        scale_factor_arr = scale_factor.astype(means.dtype)
        is_uniform_scale = False

    if scale_factor_arr.ndim == 1:
        scale_factor_arr = scale_factor_arr[np.newaxis, :]

    # Use Numba-optimized version (always available)
    # Handle center of scaling
    if center is not None:
        if not isinstance(center, np.ndarray):
            center = np.array(center, dtype=means.dtype)
        if center.ndim == 1:
            center = center[np.newaxis, :]

        # (means - center) * scale_factor + center
        centered_means = means - center
        scaled_centered = np.empty_like(means)

        if is_uniform_scale:
            elementwise_multiply_scalar_numba(centered_means, uniform_scale_value, scaled_centered)
        else:
            elementwise_multiply_vector_numba(centered_means, scale_factor_arr, scaled_centered)

        scaled_means = scaled_centered + center
    else:
        # means * scale_factor
        scaled_means = np.empty_like(means)
        if is_uniform_scale:
            elementwise_multiply_scalar_numba(means, uniform_scale_value, scaled_means)
        else:
            elementwise_multiply_vector_numba(means, scale_factor_arr, scaled_means)

    # scales * scale_factor
    scaled_scales = np.empty_like(scales)
    if is_uniform_scale:
        elementwise_multiply_scalar_numba(scales, uniform_scale_value, scaled_scales)
    else:
        elementwise_multiply_vector_numba(scales, scale_factor_arr, scaled_scales)

    return scaled_means, scaled_scales


def _scale_numpy_fallback(
    means: np.ndarray,
    scales: np.ndarray,
    scale_factor: float | ArrayLike,
    center: ArrayLike | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Pure NumPy fallback (not used - kept for reference)."""
    # Convert scale_factor to array
    if isinstance(scale_factor, (int, float)):
        scale_factor_arr = np.array([scale_factor, scale_factor, scale_factor], dtype=means.dtype)
    elif not isinstance(scale_factor, np.ndarray):
        scale_factor_arr = np.array(scale_factor, dtype=means.dtype)
    else:
        scale_factor_arr = scale_factor.astype(means.dtype)

    if scale_factor_arr.ndim == 1:
        scale_factor_arr = scale_factor_arr[np.newaxis, :]

    if center is not None:
        if not isinstance(center, np.ndarray):
            center = np.array(center, dtype=means.dtype)
        if center.ndim == 1:
            center = center[np.newaxis, :]

        scaled_means = (means - center) * scale_factor_arr + center
    else:
        scaled_means = means * scale_factor_arr

    scaled_scales = scales * scale_factor_arr

    return scaled_means, scaled_scales


def _rotate_numpy(
    means: np.ndarray,
    quaternions: np.ndarray,
    rotation: np.ndarray,
    center: ArrayLike | None = None,
    rotation_format: str = "quaternion",
) -> tuple[np.ndarray, np.ndarray]:
    """NumPy implementation of rotate."""
    # Convert rotation to quaternion if needed
    if rotation_format == "quaternion":
        rotation_quat = rotation
    elif rotation_format == "matrix":
        rotation_quat = _rotation_matrix_to_quaternion_numpy(rotation)
    elif rotation_format == "axis_angle":
        rotation_quat = _axis_angle_to_quaternion_numpy(rotation)
    elif rotation_format == "euler":
        rotation_quat = _euler_to_quaternion_numpy(rotation)
    else:
        raise ValueError(f"Unknown rotation format: {rotation_format}")

    # Reshape if needed (normalization handled in _quaternion_to_rotation_matrix_numpy)
    if rotation_quat.ndim == 1:
        rotation_quat = rotation_quat[np.newaxis, :]

    # Convert quaternion to rotation matrix for position rotation
    rot_matrix = _quaternion_to_rotation_matrix_numpy(rotation_quat.squeeze(0))

    # Handle center of rotation
    if center is not None:
        if not isinstance(center, np.ndarray):
            center = np.array(center, dtype=means.dtype)
        if center.ndim == 1:
            center = center[np.newaxis, :]

        rotated_means = (means - center) @ rot_matrix.T + center
    else:
        rotated_means = means @ rot_matrix.T

    # Rotate quaternions: q_new = q_rotation * q_old
    rotated_quaternions = _quaternion_multiply_numpy(rotation_quat, quaternions)

    return rotated_means, rotated_quaternions


def _quaternion_multiply_numpy(
    q1: np.ndarray, q2: np.ndarray, out: np.ndarray | None = None
) -> np.ndarray:
    """
    NumPy implementation of quaternion multiplication.

    Uses Numba-optimized version (200x faster than pure NumPy).

    Args:
        q1: First quaternion [N, 4] or [4] (w, x, y, z)
        q2: Second quaternion [N, 4] or [4] (w, x, y, z)
        out: Optional pre-allocated output buffer [N, 4]

    Returns:
        Product quaternion [N, 4] (same as `out` if provided)
    """
    # Use Numba-optimized version (always available)
    # Ensure inputs are 2D
    q1_was_1d = q1.ndim == 1
    q2_was_1d = q2.ndim == 1

    if q1_was_1d:
        q1 = q1[np.newaxis, :]
    if q2_was_1d:
        q2 = q2[np.newaxis, :]

    # Allocate or use provided output
    if q1.shape[0] == 1 and q2.shape[0] > 1:
        # Single quaternion broadcast to many
        if out is None:
            out = np.empty_like(q2)
        quaternion_multiply_single_numba(q1[0], q2, out)
    elif q1.shape[0] == q2.shape[0]:
        # Batched multiplication
        if out is None:
            out = np.empty_like(q1)
        quaternion_multiply_batched_numba(q1, q2, out)
    else:
        # Fall back to NumPy for other cases
        if q1.ndim == 1:
            q1 = q1[np.newaxis, :]
        if q2.ndim == 1:
            q2 = q2[np.newaxis, :]

        w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
        w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]

        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

        result = np.stack([w, x, y, z], axis=1)
        if out is not None:
            out[:] = result
            return out
        return result

    return out


def _quaternion_to_rotation_matrix_numpy(q: np.ndarray) -> np.ndarray:
    """NumPy implementation of quaternion to rotation matrix."""
    q = q / np.linalg.norm(q)
    w, x, y, z = q[0], q[1], q[2], q[3]

    R = np.zeros((3, 3), dtype=q.dtype)

    R[0, 0] = 1 - 2 * (y * y + z * z)
    R[0, 1] = 2 * (x * y - w * z)
    R[0, 2] = 2 * (x * z + w * y)

    R[1, 0] = 2 * (x * y + w * z)
    R[1, 1] = 1 - 2 * (x * x + z * z)
    R[1, 2] = 2 * (y * z - w * x)

    R[2, 0] = 2 * (x * z - w * y)
    R[2, 1] = 2 * (y * z + w * x)
    R[2, 2] = 1 - 2 * (x * x + y * y)

    return R


def _rotation_matrix_to_quaternion_numpy(R: np.ndarray) -> np.ndarray:
    """NumPy implementation of rotation matrix to quaternion."""
    trace = R[0, 0] + R[1, 1] + R[2, 2]

    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s

    q = np.array([w, x, y, z], dtype=R.dtype)
    return q / np.linalg.norm(q)


def _axis_angle_to_quaternion_numpy(axis_angle: np.ndarray) -> np.ndarray:
    """NumPy implementation of axis-angle to quaternion."""
    angle = np.linalg.norm(axis_angle)
    if angle < 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=axis_angle.dtype)

    axis = axis_angle / angle
    half_angle = angle / 2
    sin_half = np.sin(half_angle)

    w = np.cos(half_angle)
    x = axis[0] * sin_half
    y = axis[1] * sin_half
    z = axis[2] * sin_half

    return np.array([w, x, y, z], dtype=axis_angle.dtype)


def _euler_to_quaternion_numpy(euler: np.ndarray) -> np.ndarray:
    """NumPy implementation of Euler angles to quaternion."""
    roll, pitch, yaw = euler[0], euler[1], euler[2]

    cr = np.cos(roll / 2)
    sr = np.sin(roll / 2)
    cp = np.cos(pitch / 2)
    sp = np.sin(pitch / 2)
    cy = np.cos(yaw / 2)
    sy = np.sin(yaw / 2)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return np.array([w, x, y, z], dtype=euler.dtype)


def _quaternion_to_euler_numpy(q: np.ndarray) -> np.ndarray:
    """NumPy implementation of quaternion to Euler angles."""
    w, x, y, z = q[0], q[1], q[2], q[3]

    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    if np.abs(sinp) >= 1:
        pitch = np.copysign(np.pi / 2, sinp)
    else:
        pitch = np.arcsin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.array([roll, pitch, yaw], dtype=q.dtype)


# ============================================================================
# ============================================================================



# ============================================================================
# Public API - NumPy-only implementations
# ============================================================================

def translate(means: np.ndarray, translation: np.ndarray | tuple | list) -> np.ndarray:
    """
    Translate (shift) 3D positions.
    
    Args:
        means: Positions [N, 3]
        translation: Translation vector [3] or list/tuple
        
    Returns:
        Translated positions [N, 3]
    """
    translation = np.asarray(translation, dtype=np.float32)
    return _translate_numpy(means, translation)


def scale(
    means: np.ndarray,
    scales: np.ndarray,
    scale_factor: float | np.ndarray | tuple | list,
    center: np.ndarray | tuple | list | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Scale positions and Gaussian sizes.
    
    Args:
        means: Positions [N, 3]
        scales: Gaussian scales [N, 3]
        scale_factor: Uniform (float) or per-axis ([3]) scaling
        center: Optional center point for scaling
        
    Returns:
        (scaled_means, scaled_scales)
    """
    # Convert scale_factor to array
    if isinstance(scale_factor, (int, float)):
        scale_factor = np.array([scale_factor] * 3, dtype=np.float32)
    else:
        scale_factor = np.asarray(scale_factor, dtype=np.float32)
    
    if center is not None:
        center = np.asarray(center, dtype=np.float32)
    
    return _scale_numpy(means, scales, scale_factor, center)


def rotate(
    means: np.ndarray,
    quaternions: np.ndarray,
    rotation: np.ndarray,
    center: np.ndarray | tuple | list | None = None,
    rotation_format: str = "quaternion",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Rotate positions and orientations.
    
    Args:
        means: Positions [N, 3]
        quaternions: Orientations [N, 4] (w, x, y, z)
        rotation: Rotation in specified format
        center: Optional center point for rotation
        rotation_format: "quaternion", "matrix", "axis_angle", or "euler"
        
    Returns:
        (rotated_means, rotated_quaternions)
    """
    if center is not None:
        center = np.asarray(center, dtype=np.float32)
    
    return _rotate_numpy(means, quaternions, rotation, center, rotation_format)


def transform(
    means: np.ndarray,
    quaternions: np.ndarray | None = None,
    scales: np.ndarray | None = None,
    translation: np.ndarray | tuple | list | None = None,
    rotation: np.ndarray | None = None,
    rotation_format: str = "quaternion",
    scale_factor: float | np.ndarray | tuple | list | None = None,
    center: np.ndarray | tuple | list | None = None,
    out_means: np.ndarray | None = None,
    out_quaternions: np.ndarray | None = None,
    out_scales: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """
    Apply combined transformation: scale -> rotate -> translate.

    Uses fused Numba kernel for 4-5x speedup when:
    - All output buffers are pre-allocated (out_means, out_quaternions, out_scales)
    - All data is provided (quaternions, scales)
    - center is None
    - Numba is available

    Args:
        means: Positions [N, 3]
        quaternions: Optional orientations [N, 4]
        scales: Optional Gaussian scales [N, 3]
        translation: Optional translation vector
        rotation: Optional rotation
        rotation_format: Format of rotation parameter
        scale_factor: Optional scale factor
        center: Optional center for rotation/scaling
        out_means: Optional pre-allocated output for means
        out_quaternions: Optional pre-allocated output for quaternions
        out_scales: Optional pre-allocated output for scales

    Returns:
        (transformed_means, transformed_quaternions, transformed_scales)

    Example:
        >>> # Fast path with pre-allocated buffers (4-5x faster)
        >>> out_means = np.empty_like(means)
        >>> out_quats = np.empty_like(quaternions)
        >>> out_scales = np.empty_like(scales)
        >>> transform(means, quaternions, scales,
        ...           translation=[1, 0, 0], rotation=quat, scale_factor=2.0,
        ...           out_means=out_means, out_quaternions=out_quats, out_scales=out_scales)
    """
    # Validation
    if scale_factor is not None and scales is None:
        raise ValueError("scale_factor provided but scales is None")
    if rotation is not None and quaternions is None:
        raise ValueError("rotation provided but quaternions is None")

    # Validate output buffer sizes
    if out_means is not None and out_means.shape != means.shape:
        raise ValueError(
            f"out_means shape {out_means.shape} does not match means shape {means.shape}"
        )
    if out_quaternions is not None and quaternions is not None:
        if out_quaternions.shape != quaternions.shape:
            raise ValueError(
                f"out_quaternions shape {out_quaternions.shape} does not match quaternions shape {quaternions.shape}"
            )
    if out_scales is not None and scales is not None:
        if out_scales.shape != scales.shape:
            raise ValueError(
                f"out_scales shape {out_scales.shape} does not match scales shape {scales.shape}"
            )

    # Ensure contiguous arrays for optimal performance
    if not means.flags["C_CONTIGUOUS"]:
        means = np.ascontiguousarray(means)
    if quaternions is not None and not quaternions.flags["C_CONTIGUOUS"]:
        quaternions = np.ascontiguousarray(quaternions)
    if scales is not None and not scales.flags["C_CONTIGUOUS"]:
        scales = np.ascontiguousarray(scales)

    # Build transformation matrix
    transform_matrix, rotation_quat, scale_vec = _compose_transform_matrix_numpy(
        translation=translation,
        rotation=rotation,
        rotation_format=rotation_format,
        scale_factor=scale_factor,
        center=center,
        dtype=means.dtype,
    )

    # Fast path: Use fused Numba kernel when all conditions are met (4-5x faster)
    if (
        out_means is not None
        and out_quaternions is not None
        and out_scales is not None
        and quaternions is not None
        and scales is not None
        and rotation_quat is not None
        and scale_vec is not None
        and center is None  # Fused kernel doesn't support center yet
    ):
        # Extract rotation matrix from transformation matrix
        R = transform_matrix[:3, :3]
        t = transform_matrix[:3, 3]

        # Single fused kernel call (much faster than separate operations)
        fused_transform_numba(
            means,
            quaternions,
            scales,
            rotation_quat,
            scale_vec,
            t,
            R,
            out_means,
            out_quaternions,
            out_scales,
        )
        result_means = out_means
        result_quats = out_quaternions
        result_scales = out_scales

    else:
        # Standard path: Separate operations
        # Apply transformation to means
        if out_means is not None:
            result_means = out_means
            _apply_homogeneous_transform_numpy(means, transform_matrix, out=result_means)
        else:
            result_means = _apply_homogeneous_transform_numpy(means, transform_matrix)

        # Apply quaternion transformation to orientations
        if rotation_quat is not None and quaternions is not None:
            if out_quaternions is not None:
                result_quats = out_quaternions
                _quaternion_multiply_numpy(rotation_quat, quaternions, out=result_quats)
            else:
                result_quats = _quaternion_multiply_numpy(rotation_quat, quaternions)
        else:
            result_quats = quaternions

        # Apply scale to Gaussian sizes
        if scale_vec is not None and scales is not None:
            # Allocate or use provided buffer
            if out_scales is not None:
                result_scales = out_scales
            else:
                result_scales = np.empty_like(scales)

            # Use Numba-optimized elementwise multiply
            elementwise_multiply_vector_numba(scales, scale_vec, result_scales)
        else:
            result_scales = scales

    return result_means, result_quats, result_scales


def quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """
    Multiply quaternions.
    
    Args:
        q1: First quaternion(s) [4] or [N, 4]
        q2: Second quaternion(s) [4] or [N, 4]
        
    Returns:
        Product quaternion(s)
    """
    return _quaternion_multiply_numpy(q1, q2)


def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """Convert quaternion to rotation matrix."""
    return _quaternion_to_rotation_matrix_numpy(q)


def rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """Convert rotation matrix to quaternion."""
    return _rotation_matrix_to_quaternion_numpy(R)


def axis_angle_to_quaternion(axis_angle: np.ndarray) -> np.ndarray:
    """Convert axis-angle to quaternion."""
    return _axis_angle_to_quaternion_numpy(axis_angle)


def euler_to_quaternion(euler: np.ndarray) -> np.ndarray:
    """Convert Euler angles to quaternion."""
    return _euler_to_quaternion_numpy(euler)


def quaternion_to_euler(q: np.ndarray) -> np.ndarray:
    """Convert quaternion to Euler angles."""
    return _quaternion_to_euler_numpy(q)

