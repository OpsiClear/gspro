"""
3D Geometric Transformations for Gaussian Splatting

Provides efficient transformation operations for point clouds:
- Translation: Shift positions
- Rotation: Rotate positions and orientations
- Scaling: Scale positions and sizes
- Combined transforms: Apply multiple transformations efficiently

Supports both NumPy (CPU) and PyTorch (CPU/GPU) backends with automatic dispatch.
"""

from typing import Union

import numpy as np
import torch

# Type aliases for better readability
ArrayLike = Union[np.ndarray, torch.Tensor, tuple, list]

# Import Numba ops at module level to avoid import overhead on every call
try:
    from gslut.numba_ops import (
        NUMBA_AVAILABLE,
        elementwise_multiply_scalar_numba,
        elementwise_multiply_vector_numba,
        fused_transform_numba,
        quaternion_multiply_batched_numba,
        quaternion_multiply_single_numba,
    )
except ImportError:
    NUMBA_AVAILABLE = False
    quaternion_multiply_single_numba = None
    quaternion_multiply_batched_numba = None
    elementwise_multiply_scalar_numba = None
    elementwise_multiply_vector_numba = None
    fused_transform_numba = None


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

        # Normalize quaternion
        rotation_quat = rotation_quat / np.linalg.norm(rotation_quat)

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
# 4x4 Homogeneous Transformation Matrix Building (PyTorch)
# ============================================================================


def _build_translation_matrix_4x4_torch(translation: torch.Tensor) -> torch.Tensor:
    """Build 4x4 translation matrix."""
    T = torch.eye(4, dtype=translation.dtype, device=translation.device)
    T[:3, 3] = translation
    return T


def _build_rotation_matrix_4x4_torch(rotation_3x3: torch.Tensor) -> torch.Tensor:
    """Build 4x4 rotation matrix from 3x3 rotation matrix."""
    R = torch.eye(4, dtype=rotation_3x3.dtype, device=rotation_3x3.device)
    R[:3, :3] = rotation_3x3
    return R


def _build_scale_matrix_4x4_torch(scale_factor: torch.Tensor) -> torch.Tensor:
    """Build 4x4 scale matrix."""
    S = torch.eye(4, dtype=scale_factor.dtype, device=scale_factor.device)
    S[0, 0] = scale_factor[0]
    S[1, 1] = scale_factor[1]
    S[2, 2] = scale_factor[2]
    return S


def _compose_transform_matrix_torch(
    translation: ArrayLike | None,
    rotation: ArrayLike | None,
    rotation_format: str,
    scale_factor: float | ArrayLike | None,
    center: ArrayLike | None,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """
    Compose transformation matrix and auxiliary data.

    Returns:
        transform_matrix: 4x4 homogeneous transformation matrix
        rotation_quat: Quaternion for orientation updates (or None)
        scale_vec: Scale vector for size updates (or None)
    """
    # Identity matrix
    M = torch.eye(4, dtype=dtype, device=device)
    rotation_quat = None
    scale_vec = None

    # Convert inputs to torch tensors with proper dtype and device
    if scale_factor is not None:
        if isinstance(scale_factor, (int, float)):
            scale_vec = torch.tensor([scale_factor, scale_factor, scale_factor], dtype=dtype, device=device)
        elif not isinstance(scale_factor, torch.Tensor):
            scale_vec = torch.tensor(scale_factor, dtype=dtype, device=device)
        else:
            scale_vec = scale_factor.to(dtype=dtype, device=device)

        if scale_vec.ndim == 0:
            scale_vec = torch.tensor([float(scale_vec)] * 3, dtype=dtype, device=device)

    if translation is not None and not isinstance(translation, torch.Tensor):
        translation = torch.tensor(translation, dtype=dtype, device=device)
    elif translation is not None:
        translation = translation.to(dtype=dtype, device=device)

    if center is not None and not isinstance(center, torch.Tensor):
        center = torch.tensor(center, dtype=dtype, device=device)
    elif center is not None:
        center = center.to(dtype=dtype, device=device)

    # Build transformation: T @ R @ S (applied right-to-left)
    # With center: T(center) @ T @ R @ S @ T(-center)

    if center is not None:
        # Translate to origin
        T_neg_center = _build_translation_matrix_4x4_torch(-center)
        M = T_neg_center @ M

    # Scale
    if scale_vec is not None:
        S = _build_scale_matrix_4x4_torch(scale_vec)
        M = S @ M

    # Rotate
    if rotation is not None:
        # Convert rotation to quaternion and 3x3 matrix
        if rotation_format == "quaternion":
            rotation_quat = torch.tensor(rotation, dtype=dtype, device=device) if not isinstance(rotation, torch.Tensor) else rotation.to(dtype=dtype, device=device)
        elif rotation_format == "matrix":
            rotation_matrix_3x3 = torch.tensor(rotation, dtype=dtype, device=device) if not isinstance(rotation, torch.Tensor) else rotation.to(dtype=dtype, device=device)
            rotation_quat = _rotation_matrix_to_quaternion_torch(rotation_matrix_3x3)
        elif rotation_format == "axis_angle":
            rotation_quat = _axis_angle_to_quaternion_torch(torch.tensor(rotation, dtype=dtype, device=device))
        elif rotation_format == "euler":
            rotation_quat = _euler_to_quaternion_torch(torch.tensor(rotation, dtype=dtype, device=device))
        else:
            raise ValueError(f"Unknown rotation format: {rotation_format}")

        # Normalize quaternion
        rotation_quat = rotation_quat / torch.norm(rotation_quat)

        # Convert to 3x3 rotation matrix
        rotation_matrix_3x3 = _quaternion_to_rotation_matrix_torch(rotation_quat)
        R = _build_rotation_matrix_4x4_torch(rotation_matrix_3x3)
        M = R @ M

    # Translate
    if translation is not None:
        T = _build_translation_matrix_4x4_torch(translation)
        M = T @ M

    if center is not None:
        # Translate back from origin
        T_center = _build_translation_matrix_4x4_torch(center)
        M = T_center @ M

    return M, rotation_quat, scale_vec


def _apply_homogeneous_transform_torch(
    points: torch.Tensor, matrix: torch.Tensor, out: torch.Tensor | None = None
) -> torch.Tensor:
    """Apply 4x4 homogeneous transformation matrix to Nx3 points.

    Optimized to avoid homogeneous coordinate overhead by extracting
    the 3x3 rotation/scale and 3-vector translation directly.

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
        # Use PyTorch's in-place operations with output buffer
        torch.matmul(points, R.T, out=out)
        out += t
        return out
    else:
        # Apply: points @ R.T + t (single matmul + broadcast add)
        return points @ R.T + t


# ============================================================================
# NumPy Implementation
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

    # Use Numba-optimized version if available (module-level import)
    if NUMBA_AVAILABLE:
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

    # Pure NumPy fallback
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

    # Ensure quaternion is normalized
    if rotation_quat.ndim == 1:
        rotation_quat = rotation_quat[np.newaxis, :]
    rotation_quat = rotation_quat / np.linalg.norm(rotation_quat, axis=1, keepdims=True)

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

    Automatically uses Numba-optimized version when available (200x faster).
    Falls back to pure NumPy if Numba is not installed.

    Args:
        q1: First quaternion [N, 4] or [4] (w, x, y, z)
        q2: Second quaternion [N, 4] or [4] (w, x, y, z)
        out: Optional pre-allocated output buffer [N, 4]

    Returns:
        Product quaternion [N, 4] (same as `out` if provided)
    """
    # Use Numba-optimized version if available (module-level import)
    if NUMBA_AVAILABLE:
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

    # Pure NumPy fallback
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
# PyTorch Implementation
# ============================================================================


def _translate_torch(
    means: torch.Tensor,
    translation: ArrayLike,
) -> torch.Tensor:
    """PyTorch implementation of translate."""
    if not isinstance(translation, torch.Tensor):
        translation = torch.tensor(translation, dtype=means.dtype, device=means.device)

    if translation.dim() == 1:
        translation = translation.unsqueeze(0)

    return means + translation


def _scale_torch(
    means: torch.Tensor,
    scales: torch.Tensor,
    scale_factor: float | ArrayLike,
    center: ArrayLike | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """PyTorch implementation of scale."""
    # Convert scale_factor to tensor
    if isinstance(scale_factor, (int, float)):
        scale_factor = torch.tensor(
            [scale_factor, scale_factor, scale_factor],
            dtype=means.dtype,
            device=means.device,
        )
    elif not isinstance(scale_factor, torch.Tensor):
        scale_factor = torch.tensor(scale_factor, dtype=means.dtype, device=means.device)

    if scale_factor.dim() == 1:
        scale_factor = scale_factor.unsqueeze(0)

    # Handle center of scaling
    if center is not None:
        if not isinstance(center, torch.Tensor):
            center = torch.tensor(center, dtype=means.dtype, device=means.device)
        if center.dim() == 1:
            center = center.unsqueeze(0)

        scaled_means = (means - center) * scale_factor + center
    else:
        scaled_means = means * scale_factor

    scaled_scales = scales * scale_factor

    return scaled_means, scaled_scales


def _rotate_torch(
    means: torch.Tensor,
    quaternions: torch.Tensor,
    rotation: torch.Tensor,
    center: ArrayLike | None = None,
    rotation_format: str = "quaternion",
) -> tuple[torch.Tensor, torch.Tensor]:
    """PyTorch implementation of rotate."""
    # Convert rotation to quaternion if needed
    if rotation_format == "quaternion":
        rotation_quat = rotation
    elif rotation_format == "matrix":
        rotation_quat = _rotation_matrix_to_quaternion_torch(rotation)
    elif rotation_format == "axis_angle":
        rotation_quat = _axis_angle_to_quaternion_torch(rotation)
    elif rotation_format == "euler":
        rotation_quat = _euler_to_quaternion_torch(rotation)
    else:
        raise ValueError(f"Unknown rotation format: {rotation_format}")

    # Ensure quaternion is normalized and on same device
    rotation_quat = rotation_quat.to(means.device)
    if rotation_quat.dim() == 1:
        rotation_quat = rotation_quat.unsqueeze(0)
    rotation_quat = torch.nn.functional.normalize(rotation_quat, p=2, dim=1)

    # Convert quaternion to rotation matrix for position rotation
    rot_matrix = _quaternion_to_rotation_matrix_torch(rotation_quat.squeeze(0))

    # Handle center of rotation
    if center is not None:
        if not isinstance(center, torch.Tensor):
            center = torch.tensor(center, dtype=means.dtype, device=means.device)
        if center.dim() == 1:
            center = center.unsqueeze(0)

        rotated_means = (means - center) @ rot_matrix.T + center
    else:
        rotated_means = means @ rot_matrix.T

    # Rotate quaternions: q_new = q_rotation * q_old
    rotated_quaternions = _quaternion_multiply_torch(rotation_quat, quaternions)

    return rotated_means, rotated_quaternions


def _quaternion_multiply_torch(
    q1: torch.Tensor, q2: torch.Tensor, out: torch.Tensor | None = None
) -> torch.Tensor:
    """
    PyTorch implementation of quaternion multiplication.

    Args:
        q1: First quaternion [N, 4] or [4] (w, x, y, z)
        q2: Second quaternion [N, 4] or [4] (w, x, y, z)
        out: Optional pre-allocated output buffer [N, 4]

    Returns:
        Product quaternion [N, 4] (same as `out` if provided)
    """
    if q1.dim() == 1:
        q1 = q1.unsqueeze(0)
    if q2.dim() == 1:
        q2 = q2.unsqueeze(0)

    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]

    if out is not None:
        # Compute in-place to provided buffer
        out[:, 0] = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        out[:, 1] = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        out[:, 2] = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        out[:, 3] = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        return out
    else:
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        return torch.stack([w, x, y, z], dim=1)


def _quaternion_to_rotation_matrix_torch(q: torch.Tensor) -> torch.Tensor:
    """PyTorch implementation of quaternion to rotation matrix."""
    q = torch.nn.functional.normalize(q, p=2, dim=0)
    w, x, y, z = q[0], q[1], q[2], q[3]

    R = torch.zeros(3, 3, dtype=q.dtype, device=q.device)

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


def _rotation_matrix_to_quaternion_torch(R: torch.Tensor) -> torch.Tensor:
    """PyTorch implementation of rotation matrix to quaternion."""
    trace = R[0, 0] + R[1, 1] + R[2, 2]

    if trace > 0:
        s = 0.5 / torch.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * torch.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * torch.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * torch.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s

    q = torch.tensor([w, x, y, z], dtype=R.dtype, device=R.device)
    return torch.nn.functional.normalize(q, p=2, dim=0)


def _axis_angle_to_quaternion_torch(axis_angle: torch.Tensor) -> torch.Tensor:
    """PyTorch implementation of axis-angle to quaternion."""
    angle = torch.norm(axis_angle)
    if angle < 1e-8:
        return torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=axis_angle.dtype, device=axis_angle.device)

    axis = axis_angle / angle
    half_angle = angle / 2
    sin_half = torch.sin(half_angle)

    w = torch.cos(half_angle)
    x = axis[0] * sin_half
    y = axis[1] * sin_half
    z = axis[2] * sin_half

    return torch.tensor([w, x, y, z], dtype=axis_angle.dtype, device=axis_angle.device)


def _euler_to_quaternion_torch(euler: torch.Tensor) -> torch.Tensor:
    """PyTorch implementation of Euler angles to quaternion."""
    roll, pitch, yaw = euler[0], euler[1], euler[2]

    cr = torch.cos(roll / 2)
    sr = torch.sin(roll / 2)
    cp = torch.cos(pitch / 2)
    sp = torch.sin(pitch / 2)
    cy = torch.cos(yaw / 2)
    sy = torch.sin(yaw / 2)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return torch.tensor([w, x, y, z], dtype=euler.dtype, device=euler.device)


def _quaternion_to_euler_torch(q: torch.Tensor) -> torch.Tensor:
    """PyTorch implementation of quaternion to Euler angles."""
    w, x, y, z = q[0], q[1], q[2], q[3]

    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    if torch.abs(sinp) >= 1:
        pitch = torch.copysign(torch.tensor(torch.pi / 2), sinp)
    else:
        pitch = torch.asin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    return torch.tensor([roll, pitch, yaw], dtype=q.dtype, device=q.device)


# ============================================================================
# Public API with Automatic Dispatch
# ============================================================================


def translate(
    means: np.ndarray | torch.Tensor,
    translation: ArrayLike,
) -> np.ndarray | torch.Tensor:
    """
    Translate (shift) 3D positions.

    Supports both NumPy arrays and PyTorch tensors with automatic dispatch.

    Args:
        means: Point positions [N, 3] (numpy.ndarray or torch.Tensor)
        translation: Translation vector [3] or (x, y, z)

    Returns:
        Translated positions [N, 3] (same type as input)

    Example:
        >>> # NumPy
        >>> means_np = np.random.rand(1000, 3)
        >>> translated = translate(means_np, [1.0, 0.0, -0.5])
        >>>
        >>> # PyTorch
        >>> means_torch = torch.rand(1000, 3)
        >>> translated = translate(means_torch, [1.0, 0.0, -0.5])
    """
    if isinstance(means, np.ndarray):
        return _translate_numpy(means, translation)
    else:
        return _translate_torch(means, translation)


def scale(
    means: np.ndarray | torch.Tensor,
    scales: np.ndarray | torch.Tensor,
    scale_factor: float | ArrayLike,
    center: ArrayLike | None = None,
) -> tuple[np.ndarray | torch.Tensor, np.ndarray | torch.Tensor]:
    """
    Scale positions and Gaussian sizes.

    Supports both NumPy arrays and PyTorch tensors with automatic dispatch.

    Args:
        means: Point positions [N, 3] (numpy.ndarray or torch.Tensor)
        scales: Gaussian scales [N, 3] (same type as means)
        scale_factor: Uniform scale (float) or per-axis scale [3]
        center: Center of scaling [3]. If None, scales around origin.

    Returns:
        Tuple of (scaled_means, scaled_scales) (same type as input)

    Example:
        >>> # NumPy
        >>> means = np.random.rand(1000, 3)
        >>> scales = np.random.rand(1000, 3)
        >>> new_means, new_scales = scale(means, scales, 2.0)
        >>>
        >>> # PyTorch
        >>> means = torch.rand(1000, 3)
        >>> scales = torch.rand(1000, 3)
        >>> new_means, new_scales = scale(means, scales, 2.0)
    """
    if isinstance(means, np.ndarray):
        return _scale_numpy(means, scales, scale_factor, center)
    else:
        return _scale_torch(means, scales, scale_factor, center)


def rotate(
    means: np.ndarray | torch.Tensor,
    quaternions: np.ndarray | torch.Tensor,
    rotation: np.ndarray | torch.Tensor,
    center: ArrayLike | None = None,
    rotation_format: str = "quaternion",
) -> tuple[np.ndarray | torch.Tensor, np.ndarray | torch.Tensor]:
    """
    Rotate positions and orientations.

    Supports both NumPy arrays and PyTorch tensors with automatic dispatch.

    Args:
        means: Point positions [N, 3] (numpy.ndarray or torch.Tensor)
        quaternions: Gaussian orientations [N, 4] (w, x, y, z) (same type as means)
        rotation: Rotation in specified format (same type as means)
            - "quaternion": [4] (w, x, y, z)
            - "matrix": [3, 3] rotation matrix
            - "axis_angle": [3] axis-angle representation
            - "euler": [3] Euler angles (roll, pitch, yaw) in radians
        center: Center of rotation [3]. If None, rotates around origin.
        rotation_format: Format of rotation parameter

    Returns:
        Tuple of (rotated_means, rotated_quaternions) (same type as input)

    Example:
        >>> # NumPy
        >>> means = np.random.rand(1000, 3)
        >>> quats = np.random.rand(1000, 4)
        >>> quats = quats / np.linalg.norm(quats, axis=1, keepdims=True)
        >>> rotation_quat = np.array([0.7071, 0, 0, 0.7071])  # 90° Z
        >>> new_means, new_quats = rotate(means, quats, rotation_quat)
        >>>
        >>> # PyTorch
        >>> means = torch.rand(1000, 3)
        >>> quats = torch.nn.functional.normalize(torch.rand(1000, 4), p=2, dim=1)
        >>> rotation_quat = torch.tensor([0.7071, 0, 0, 0.7071])
        >>> new_means, new_quats = rotate(means, quats, rotation_quat)
    """
    if isinstance(means, np.ndarray):
        return _rotate_numpy(means, quaternions, rotation, center, rotation_format)
    else:
        return _rotate_torch(means, quaternions, rotation, center, rotation_format)


def transform(
    means: np.ndarray | torch.Tensor,
    quaternions: np.ndarray | torch.Tensor | None = None,
    scales: np.ndarray | torch.Tensor | None = None,
    translation: ArrayLike | None = None,
    rotation: np.ndarray | torch.Tensor | None = None,
    rotation_format: str = "quaternion",
    scale_factor: float | ArrayLike | None = None,
    center: ArrayLike | None = None,
    out_means: np.ndarray | torch.Tensor | None = None,
    out_quaternions: np.ndarray | torch.Tensor | None = None,
    out_scales: np.ndarray | torch.Tensor | None = None,
) -> tuple[
    np.ndarray | torch.Tensor,
    np.ndarray | torch.Tensor | None,
    np.ndarray | torch.Tensor | None,
]:
    """
    Apply combined transformation: scale -> rotate -> translate.

    Uses fused 4x4 homogeneous transformation matrix for optimal performance.
    Supports both NumPy arrays and PyTorch tensors with automatic dispatch.

    Args:
        means: Point positions [N, 3] (numpy.ndarray or torch.Tensor)
        quaternions: Gaussian orientations [N, 4] (optional, same type as means)
        scales: Gaussian scales [N, 3] (optional, same type as means)
        translation: Translation vector [3] or (x, y, z)
        rotation: Rotation in specified format (see rotate())
        rotation_format: Format of rotation parameter
        scale_factor: Uniform scale or per-axis scale [3]
        center: Center for rotation and scaling
        out_means: Optional pre-allocated output buffer for means [N, 3].
            Must match input shape. Reduces allocation overhead by ~30-40%.
        out_quaternions: Optional pre-allocated output buffer for quaternions [N, 4].
            Ignored if quaternions is None.
        out_scales: Optional pre-allocated output buffer for scales [N, 3].
            Ignored if scales is None.

    Returns:
        Tuple of (transformed_means, transformed_quaternions, transformed_scales)
        (same type as input). Quaternions and scales are None if not provided as input.
        If output buffers are provided, they are returned (modified in-place).

    Example:
        >>> # NumPy
        >>> means = np.random.rand(1000, 3)
        >>> quats = np.random.rand(1000, 4)
        >>> quats = quats / np.linalg.norm(quats, axis=1, keepdims=True)
        >>> scales = np.random.rand(1000, 3)
        >>> new_means, new_quats, new_scales = transform(
        ...     means, quats, scales,
        ...     scale_factor=2.0,
        ...     rotation=np.array([0.9239, 0, 0, 0.3827]),  # 45° Z
        ...     translation=[1.0, 0.0, 0.0]
        ... )
        >>>
        >>> # PyTorch
        >>> means = torch.rand(1000, 3)
        >>> quats = torch.nn.functional.normalize(torch.rand(1000, 4), p=2, dim=1)
        >>> scales = torch.rand(1000, 3)
        >>> new_means, new_quats, new_scales = transform(
        ...     means, quats, scales,
        ...     scale_factor=2.0,
        ...     rotation=torch.tensor([0.9239, 0, 0, 0.3827]),
        ...     translation=[1.0, 0.0, 0.0]
        ... )

    Note:
        This function now uses a fused 4x4 transformation matrix approach,
        which provides 2-3x speedup compared to sequential operations for
        large point clouds (>100K points).
    """
    # Validation
    if scale_factor is not None and scales is None:
        raise ValueError("scale_factor provided but scales is None")
    if rotation is not None and quaternions is None:
        raise ValueError("rotation provided but quaternions is None")

    # Validate output buffer sizes (handle variable-size frames)
    if out_means is not None:
        if out_means.shape != means.shape:
            raise ValueError(
                f"out_means shape {out_means.shape} does not match means shape {means.shape}. "
                f"For variable-size frames, allocate new buffers for each size."
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

    # Ensure contiguous arrays/tensors for optimal performance
    if isinstance(means, np.ndarray):
        if not means.flags["C_CONTIGUOUS"]:
            means = np.ascontiguousarray(means)
        if quaternions is not None and not quaternions.flags["C_CONTIGUOUS"]:
            quaternions = np.ascontiguousarray(quaternions)
        if scales is not None and not scales.flags["C_CONTIGUOUS"]:
            scales = np.ascontiguousarray(scales)
    else:  # PyTorch tensors
        if not means.is_contiguous():
            means = means.contiguous()
        if quaternions is not None and not quaternions.is_contiguous():
            quaternions = quaternions.contiguous()
        if scales is not None and not scales.is_contiguous():
            scales = scales.contiguous()

    # Dispatch to NumPy or PyTorch implementation
    if isinstance(means, np.ndarray):
        # Build transformation matrix (NumPy)
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
            NUMBA_AVAILABLE
            and fused_transform_numba is not None
            and out_means is not None
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

                # Use Numba-optimized elementwise multiply when available (module-level import)
                if NUMBA_AVAILABLE:
                    elementwise_multiply_vector_numba(scales, scale_vec, result_scales)
                else:
                    result_scales[:] = scales * scale_vec
            else:
                result_scales = scales

    else:
        # Build transformation matrix (PyTorch)
        transform_matrix, rotation_quat, scale_vec = _compose_transform_matrix_torch(
            translation=translation,
            rotation=rotation,
            rotation_format=rotation_format,
            scale_factor=scale_factor,
            center=center,
            device=means.device,
            dtype=means.dtype,
        )

        # Apply transformation to means
        if out_means is not None:
            result_means = out_means
            _apply_homogeneous_transform_torch(means, transform_matrix, out=result_means)
        else:
            result_means = _apply_homogeneous_transform_torch(means, transform_matrix)

        # Apply quaternion transformation to orientations
        if rotation_quat is not None and quaternions is not None:
            if out_quaternions is not None:
                result_quats = out_quaternions
                _quaternion_multiply_torch(rotation_quat, quaternions, out=result_quats)
            else:
                result_quats = _quaternion_multiply_torch(rotation_quat, quaternions)
        else:
            result_quats = quaternions

        # Apply scale to Gaussian sizes
        if scale_vec is not None and scales is not None:
            if out_scales is not None:
                result_scales = out_scales
                torch.mul(scales, scale_vec, out=result_scales)
            else:
                result_scales = scales * scale_vec
        else:
            result_scales = scales

    return result_means, result_quats, result_scales


def transform_torch_fast(
    means: torch.Tensor,
    quaternions: torch.Tensor,
    scales: torch.Tensor,
    translation: torch.Tensor,
    rotation: torch.Tensor,
    scale_factor: float,
    out_means: torch.Tensor,
    out_quaternions: torch.Tensor,
    out_scales: torch.Tensor,
    device: str | torch.device = "cpu",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Fast-path transform with minimal overhead (PyTorch only).

    REQUIREMENTS (no validation performed):
    - All inputs must be contiguous PyTorch float32 tensors on same device
    - means: [N, 3], quaternions: [N, 4], scales: [N, 3]
    - rotation: [4] quaternion (w, x, y, z)
    - translation: [3] vector
    - scale_factor: uniform float
    - Output buffers must be pre-allocated with correct shapes on same device

    This function skips:
    - Type checking
    - Shape validation
    - Contiguity checks
    - Input conversion
    - Broadcasting logic
    - Output allocation

    Performance: ~1.5-2x faster than transform() for large batches on CPU/GPU.

    Args:
        means: Point positions [N, 3]
        quaternions: Gaussian orientations [N, 4]
        scales: Gaussian scales [N, 3]
        translation: Translation vector [3]
        rotation: Rotation quaternion [4] (w, x, y, z)
        scale_factor: Uniform scale factor
        out_means: Pre-allocated output [N, 3]
        out_quaternions: Pre-allocated output [N, 4]
        out_scales: Pre-allocated output [N, 3]
        device: Device for tensor operations (default: "cpu")

    Returns:
        Tuple of (out_means, out_quaternions, out_scales) - same objects passed in

    Example:
        >>> N = 1_000_000
        >>> device = "cuda" if torch.cuda.is_available() else "cpu"
        >>> means = torch.randn(N, 3, device=device)
        >>> quats = F.normalize(torch.randn(N, 4, device=device), p=2, dim=1)
        >>> scales = torch.rand(N, 3, device=device)
        >>>
        >>> # Pre-allocate output buffers (reuse for multiple frames)
        >>> out_means = torch.empty_like(means)
        >>> out_quats = torch.empty_like(quats)
        >>> out_scales = torch.empty_like(scales)
        >>>
        >>> # Fast transform (no validation overhead)
        >>> transform_torch_fast(
        ...     means, quats, scales,
        ...     translation=torch.tensor([1.0, 2.0, 3.0], device=device),
        ...     rotation=torch.tensor([0.9239, 0.0, 0.0, 0.3827], device=device),
        ...     scale_factor=2.0,
        ...     out_means=out_means, out_quaternions=out_quats, out_scales=out_scales,
        ...     device=device
        ... )
    """
    # Build transformation matrix (minimal overhead)
    rotation_quat = rotation  # Already quaternion format
    scale_vec = torch.tensor(
        [scale_factor, scale_factor, scale_factor], dtype=means.dtype, device=device
    )

    # Build combined transform matrix
    M = torch.eye(4, dtype=means.dtype, device=device)

    # Scale
    M[0, 0] = scale_factor
    M[1, 1] = scale_factor
    M[2, 2] = scale_factor

    # Rotation (quaternion to matrix) - inline for speed
    w, x, y, z = rotation[0], rotation[1], rotation[2], rotation[3]
    M[0, 0] *= 1 - 2 * (y * y + z * z)
    M[0, 1] = 2 * (x * y - w * z) * scale_factor
    M[0, 2] = 2 * (x * z + w * y) * scale_factor
    M[1, 0] = 2 * (x * y + w * z) * scale_factor
    M[1, 1] *= 1 - 2 * (x * x + z * z)
    M[1, 2] = 2 * (y * z - w * x) * scale_factor
    M[2, 0] = 2 * (x * z - w * y) * scale_factor
    M[2, 1] = 2 * (y * z + w * x) * scale_factor
    M[2, 2] *= 1 - 2 * (x * x + y * y)

    # Translation
    M[:3, 3] = translation

    # Apply transformation to means (direct operations)
    R = M[:3, :3]
    t = M[:3, 3]
    torch.matmul(means, R.T, out=out_means)
    out_means += t

    # Apply quaternion transformation (direct call)
    _quaternion_multiply_torch(rotation_quat, quaternions, out=out_quaternions)

    # Apply scale to Gaussian sizes (direct operation)
    torch.mul(scales, scale_vec, out=out_scales)

    return out_means, out_quaternions, out_scales


# ============================================================================
# Quaternion Math Utilities
# ============================================================================


def quaternion_multiply(
    q1: np.ndarray | torch.Tensor, q2: np.ndarray | torch.Tensor
) -> np.ndarray | torch.Tensor:
    """
    Multiply two quaternions: q_result = q1 * q2.

    Supports both NumPy arrays and PyTorch tensors with automatic dispatch.

    Args:
        q1: First quaternion [N, 4] or [4] (w, x, y, z)
        q2: Second quaternion [N, 4] or [4] (w, x, y, z)

    Returns:
        Product quaternion [N, 4] (same type as input)

    Note: Quaternion multiplication is NOT commutative!

    Example:
        >>> # NumPy
        >>> q1 = np.array([1.0, 0.0, 0.0, 0.0])
        >>> q2 = np.array([0.7071, 0.0, 0.0, 0.7071])
        >>> result = quaternion_multiply(q1, q2)
        >>>
        >>> # PyTorch
        >>> q1 = torch.tensor([1.0, 0.0, 0.0, 0.0])
        >>> q2 = torch.tensor([0.7071, 0.0, 0.0, 0.7071])
        >>> result = quaternion_multiply(q1, q2)
    """
    if isinstance(q1, np.ndarray):
        return _quaternion_multiply_numpy(q1, q2)
    else:
        return _quaternion_multiply_torch(q1, q2)


def quaternion_to_rotation_matrix(
    q: np.ndarray | torch.Tensor,
) -> np.ndarray | torch.Tensor:
    """
    Convert quaternion to 3x3 rotation matrix.

    Supports both NumPy arrays and PyTorch tensors with automatic dispatch.

    Args:
        q: Quaternion [4] (w, x, y, z)

    Returns:
        Rotation matrix [3, 3] (same type as input)

    Example:
        >>> # NumPy
        >>> q = np.array([1.0, 0.0, 0.0, 0.0])
        >>> R = quaternion_to_rotation_matrix(q)
        >>>
        >>> # PyTorch
        >>> q = torch.tensor([1.0, 0.0, 0.0, 0.0])
        >>> R = quaternion_to_rotation_matrix(q)
    """
    if isinstance(q, np.ndarray):
        return _quaternion_to_rotation_matrix_numpy(q)
    else:
        return _quaternion_to_rotation_matrix_torch(q)


def rotation_matrix_to_quaternion(
    R: np.ndarray | torch.Tensor,
) -> np.ndarray | torch.Tensor:
    """
    Convert 3x3 rotation matrix to quaternion.

    Supports both NumPy arrays and PyTorch tensors with automatic dispatch.

    Args:
        R: Rotation matrix [3, 3]

    Returns:
        Quaternion [4] (w, x, y, z) (same type as input)

    Example:
        >>> # NumPy
        >>> R = np.eye(3)
        >>> q = rotation_matrix_to_quaternion(R)
        >>>
        >>> # PyTorch
        >>> R = torch.eye(3)
        >>> q = rotation_matrix_to_quaternion(R)
    """
    if isinstance(R, np.ndarray):
        return _rotation_matrix_to_quaternion_numpy(R)
    else:
        return _rotation_matrix_to_quaternion_torch(R)


def axis_angle_to_quaternion(
    axis_angle: np.ndarray | torch.Tensor,
) -> np.ndarray | torch.Tensor:
    """
    Convert axis-angle representation to quaternion.

    Supports both NumPy arrays and PyTorch tensors with automatic dispatch.

    Args:
        axis_angle: Axis-angle [3]. Magnitude is angle in radians,
                   direction is rotation axis.

    Returns:
        Quaternion [4] (w, x, y, z) (same type as input)

    Example:
        >>> # NumPy
        >>> axis_angle = np.array([0.0, 0.0, np.pi/2])  # 90° around Z
        >>> q = axis_angle_to_quaternion(axis_angle)
        >>>
        >>> # PyTorch
        >>> axis_angle = torch.tensor([0.0, 0.0, 3.14159/2])
        >>> q = axis_angle_to_quaternion(axis_angle)
    """
    if isinstance(axis_angle, np.ndarray):
        return _axis_angle_to_quaternion_numpy(axis_angle)
    else:
        return _axis_angle_to_quaternion_torch(axis_angle)


def euler_to_quaternion(euler: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    """
    Convert Euler angles to quaternion.

    Supports both NumPy arrays and PyTorch tensors with automatic dispatch.

    Args:
        euler: Euler angles [3] (roll, pitch, yaw) in radians
               roll: rotation around X axis
               pitch: rotation around Y axis
               yaw: rotation around Z axis

    Returns:
        Quaternion [4] (w, x, y, z) (same type as input)

    Example:
        >>> # NumPy
        >>> euler = np.array([0.0, 0.0, np.pi/2])  # 90° yaw
        >>> q = euler_to_quaternion(euler)
        >>>
        >>> # PyTorch
        >>> euler = torch.tensor([0.0, 0.0, 3.14159/2])
        >>> q = euler_to_quaternion(euler)
    """
    if isinstance(euler, np.ndarray):
        return _euler_to_quaternion_numpy(euler)
    else:
        return _euler_to_quaternion_torch(euler)


def quaternion_to_euler(q: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    """
    Convert quaternion to Euler angles.

    Supports both NumPy arrays and PyTorch tensors with automatic dispatch.

    Args:
        q: Quaternion [4] (w, x, y, z)

    Returns:
        Euler angles [3] (roll, pitch, yaw) in radians (same type as input)

    Example:
        >>> # NumPy
        >>> q = np.array([1.0, 0.0, 0.0, 0.0])
        >>> euler = quaternion_to_euler(q)
        >>>
        >>> # PyTorch
        >>> q = torch.tensor([1.0, 0.0, 0.0, 0.0])
        >>> euler = quaternion_to_euler(q)
    """
    if isinstance(q, np.ndarray):
        return _quaternion_to_euler_numpy(q)
    else:
        return _quaternion_to_euler_torch(q)
