"""
Pre-activation utilities for GSData attributes.

Provides a fused Numba kernel that transforms:
  - log-scale triples -> clipped scale factors via exp + clip
  - logit opacities  -> sigmoid opacities in [0, 1]
  - arbitrary quaternions -> normalized unit quaternions
"""

from __future__ import annotations

import logging
from typing import Final

import numpy as np
from gsply import GSData
from numba import njit, prange
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

Float32Array = NDArray[np.float32]

# Default clamp values recommended by rendering pipeline
_DEFAULT_MIN_SCALE: Final[np.float32] = np.float32(1e-4)
_DEFAULT_MAX_SCALE: Final[np.float32] = np.float32(100.0)
_DEFAULT_MIN_NORM: Final[np.float32] = np.float32(1e-8)


@njit(parallel=True, fastmath=True, cache=True, nogil=True)
def _activate_gaussians_numba(
    scales: Float32Array,
    opacities: Float32Array,
    quats: Float32Array,
    min_scale: np.float32,
    max_scale: np.float32,
    min_quat_norm: np.float32,
) -> None:
    """
    Fused attribute activation.

    Args:
        scales: Log-scale values, shape [N, 3]
        opacities: Logit opacities, shape [N]
        quats: Raw quaternions, shape [N, 4]
        min_scale: Minimum clamp value post-exp
        max_scale: Maximum clamp value post-exp
        min_quat_norm: Minimum allowable quaternion norm (safety floor)
    """
    count = scales.shape[0]

    for i in prange(count):
        # ------------------------------------------------------------------
        # Scale activation: exp + clamp
        # ------------------------------------------------------------------
        sx = np.exp(scales[i, 0])
        sy = np.exp(scales[i, 1])
        sz = np.exp(scales[i, 2])

        sx = min(max(sx, min_scale), max_scale)
        sy = min(max(sy, min_scale), max_scale)
        sz = min(max(sz, min_scale), max_scale)

        scales[i, 0] = sx
        scales[i, 1] = sy
        scales[i, 2] = sz

        # ------------------------------------------------------------------
        # Opacity activation: numerically-stable sigmoid
        # ------------------------------------------------------------------
        logit = opacities[i]
        if logit >= 0.0:
            exp_term = np.exp(-logit)
            sigmoid = 1.0 / (1.0 + exp_term)
        else:
            exp_term = np.exp(logit)
            sigmoid = exp_term / (1.0 + exp_term)
        opacities[i] = sigmoid

        # ------------------------------------------------------------------
        # Quaternion activation: normalize with safety floor
        # ------------------------------------------------------------------
        qx = quats[i, 0]
        qy = quats[i, 1]
        qz = quats[i, 2]
        qw = quats[i, 3]

        norm = np.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
        if norm < min_quat_norm:
            quats[i, 0] = np.float32(0.0)
            quats[i, 1] = np.float32(0.0)
            quats[i, 2] = np.float32(0.0)
            quats[i, 3] = np.float32(1.0)
        else:
            inv = 1.0 / norm
            quats[i, 0] = qx * inv
            quats[i, 1] = qy * inv
            quats[i, 2] = qz * inv
            quats[i, 3] = qw * inv


def apply_pre_activations(
    data: GSData,
    *,
    min_scale: float = float(_DEFAULT_MIN_SCALE),
    max_scale: float = float(_DEFAULT_MAX_SCALE),
    min_quat_norm: float = float(_DEFAULT_MIN_NORM),
    inplace: bool = True,
) -> GSData:
    """
    Activate GSData attributes (scales, opacities, quaternions) in a single pass.

    Args:
        data: GSData instance to process
        min_scale: Minimum allowed scale value after exponentiation
        max_scale: Maximum allowed scale value after exponentiation
        min_quat_norm: Norm floor for normalizing quaternions (avoids NaNs)
        inplace: If False, returns a copy before activation

    Returns:
        GSData with activated attributes (either modified in-place or copy)
    """
    if min_scale <= 0:
        raise ValueError("min_scale must be positive to avoid degenerate exponentiation results.")
    if max_scale <= 0 or max_scale < min_scale:
        raise ValueError("max_scale must be positive and >= min_scale.")
    if min_quat_norm <= 0:
        raise ValueError("min_quat_norm must be positive.")

    if not inplace:
        data = data.copy()

    scales = _ensure_float32_contiguous(data.scales, "scales")
    opacities = _ensure_float32_contiguous(data.opacities, "opacities")
    quats = _ensure_float32_contiguous(data.quats, "quats")

    if scales.ndim != 2 or scales.shape[1] != 3:
        raise ValueError("scales must have shape [N, 3].")
    if quats.ndim != 2 or quats.shape[1] != 4:
        raise ValueError("quats must have shape [N, 4].")

    if opacities.ndim == 2 and opacities.shape[1] == 1:
        opacity_view = opacities.reshape(opacities.shape[0])
    elif opacities.ndim == 1:
        opacity_view = opacities
    else:
        raise ValueError("opacities must be 1D or have shape [N, 1].")

    n_gaussians = scales.shape[0]
    if quats.shape[0] != n_gaussians or opacity_view.shape[0] != n_gaussians:
        raise ValueError("scales, opacities, and quats must have matching lengths.")

    _activate_gaussians_numba(
        scales,
        opacity_view,
        quats,
        np.float32(min_scale),
        np.float32(max_scale),
        np.float32(min_quat_norm),
    )

    data.scales = scales
    data.opacities = opacities
    data.quats = quats

    logger.debug(
        "[PreActivation] Activated %d Gaussians (min_scale=%.2e, max_scale=%.2f, min_quat_norm=%.2e)",
        scales.shape[0],
        min_scale,
        max_scale,
        min_quat_norm,
    )

    return data


def _ensure_float32_contiguous(array: Float32Array | None, name: str) -> Float32Array:
    """
    Ensure arrays passed to kernels are float32 and C-contiguous.

    Args:
        array: Array to validate
        name: Attribute name (for error messages)

    Returns:
        Array guaranteed to be float32 and contiguous
    """
    if array is None:
        raise ValueError(f"GSData.{name} is required for pre-activation.")

    if array.dtype != np.float32:
        array = array.astype(np.float32, copy=False)

    if not array.flags["C_CONTIGUOUS"]:
        array = np.ascontiguousarray(array)

    return array


__all__ = ["apply_pre_activations"]
