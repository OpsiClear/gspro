"""Tests for GSData pre-activation utilities."""

from __future__ import annotations

import numpy as np
import pytest
from gsply import GSData

from gspro.activations import apply_pre_activations


def _create_test_data(n: int = 512) -> GSData:
    """Create GSData with log-domain attributes for activation tests."""
    rng = np.random.default_rng(123)

    means = rng.standard_normal((n, 3), dtype=np.float32)
    log_scales = rng.normal(loc=-2.0, scale=0.5, size=(n, 3)).astype(np.float32)
    logit_opacities = rng.normal(size=n).astype(np.float32)
    quats = rng.normal(size=(n, 4)).astype(np.float32)

    # Force a degenerate quaternion to exercise safety floor
    quats[0] = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)

    sh0 = rng.random((n, 3), dtype=np.float32)

    return GSData(
        means=means,
        scales=log_scales,
        quats=quats,
        opacities=logit_opacities,
        sh0=sh0,
        shN=None,
    )


def test_apply_pre_activations_matches_numpy():
    """Fused kernel should match standalone NumPy activations."""
    data = _create_test_data()
    reference = GSData(
        means=data.means.copy(),
        scales=data.scales.copy(),
        quats=data.quats.copy(),
        opacities=data.opacities.copy(),
        sh0=data.sh0.copy(),
        shN=data.shN,
    )

    # Manual NumPy activations
    np.exp(reference.scales, out=reference.scales)
    np.clip(reference.scales, 1e-4, 100.0, out=reference.scales)

    np.negative(reference.opacities, out=reference.opacities)
    np.exp(reference.opacities, out=reference.opacities)
    reference.opacities[:] = 1.0 / (1.0 + reference.opacities)
    quat_norms = np.linalg.norm(reference.quats, axis=1, keepdims=True).astype(np.float32)
    small_norms = quat_norms < 1e-8
    quat_norms[small_norms] = 1.0
    reference.quats[:] = reference.quats / quat_norms
    reference.quats[small_norms[:, 0]] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

    apply_pre_activations(data)

    np.testing.assert_allclose(data.scales, reference.scales, rtol=5e-5, atol=5e-5)
    np.testing.assert_allclose(data.opacities, reference.opacities, rtol=5e-5, atol=5e-5)
    np.testing.assert_allclose(data.quats, reference.quats, rtol=5e-5, atol=5e-5)


def test_apply_pre_activations_preserves_inputs_when_copy():
    """inplace=False should keep the original GSData untouched."""
    data = _create_test_data()
    original_scales = data.scales.copy()
    original_opacities = data.opacities.copy()
    original_quats = data.quats.copy()

    result = apply_pre_activations(
        data,
        min_scale=1e-3,
        max_scale=10.0,
        min_quat_norm=1e-6,
        inplace=False,
    )

    # Original arrays untouched
    np.testing.assert_allclose(data.scales, original_scales)
    np.testing.assert_allclose(data.opacities, original_opacities)
    np.testing.assert_allclose(data.quats, original_quats)

    # Copy receives activations
    assert result is not data
    assert not np.allclose(result.scales, original_scales)
    assert not np.allclose(result.opacities, original_opacities)
    assert not np.allclose(result.quats, original_quats)


def test_invalid_parameters_raise():
    """Parameter validation should reject degenerate ranges."""
    data = _create_test_data()

    with pytest.raises(ValueError):
        apply_pre_activations(data, min_scale=-1.0)
    with pytest.raises(ValueError):
        apply_pre_activations(data, min_scale=2.0, max_scale=1.0)
    with pytest.raises(ValueError):
        apply_pre_activations(data, min_quat_norm=0.0)


def test_apply_pre_activations_handles_non_contiguous():
    """Kernel should coerce non-contiguous buffers for best performance."""
    data = _create_test_data(16)
    data.scales = np.asfortranarray(data.scales)  # Non C-contiguous
    data.opacities = data.opacities.reshape(-1, 1)  # 2D view with strides
    data.quats = np.asfortranarray(data.quats)  # Non C-contiguous

    apply_pre_activations(data)

    assert data.scales.flags["C_CONTIGUOUS"]
    assert data.opacities.flags["C_CONTIGUOUS"]
    assert data.quats.flags["C_CONTIGUOUS"]
