"""
Tests for geometric transformation functions.
"""

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from gslut.transforms import (
    axis_angle_to_quaternion,
    euler_to_quaternion,
    quaternion_multiply,
    quaternion_to_euler,
    quaternion_to_rotation_matrix,
    rotate,
    rotation_matrix_to_quaternion,
    scale,
    transform,
    translate,
)


@pytest.fixture
def device():
    """Test device (CPU for CI compatibility)."""
    return "cpu"


@pytest.fixture
def sample_data(device):
    """Generate sample Gaussian data for testing."""
    n = 100
    means = torch.randn(n, 3, device=device)
    quaternions = torch.randn(n, 4, device=device)
    quaternions = F.normalize(quaternions, p=2, dim=1)
    scales = torch.rand(n, 3, device=device) + 0.1
    return means, quaternions, scales


# ============================================================================
# Translation Tests
# ============================================================================


def test_translate_basic(device):
    """Test basic translation."""
    means = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], device=device)
    translation = [1.0, 2.0, 3.0]

    result = translate(means, translation)

    expected = torch.tensor([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]], device=device)
    assert torch.allclose(result, expected, atol=1e-6)


def test_translate_tensor_input(device):
    """Test translation with tensor input."""
    means = torch.randn(50, 3, device=device)
    translation = torch.tensor([0.5, -0.5, 1.0], device=device)

    result = translate(means, translation)

    expected = means + translation
    assert torch.allclose(result, expected, atol=1e-6)


def test_translate_batched(sample_data):
    """Test translation with batched data."""
    means, _, _ = sample_data
    translation = [10.0, -5.0, 3.0]

    result = translate(means, translation)

    assert result.shape == means.shape
    assert torch.allclose(result - means, torch.tensor(translation, device=means.device), atol=1e-6)


# ============================================================================
# Scaling Tests
# ============================================================================


def test_scale_uniform(device):
    """Test uniform scaling."""
    means = torch.tensor([[1.0, 2.0, 3.0]], device=device)
    scales = torch.tensor([[0.1, 0.2, 0.3]], device=device)
    scale_factor = 2.0

    new_means, new_scales = scale(means, scales, scale_factor)

    expected_means = torch.tensor([[2.0, 4.0, 6.0]], device=device)
    expected_scales = torch.tensor([[0.2, 0.4, 0.6]], device=device)

    assert torch.allclose(new_means, expected_means, atol=1e-6)
    assert torch.allclose(new_scales, expected_scales, atol=1e-6)


def test_scale_per_axis(device):
    """Test per-axis scaling."""
    means = torch.tensor([[1.0, 2.0, 3.0]], device=device)
    scales = torch.tensor([[0.1, 0.2, 0.3]], device=device)
    scale_factor = [2.0, 3.0, 0.5]

    new_means, new_scales = scale(means, scales, scale_factor)

    expected_means = torch.tensor([[2.0, 6.0, 1.5]], device=device)
    expected_scales = torch.tensor([[0.2, 0.6, 0.15]], device=device)

    assert torch.allclose(new_means, expected_means, atol=1e-6)
    assert torch.allclose(new_scales, expected_scales, atol=1e-6)


def test_scale_with_center(device):
    """Test scaling around a center point."""
    means = torch.tensor([[2.0, 2.0, 2.0]], device=device)
    scales = torch.tensor([[0.1, 0.1, 0.1]], device=device)
    scale_factor = 2.0
    center = [1.0, 1.0, 1.0]

    new_means, new_scales = scale(means, scales, scale_factor, center=center)

    # (2,2,2) - (1,1,1) = (1,1,1) -> *2 = (2,2,2) -> +(1,1,1) = (3,3,3)
    expected_means = torch.tensor([[3.0, 3.0, 3.0]], device=device)
    expected_scales = torch.tensor([[0.2, 0.2, 0.2]], device=device)

    assert torch.allclose(new_means, expected_means, atol=1e-6)
    assert torch.allclose(new_scales, expected_scales, atol=1e-6)


def test_scale_batched(sample_data):
    """Test scaling with batched data."""
    means, _, scales = sample_data
    scale_factor = 0.5

    new_means, new_scales = scale(means, scales, scale_factor)

    assert new_means.shape == means.shape
    assert new_scales.shape == scales.shape
    assert torch.allclose(new_means, means * 0.5, atol=1e-6)
    assert torch.allclose(new_scales, scales * 0.5, atol=1e-6)


# ============================================================================
# Quaternion Utility Tests
# ============================================================================


def test_quaternion_multiply(device):
    """Test quaternion multiplication."""
    # Identity quaternion
    q1 = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)
    q2 = torch.tensor([0.7071, 0.0, 0.0, 0.7071], device=device)  # 90° Z rotation

    result = quaternion_multiply(q1, q2)

    assert torch.allclose(result, q2, atol=1e-4)


def test_quaternion_multiply_batched(device):
    """Test batched quaternion multiplication."""
    q1 = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.7071, 0.0, 0.0, 0.7071]], device=device)
    q2 = torch.tensor([[0.7071, 0.0, 0.0, 0.7071], [0.7071, 0.0, 0.0, 0.7071]], device=device)

    result = quaternion_multiply(q1, q2)

    assert result.shape == (2, 4)
    # First: identity * 90°Z = 90°Z
    assert torch.allclose(result[0], q2[0], atol=1e-4)
    # Second: 90°Z * 90°Z = 180°Z
    expected = torch.tensor([0.0, 0.0, 0.0, 1.0], device=device)
    assert torch.allclose(result[1], expected, atol=1e-4)


def test_quaternion_to_rotation_matrix(device):
    """Test quaternion to rotation matrix conversion."""
    # Identity quaternion -> identity matrix
    q = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)
    R = quaternion_to_rotation_matrix(q)

    expected = torch.eye(3, device=device)
    assert torch.allclose(R, expected, atol=1e-6)


def test_rotation_matrix_to_quaternion(device):
    """Test rotation matrix to quaternion conversion."""
    # Identity matrix -> identity quaternion
    R = torch.eye(3, device=device)
    q = rotation_matrix_to_quaternion(R)

    expected = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)
    assert torch.allclose(q, expected, atol=1e-6)


def test_quaternion_rotation_roundtrip(device):
    """Test quaternion <-> rotation matrix roundtrip."""
    # Random quaternion
    q_original = torch.randn(4, device=device)
    q_original = F.normalize(q_original, p=2, dim=0)

    # Convert to matrix and back
    R = quaternion_to_rotation_matrix(q_original)
    q_reconstructed = rotation_matrix_to_quaternion(R)

    # Quaternions q and -q represent same rotation
    assert torch.allclose(q_reconstructed, q_original, atol=1e-5) or torch.allclose(
        q_reconstructed, -q_original, atol=1e-5
    )


def test_axis_angle_to_quaternion(device):
    """Test axis-angle to quaternion conversion."""
    # No rotation
    axis_angle = torch.tensor([0.0, 0.0, 0.0], device=device)
    q = axis_angle_to_quaternion(axis_angle)

    expected = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)
    assert torch.allclose(q, expected, atol=1e-6)


def test_axis_angle_90_degrees(device):
    """Test 90 degree rotation around Z axis."""
    # 90 degrees around Z
    axis_angle = torch.tensor([0.0, 0.0, np.pi / 2], device=device)
    q = axis_angle_to_quaternion(axis_angle)

    expected = torch.tensor([0.7071, 0.0, 0.0, 0.7071], device=device)
    assert torch.allclose(q, expected, atol=1e-4)


def test_euler_to_quaternion(device):
    """Test Euler angles to quaternion conversion."""
    # No rotation
    euler = torch.tensor([0.0, 0.0, 0.0], device=device)
    q = euler_to_quaternion(euler)

    expected = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)
    assert torch.allclose(q, expected, atol=1e-6)


def test_quaternion_to_euler(device):
    """Test quaternion to Euler angles conversion."""
    # Identity quaternion
    q = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)
    euler = quaternion_to_euler(q)

    expected = torch.tensor([0.0, 0.0, 0.0], device=device)
    assert torch.allclose(euler, expected, atol=1e-6)


def test_euler_quaternion_roundtrip(device):
    """Test Euler <-> quaternion roundtrip."""
    # Random Euler angles (avoid gimbal lock)
    euler_original = torch.tensor([0.5, 0.3, 0.7], device=device)

    # Convert to quaternion and back
    q = euler_to_quaternion(euler_original)
    euler_reconstructed = quaternion_to_euler(q)

    assert torch.allclose(euler_reconstructed, euler_original, atol=1e-5)


# ============================================================================
# Rotation Tests
# ============================================================================


def test_rotate_quaternion_format(device):
    """Test rotation with quaternion format."""
    means = torch.tensor([[1.0, 0.0, 0.0]], device=device)
    quaternions = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)

    # 90 degree rotation around Z axis
    rotation = torch.tensor([0.7071, 0.0, 0.0, 0.7071], device=device)

    new_means, new_quats = rotate(means, quaternions, rotation, rotation_format="quaternion")

    # Point at (1,0,0) rotated 90° around Z should be at (0,1,0)
    expected_means = torch.tensor([[0.0, 1.0, 0.0]], device=device)
    assert torch.allclose(new_means, expected_means, atol=1e-4)


def test_rotate_euler_format(device):
    """Test rotation with Euler angle format."""
    means = torch.tensor([[1.0, 0.0, 0.0]], device=device)
    quaternions = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)

    # 90 degree rotation around Z axis (yaw)
    rotation = torch.tensor([0.0, 0.0, np.pi / 2], device=device)

    new_means, new_quats = rotate(means, quaternions, rotation, rotation_format="euler")

    # Point at (1,0,0) rotated 90° around Z should be at (0,1,0)
    expected_means = torch.tensor([[0.0, 1.0, 0.0]], device=device)
    assert torch.allclose(new_means, expected_means, atol=1e-4)


def test_rotate_axis_angle_format(device):
    """Test rotation with axis-angle format."""
    means = torch.tensor([[1.0, 0.0, 0.0]], device=device)
    quaternions = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)

    # 90 degree rotation around Z axis
    rotation = torch.tensor([0.0, 0.0, np.pi / 2], device=device)

    new_means, new_quats = rotate(means, quaternions, rotation, rotation_format="axis_angle")

    # Point at (1,0,0) rotated 90° around Z should be at (0,1,0)
    expected_means = torch.tensor([[0.0, 1.0, 0.0]], device=device)
    assert torch.allclose(new_means, expected_means, atol=1e-4)


def test_rotate_matrix_format(device):
    """Test rotation with rotation matrix format."""
    means = torch.tensor([[1.0, 0.0, 0.0]], device=device)
    quaternions = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)

    # 90 degree rotation around Z axis
    cos_90, sin_90 = 0.0, 1.0
    rotation = torch.tensor(
        [[cos_90, -sin_90, 0.0], [sin_90, cos_90, 0.0], [0.0, 0.0, 1.0]], device=device
    )

    new_means, new_quats = rotate(means, quaternions, rotation, rotation_format="matrix")

    # Point at (1,0,0) rotated 90° around Z should be at (0,1,0)
    expected_means = torch.tensor([[0.0, 1.0, 0.0]], device=device)
    assert torch.allclose(new_means, expected_means, atol=1e-4)


def test_rotate_with_center(device):
    """Test rotation around a center point."""
    means = torch.tensor([[2.0, 0.0, 0.0]], device=device)
    quaternions = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)

    # 90 degree rotation around Z axis
    rotation = torch.tensor([0.7071, 0.0, 0.0, 0.7071], device=device)
    center = [1.0, 0.0, 0.0]

    new_means, new_quats = rotate(
        means, quaternions, rotation, center=center, rotation_format="quaternion"
    )

    # Point at (2,0,0) relative to (1,0,0) is (1,0,0)
    # Rotated 90° around Z: (0,1,0)
    # Add back center: (1,1,0)
    expected_means = torch.tensor([[1.0, 1.0, 0.0]], device=device)
    assert torch.allclose(new_means, expected_means, atol=1e-4)


def test_rotate_batched(sample_data):
    """Test rotation with batched data."""
    means, quaternions, _ = sample_data

    rotation = torch.tensor([0.7071, 0.0, 0.0, 0.7071], device=means.device)

    new_means, new_quats = rotate(means, quaternions, rotation, rotation_format="quaternion")

    assert new_means.shape == means.shape
    assert new_quats.shape == quaternions.shape


def test_rotate_invalid_format(device):
    """Test rotation with invalid format."""
    means = torch.tensor([[1.0, 0.0, 0.0]], device=device)
    quaternions = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)
    rotation = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)

    with pytest.raises(ValueError, match="Unknown rotation format"):
        rotate(means, quaternions, rotation, rotation_format="invalid")


# ============================================================================
# Combined Transform Tests
# ============================================================================


def test_transform_translation_only(device):
    """Test combined transform with translation only."""
    means = torch.randn(50, 3, device=device)
    translation = [1.0, 2.0, 3.0]

    new_means, new_quats, new_scales = transform(means, translation=translation)

    expected = means + torch.tensor(translation, device=device)
    assert torch.allclose(new_means, expected, atol=1e-6)
    assert new_quats is None
    assert new_scales is None


def test_transform_scale_only(device):
    """Test combined transform with scale only."""
    means = torch.randn(50, 3, device=device)
    scales = torch.rand(50, 3, device=device)
    scale_factor = 2.0

    new_means, new_quats, new_scales = transform(means, scales=scales, scale_factor=scale_factor)

    assert torch.allclose(new_means, means * 2.0, atol=1e-6)
    assert torch.allclose(new_scales, scales * 2.0, atol=1e-6)
    assert new_quats is None


def test_transform_rotation_only(device):
    """Test combined transform with rotation only."""
    means = torch.tensor([[1.0, 0.0, 0.0]], device=device)
    quaternions = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)

    # 90 degree rotation around Z
    rotation = torch.tensor([0.7071, 0.0, 0.0, 0.7071], device=device)

    new_means, new_quats, new_scales = transform(means, quaternions=quaternions, rotation=rotation)

    expected_means = torch.tensor([[0.0, 1.0, 0.0]], device=device)
    assert torch.allclose(new_means, expected_means, atol=1e-4)
    assert new_scales is None


def test_transform_combined_all(device):
    """Test combined transform with all transformations."""
    means = torch.tensor([[1.0, 0.0, 0.0]], device=device)
    quaternions = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)
    scales = torch.tensor([[0.1, 0.1, 0.1]], device=device)

    # Scale 2x, rotate 90° Z, translate [1,0,0]
    rotation = torch.tensor([0.7071, 0.0, 0.0, 0.7071], device=device)

    new_means, new_quats, new_scales = transform(
        means,
        quaternions=quaternions,
        scales=scales,
        scale_factor=2.0,
        rotation=rotation,
        translation=[1.0, 0.0, 0.0],
    )

    # Order: scale -> rotate -> translate
    # (1,0,0) -> scale 2x -> (2,0,0)
    # (2,0,0) -> rotate 90°Z -> (0,2,0)
    # (0,2,0) -> translate [1,0,0] -> (1,2,0)
    expected_means = torch.tensor([[1.0, 2.0, 0.0]], device=device)
    assert torch.allclose(new_means, expected_means, atol=1e-4)

    # Scales should be 2x
    expected_scales = torch.tensor([[0.2, 0.2, 0.2]], device=device)
    assert torch.allclose(new_scales, expected_scales, atol=1e-6)


def test_transform_with_center(device):
    """Test combined transform with center point."""
    means = torch.tensor([[2.0, 0.0, 0.0]], device=device)
    quaternions = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)
    scales = torch.tensor([[0.1, 0.1, 0.1]], device=device)

    center = [1.0, 0.0, 0.0]
    rotation = torch.tensor([0.7071, 0.0, 0.0, 0.7071], device=device)

    new_means, new_quats, new_scales = transform(
        means,
        quaternions=quaternions,
        scales=scales,
        scale_factor=2.0,
        rotation=rotation,
        center=center,
    )

    # Scale around (1,0,0): (2,0,0) - (1,0,0) = (1,0,0) -> *2 = (2,0,0) -> +(1,0,0) = (3,0,0)
    # Rotate around (1,0,0): (3,0,0) - (1,0,0) = (2,0,0) -> rot 90°Z -> (0,2,0) -> +(1,0,0) = (1,2,0)
    expected_means = torch.tensor([[1.0, 2.0, 0.0]], device=device)
    assert torch.allclose(new_means, expected_means, atol=1e-4)


def test_transform_error_scale_without_scales(device):
    """Test that providing scale_factor without scales raises error."""
    means = torch.randn(10, 3, device=device)

    with pytest.raises(ValueError, match="scale_factor provided but scales is None"):
        transform(means, scale_factor=2.0)


def test_transform_error_rotation_without_quaternions(device):
    """Test that providing rotation without quaternions raises error."""
    means = torch.randn(10, 3, device=device)
    rotation = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)

    with pytest.raises(ValueError, match="rotation provided but quaternions is None"):
        transform(means, rotation=rotation)


# ============================================================================
# Edge Cases and Numerical Stability
# ============================================================================


def test_identity_transform(sample_data):
    """Test that identity transforms preserve data."""
    means, quaternions, scales = sample_data

    # Identity rotation (w=1, x=y=z=0)
    rotation = torch.tensor([1.0, 0.0, 0.0, 0.0], device=means.device)

    new_means, new_quats, new_scales = transform(
        means,
        quaternions=quaternions,
        scales=scales,
        scale_factor=1.0,
        rotation=rotation,
        translation=[0.0, 0.0, 0.0],
    )

    assert torch.allclose(new_means, means, atol=1e-5)
    assert torch.allclose(new_scales, scales, atol=1e-5)
    # Quaternions should be close (accounting for normalization)
    assert torch.allclose(
        F.normalize(new_quats, p=2, dim=1), F.normalize(quaternions, p=2, dim=1), atol=1e-5
    )


def test_large_scale_factor(device):
    """Test very large scale factor."""
    means = torch.randn(10, 3, device=device)
    scales = torch.rand(10, 3, device=device)
    scale_factor = 1000.0

    new_means, new_scales = scale(means, scales, scale_factor)

    assert torch.allclose(new_means, means * 1000.0, atol=1e-3)
    assert torch.allclose(new_scales, scales * 1000.0, atol=1e-3)


def test_small_scale_factor(device):
    """Test very small scale factor."""
    means = torch.randn(10, 3, device=device)
    scales = torch.rand(10, 3, device=device)
    scale_factor = 0.001

    new_means, new_scales = scale(means, scales, scale_factor)

    assert torch.allclose(new_means, means * 0.001, atol=1e-6)
    assert torch.allclose(new_scales, scales * 0.001, atol=1e-6)


def test_zero_rotation_axis_angle(device):
    """Test that zero-length axis-angle gives identity quaternion."""
    axis_angle = torch.tensor([0.0, 0.0, 0.0], device=device)
    q = axis_angle_to_quaternion(axis_angle)

    expected = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)
    assert torch.allclose(q, expected, atol=1e-6)


def test_quaternion_normalization(device):
    """Test that quaternions are properly normalized after operations."""
    means = torch.randn(20, 3, device=device)
    quaternions = torch.randn(20, 4, device=device)  # Unnormalized
    quaternions = F.normalize(quaternions, p=2, dim=1)

    rotation = torch.tensor([0.7071, 0.0, 0.0, 0.7071], device=device)

    _, new_quats = rotate(means, quaternions, rotation, rotation_format="quaternion")

    # Check all quaternions have unit length
    norms = torch.norm(new_quats, p=2, dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
