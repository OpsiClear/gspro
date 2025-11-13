"""
Tests for NumPy implementation of geometric transformation functions.
"""

import numpy as np
import pytest

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
def sample_data_numpy():
    """Generate sample Gaussian data for testing (NumPy)."""
    n = 100
    means = np.random.randn(n, 3)
    quaternions = np.random.randn(n, 4)
    quaternions = quaternions / np.linalg.norm(quaternions, axis=1, keepdims=True)
    scales = np.random.rand(n, 3) + 0.1
    return means, quaternions, scales


# ============================================================================
# Translation Tests
# ============================================================================


def test_translate_basic_numpy():
    """Test basic translation with NumPy."""
    means = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    translation = [1.0, 2.0, 3.0]

    result = translate(means, translation)

    assert isinstance(result, np.ndarray)
    expected = np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]])
    np.testing.assert_allclose(result, expected, atol=1e-6)


def test_translate_batched_numpy(sample_data_numpy):
    """Test translation with batched data (NumPy)."""
    means, _, _ = sample_data_numpy
    translation = [10.0, -5.0, 3.0]

    result = translate(means, translation)

    assert isinstance(result, np.ndarray)
    assert result.shape == means.shape
    # Broadcasting comparison - result is [N, 3], translation broadcasts to [N, 3]
    expected_diff = np.tile(np.array(translation), (len(means), 1))
    np.testing.assert_allclose(result - means, expected_diff, atol=1e-6)


# ============================================================================
# Scaling Tests
# ============================================================================


def test_scale_uniform_numpy():
    """Test uniform scaling (NumPy)."""
    means = np.array([[1.0, 2.0, 3.0]])
    scales = np.array([[0.1, 0.2, 0.3]])
    scale_factor = 2.0

    new_means, new_scales = scale(means, scales, scale_factor)

    assert isinstance(new_means, np.ndarray)
    assert isinstance(new_scales, np.ndarray)

    expected_means = np.array([[2.0, 4.0, 6.0]])
    expected_scales = np.array([[0.2, 0.4, 0.6]])

    np.testing.assert_allclose(new_means, expected_means, atol=1e-6)
    np.testing.assert_allclose(new_scales, expected_scales, atol=1e-6)


def test_scale_per_axis_numpy():
    """Test per-axis scaling (NumPy)."""
    means = np.array([[1.0, 2.0, 3.0]])
    scales = np.array([[0.1, 0.2, 0.3]])
    scale_factor = [2.0, 3.0, 0.5]

    new_means, new_scales = scale(means, scales, scale_factor)

    expected_means = np.array([[2.0, 6.0, 1.5]])
    expected_scales = np.array([[0.2, 0.6, 0.15]])

    np.testing.assert_allclose(new_means, expected_means, atol=1e-6)
    np.testing.assert_allclose(new_scales, expected_scales, atol=1e-6)


def test_scale_with_center_numpy():
    """Test scaling around a center point (NumPy)."""
    means = np.array([[2.0, 2.0, 2.0]])
    scales = np.array([[0.1, 0.1, 0.1]])
    scale_factor = 2.0
    center = [1.0, 1.0, 1.0]

    new_means, new_scales = scale(means, scales, scale_factor, center=center)

    expected_means = np.array([[3.0, 3.0, 3.0]])
    expected_scales = np.array([[0.2, 0.2, 0.2]])

    np.testing.assert_allclose(new_means, expected_means, atol=1e-6)
    np.testing.assert_allclose(new_scales, expected_scales, atol=1e-6)


# ============================================================================
# Quaternion Utility Tests
# ============================================================================


def test_quaternion_multiply_numpy():
    """Test quaternion multiplication (NumPy)."""
    q1 = np.array([1.0, 0.0, 0.0, 0.0])
    q2 = np.array([0.7071, 0.0, 0.0, 0.7071])

    result = quaternion_multiply(q1, q2)

    assert isinstance(result, np.ndarray)
    # Result is [1, 4] due to batching, squeeze or compare with expanded q2
    np.testing.assert_allclose(result.squeeze(), q2, atol=1e-4)


def test_quaternion_to_rotation_matrix_numpy():
    """Test quaternion to rotation matrix conversion (NumPy)."""
    q = np.array([1.0, 0.0, 0.0, 0.0])
    R = quaternion_to_rotation_matrix(q)

    assert isinstance(R, np.ndarray)
    expected = np.eye(3)
    np.testing.assert_allclose(R, expected, atol=1e-6)


def test_rotation_matrix_to_quaternion_numpy():
    """Test rotation matrix to quaternion conversion (NumPy)."""
    R = np.eye(3)
    q = rotation_matrix_to_quaternion(R)

    assert isinstance(q, np.ndarray)
    expected = np.array([1.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(q, expected, atol=1e-6)


def test_quaternion_rotation_roundtrip_numpy():
    """Test quaternion <-> rotation matrix roundtrip (NumPy)."""
    q_original = np.random.randn(4)
    q_original = q_original / np.linalg.norm(q_original)

    R = quaternion_to_rotation_matrix(q_original)
    q_reconstructed = rotation_matrix_to_quaternion(R)

    # Quaternions q and -q represent same rotation
    assert np.allclose(q_reconstructed, q_original, atol=1e-5) or np.allclose(
        q_reconstructed, -q_original, atol=1e-5
    )


def test_axis_angle_to_quaternion_numpy():
    """Test axis-angle to quaternion conversion (NumPy)."""
    axis_angle = np.array([0.0, 0.0, 0.0])
    q = axis_angle_to_quaternion(axis_angle)

    assert isinstance(q, np.ndarray)
    expected = np.array([1.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(q, expected, atol=1e-6)


def test_axis_angle_90_degrees_numpy():
    """Test 90 degree rotation around Z axis (NumPy)."""
    axis_angle = np.array([0.0, 0.0, np.pi / 2])
    q = axis_angle_to_quaternion(axis_angle)

    expected = np.array([0.7071, 0.0, 0.0, 0.7071])
    np.testing.assert_allclose(q, expected, atol=1e-4)


def test_euler_to_quaternion_numpy():
    """Test Euler angles to quaternion conversion (NumPy)."""
    euler = np.array([0.0, 0.0, 0.0])
    q = euler_to_quaternion(euler)

    assert isinstance(q, np.ndarray)
    expected = np.array([1.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(q, expected, atol=1e-6)


def test_quaternion_to_euler_numpy():
    """Test quaternion to Euler angles conversion (NumPy)."""
    q = np.array([1.0, 0.0, 0.0, 0.0])
    euler = quaternion_to_euler(q)

    assert isinstance(euler, np.ndarray)
    expected = np.array([0.0, 0.0, 0.0])
    np.testing.assert_allclose(euler, expected, atol=1e-6)


def test_euler_quaternion_roundtrip_numpy():
    """Test Euler <-> quaternion roundtrip (NumPy)."""
    euler_original = np.array([0.5, 0.3, 0.7])

    q = euler_to_quaternion(euler_original)
    euler_reconstructed = quaternion_to_euler(q)

    np.testing.assert_allclose(euler_reconstructed, euler_original, atol=1e-5)


# ============================================================================
# Rotation Tests
# ============================================================================


def test_rotate_quaternion_format_numpy():
    """Test rotation with quaternion format (NumPy)."""
    means = np.array([[1.0, 0.0, 0.0]])
    quaternions = np.array([[1.0, 0.0, 0.0, 0.0]])

    rotation = np.array([0.7071, 0.0, 0.0, 0.7071])

    new_means, new_quats = rotate(means, quaternions, rotation, rotation_format="quaternion")

    assert isinstance(new_means, np.ndarray)
    assert isinstance(new_quats, np.ndarray)

    expected_means = np.array([[0.0, 1.0, 0.0]])
    np.testing.assert_allclose(new_means, expected_means, atol=1e-4)


def test_rotate_euler_format_numpy():
    """Test rotation with Euler angle format (NumPy)."""
    means = np.array([[1.0, 0.0, 0.0]])
    quaternions = np.array([[1.0, 0.0, 0.0, 0.0]])

    rotation = np.array([0.0, 0.0, np.pi / 2])

    new_means, new_quats = rotate(means, quaternions, rotation, rotation_format="euler")

    expected_means = np.array([[0.0, 1.0, 0.0]])
    np.testing.assert_allclose(new_means, expected_means, atol=1e-4)


def test_rotate_with_center_numpy():
    """Test rotation around a center point (NumPy)."""
    means = np.array([[2.0, 0.0, 0.0]])
    quaternions = np.array([[1.0, 0.0, 0.0, 0.0]])

    rotation = np.array([0.7071, 0.0, 0.0, 0.7071])
    center = [1.0, 0.0, 0.0]

    new_means, new_quats = rotate(
        means, quaternions, rotation, center=center, rotation_format="quaternion"
    )

    expected_means = np.array([[1.0, 1.0, 0.0]])
    np.testing.assert_allclose(new_means, expected_means, atol=1e-4)


# ============================================================================
# Combined Transform Tests
# ============================================================================


def test_transform_translation_only_numpy():
    """Test combined transform with translation only (NumPy)."""
    means = np.random.randn(50, 3)
    translation = [1.0, 2.0, 3.0]

    new_means, new_quats, new_scales = transform(means, translation=translation)

    assert isinstance(new_means, np.ndarray)
    expected = means + np.array(translation)
    np.testing.assert_allclose(new_means, expected, atol=1e-6)
    assert new_quats is None
    assert new_scales is None


def test_transform_scale_only_numpy():
    """Test combined transform with scale only (NumPy)."""
    means = np.random.randn(50, 3)
    scales = np.random.rand(50, 3)
    scale_factor = 2.0

    new_means, new_quats, new_scales = transform(means, scales=scales, scale_factor=scale_factor)

    assert isinstance(new_means, np.ndarray)
    np.testing.assert_allclose(new_means, means * 2.0, atol=1e-6)
    np.testing.assert_allclose(new_scales, scales * 2.0, atol=1e-6)
    assert new_quats is None


def test_transform_combined_all_numpy():
    """Test combined transform with all transformations (NumPy)."""
    means = np.array([[1.0, 0.0, 0.0]])
    quaternions = np.array([[1.0, 0.0, 0.0, 0.0]])
    scales = np.array([[0.1, 0.1, 0.1]])

    rotation = np.array([0.7071, 0.0, 0.0, 0.7071])

    new_means, new_quats, new_scales = transform(
        means,
        quaternions=quaternions,
        scales=scales,
        scale_factor=2.0,
        rotation=rotation,
        translation=[1.0, 0.0, 0.0],
    )

    assert isinstance(new_means, np.ndarray)
    expected_means = np.array([[1.0, 2.0, 0.0]])
    np.testing.assert_allclose(new_means, expected_means, atol=1e-4)

    expected_scales = np.array([[0.2, 0.2, 0.2]])
    np.testing.assert_allclose(new_scales, expected_scales, atol=1e-6)


def test_transform_with_center_numpy():
    """Test combined transform with center point (NumPy)."""
    means = np.array([[2.0, 0.0, 0.0]])
    quaternions = np.array([[1.0, 0.0, 0.0, 0.0]])
    scales = np.array([[0.1, 0.1, 0.1]])

    center = [1.0, 0.0, 0.0]
    rotation = np.array([0.7071, 0.0, 0.0, 0.7071])

    new_means, new_quats, new_scales = transform(
        means,
        quaternions=quaternions,
        scales=scales,
        scale_factor=2.0,
        rotation=rotation,
        center=center,
    )

    expected_means = np.array([[1.0, 2.0, 0.0]])
    np.testing.assert_allclose(new_means, expected_means, atol=1e-4)


# ============================================================================
# Edge Cases
# ============================================================================


def test_identity_transform_numpy(sample_data_numpy):
    """Test that identity transforms preserve data (NumPy)."""
    means, quaternions, scales = sample_data_numpy

    rotation = np.array([1.0, 0.0, 0.0, 0.0])

    new_means, new_quats, new_scales = transform(
        means,
        quaternions=quaternions,
        scales=scales,
        scale_factor=1.0,
        rotation=rotation,
        translation=[0.0, 0.0, 0.0],
    )

    np.testing.assert_allclose(new_means, means, atol=1e-5)
    np.testing.assert_allclose(new_scales, scales, atol=1e-5)


def test_large_scale_factor_numpy():
    """Test very large scale factor (NumPy)."""
    means = np.random.randn(10, 3)
    scales = np.random.rand(10, 3)
    scale_factor = 1000.0

    new_means, new_scales = scale(means, scales, scale_factor)

    np.testing.assert_allclose(new_means, means * 1000.0, atol=1e-3)
    np.testing.assert_allclose(new_scales, scales * 1000.0, atol=1e-3)


def test_small_scale_factor_numpy():
    """Test very small scale factor (NumPy)."""
    means = np.random.randn(10, 3)
    scales = np.random.rand(10, 3)
    scale_factor = 0.001

    new_means, new_scales = scale(means, scales, scale_factor)

    np.testing.assert_allclose(new_means, means * 0.001, atol=1e-6)
    np.testing.assert_allclose(new_scales, scales * 0.001, atol=1e-6)


def test_zero_rotation_axis_angle_numpy():
    """Test that zero-length axis-angle gives identity quaternion (NumPy)."""
    axis_angle = np.array([0.0, 0.0, 0.0])
    q = axis_angle_to_quaternion(axis_angle)

    expected = np.array([1.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(q, expected, atol=1e-6)


def test_quaternion_normalization_numpy():
    """Test that quaternions are properly normalized after operations (NumPy)."""
    means = np.random.randn(20, 3)
    quaternions = np.random.randn(20, 4)
    quaternions = quaternions / np.linalg.norm(quaternions, axis=1, keepdims=True)

    rotation = np.array([0.7071, 0.0, 0.0, 0.7071])

    _, new_quats = rotate(means, quaternions, rotation, rotation_format="quaternion")

    # Check all quaternions have unit length
    norms = np.linalg.norm(new_quats, axis=1)
    np.testing.assert_allclose(norms, np.ones_like(norms), atol=1e-5)
