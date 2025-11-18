"""Tests for unified Pipeline (composing Color, Transform, and Filter operations)."""

import numpy as np
import pytest
from gsply import GSData

from gspro import Pipeline


@pytest.fixture
def sample_gsdata():
    """Generate sample GSData for testing."""
    rng = np.random.default_rng(42)
    n = 1000

    # Create sample Gaussian data
    means = rng.random((n, 3), dtype=np.float32) * 2 - 1  # [-1, 1]
    scales = rng.random((n, 3), dtype=np.float32) * 0.1
    quats = rng.random((n, 4), dtype=np.float32)
    # Normalize quaternions
    quats = quats / np.linalg.norm(quats, axis=1, keepdims=True)
    opacities = rng.random(n, dtype=np.float32)
    sh0 = rng.random((n, 3), dtype=np.float32)  # RGB colors

    return GSData(
        means=means,
        scales=scales,
        quats=quats,
        opacities=opacities,
        sh0=sh0,
        shN=None,
    )


@pytest.fixture
def sample_gsdata_with_shn():
    """Generate sample GSData with higher-order SH coefficients."""
    rng = np.random.default_rng(42)
    n = 1000

    means = rng.random((n, 3), dtype=np.float32) * 2 - 1
    scales = rng.random((n, 3), dtype=np.float32) * 0.1
    quats = rng.random((n, 4), dtype=np.float32)
    quats = quats / np.linalg.norm(quats, axis=1, keepdims=True)
    opacities = rng.random(n, dtype=np.float32)
    sh0 = rng.random((n, 3), dtype=np.float32)
    shN = rng.random((n, 15, 3), dtype=np.float32)  # Higher-order SH

    return GSData(
        means=means,
        scales=scales,
        quats=quats,
        opacities=opacities,
        sh0=sh0,
        shN=shN,
    )


class TestPipeline:
    """Test unified Pipeline functionality."""

    def test_initialization(self):
        """Test Pipeline initialization."""
        pipeline = Pipeline()

        assert pipeline.has_color is False
        assert pipeline.has_transform is False
        assert pipeline.has_filter is False
        assert len(pipeline) == 0

    def test_method_chaining(self):
        """Test that all methods return self for chaining."""
        pipeline = Pipeline()

        # Test chaining all types of operations
        result = (
            pipeline.brightness(1.2).saturation(1.3).translate([1, 0, 0]).within_sphere(radius=0.8)
        )

        assert result is pipeline

    # ========================================================================
    # Color Operations
    # ========================================================================

    def test_color_operations(self, sample_gsdata):
        """Test color operations through unified pipeline."""
        pipeline = Pipeline().brightness(1.2).saturation(1.3).contrast(1.1)

        assert pipeline.has_color is True
        assert len(pipeline) >= 1

        # Apply to GSData
        result = pipeline(sample_gsdata, inplace=False)

        assert isinstance(result, GSData)
        assert len(result) == len(sample_gsdata)
        # Colors should be modified
        assert not np.allclose(result.sh0, sample_gsdata.sh0)
        # Geometry should be unchanged
        assert np.allclose(result.means, sample_gsdata.means)

    def test_all_color_methods(self, sample_gsdata):
        """Test all color adjustment methods."""
        pipeline = (
            Pipeline()
            .temperature(0.6)
            .brightness(1.2)
            .contrast(1.1)
            .gamma(1.05)
            .saturation(1.3)
            .shadows(1.1)
            .highlights(0.9)
        )

        result = pipeline(sample_gsdata, inplace=False)

        assert not np.allclose(result.sh0, sample_gsdata.sh0)
        assert np.all(result.sh0 >= 0.0)
        assert np.all(result.sh0 <= 1.0)

    # ========================================================================
    # Transform Operations
    # ========================================================================

    def test_transform_operations(self, sample_gsdata):
        """Test transform operations through unified pipeline."""
        pipeline = Pipeline().translate([1, 0, 0]).scale(2.0)

        assert pipeline.has_transform is True
        assert len(pipeline) >= 1

        result = pipeline(sample_gsdata, inplace=False)

        assert isinstance(result, GSData)
        assert len(result) == len(sample_gsdata)
        # Positions should be modified
        assert not np.allclose(result.means, sample_gsdata.means)
        # Scales should be modified
        assert not np.allclose(result.scales, sample_gsdata.scales)

    def test_rotation_transform(self, sample_gsdata):
        """Test rotation transform."""
        # Identity quaternion (no rotation)
        quat = np.array([1, 0, 0, 0], dtype=np.float32)

        pipeline = Pipeline().rotate_quat(quat)

        result = pipeline(sample_gsdata, inplace=False)

        # With identity rotation, positions should be very close
        assert np.allclose(result.means, sample_gsdata.means, atol=1e-5)

    def test_transform_with_center(self, sample_gsdata):
        """Test transform operations with center point."""
        center = np.array([0.5, 0.5, 0.5], dtype=np.float32)

        pipeline = Pipeline().set_center(center).scale(2.0)

        result = pipeline(sample_gsdata, inplace=False)

        # Positions should be scaled around center
        assert not np.allclose(result.means, sample_gsdata.means)

    # ========================================================================
    # Filter Operations
    # ========================================================================

    def test_filter_operations(self, sample_gsdata):
        """Test filter operations through unified pipeline."""
        pipeline = Pipeline().within_sphere(radius=0.5).min_opacity(0.3)

        assert pipeline.has_filter is True
        assert len(pipeline) >= 1

        result = pipeline(sample_gsdata, inplace=False)

        assert isinstance(result, GSData)
        # Some Gaussians should be filtered out
        assert len(result) < len(sample_gsdata)

    def test_sphere_filter(self, sample_gsdata):
        """Test spherical volume filter."""
        pipeline = Pipeline().within_sphere(radius=0.5)

        result = pipeline(sample_gsdata, inplace=False)

        # Should filter out Gaussians outside sphere
        assert len(result) < len(sample_gsdata)

    def test_cuboid_filter(self, sample_gsdata):
        """Test cuboid volume filter."""
        pipeline = Pipeline().within_box(size=[0.5, 0.5, 0.5])

        result = pipeline(sample_gsdata, inplace=False)

        # Should filter out Gaussians outside cuboid
        assert len(result) < len(sample_gsdata)

    def test_opacity_filter(self, sample_gsdata):
        """Test opacity threshold filter."""
        pipeline = Pipeline().min_opacity(0.5)

        result = pipeline(sample_gsdata, inplace=False)

        # Should only keep high-opacity Gaussians
        assert len(result) < len(sample_gsdata)
        assert np.all(result.opacities >= 0.5)

    def test_scale_filter(self, sample_gsdata):
        """Test maximum scale filter."""
        pipeline = Pipeline().max_scale(0.05)

        result = pipeline(sample_gsdata, inplace=False)

        # Should filter out large Gaussians
        assert len(result) <= len(sample_gsdata)
        # All scales should be below threshold
        max_scales = np.max(result.scales, axis=1)
        assert np.all(max_scales <= 0.05)

    def test_filter_preserves_shn(self, sample_gsdata_with_shn):
        """Test that filtering preserves higher-order SH coefficients."""
        pipeline = Pipeline().within_sphere(radius=0.5)

        result = pipeline(sample_gsdata_with_shn, inplace=False)

        # shN should be present and filtered
        assert result.shN is not None
        assert len(result.shN) == len(result)
        assert result.shN.shape == (len(result), 15, 3)

    # ========================================================================
    # Combined Operations
    # ========================================================================

    def test_combined_operations(self, sample_gsdata):
        """Test combining filter, transform, and color operations."""
        quat = np.array([1, 0, 0, 0], dtype=np.float32)

        pipeline = (
            Pipeline()
            .within_sphere(radius=0.8)
            .min_opacity(0.2)
            .rotate_quat(quat)
            .translate([0.5, 0, 0])
            .scale(1.5)
            .brightness(1.2)
            .saturation(1.3)
        )

        result = pipeline(sample_gsdata, inplace=False)

        # Should have all three operation types
        assert pipeline.has_filter is True
        assert pipeline.has_transform is True
        assert pipeline.has_color is True

        # Result should be filtered
        assert len(result) < len(sample_gsdata)
        # Positions should be transformed
        assert not np.allclose(result.means, sample_gsdata.means[: len(result)])
        # Colors should be adjusted
        assert not np.allclose(result.sh0, sample_gsdata.sh0[: len(result)])

    def test_operation_order(self, sample_gsdata):
        """Test that operations are applied in correct order: Filter -> Transform -> Color."""
        pipeline = (
            Pipeline()
            .brightness(1.2)  # Color (added first)
            .within_sphere(radius=0.8)  # Filter (added second)
            .translate([1, 0, 0])  # Transform (added third)
        )

        # Should still execute as: Filter -> Transform -> Color
        result = pipeline(sample_gsdata, inplace=False)

        assert isinstance(result, GSData)
        assert len(result) < len(sample_gsdata)  # Filtered

    def test_empty_pipeline(self, sample_gsdata):
        """Test empty pipeline (no operations)."""
        pipeline = Pipeline()

        result = pipeline(sample_gsdata, inplace=False)

        # Should return unmodified copy
        assert len(result) == len(sample_gsdata)
        assert np.allclose(result.means, sample_gsdata.means)
        assert np.allclose(result.sh0, sample_gsdata.sh0)

    # ========================================================================
    # Inplace vs Copy
    # ========================================================================

    def test_inplace_mode(self, sample_gsdata):
        """Test inplace modification."""
        original_means = sample_gsdata.means.copy()
        original_colors = sample_gsdata.sh0.copy()

        pipeline = Pipeline().translate([1, 0, 0]).brightness(1.2)

        result = pipeline(sample_gsdata, inplace=True)

        # Should return same GSData object
        assert result is sample_gsdata
        # Data should be modified
        assert not np.allclose(sample_gsdata.means, original_means)
        assert not np.allclose(sample_gsdata.sh0, original_colors)

    def test_copy_mode(self, sample_gsdata):
        """Test copy mode (non-inplace)."""
        original_means = sample_gsdata.means.copy()
        original_colors = sample_gsdata.sh0.copy()

        pipeline = Pipeline().translate([1, 0, 0]).brightness(1.2)

        result = pipeline(sample_gsdata, inplace=False)

        # Should return new GSData object
        assert result is not sample_gsdata
        # Original should be unchanged
        assert np.allclose(sample_gsdata.means, original_means)
        assert np.allclose(sample_gsdata.sh0, original_colors)
        # Result should be modified
        assert not np.allclose(result.means, original_means)
        assert not np.allclose(result.sh0, original_colors)

    def test_filter_inplace_replaces_arrays(self, sample_gsdata):
        """Test that filter inplace mode replaces GSData arrays."""
        original_len = len(sample_gsdata)
        original_id = id(sample_gsdata.means)

        pipeline = Pipeline().within_sphere(radius=0.5)

        result = pipeline(sample_gsdata, inplace=True)

        # Should return same GSData object
        assert result is sample_gsdata
        # Length should change (filtering)
        assert len(sample_gsdata) < original_len
        # Array should be replaced (different id)
        assert id(sample_gsdata.means) != original_id

    # ========================================================================
    # Reset and Utilities
    # ========================================================================

    def test_reset(self, sample_gsdata):
        """Test reset functionality."""
        pipeline = Pipeline().brightness(1.2).translate([1, 0, 0]).within_sphere(radius=0.8)

        assert pipeline.has_color is True
        assert pipeline.has_transform is True
        assert pipeline.has_filter is True
        assert len(pipeline) > 0

        # Reset
        pipeline.reset()

        # Should clear all operations
        assert pipeline.has_color is False
        assert pipeline.has_transform is False
        assert pipeline.has_filter is False
        assert len(pipeline) == 0

        # Applying reset pipeline should be no-op
        result = pipeline(sample_gsdata, inplace=False)
        assert len(result) == len(sample_gsdata)

    def test_repr(self):
        """Test string representation."""
        # Empty pipeline
        pipeline = Pipeline()
        repr_str = repr(pipeline)
        assert "Pipeline" in repr_str
        assert "empty" in repr_str

        # Pipeline with operations
        pipeline.brightness(1.2).translate([1, 0, 0]).within_sphere(radius=0.8)
        repr_str = repr(pipeline)
        assert "Pipeline" in repr_str
        assert "filter" in repr_str
        assert "transform" in repr_str
        assert "color" in repr_str

    def test_len(self):
        """Test length calculation."""
        pipeline = Pipeline()
        assert len(pipeline) == 0

        # Add color operation
        pipeline.brightness(1.2)
        assert len(pipeline) == 1

        # Add transform operations
        pipeline.translate([1, 0, 0])
        assert len(pipeline) == 2

        # Add filter operations
        pipeline.within_sphere(radius=0.8)
        assert len(pipeline) == 3

        pipeline.min_opacity(0.1)
        assert len(pipeline) == 4

    def test_apply_method(self, sample_gsdata):
        """Test apply() method (alternative to __call__)."""
        pipeline = Pipeline().brightness(1.2)

        # Both should work identically
        result1 = pipeline.apply(sample_gsdata, inplace=False)
        result2 = pipeline(sample_gsdata, inplace=False)

        assert np.allclose(result1.sh0, result2.sh0)
        assert np.allclose(result1.means, result2.means)

    # ========================================================================
    # Edge Cases
    # ========================================================================

    def test_multiple_same_operation_type(self, sample_gsdata):
        """Test adding multiple operations of the same type."""
        pipeline = Pipeline().brightness(1.2).saturation(1.3).contrast(1.1)

        result = pipeline(sample_gsdata, inplace=False)

        # All operations should be applied
        assert not np.allclose(result.sh0, sample_gsdata.sh0)

    def test_bounds_precomputation(self, sample_gsdata):
        """Test using pre-computed bounds for volume filters."""
        from gspro.filter.bounds import calculate_scene_bounds

        bounds = calculate_scene_bounds(sample_gsdata.means)

        pipeline = Pipeline().bounds(bounds).within_sphere(radius=0.8)

        result = pipeline(sample_gsdata, inplace=False)

        # Should use the pre-computed bounds
        assert len(result) < len(sample_gsdata)

    def test_quaternion_formats(self, sample_gsdata):
        """Test different rotation formats."""
        # Test quaternion
        quat = np.array([1, 0, 0, 0], dtype=np.float32)
        pipeline1 = Pipeline().rotate_quat(quat)
        result1 = pipeline1(sample_gsdata, inplace=False)

        # Test euler angles (identity rotation)
        euler = np.array([0, 0, 0], dtype=np.float32)
        pipeline2 = Pipeline().rotate_euler(euler)
        result2 = pipeline2(sample_gsdata, inplace=False)

        # Both should produce similar results (identity rotation)
        assert np.allclose(result1.means, result2.means, atol=1e-5)

    def test_large_pipeline(self, sample_gsdata):
        """Test a large pipeline with many operations."""
        quat = np.array([1, 0, 0, 0], dtype=np.float32)

        pipeline = (
            Pipeline()
            # Filters
            .within_sphere(radius=0.8)
            .min_opacity(0.1)
            .max_scale(0.1)
            # Transforms
            .translate([0.5, 0, 0])
            .rotate_quat(quat)
            .scale(1.5)
            # Colors
            .temperature(0.6)
            .brightness(1.2)
            .contrast(1.1)
            .gamma(1.05)
            .saturation(1.3)
            .shadows(1.1)
            .highlights(0.9)
        )

        result = pipeline(sample_gsdata, inplace=False)

        # Should apply all operations successfully
        assert isinstance(result, GSData)
        assert len(result) < len(sample_gsdata)  # Filtered
        assert len(pipeline) >= 7  # Many operations (3 filter + 3 transform + 1 color)
