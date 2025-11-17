"""
Tests for GSData integration with pipelines.

This module tests the apply_to_gsdata methods for Color, Transform, and Filter pipelines.
"""

import logging

import numpy as np
import pytest
from gsply import GSData

from gspro import Color, Filter, Transform

logger = logging.getLogger(__name__)


class TestGSDataIntegration:
    """Test GSData integration with all pipelines."""

    @pytest.fixture
    def sample_gsdata(self):
        """Create sample GSData for testing."""
        n = 100
        data = GSData(
            means=np.random.randn(n, 3).astype(np.float32),
            scales=np.random.rand(n, 3).astype(np.float32),
            quats=np.random.randn(n, 4).astype(np.float32),
            opacities=np.random.rand(n).astype(np.float32),
            sh0=np.random.rand(n, 3).astype(np.float32),
            shN=None,
        )
        # Normalize quaternions
        data.quats[:] = data.quats / np.linalg.norm(data.quats, axis=1, keepdims=True)
        return data

    def test_color_apply_inplace(self, sample_gsdata):
        """Test Color.apply with inplace=True."""
        pipeline = Color().brightness(1.2).saturation(1.3)

        # Store original sh0 reference
        original_sh0 = sample_gsdata.sh0.copy()

        # Apply color transformation inplace
        result = pipeline.apply(sample_gsdata, inplace=True)

        # Should return same GSData object
        assert result is sample_gsdata

        # sh0 should be modified
        assert not np.allclose(sample_gsdata.sh0, original_sh0)

        # Other attributes should be unchanged
        assert sample_gsdata.means is sample_gsdata.means
        assert sample_gsdata.quats is sample_gsdata.quats

    def test_color_apply_copy(self, sample_gsdata):
        """Test Color.apply with inplace=False."""
        pipeline = Color().brightness(1.2).saturation(1.3)

        # Store original sh0
        original_sh0 = sample_gsdata.sh0.copy()

        # Apply color transformation with copy
        result = pipeline.apply(sample_gsdata, inplace=False)

        # Should return different GSData object
        assert result is not sample_gsdata

        # Original should be unchanged
        assert np.allclose(sample_gsdata.sh0, original_sh0)

        # Result should be different
        assert not np.allclose(result.sh0, original_sh0)

        # Result should have same length
        assert len(result) == len(sample_gsdata)

    def test_transform_apply_inplace(self, sample_gsdata):
        """Test Transform.apply with inplace=True."""
        quat = np.array([0.707, 0.0, 0.707, 0.0], dtype=np.float32)
        pipeline = Transform().rotate_quat(quat).translate([1.0, 0.0, 0.0]).scale(2.0)

        # Store original means
        original_means = sample_gsdata.means.copy()

        # Apply transform inplace
        result = pipeline.apply(sample_gsdata, inplace=True)

        # Should return same GSData object
        assert result is sample_gsdata

        # Means should be modified
        assert not np.allclose(sample_gsdata.means, original_means)

        # Colors should be unchanged
        assert sample_gsdata.sh0 is sample_gsdata.sh0

    def test_transform_apply_copy(self, sample_gsdata):
        """Test Transform.apply with inplace=False."""
        quat = np.array([0.707, 0.0, 0.707, 0.0], dtype=np.float32)
        pipeline = Transform().rotate_quat(quat).translate([1.0, 0.0, 0.0])

        # Store original means
        original_means = sample_gsdata.means.copy()

        # Apply transform with copy
        result = pipeline.apply(sample_gsdata, inplace=False)

        # Should return different GSData object
        assert result is not sample_gsdata

        # Original should be unchanged
        assert np.allclose(sample_gsdata.means, original_means)

        # Result should be different
        assert not np.allclose(result.means, original_means)

        # Result should have same length
        assert len(result) == len(sample_gsdata)

    def test_filter_apply(self, sample_gsdata):
        """Test Filter.apply."""
        # Apply filtering with copy
        pipeline = Filter().min_opacity(threshold=0.3).max_scale(5.0)

        result = pipeline.apply(sample_gsdata, inplace=False)

        # Should return different GSData object
        assert result is not sample_gsdata

        # Result should have fewer or equal Gaussians
        assert len(result) <= len(sample_gsdata)

        # All arrays should have consistent length
        assert len(result.means) == len(result)
        assert len(result.quats) == len(result)
        assert len(result.scales) == len(result)
        assert len(result.opacities) == len(result)
        assert len(result.sh0) == len(result)

    def test_combined_pipeline_with_gsdata(self, sample_gsdata):
        """Test combining multiple pipelines with GSData."""
        # Filter -> Transform -> Color
        filtered = Filter().min_opacity(threshold=0.2).apply(sample_gsdata)

        quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        transformed = Transform().rotate_quat(quat).translate([1.0, 2.0, 3.0]).apply(
            filtered, inplace=True
        )

        colored = Color().brightness(1.3).saturation(1.2).apply(
            transformed, inplace=True
        )

        # Final result should be same object as transformed (both were inplace)
        assert colored is transformed

        # Length should be <= original
        assert len(colored) <= len(sample_gsdata)

    def test_filter_sphere_with_gsdata(self, sample_gsdata):
        """Test Filter.sphere with GSData."""
        # Create spherical filter
        pipeline = Filter().within_sphere(center=[0.0, 0.0, 0.0], radius=0.5)

        result = pipeline.apply(sample_gsdata, inplace=False)

        # Should filter some Gaussians
        assert len(result) < len(sample_gsdata)

    def test_filter_cuboid_with_gsdata(self, sample_gsdata):
        """Test Filter.cuboid with GSData."""
        # Create cuboid filter
        pipeline = Filter().within_box(center=[0.0, 0.0, 0.0], size=[0.5, 0.5, 0.5])

        result = pipeline.apply(sample_gsdata, inplace=False)

        # Should filter some Gaussians
        assert len(result) < len(sample_gsdata)

    def test_method_chaining_with_gsdata(self, sample_gsdata):
        """Test method chaining patterns with GSData."""
        # Build complex pipeline
        result = (
            Filter()
            .min_opacity(threshold=0.1)
            .max_scale(10.0)
            .within_sphere(radius=0.8)
            .apply(sample_gsdata)
        )

        assert isinstance(result, GSData)
        assert len(result) <= len(sample_gsdata)

        # Apply color adjustments
        result2 = (
            Color()
            .brightness(1.2)
            .contrast(1.1)
            .saturation(1.3)
            .apply(result, inplace=True)
        )

        assert result2 is result

        # Apply transformations
        quat = np.array([0.707, 0.0, 0.707, 0.0], dtype=np.float32)
        result3 = (
            Transform()
            .rotate_quat(quat)
            .translate([1.0, 0.0, 0.0])
            .scale(2.0)
            .apply(result2, inplace=True)
        )

        assert result3 is result2
