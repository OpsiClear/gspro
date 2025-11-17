"""
Tests for Gaussian splat filtering system.
"""

import numpy as np
import pytest
from gsply import GSData

from gspro import Filter, calculate_recommended_max_scale, calculate_scene_bounds
from gspro.filter import FilterConfig, SceneBounds
from gspro.filter.api import _apply_filter  # Internal testing only


@pytest.fixture
def sample_gsdata():
    """Generate sample GSData for testing."""
    n = 1000
    rng = np.random.default_rng(42)

    means = rng.standard_normal((n, 3)).astype(np.float32)
    quats = rng.standard_normal((n, 4)).astype(np.float32)
    quats = quats / np.linalg.norm(quats, axis=1, keepdims=True)
    scales = rng.random((n, 3), dtype=np.float32) * 2.0
    opacities = rng.random(n, dtype=np.float32)
    sh0 = rng.random((n, 3), dtype=np.float32)

    return GSData(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        sh0=sh0,
        shN=None,
    )


# ============================================================================
# FilterConfig Tests
# ============================================================================


class TestFilterConfig:
    """Test FilterConfig dataclass."""

    def test_default_initialization(self):
        """Test default FilterConfig initialization."""
        config = FilterConfig()

        assert config.filter_type == "none"
        assert config.sphere_center == (0.0, 0.0, 0.0)
        assert config.sphere_radius_factor == 1.0
        assert config.cuboid_center == (0.0, 0.0, 0.0)
        assert config.cuboid_size_factor_x == 1.0
        assert config.cuboid_size_factor_y == 1.0
        assert config.cuboid_size_factor_z == 1.0
        assert config.opacity_threshold == 0.05
        assert config.max_scale == 10.0

    def test_custom_initialization(self):
        """Test custom FilterConfig initialization."""
        config = FilterConfig(
            filter_type="sphere",
            sphere_center=(1.0, 2.0, 3.0),
            sphere_radius_factor=0.5,
            opacity_threshold=0.1,
            max_scale=2.5,
        )

        assert config.filter_type == "sphere"
        assert config.sphere_center == (1.0, 2.0, 3.0)
        assert config.sphere_radius_factor == 0.5
        assert config.opacity_threshold == 0.1
        assert config.max_scale == 2.5

    def test_invalid_filter_type(self):
        """Test that invalid filter_type raises error."""
        with pytest.raises(ValueError, match="Invalid filter_type"):
            FilterConfig(filter_type="invalid")

    def test_invalid_sphere_radius_factor(self):
        """Test that invalid sphere_radius_factor raises error."""
        with pytest.raises(ValueError, match="sphere_radius_factor"):
            FilterConfig(sphere_radius_factor=-0.1)

        with pytest.raises(ValueError, match="sphere_radius_factor"):
            FilterConfig(sphere_radius_factor=1.5)

    def test_invalid_opacity_threshold(self):
        """Test that invalid opacity_threshold raises error."""
        with pytest.raises(ValueError, match="opacity_threshold"):
            FilterConfig(opacity_threshold=-0.1)

        with pytest.raises(ValueError, match="opacity_threshold"):
            FilterConfig(opacity_threshold=1.5)

    def test_invalid_max_scale(self):
        """Test that invalid max_scale raises error."""
        with pytest.raises(ValueError, match="max_scale"):
            FilterConfig(max_scale=-1.0)


# ============================================================================
# SceneBounds Tests
# ============================================================================


class TestSceneBounds:
    """Test SceneBounds calculation."""

    def test_calculate_scene_bounds(self, sample_gsdata):
        """Test basic scene bounds calculation."""
        positions = sample_gsdata.means
        bounds = calculate_scene_bounds(positions)

        assert isinstance(bounds, SceneBounds)
        assert bounds.min.shape == (3,)
        assert bounds.max.shape == (3,)
        assert bounds.sizes.shape == (3,)
        assert isinstance(bounds.max_size, float)
        assert bounds.center.shape == (3,)

        # Check that min/max are correct
        np.testing.assert_allclose(bounds.min, positions.min(axis=0), atol=1e-6)
        np.testing.assert_allclose(bounds.max, positions.max(axis=0), atol=1e-6)

    def test_scene_bounds_properties(self, sample_gsdata):
        """Test SceneBounds derived properties."""
        positions = sample_gsdata.means
        bounds = calculate_scene_bounds(positions)

        # Check sizes
        expected_sizes = bounds.max - bounds.min
        np.testing.assert_allclose(bounds.sizes, expected_sizes, atol=1e-6)

        # Check max_size
        expected_max_size = expected_sizes.max()
        assert abs(bounds.max_size - expected_max_size) < 1e-6

        # Check center
        expected_center = (bounds.min + bounds.max) * 0.5
        np.testing.assert_allclose(bounds.center, expected_center, atol=1e-6)

    def test_empty_positions(self):
        """Test that empty positions raises error."""
        with pytest.raises(ValueError, match="cannot be empty"):
            calculate_scene_bounds(np.array([]))

    def test_invalid_shape(self):
        """Test that invalid position shape raises error."""
        with pytest.raises(ValueError, match="must be"):
            calculate_scene_bounds(np.random.randn(10, 2))

        with pytest.raises(ValueError, match="must be"):
            calculate_scene_bounds(np.random.randn(10))


# ============================================================================
# Recommended Max Scale Tests
# ============================================================================


class TestRecommendedMaxScale:
    """Test recommended max scale calculation."""

    def test_calculate_recommended_max_scale(self, sample_gsdata):
        """Test basic recommended max scale calculation."""
        scales = sample_gsdata.scales
        threshold = calculate_recommended_max_scale(scales)

        assert isinstance(threshold, float)
        assert threshold > 0.0

        # Check that it's approximately the 99.5th percentile
        max_scales = scales.max(axis=1)
        expected = np.percentile(max_scales, 99.5)
        assert abs(threshold - expected) < 1e-6

    def test_custom_percentile(self, sample_gsdata):
        """Test with custom percentile."""
        scales = sample_gsdata.scales

        threshold_95 = calculate_recommended_max_scale(scales, percentile=95.0)
        threshold_99 = calculate_recommended_max_scale(scales, percentile=99.0)

        assert threshold_95 < threshold_99

    def test_empty_scales(self):
        """Test that empty scales raises error."""
        with pytest.raises(ValueError, match="cannot be empty"):
            calculate_recommended_max_scale(np.array([]))

    def test_invalid_percentile(self, sample_gsdata):
        """Test that invalid percentile raises error."""
        scales = sample_gsdata.scales

        with pytest.raises(ValueError, match="percentile"):
            calculate_recommended_max_scale(scales, percentile=-10.0)

        with pytest.raises(ValueError, match="percentile"):
            calculate_recommended_max_scale(scales, percentile=150.0)


# ============================================================================
# apply_filter Tests
# ============================================================================


class TestApplyFilter:
    """Test apply_filter function."""

    def test_no_filtering(self, sample_gsdata):
        """Test that default config applies no filtering."""
        positions = sample_gsdata.means
        config = FilterConfig()

        mask = _apply_filter(positions, config=config)

        assert mask.shape == (len(positions),)
        assert mask.dtype == bool
        assert mask.all()  # All True

    def test_opacity_filtering(self, sample_gsdata):
        """Test opacity-based filtering."""
        positions = sample_gsdata.means
        opacities = sample_gsdata.opacities

        config = FilterConfig(opacity_threshold=0.5)
        mask = _apply_filter(positions, opacities=opacities, config=config)

        # Check that mask is correct
        expected_mask = opacities >= 0.5
        np.testing.assert_array_equal(mask, expected_mask)

    def test_scale_filtering(self, sample_gsdata):
        """Test scale-based filtering."""
        positions = sample_gsdata.means
        scales = sample_gsdata.scales

        config = FilterConfig(max_scale=1.0)
        mask = _apply_filter(positions, scales=scales, config=config)

        # Check that mask is correct
        max_scales = scales.max(axis=1)
        expected_mask = max_scales <= 1.0
        np.testing.assert_array_equal(mask, expected_mask)

    def test_sphere_filtering(self, sample_gsdata):
        """Test sphere volume filtering."""
        positions = sample_gsdata.means

        config = FilterConfig(
            filter_type="sphere", sphere_center=(0.0, 0.0, 0.0), sphere_radius_factor=0.5
        )
        mask = _apply_filter(positions, config=config)

        # Calculate expected mask
        bounds = calculate_scene_bounds(positions)
        radius = bounds.max_size * 0.5 * 0.5
        distances_sq = np.sum(positions**2, axis=1)
        expected_mask = distances_sq <= radius**2

        np.testing.assert_array_equal(mask, expected_mask)

    def test_cuboid_filtering(self, sample_gsdata):
        """Test cuboid volume filtering."""
        positions = sample_gsdata.means

        config = FilterConfig(
            filter_type="cuboid",
            cuboid_center=(0.0, 0.0, 0.0),
            cuboid_size_factor_x=0.5,
            cuboid_size_factor_y=0.5,
            cuboid_size_factor_z=0.5,
        )
        mask = _apply_filter(positions, config=config)

        # Calculate expected mask
        bounds = calculate_scene_bounds(positions)
        half_sizes = bounds.sizes * 0.5 * 0.5
        expected_mask = np.all(
            (positions >= -half_sizes) & (positions <= half_sizes), axis=1
        )

        np.testing.assert_array_equal(mask, expected_mask)

    def test_combined_filtering(self, sample_gsdata):
        """Test combined filtering (all filters)."""
        positions = sample_gsdata.means
        opacities = sample_gsdata.opacities
        scales = sample_gsdata.scales

        config = FilterConfig(
            filter_type="sphere",
            sphere_radius_factor=0.8,
            opacity_threshold=0.3,
            max_scale=1.5,
        )
        mask = _apply_filter(
            positions, opacities=opacities, scales=scales, config=config
        )

        # All filters use AND logic, so result should be subset
        opacity_mask = opacities >= 0.3
        scale_mask = scales.max(axis=1) <= 1.5

        assert mask.sum() <= opacity_mask.sum()
        assert mask.sum() <= scale_mask.sum()

    def test_empty_positions(self):
        """Test with empty positions."""
        mask = _apply_filter(np.array([]).reshape(0, 3))
        assert mask.shape == (0,)

    def test_invalid_opacities_shape(self, sample_gsdata):
        """Test that mismatched opacities shape raises error."""
        positions = sample_gsdata.means
        opacities = np.random.rand(len(positions) - 10)

        config = FilterConfig(opacity_threshold=0.5)

        with pytest.raises(ValueError, match="doesn't match"):
            _apply_filter(positions, opacities=opacities, config=config)

    def test_invalid_scales_shape(self, sample_gsdata):
        """Test that mismatched scales shape raises error."""
        positions = sample_gsdata.means
        scales = np.random.rand(len(positions) - 10, 3)

        config = FilterConfig(max_scale=1.0)

        with pytest.raises(ValueError, match="doesn't match"):
            _apply_filter(positions, scales=scales, config=config)


# ============================================================================
# Integration Tests
# ============================================================================


class TestFilteringIntegration:
    """Integration tests for filtering system."""

    def test_realistic_workflow(self, sample_gsdata):
        """Test realistic filtering workflow with GSData."""
        # Step 1: Calculate scene bounds (now accepts GSData directly)
        bounds = calculate_scene_bounds(sample_gsdata)
        assert bounds.max_size > 0

        # Step 2: Calculate recommended max scale (now accepts GSData directly)
        threshold = calculate_recommended_max_scale(sample_gsdata)
        assert threshold > 0

        # Step 3: Apply filtering using Filter pipeline (recommended API)
        filtered = (
            Filter()
            .within_sphere(radius=bounds.max_size * 0.8)
            .min_opacity(0.1)
            .max_scale(threshold)
            (sample_gsdata, inplace=False)
        )

        # Verify results
        assert len(filtered) < len(sample_gsdata)
        assert len(filtered) > 0  # Should keep some Gaussians

    def test_reuse_scene_bounds(self, sample_gsdata):
        """Test that scene bounds can be reused across frames."""
        # Calculate once (now accepts GSData)
        bounds = calculate_scene_bounds(sample_gsdata)

        # Use for multiple "frames" with same params (internal _apply_filter)
        mask1 = _apply_filter(
            sample_gsdata.means,
            filter_type="sphere",
            sphere_radius_factor=0.5,
            scene_bounds=bounds,
        )
        mask2 = _apply_filter(
            sample_gsdata.means,
            filter_type="sphere",
            sphere_radius_factor=0.5,
            scene_bounds=bounds,
        )

        # Should give same results
        np.testing.assert_array_equal(mask1, mask2)

    def test_extreme_filtering(self, sample_gsdata):
        """Test extreme filtering (filter everything)."""
        config = FilterConfig(
            filter_type="sphere",
            sphere_radius_factor=0.0,  # Zero radius
        )

        mask = _apply_filter(sample_gsdata.means, config=config)

        # Should filter everything (or very close to center)
        assert mask.sum() < 10  # Very few if any
