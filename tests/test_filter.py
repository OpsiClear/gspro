"""
Tests for Gaussian splat filtering system.
"""

import numpy as np
import pytest

from gspro.filter import (
    FilterConfig,
    SceneBounds,
    apply_filter,
    calculate_recommended_max_scale,
    calculate_scene_bounds,
    filter_gaussians,
)


@pytest.fixture
def sample_gaussians():
    """Generate sample Gaussian data for testing."""
    n = 1000
    np.random.seed(42)

    positions = np.random.randn(n, 3).astype(np.float32)
    quaternions = np.random.randn(n, 4).astype(np.float32)
    quaternions = quaternions / np.linalg.norm(quaternions, axis=1, keepdims=True)
    scales = np.random.rand(n, 3).astype(np.float32) * 2.0
    opacities = np.random.rand(n).astype(np.float32)
    colors = np.random.rand(n, 3).astype(np.float32)

    return {
        "positions": positions,
        "quaternions": quaternions,
        "scales": scales,
        "opacities": opacities,
        "colors": colors,
    }


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

    def test_calculate_scene_bounds(self, sample_gaussians):
        """Test basic scene bounds calculation."""
        positions = sample_gaussians["positions"]
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

    def test_scene_bounds_properties(self, sample_gaussians):
        """Test SceneBounds derived properties."""
        positions = sample_gaussians["positions"]
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

    def test_calculate_recommended_max_scale(self, sample_gaussians):
        """Test basic recommended max scale calculation."""
        scales = sample_gaussians["scales"]
        threshold = calculate_recommended_max_scale(scales)

        assert isinstance(threshold, float)
        assert threshold > 0.0

        # Check that it's approximately the 99.5th percentile
        max_scales = scales.max(axis=1)
        expected = np.percentile(max_scales, 99.5)
        assert abs(threshold - expected) < 1e-6

    def test_custom_percentile(self, sample_gaussians):
        """Test with custom percentile."""
        scales = sample_gaussians["scales"]

        threshold_95 = calculate_recommended_max_scale(scales, percentile=95.0)
        threshold_99 = calculate_recommended_max_scale(scales, percentile=99.0)

        assert threshold_95 < threshold_99

    def test_empty_scales(self):
        """Test that empty scales raises error."""
        with pytest.raises(ValueError, match="cannot be empty"):
            calculate_recommended_max_scale(np.array([]))

    def test_invalid_percentile(self, sample_gaussians):
        """Test that invalid percentile raises error."""
        scales = sample_gaussians["scales"]

        with pytest.raises(ValueError, match="percentile"):
            calculate_recommended_max_scale(scales, percentile=-10.0)

        with pytest.raises(ValueError, match="percentile"):
            calculate_recommended_max_scale(scales, percentile=150.0)


# ============================================================================
# apply_filter Tests
# ============================================================================


class TestApplyFilter:
    """Test apply_filter function."""

    def test_no_filtering(self, sample_gaussians):
        """Test that default config applies no filtering."""
        positions = sample_gaussians["positions"]
        config = FilterConfig()

        mask = apply_filter(positions, config=config)

        assert mask.shape == (len(positions),)
        assert mask.dtype == bool
        assert mask.all()  # All True

    def test_opacity_filtering(self, sample_gaussians):
        """Test opacity-based filtering."""
        positions = sample_gaussians["positions"]
        opacities = sample_gaussians["opacities"]

        config = FilterConfig(opacity_threshold=0.5)
        mask = apply_filter(positions, opacities=opacities, config=config)

        # Check that mask is correct
        expected_mask = opacities >= 0.5
        np.testing.assert_array_equal(mask, expected_mask)

    def test_scale_filtering(self, sample_gaussians):
        """Test scale-based filtering."""
        positions = sample_gaussians["positions"]
        scales = sample_gaussians["scales"]

        config = FilterConfig(max_scale=1.0)
        mask = apply_filter(positions, scales=scales, config=config)

        # Check that mask is correct
        max_scales = scales.max(axis=1)
        expected_mask = max_scales <= 1.0
        np.testing.assert_array_equal(mask, expected_mask)

    def test_sphere_filtering(self, sample_gaussians):
        """Test sphere volume filtering."""
        positions = sample_gaussians["positions"]

        config = FilterConfig(
            filter_type="sphere", sphere_center=(0.0, 0.0, 0.0), sphere_radius_factor=0.5
        )
        mask = apply_filter(positions, config=config)

        # Calculate expected mask
        bounds = calculate_scene_bounds(positions)
        radius = bounds.max_size * 0.5 * 0.5
        distances_sq = np.sum(positions**2, axis=1)
        expected_mask = distances_sq <= radius**2

        np.testing.assert_array_equal(mask, expected_mask)

    def test_cuboid_filtering(self, sample_gaussians):
        """Test cuboid volume filtering."""
        positions = sample_gaussians["positions"]

        config = FilterConfig(
            filter_type="cuboid",
            cuboid_center=(0.0, 0.0, 0.0),
            cuboid_size_factor_x=0.5,
            cuboid_size_factor_y=0.5,
            cuboid_size_factor_z=0.5,
        )
        mask = apply_filter(positions, config=config)

        # Calculate expected mask
        bounds = calculate_scene_bounds(positions)
        half_sizes = bounds.sizes * 0.5 * 0.5
        expected_mask = np.all(
            (positions >= -half_sizes) & (positions <= half_sizes), axis=1
        )

        np.testing.assert_array_equal(mask, expected_mask)

    def test_combined_filtering(self, sample_gaussians):
        """Test combined filtering (all filters)."""
        positions = sample_gaussians["positions"]
        opacities = sample_gaussians["opacities"]
        scales = sample_gaussians["scales"]

        config = FilterConfig(
            filter_type="sphere",
            sphere_radius_factor=0.8,
            opacity_threshold=0.3,
            max_scale=1.5,
        )
        mask = apply_filter(
            positions, opacities=opacities, scales=scales, config=config
        )

        # All filters use AND logic, so result should be subset
        opacity_mask = opacities >= 0.3
        scale_mask = scales.max(axis=1) <= 1.5

        assert mask.sum() <= opacity_mask.sum()
        assert mask.sum() <= scale_mask.sum()

    def test_empty_positions(self):
        """Test with empty positions."""
        mask = apply_filter(np.array([]).reshape(0, 3))
        assert mask.shape == (0,)

    def test_invalid_opacities_shape(self, sample_gaussians):
        """Test that mismatched opacities shape raises error."""
        positions = sample_gaussians["positions"]
        opacities = np.random.rand(len(positions) - 10)

        config = FilterConfig(opacity_threshold=0.5)

        with pytest.raises(ValueError, match="doesn't match"):
            apply_filter(positions, opacities=opacities, config=config)

    def test_invalid_scales_shape(self, sample_gaussians):
        """Test that mismatched scales shape raises error."""
        positions = sample_gaussians["positions"]
        scales = np.random.rand(len(positions) - 10, 3)

        config = FilterConfig(max_scale=1.0)

        with pytest.raises(ValueError, match="doesn't match"):
            apply_filter(positions, scales=scales, config=config)


# ============================================================================
# filter_gaussians Tests
# ============================================================================


class TestFilterGaussians:
    """Test filter_gaussians convenience function."""

    def test_filter_all_attributes(self, sample_gaussians):
        """Test filtering all Gaussian attributes."""
        # Test with kwargs (like transform module)
        new_pos, new_quats, new_scales, new_opac, new_colors = filter_gaussians(
            positions=sample_gaussians["positions"],
            quaternions=sample_gaussians["quaternions"],
            scales=sample_gaussians["scales"],
            opacities=sample_gaussians["opacities"],
            colors=sample_gaussians["colors"],
            opacity_threshold=0.5,
        )

        # Check that all attributes have consistent length
        n_filtered = len(new_pos)
        assert len(new_quats) == n_filtered
        assert len(new_scales) == n_filtered
        assert len(new_opac) == n_filtered
        assert len(new_colors) == n_filtered

        # Should have filtered some Gaussians
        assert n_filtered < len(sample_gaussians["positions"])

    def test_filter_partial_attributes(self, sample_gaussians):
        """Test filtering with only some attributes."""
        # Filter with only some attributes (kwargs like transform)
        new_pos, new_quats, new_scales, new_opac, new_colors = filter_gaussians(
            positions=sample_gaussians["positions"],
            opacities=sample_gaussians["opacities"],
            opacity_threshold=0.5,
        )

        # Check return values
        assert new_pos is not None
        assert new_quats is None  # Not provided
        assert new_scales is None  # Not provided
        assert new_opac is not None  # Was provided
        assert new_colors is None  # Not provided

    def test_no_filtering(self, sample_gaussians):
        """Test that default params return all Gaussians."""
        new_pos, *_ = filter_gaussians(positions=sample_gaussians["positions"])

        assert len(new_pos) == len(sample_gaussians["positions"])

    def test_1d_colors(self, sample_gaussians):
        """Test filtering with 1D color arrays (grayscale)."""
        # Create 1D colors (e.g., grayscale or single-channel)
        colors_1d = np.random.rand(len(sample_gaussians["positions"])).astype(
            np.float32
        )

        # Test with 1D colors
        new_pos, new_quats, new_scales, new_opac, new_colors = filter_gaussians(
            positions=sample_gaussians["positions"],
            colors=colors_1d,
            opacity_threshold=0.5,
        )

        # Should work without error
        assert new_colors is not None
        assert new_colors.ndim == 1  # Should preserve 1D shape
        assert len(new_colors) == len(new_pos)

    def test_multichannel_colors(self, sample_gaussians):
        """Test filtering with different color channel counts."""
        for n_channels in [1, 3, 4, 8]:
            colors_nd = np.random.rand(
                len(sample_gaussians["positions"]), n_channels
            ).astype(np.float32)

            # Test with n-channel colors
            new_pos, _, _, _, new_colors = filter_gaussians(
                positions=sample_gaussians["positions"],
                colors=colors_nd,
                opacity_threshold=0.5,
            )

            # Should work without error
            assert new_colors is not None
            assert new_colors.shape[1] == n_channels
            assert len(new_colors) == len(new_pos)


# ============================================================================
# Integration Tests
# ============================================================================


class TestFilteringIntegration:
    """Integration tests for filtering system."""

    def test_realistic_workflow(self, sample_gaussians):
        """Test realistic filtering workflow."""
        # Step 1: Calculate scene bounds
        bounds = calculate_scene_bounds(sample_gaussians["positions"])
        assert bounds.max_size > 0

        # Step 2: Calculate recommended max scale
        threshold = calculate_recommended_max_scale(sample_gaussians["scales"])
        assert threshold > 0

        # Step 3: Apply filtering with kwargs (like transform module)
        new_pos, new_quats, new_scales, new_opac, new_colors = filter_gaussians(
            positions=sample_gaussians["positions"],
            quaternions=sample_gaussians["quaternions"],
            scales=sample_gaussians["scales"],
            opacities=sample_gaussians["opacities"],
            colors=sample_gaussians["colors"],
            filter_type="sphere",
            sphere_center=tuple(bounds.center),
            sphere_radius_factor=0.8,
            opacity_threshold=0.1,
            max_scale=threshold,
            scene_bounds=bounds,
        )

        # Verify results
        assert len(new_pos) < len(sample_gaussians["positions"])
        assert len(new_pos) > 0  # Should keep some Gaussians

    def test_reuse_scene_bounds(self, sample_gaussians):
        """Test that scene bounds can be reused across frames."""
        # Calculate once
        bounds = calculate_scene_bounds(sample_gaussians["positions"])

        # Use for multiple "frames" with same params (kwargs like transform)
        mask1 = apply_filter(
            sample_gaussians["positions"],
            filter_type="sphere",
            sphere_radius_factor=0.5,
            scene_bounds=bounds,
        )
        mask2 = apply_filter(
            sample_gaussians["positions"],
            filter_type="sphere",
            sphere_radius_factor=0.5,
            scene_bounds=bounds,
        )

        # Should give same results
        np.testing.assert_array_equal(mask1, mask2)

    def test_extreme_filtering(self, sample_gaussians):
        """Test extreme filtering (filter everything)."""
        config = FilterConfig(
            filter_type="sphere",
            sphere_radius_factor=0.0,  # Zero radius
        )

        mask = apply_filter(sample_gaussians["positions"], config=config)

        # Should filter everything (or very close to center)
        assert mask.sum() < 10  # Very few if any
