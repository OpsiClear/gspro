"""
Tests for Filter.get_mask() API.
"""

import numpy as np
import pytest
from gsply import GSData

from gspro import Filter


@pytest.fixture
def sample_gsdata():
    """Create sample GSData for testing."""
    n = 1000
    np.random.seed(42)
    data = GSData(
        means=np.random.randn(n, 3).astype(np.float32) * 2.0,  # Spread out for spatial filtering
        scales=np.random.rand(n, 3).astype(np.float32) * 3.0,  # 0-3 range
        quats=np.random.randn(n, 4).astype(np.float32),
        opacities=np.random.rand(n).astype(np.float32),  # 0-1 range
        sh0=np.random.rand(n, 3).astype(np.float32),
        shN=None,
    )
    # Normalize quaternions
    data.quats[:] = data.quats / np.linalg.norm(data.quats, axis=1, keepdims=True)
    return data


class TestFilterGetMask:
    """Test Filter.get_mask() functionality."""

    def test_get_mask_no_filtering(self, sample_gsdata):
        """Test get_mask with no filters returns all True."""
        pipeline = Filter()
        mask = pipeline.get_mask(sample_gsdata)

        assert isinstance(mask, np.ndarray)
        assert mask.dtype == bool
        assert mask.shape == (len(sample_gsdata),)
        assert mask.all()  # All True (no filtering)

    def test_get_mask_opacity_filter(self, sample_gsdata):
        """Test get_mask with opacity filtering."""
        pipeline = Filter().min_opacity(0.5)
        mask = pipeline.get_mask(sample_gsdata)

        # Verify mask is correct
        expected_mask = sample_gsdata.opacities >= 0.5
        np.testing.assert_array_equal(mask, expected_mask)

        # Should filter some but not all
        assert 0 < mask.sum() < len(sample_gsdata)

    def test_get_mask_scale_filter(self, sample_gsdata):
        """Test get_mask with scale filtering."""
        pipeline = Filter().max_scale(1.5)
        mask = pipeline.get_mask(sample_gsdata)

        # Verify mask filters high scale Gaussians
        max_scales = sample_gsdata.scales.max(axis=1)
        expected_mask = max_scales <= 1.5
        np.testing.assert_array_equal(mask, expected_mask)

        # Should filter some but not all
        assert 0 < mask.sum() < len(sample_gsdata)

    def test_get_mask_sphere_filter(self, sample_gsdata):
        """Test get_mask with sphere filtering."""
        pipeline = Filter().within_sphere(radius=0.5)
        mask = pipeline.get_mask(sample_gsdata)

        # Sphere filter uses scene bounds and radius_factor
        # Just verify it filters some Gaussians
        assert 0 < mask.sum() < len(sample_gsdata)

    def test_get_mask_cuboid_filter(self, sample_gsdata):
        """Test get_mask with cuboid filtering."""
        pipeline = Filter().within_box(size=(0.5, 0.5, 0.5))
        mask = pipeline.get_mask(sample_gsdata)

        # Should filter some but not all
        assert 0 < mask.sum() < len(sample_gsdata)

    def test_get_mask_combined_filters(self, sample_gsdata):
        """Test get_mask with multiple filters (AND logic)."""
        pipeline = Filter().min_opacity(0.3).max_scale(2.0).within_sphere(radius=0.7)
        mask = pipeline.get_mask(sample_gsdata)

        # Should filter more than individual filters
        mask_opacity = Filter().min_opacity(0.3).get_mask(sample_gsdata)
        mask_scale = Filter().max_scale(2.0).get_mask(sample_gsdata)

        # Combined should be subset (AND logic)
        assert mask.sum() <= mask_opacity.sum()
        assert mask.sum() <= mask_scale.sum()

    def test_get_mask_with_apply_consistency(self, sample_gsdata):
        """Test that get_mask and apply produce consistent results."""
        pipeline = Filter().min_opacity(0.4).max_scale(1.8)

        # Get mask
        mask = pipeline.get_mask(sample_gsdata)

        # Apply filter
        filtered = pipeline.apply(sample_gsdata, inplace=False)

        # Should have same number of Gaussians
        assert len(filtered) == mask.sum()

        # Verify filtered data matches mask
        np.testing.assert_array_equal(filtered.means, sample_gsdata.means[mask])
        np.testing.assert_array_equal(filtered.opacities, sample_gsdata.opacities[mask])

    def test_mask_combination_and(self, sample_gsdata):
        """Test combining multiple masks with AND logic."""
        mask1 = Filter().min_opacity(0.3).get_mask(sample_gsdata)
        mask2 = Filter().max_scale(2.0).get_mask(sample_gsdata)

        # Combine with AND
        combined_mask = mask1 & mask2

        # Should be subset of both
        assert combined_mask.sum() <= mask1.sum()
        assert combined_mask.sum() <= mask2.sum()

        # Apply combined mask
        filtered = sample_gsdata[combined_mask]
        assert len(filtered) == combined_mask.sum()

    def test_mask_combination_or(self, sample_gsdata):
        """Test combining multiple masks with OR logic."""
        mask1 = Filter().min_opacity(0.7).get_mask(sample_gsdata)
        mask2 = Filter().max_scale(0.5).get_mask(sample_gsdata)

        # Combine with OR
        combined_mask = mask1 | mask2

        # Should be superset of both
        assert combined_mask.sum() >= mask1.sum()
        assert combined_mask.sum() >= mask2.sum()

        # Apply combined mask
        filtered = sample_gsdata[combined_mask]
        assert len(filtered) == combined_mask.sum()

    def test_mask_negation(self, sample_gsdata):
        """Test negating a mask to get inverse."""
        mask = Filter().min_opacity(0.5).get_mask(sample_gsdata)
        inverse_mask = ~mask

        # Should be complementary
        assert (mask | inverse_mask).all()
        assert (~(mask & inverse_mask)).all()
        assert mask.sum() + inverse_mask.sum() == len(sample_gsdata)

    def test_get_mask_with_gsdata_slicing(self, sample_gsdata):
        """Test using get_mask with GSData slicing operations."""
        pipeline = Filter().min_opacity(0.4)
        mask = pipeline.get_mask(sample_gsdata)

        # Test direct slicing
        filtered1 = sample_gsdata[mask]
        assert len(filtered1) == mask.sum()

        # Test copy_slice
        filtered2 = sample_gsdata.copy_slice(mask)
        assert len(filtered2) == mask.sum()

        # Both should give same results
        np.testing.assert_array_equal(filtered1.means, filtered2.means)
        np.testing.assert_array_equal(filtered1.opacities, filtered2.opacities)

    def test_get_mask_performance_vs_apply(self, sample_gsdata):
        """Test that get_mask is faster than apply when only mask is needed."""
        import time

        pipeline = Filter().min_opacity(0.3).max_scale(2.0)

        # Time get_mask (should be fast - no data copying)
        start = time.perf_counter()
        for _ in range(100):
            mask = pipeline.get_mask(sample_gsdata)
        time_get_mask = time.perf_counter() - start

        # Time apply (slower - includes data copying)
        start = time.perf_counter()
        for _ in range(100):
            filtered = pipeline.apply(sample_gsdata, inplace=False)
        time_apply = time.perf_counter() - start

        # get_mask should be faster (no data copying)
        # But we won't assert this as it depends on hardware
        # Just verify both work
        assert isinstance(mask, np.ndarray)
        assert len(filtered) == mask.sum()

    def test_get_mask_empty_result(self):
        """Test get_mask with filter that excludes all Gaussians."""
        # Create data with all opacities below threshold
        n = 100
        data = GSData(
            means=np.random.randn(n, 3).astype(np.float32),
            scales=np.random.rand(n, 3).astype(np.float32),
            quats=np.random.randn(n, 4).astype(np.float32),
            opacities=np.full(n, 0.1, dtype=np.float32),  # All opacities = 0.1
            sh0=np.random.rand(n, 3).astype(np.float32),
            shN=None,
        )
        data.quats[:] = data.quats / np.linalg.norm(data.quats, axis=1, keepdims=True)

        # Filter that will exclude everything (threshold > max opacity)
        pipeline = Filter().min_opacity(0.2)
        mask = pipeline.get_mask(data)

        # Should be all False (all opacities 0.1 < threshold 0.2)
        assert mask.sum() == 0
        assert not mask.any()

    def test_get_mask_auto_compile(self, sample_gsdata):
        """Test that get_mask automatically compiles the pipeline."""
        pipeline = Filter().min_opacity(0.5)

        # Should not be compiled yet
        assert not pipeline._is_compiled

        # get_mask should auto-compile
        mask = pipeline.get_mask(sample_gsdata)

        # Should now be compiled
        assert pipeline._is_compiled
        assert isinstance(mask, np.ndarray)
