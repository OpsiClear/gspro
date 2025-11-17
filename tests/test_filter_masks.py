"""Tests for FilterMasks convenience wrapper."""

import numpy as np
import pytest
from gsply import GSData

from gspro import Filter, FilterMasks


@pytest.fixture
def sample_data():
    """Create sample GSData for testing."""
    np.random.seed(42)
    n = 100
    return GSData(
        means=np.random.randn(n, 3).astype(np.float32) * 2,
        scales=np.random.rand(n, 3).astype(np.float32) * 3,
        quats=np.tile([1, 0, 0, 0], (n, 1)).astype(np.float32),
        opacities=np.random.rand(n).astype(np.float32),
        sh0=np.random.rand(n, 3).astype(np.float32),
        shN=None,
    )


class TestFilterMasksBasic:
    """Test basic FilterMasks functionality."""

    def test_initialization(self, sample_data):
        """Test FilterMasks initialization."""
        masks = FilterMasks(sample_data)

        assert masks.data is sample_data
        assert len(masks) == 0
        assert masks.names is None

    def test_add_filter_layer(self, sample_data):
        """Test adding filter as mask layer."""
        masks = FilterMasks(sample_data)
        filter_obj = Filter().min_opacity(0.5)

        masks.add("opacity", filter_obj)

        assert len(masks) == 1
        assert "opacity" in masks
        assert masks.names == ["opacity"]

    def test_add_multiple_filters(self, sample_data):
        """Test adding multiple filter layers."""
        masks = FilterMasks(sample_data)

        masks.add("opacity", Filter().min_opacity(0.3))
        masks.add("sphere", Filter().within_sphere(radius=0.8))
        masks.add("scale", Filter().max_scale(2.0))

        assert len(masks) == 3
        assert set(masks.names) == {"opacity", "sphere", "scale"}

    def test_remove_layer(self, sample_data):
        """Test removing a mask layer."""
        masks = FilterMasks(sample_data)
        masks.add("opacity", Filter().min_opacity(0.5))
        masks.add("sphere", Filter().within_sphere(radius=0.8))

        masks.remove("opacity")

        assert len(masks) == 1
        assert "opacity" not in masks
        assert "sphere" in masks

    def test_get_layer(self, sample_data):
        """Test retrieving mask layer."""
        masks = FilterMasks(sample_data)
        masks.add("opacity", Filter().min_opacity(0.5))

        mask = masks.get("opacity")

        assert isinstance(mask, np.ndarray)
        assert mask.dtype == bool
        assert len(mask) == len(sample_data)

    def test_getitem(self, sample_data):
        """Test __getitem__ access."""
        masks = FilterMasks(sample_data)
        masks.add("test", Filter().min_opacity(0.5))

        mask1 = masks["test"]
        mask2 = masks.get("test")

        np.testing.assert_array_equal(mask1, mask2)

    def test_contains(self, sample_data):
        """Test 'in' operator."""
        masks = FilterMasks(sample_data)
        masks.add("opacity", Filter().min_opacity(0.5))

        assert "opacity" in masks
        assert "nonexistent" not in masks

    def test_len(self, sample_data):
        """Test len() function."""
        masks = FilterMasks(sample_data)

        assert len(masks) == 0

        masks.add("layer1", Filter().min_opacity(0.5))
        assert len(masks) == 1

        masks.add("layer2", Filter().within_sphere(radius=0.8))
        assert len(masks) == 2

    def test_repr(self, sample_data):
        """Test string representation."""
        masks = FilterMasks(sample_data)

        # Empty
        assert repr(masks) == "FilterMasks(0 layers)"

        # With layers
        masks.add("opacity", Filter().min_opacity(0.5))
        masks.add("sphere", Filter().within_sphere(radius=0.8))

        repr_str = repr(masks)
        assert "FilterMasks(2 layers:" in repr_str
        assert "opacity" in repr_str
        assert "sphere" in repr_str


class TestFilterMasksCombination:
    """Test mask combination operations."""

    def test_combine_and(self, sample_data):
        """Test AND combination."""
        masks = FilterMasks(sample_data)
        masks.add("opacity", Filter().min_opacity(0.3))
        masks.add("sphere", Filter().within_sphere(radius=0.8))

        combined = masks.combine(mode="and")

        # Manually compute expected
        opacity_mask = masks["opacity"]
        sphere_mask = masks["sphere"]
        expected = opacity_mask & sphere_mask

        np.testing.assert_array_equal(combined, expected)

    def test_combine_or(self, sample_data):
        """Test OR combination."""
        masks = FilterMasks(sample_data)
        masks.add("opacity", Filter().min_opacity(0.7))
        masks.add("sphere", Filter().within_sphere(radius=0.3))

        combined = masks.combine(mode="or")

        opacity_mask = masks["opacity"]
        sphere_mask = masks["sphere"]
        expected = opacity_mask | sphere_mask

        np.testing.assert_array_equal(combined, expected)

    def test_combine_specific_layers(self, sample_data):
        """Test combining only specific layers."""
        masks = FilterMasks(sample_data)
        masks.add("layer1", Filter().min_opacity(0.5))
        masks.add("layer2", Filter().within_sphere(radius=0.8))
        masks.add("layer3", Filter().max_scale(2.0))

        combined = masks.combine(layers=["layer1", "layer3"], mode="and")

        expected = masks["layer1"] & masks["layer3"]
        np.testing.assert_array_equal(combined, expected)


class TestFilterMasksApplication:
    """Test applying masks to filter data."""

    def test_apply_and(self, sample_data):
        """Test applying with AND logic."""
        masks = FilterMasks(sample_data)
        masks.add("opacity", Filter().min_opacity(0.5))
        masks.add("sphere", Filter().within_sphere(radius=0.8))

        filtered = masks.apply(mode="and", inplace=False)

        # Should match manual filtering
        combined_mask = masks["opacity"] & masks["sphere"]
        assert len(filtered) == combined_mask.sum()

    def test_apply_or(self, sample_data):
        """Test applying with OR logic."""
        masks = FilterMasks(sample_data)
        masks.add("high_opacity", Filter().min_opacity(0.7))
        masks.add("low_opacity", Filter().min_opacity(0.0).max_scale(0.5))

        filtered = masks.apply(mode="or", inplace=False)

        combined_mask = masks.combine(mode="or")
        assert len(filtered) == combined_mask.sum()

    def test_apply_specific_layers(self, sample_data):
        """Test applying only specific layers."""
        masks = FilterMasks(sample_data)
        masks.add("layer1", Filter().min_opacity(0.5))
        masks.add("layer2", Filter().within_sphere(radius=0.8))
        masks.add("layer3", Filter().max_scale(1.5))

        filtered = masks.apply(layers=["layer1", "layer2"], mode="and", inplace=False)

        expected_mask = masks["layer1"] & masks["layer2"]
        assert len(filtered) == expected_mask.sum()

    def test_apply_inplace(self, sample_data):
        """Test in-place application."""
        original_len = len(sample_data)
        masks = FilterMasks(sample_data)
        masks.add("test", Filter().min_opacity(0.5))

        result = masks.apply(mode="and", inplace=True)

        assert result is sample_data
        assert len(sample_data) < original_len


class TestFilterMasksSummary:
    """Test summary/inspection functionality."""

    def test_summary_empty(self, sample_data, capsys):
        """Test summary with no layers."""
        masks = FilterMasks(sample_data)
        masks.summary()

        captured = capsys.readouterr()
        assert "No mask layers" in captured.out

    def test_summary_with_layers(self, sample_data, capsys):
        """Test summary with multiple layers."""
        masks = FilterMasks(sample_data)
        masks.add("opacity", Filter().min_opacity(0.5))
        masks.add("sphere", Filter().within_sphere(radius=0.8))

        masks.summary()

        captured = capsys.readouterr()
        assert "opacity:" in captured.out
        assert "sphere:" in captured.out
        assert "/100" in captured.out  # Should show counts

    def test_names_property(self, sample_data):
        """Test names property."""
        masks = FilterMasks(sample_data)

        assert masks.names is None

        masks.add("layer1", Filter().min_opacity(0.5))
        masks.add("layer2", Filter().within_sphere(radius=0.8))

        assert masks.names == ["layer1", "layer2"]


class TestFilterMasksIntegration:
    """Test integration with Filter.get_mask()."""

    def test_filter_get_mask_integration(self, sample_data):
        """Test that FilterMasks correctly uses Filter.get_mask()."""
        # Create filter
        filter_obj = Filter().min_opacity(0.5).max_scale(2.0)

        # Add to FilterMasks
        masks = FilterMasks(sample_data)
        masks.add("combined", filter_obj)

        # Compare with direct get_mask()
        direct_mask = filter_obj.get_mask(sample_data)
        from_filtermasks = masks["combined"]

        np.testing.assert_array_equal(from_filtermasks, direct_mask)

    def test_complex_filter_pipeline(self, sample_data):
        """Test with complex filter pipeline."""
        masks = FilterMasks(sample_data)

        # Add complex filters
        masks.add(
            "strict",
            Filter().min_opacity(0.7).max_scale(1.5).within_sphere(radius=0.6)
        )
        masks.add(
            "loose",
            Filter().min_opacity(0.2).max_scale(3.0)
        )

        # Apply with OR (either strict or loose passes)
        filtered = masks.apply(mode="or", inplace=False)

        # Verify results make sense
        strict_count = masks["strict"].sum()
        loose_count = masks["loose"].sum()
        combined_count = len(filtered)

        # OR should have at least as many as the larger individual filter
        assert combined_count >= max(strict_count, loose_count)
        # But no more than the sum (accounting for overlap)
        assert combined_count <= len(sample_data)
