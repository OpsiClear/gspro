"""Tests for ColorLUT (NumPy/Numba CPU implementation)."""

import numpy as np
import pytest

from gspro import ColorLUT


@pytest.fixture
def sample_colors():
    """Generate sample RGB colors."""
    rng = np.random.default_rng(42)
    return rng.random((1000, 3), dtype=np.float32)  # Random colors in [0, 1]


@pytest.fixture
def device():
    """Test device fixture (CPU-only)."""
    return "cpu"


class TestColorLUT:
    """Test ColorLUT functionality."""

    def test_initialization(self, device):
        """Test ColorLUT initialization."""
        color_lut = ColorLUT(device=device, lut_size=1024)

        assert color_lut.lut_size == 1024
        assert color_lut.r_lut is None
        assert color_lut.g_lut is None
        assert color_lut.b_lut is None

    def test_apply_default_parameters(self, device, sample_colors):
        """Test applying with default parameters (no change)."""
        color_lut = ColorLUT(device=device)

        result = color_lut.apply(sample_colors)

        # With default parameters, output should be very close to input
        assert np.allclose(result, sample_colors, atol=0.01)

    def test_apply_temperature(self, device, sample_colors):
        """Test temperature adjustment."""
        color_lut = ColorLUT(device=device)

        # Warm temperature (should increase red, decrease blue)
        warm = color_lut.apply(sample_colors, temperature=1.0)
        assert np.mean(warm[:, 0]) > np.mean(sample_colors[:, 0])  # More red

        # Cool temperature (should decrease red, increase blue)
        cool = color_lut.apply(sample_colors, temperature=0.0)
        assert np.mean(cool[:, 0]) < np.mean(sample_colors[:, 0])  # Less red

    def test_apply_brightness(self, device, sample_colors):
        """Test brightness adjustment."""
        color_lut = ColorLUT(device=device)

        # Increase brightness
        bright = color_lut.apply(sample_colors, brightness=1.5)
        assert np.mean(bright) > np.mean(sample_colors)

        # Decrease brightness
        dark = color_lut.apply(sample_colors, brightness=0.5)
        assert np.mean(dark) < np.mean(sample_colors)

    def test_apply_contrast(self, device, sample_colors):
        """Test contrast adjustment."""
        color_lut = ColorLUT(device=device)

        # Increase contrast (should increase range)
        high_contrast = color_lut.apply(sample_colors, contrast=2.0)
        assert np.std(high_contrast) > np.std(sample_colors)

        # Decrease contrast (should decrease range)
        low_contrast = color_lut.apply(sample_colors, contrast=0.5)
        assert np.std(low_contrast) < np.std(sample_colors)

    def test_apply_gamma(self, device, sample_colors):
        """Test gamma correction."""
        color_lut = ColorLUT(device=device)

        # Gamma > 1 (darken)
        dark = color_lut.apply(sample_colors, gamma=2.0)
        assert np.mean(dark) < np.mean(sample_colors)

        # Gamma < 1 (brighten)
        bright = color_lut.apply(sample_colors, gamma=0.5)
        assert np.mean(bright) > np.mean(sample_colors)

    def test_apply_saturation(self, device):
        """Test saturation adjustment."""
        color_lut = ColorLUT(device=device)

        # Create colorful test image
        colors = np.array(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32
        )

        # Desaturate (should make colors more gray)
        desaturated = color_lut.apply(colors, saturation=0.0)
        # All channels should be equal (grayscale)
        assert np.allclose(desaturated[:, 0], desaturated[:, 1], atol=0.01)
        assert np.allclose(desaturated[:, 1], desaturated[:, 2], atol=0.01)

        # Over-saturate
        saturated = color_lut.apply(colors, saturation=2.0)
        # Colors should be more extreme (or at least as extreme for already-saturated colors)
        assert np.sum(np.abs(saturated - 0.5)) >= np.sum(np.abs(colors - 0.5))

    def test_apply_shadows_highlights(self, device):
        """Test shadows and highlights adjustment."""
        color_lut = ColorLUT(device=device)

        # Create gradient from dark to bright
        colors = np.linspace(0, 1, 100, dtype=np.float32)[:, np.newaxis]
        colors = np.repeat(colors, 3, axis=1)

        # Boost shadows (darks should get brighter)
        shadow_boost = color_lut.apply(colors, shadows=1.5)
        dark_region = colors[:20].mean()
        dark_boosted = shadow_boost[:20].mean()
        assert dark_boosted > dark_region

        # Reduce highlights (brights should get darker)
        highlight_reduce = color_lut.apply(colors, highlights=0.5)
        bright_region = colors[-20:].mean()
        bright_reduced = highlight_reduce[-20:].mean()
        assert bright_reduced < bright_region

    def test_lut_caching(self, device, sample_colors):
        """Test that LUTs are cached and reused."""
        color_lut = ColorLUT(device=device)

        # First call compiles LUTs
        result1 = color_lut.apply(sample_colors, temperature=0.7, brightness=1.2)
        assert color_lut.r_lut is not None

        # Second call with same parameters should reuse LUTs
        result2 = color_lut.apply(sample_colors, temperature=0.7, brightness=1.2)
        assert np.allclose(result1, result2)

        # Call with different parameters should recompile
        result3 = color_lut.apply(sample_colors, temperature=0.8, brightness=1.2)
        assert not np.allclose(result1, result3)

    def test_reset(self, device, sample_colors):
        """Test LUT cache reset."""
        color_lut = ColorLUT(device=device)

        # Build LUTs
        color_lut.apply(sample_colors)
        assert color_lut.r_lut is not None

        # Reset
        color_lut.reset()
        assert color_lut.r_lut is None
        assert color_lut.g_lut is None
        assert color_lut.b_lut is None
        assert color_lut._cached_params_numpy is None

    def test_output_range(self, device, sample_colors):
        """Test that output is always in valid [0, 1] range."""
        color_lut = ColorLUT(device=device)

        # Apply extreme adjustments
        result = color_lut.apply(
            sample_colors,
            temperature=1.0,
            brightness=2.0,
            contrast=2.0,
            gamma=0.5,
            saturation=2.0,
            shadows=2.0,
            highlights=2.0,
        )

        # Output should still be in [0, 1]
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    def test_identity_transform(self, device, sample_colors):
        """Test that default parameters act as identity."""
        color_lut = ColorLUT(device=device)

        result = color_lut.apply(
            sample_colors,
            temperature=0.5,  # Neutral
            brightness=1.0,  # No change
            contrast=1.0,  # No change
            gamma=1.0,  # Linear
            saturation=1.0,  # No change
            shadows=1.0,  # No change
            highlights=1.0,  # No change
        )

        # Should be very close to original
        assert np.allclose(result, sample_colors, atol=0.01)

    def test_different_lut_sizes(self, device):
        """Test different LUT resolutions."""
        rng = np.random.default_rng(42)
        colors = rng.random((100, 3), dtype=np.float32)

        # Test different LUT sizes
        for lut_size in [256, 512, 1024, 2048]:
            color_lut = ColorLUT(device=device, lut_size=lut_size)
            result = color_lut.apply(colors, brightness=1.2)

            assert result.shape == colors.shape
            assert np.all(result >= 0.0)
            assert np.all(result <= 1.0)

    def test_batch_processing(self, device):
        """Test processing different batch sizes."""
        color_lut = ColorLUT(device=device)
        rng = np.random.default_rng(42)

        for batch_size in [1, 10, 100, 1000, 10000]:
            colors = rng.random((batch_size, 3), dtype=np.float32)

            result = color_lut.apply(colors, brightness=1.2)

            assert result.shape == (batch_size, 3)
            assert np.all(result >= 0.0)
            assert np.all(result <= 1.0)

    def test_apply_numpy(self):
        """Test the pure NumPy API."""
        color_lut = ColorLUT(device="cpu")
        rng = np.random.default_rng(42)
        colors = rng.random((1000, 3), dtype=np.float32)

        result = color_lut.apply_numpy(colors, brightness=1.2, saturation=1.3)

        # Should return NumPy array
        assert isinstance(result, np.ndarray)
        assert result.shape == colors.shape
        assert result.dtype == np.float32
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    def test_apply_numpy_inplace(self):
        """Test the zero-copy in-place API."""
        color_lut = ColorLUT(device="cpu")
        rng = np.random.default_rng(42)
        colors = rng.random((1000, 3), dtype=np.float32)
        out = np.empty_like(colors)

        # In-place operation
        color_lut.apply_numpy_inplace(colors, out, brightness=1.2, saturation=1.3)

        # Should modify out buffer
        assert isinstance(out, np.ndarray)
        assert out.shape == colors.shape
        assert out.dtype == np.float32
        assert np.all(out >= 0.0)
        assert np.all(out <= 1.0)

        # Result should match regular apply_numpy
        result_regular = color_lut.apply_numpy(colors, brightness=1.2, saturation=1.3)
        assert np.allclose(out, result_regular, atol=1e-6)

    def test_apply_apply_numpy_consistency(self):
        """Test that apply() and apply_numpy() produce same results."""
        color_lut = ColorLUT(device="cpu")
        rng = np.random.default_rng(42)
        colors = rng.random((1000, 3), dtype=np.float32)

        result1 = color_lut.apply(colors, brightness=1.2, saturation=1.3, temperature=0.7)
        result2 = color_lut.apply_numpy(
            colors, brightness=1.2, saturation=1.3, temperature=0.7
        )

        # Both should produce identical results
        assert np.allclose(result1, result2, atol=1e-6)
