"""Tests for ColorLUT."""

import pytest
import torch

from gslut import ColorLUT


@pytest.fixture
def sample_colors():
    """Generate sample RGB colors."""
    return torch.rand(1000, 3)  # Random colors in [0, 1]


@pytest.fixture(params=["cpu"])
def device(request):
    """Test device fixture."""
    return request.param


class TestColorLUT:
    """Test ColorLUT functionality."""

    def test_initialization(self, device):
        """Test ColorLUT initialization."""
        color_lut = ColorLUT(device=device, lut_size=1024)

        assert color_lut.device == device
        assert color_lut.lut_size == 1024
        assert color_lut.r_lut is None
        assert color_lut.g_lut is None
        assert color_lut.b_lut is None

    def test_initialization_cpu_numpy(self):
        """Test CPU initialization uses NumPy mode."""
        color_lut = ColorLUT(device="cpu")
        assert color_lut.use_numpy

    def test_apply_default_parameters(self, device, sample_colors):
        """Test applying with default parameters (no change)."""
        color_lut = ColorLUT(device=device)

        colors = sample_colors.to(device) if device != "cpu" else sample_colors
        result = color_lut.apply(colors)

        # With default parameters, output should be very close to input
        assert torch.allclose(result, colors, atol=0.01)

    def test_apply_temperature(self, device, sample_colors):
        """Test temperature adjustment."""
        color_lut = ColorLUT(device=device)

        colors = sample_colors.to(device) if device != "cpu" else sample_colors

        # Warm temperature (should increase red, decrease blue)
        warm = color_lut.apply(colors, temperature=1.0)
        assert torch.mean(warm[:, 0]) > torch.mean(colors[:, 0])  # More red

        # Cool temperature (should decrease red, increase blue)
        cool = color_lut.apply(colors, temperature=0.0)
        assert torch.mean(cool[:, 0]) < torch.mean(colors[:, 0])  # Less red

    def test_apply_brightness(self, device, sample_colors):
        """Test brightness adjustment."""
        color_lut = ColorLUT(device=device)

        colors = sample_colors.to(device) if device != "cpu" else sample_colors

        # Increase brightness
        bright = color_lut.apply(colors, brightness=1.5)
        assert torch.mean(bright) > torch.mean(colors)

        # Decrease brightness
        dark = color_lut.apply(colors, brightness=0.5)
        assert torch.mean(dark) < torch.mean(colors)

    def test_apply_contrast(self, device, sample_colors):
        """Test contrast adjustment."""
        color_lut = ColorLUT(device=device)

        colors = sample_colors.to(device) if device != "cpu" else sample_colors

        # Increase contrast (should increase range)
        high_contrast = color_lut.apply(colors, contrast=2.0)
        assert torch.std(high_contrast) > torch.std(colors)

        # Decrease contrast (should decrease range)
        low_contrast = color_lut.apply(colors, contrast=0.5)
        assert torch.std(low_contrast) < torch.std(colors)

    def test_apply_gamma(self, device, sample_colors):
        """Test gamma correction."""
        color_lut = ColorLUT(device=device)

        colors = sample_colors.to(device) if device != "cpu" else sample_colors

        # Gamma > 1 (darken)
        dark = color_lut.apply(colors, gamma=2.0)
        assert torch.mean(dark) < torch.mean(colors)

        # Gamma < 1 (brighten)
        bright = color_lut.apply(colors, gamma=0.5)
        assert torch.mean(bright) > torch.mean(colors)

    def test_apply_saturation(self, device):
        """Test saturation adjustment."""
        color_lut = ColorLUT(device=device)

        # Create colorful test image
        colors = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        if device != "cpu":
            colors = colors.to(device)

        # Desaturate (should make colors more gray)
        desaturated = color_lut.apply(colors, saturation=0.0)
        # All channels should be equal (grayscale)
        assert torch.allclose(
            desaturated[:, 0], desaturated[:, 1], atol=0.01
        )
        assert torch.allclose(
            desaturated[:, 1], desaturated[:, 2], atol=0.01
        )

        # Over-saturate
        saturated = color_lut.apply(colors, saturation=2.0)
        # Colors should be more extreme
        assert torch.sum(torch.abs(saturated - 0.5)) > torch.sum(
            torch.abs(colors - 0.5)
        )

    def test_apply_shadows_highlights(self, device):
        """Test shadows and highlights adjustment."""
        color_lut = ColorLUT(device=device)

        # Create gradient from dark to bright
        colors = torch.linspace(0, 1, 100).unsqueeze(1).expand(-1, 3)
        if device != "cpu":
            colors = colors.to(device)

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

        colors = sample_colors.to(device) if device != "cpu" else sample_colors

        # First call compiles LUTs
        result1 = color_lut.apply(colors, temperature=0.7, brightness=1.2)
        assert color_lut.r_lut is not None

        # Second call with same parameters should reuse LUTs
        result2 = color_lut.apply(colors, temperature=0.7, brightness=1.2)
        assert torch.allclose(result1, result2)

        # Call with different parameters should recompile
        result3 = color_lut.apply(colors, temperature=0.8, brightness=1.2)
        assert not torch.allclose(result1, result3)

    def test_reset(self, device, sample_colors):
        """Test LUT cache reset."""
        color_lut = ColorLUT(device=device)

        colors = sample_colors.to(device) if device != "cpu" else sample_colors

        # Build LUTs
        color_lut.apply(colors)
        assert color_lut.r_lut is not None

        # Reset
        color_lut.reset()
        assert color_lut.r_lut is None
        assert color_lut.g_lut is None
        assert color_lut.b_lut is None
        assert color_lut._cached_params is None

    def test_output_range(self, device, sample_colors):
        """Test that output is always in valid [0, 1] range."""
        color_lut = ColorLUT(device=device)

        colors = sample_colors.to(device) if device != "cpu" else sample_colors

        # Apply extreme adjustments
        result = color_lut.apply(
            colors,
            temperature=1.0,
            brightness=2.0,
            contrast=2.0,
            gamma=0.5,
            saturation=2.0,
            shadows=2.0,
            highlights=2.0,
        )

        # Output should still be in [0, 1]
        assert torch.all(result >= 0.0)
        assert torch.all(result <= 1.0)

    def test_identity_transform(self, device, sample_colors):
        """Test that default parameters act as identity."""
        color_lut = ColorLUT(device=device)

        colors = sample_colors.to(device) if device != "cpu" else sample_colors

        result = color_lut.apply(
            colors,
            temperature=0.5,  # Neutral
            brightness=1.0,  # No change
            contrast=1.0,  # No change
            gamma=1.0,  # Linear
            saturation=1.0,  # No change
            shadows=1.0,  # No change
            highlights=1.0,  # No change
        )

        # Should be very close to original
        assert torch.allclose(result, colors, atol=0.01)

    def test_different_lut_sizes(self, device):
        """Test different LUT resolutions."""
        colors = torch.rand(100, 3)
        if device != "cpu":
            colors = colors.to(device)

        # Test different LUT sizes
        for lut_size in [256, 512, 1024, 2048]:
            color_lut = ColorLUT(device=device, lut_size=lut_size)
            result = color_lut.apply(colors, brightness=1.2)

            assert result.shape == colors.shape
            assert torch.all(result >= 0.0)
            assert torch.all(result <= 1.0)

    def test_batch_processing(self, device):
        """Test processing different batch sizes."""
        color_lut = ColorLUT(device=device)

        for batch_size in [1, 10, 100, 1000, 10000]:
            colors = torch.rand(batch_size, 3)
            if device != "cpu":
                colors = colors.to(device)

            result = color_lut.apply(colors, brightness=1.2)

            assert result.shape == (batch_size, 3)
            assert torch.all(result >= 0.0)
            assert torch.all(result <= 1.0)

    def test_numpy_cpu_optimization(self):
        """Test that CPU mode uses NumPy for faster processing."""
        color_lut = ColorLUT(device="cpu")

        colors = torch.rand(1000, 3)
        result = color_lut.apply(colors, brightness=1.2)

        # Should use NumPy internally but return torch.Tensor
        assert isinstance(result, torch.Tensor)
        assert result.shape == colors.shape
