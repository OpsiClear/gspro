"""Tests for SH and RGB conversion functions."""

import torch

from gslut import SH_C0, get_sh_c0_constant, rgb2sh, sh2rgb


class TestConversions:
    """Test SH and RGB conversion functions."""

    def test_sh_c0_constant(self):
        """Test SH C0 constant value."""
        assert SH_C0 == 0.28209479177387814
        assert get_sh_c0_constant() == SH_C0

    def test_sh2rgb_basic(self):
        """Test basic sh2rgb conversion."""
        sh = torch.tensor([[0.0, 0.0, 0.0]])
        rgb = sh2rgb(sh)

        # SH=0 should map to RGB=0.5 (middle gray)
        expected = torch.tensor([[0.5, 0.5, 0.5]])
        assert torch.allclose(rgb, expected, atol=1e-6)

    def test_rgb2sh_basic(self):
        """Test basic rgb2sh conversion."""
        rgb = torch.tensor([[0.5, 0.5, 0.5]])
        sh = rgb2sh(rgb)

        # RGB=0.5 should map to SH=0
        expected = torch.tensor([[0.0, 0.0, 0.0]])
        assert torch.allclose(sh, expected, atol=1e-6)

    def test_roundtrip_conversion(self):
        """Test that sh2rgb(rgb2sh(x)) == x."""
        rgb = torch.tensor(
            [
                [0.0, 0.0, 0.0],  # Black
                [1.0, 1.0, 1.0],  # White
                [0.5, 0.5, 0.5],  # Gray
                [1.0, 0.0, 0.0],  # Red
                [0.0, 1.0, 0.0],  # Green
                [0.0, 0.0, 1.0],  # Blue
                [0.3, 0.7, 0.2],  # Random
            ]
        )

        sh = rgb2sh(rgb)
        rgb_reconstructed = sh2rgb(sh)

        assert torch.allclose(rgb, rgb_reconstructed, atol=1e-6)

    def test_roundtrip_sh_first(self):
        """Test that rgb2sh(sh2rgb(x)) == x."""
        sh = torch.tensor(
            [
                [-1.77, -1.77, -1.77],
                [1.77, 1.77, 1.77],
                [0.0, 0.0, 0.0],
                [1.0, -0.5, 0.2],
            ]
        )

        rgb = sh2rgb(sh)
        sh_reconstructed = rgb2sh(rgb)

        assert torch.allclose(sh, sh_reconstructed, atol=1e-6)

    def test_batch_processing(self):
        """Test batch processing of conversions."""
        batch_size = 1000
        rgb = torch.rand(batch_size, 3)

        sh = rgb2sh(rgb)
        rgb_reconstructed = sh2rgb(sh)

        assert sh.shape == (batch_size, 3)
        assert rgb_reconstructed.shape == (batch_size, 3)
        assert torch.allclose(rgb, rgb_reconstructed, atol=1e-6)

    def test_rgb_range(self):
        """Test that RGB values stay in [0, 1] range."""
        # Test SH values that should produce valid RGB
        sh = torch.linspace(-1.77, 1.77, 100).unsqueeze(1).expand(-1, 3)

        rgb = sh2rgb(sh)

        # RGB should be in [0, 1] range for reasonable SH values
        assert torch.all(rgb >= 0.0)
        assert torch.all(rgb <= 1.0)

    def test_extreme_rgb_values(self):
        """Test conversion with extreme RGB values."""
        # Black
        rgb_black = torch.tensor([[0.0, 0.0, 0.0]])
        sh_black = rgb2sh(rgb_black)
        expected_sh = torch.tensor([[-1.7725, -1.7725, -1.7725]])
        assert torch.allclose(sh_black, expected_sh, atol=0.01)

        # White
        rgb_white = torch.tensor([[1.0, 1.0, 1.0]])
        sh_white = rgb2sh(rgb_white)
        expected_sh = torch.tensor([[1.7725, 1.7725, 1.7725]])
        assert torch.allclose(sh_white, expected_sh, atol=0.01)

    def test_multidimensional_tensors(self):
        """Test conversions with multidimensional tensors."""
        # Test with shape [H, W, 3] (image-like)
        rgb_image = torch.rand(10, 10, 3)

        sh_image = rgb2sh(rgb_image)
        rgb_reconstructed = sh2rgb(sh_image)

        assert sh_image.shape == (10, 10, 3)
        assert torch.allclose(rgb_image, rgb_reconstructed, atol=1e-6)

    def test_single_value(self):
        """Test conversion with single value."""
        rgb = torch.tensor([[0.7, 0.3, 0.5]])
        sh = rgb2sh(rgb)
        rgb_back = sh2rgb(sh)

        assert rgb.shape == sh.shape == rgb_back.shape
        assert torch.allclose(rgb, rgb_back, atol=1e-6)

    def test_dtype_preservation(self):
        """Test that dtype is preserved in conversions."""
        for dtype in [torch.float32, torch.float64]:
            rgb = torch.tensor([[0.5, 0.5, 0.5]], dtype=dtype)
            sh = rgb2sh(rgb)
            rgb_back = sh2rgb(sh)

            assert sh.dtype == dtype
            assert rgb_back.dtype == dtype

    def test_device_compatibility(self):
        """Test conversions work on CPU."""
        rgb = torch.rand(100, 3)

        sh = rgb2sh(rgb)
        rgb_back = sh2rgb(sh)

        assert sh.device == rgb.device
        assert rgb_back.device == rgb.device
        assert torch.allclose(rgb, rgb_back, atol=1e-6)

    def test_gradient_flow(self):
        """Test that gradients flow through conversions."""
        rgb = torch.tensor([[0.5, 0.7, 0.3]], requires_grad=True)

        sh = rgb2sh(rgb)
        loss = sh.sum()
        loss.backward()

        # Gradient should flow back to rgb
        assert rgb.grad is not None
        assert rgb.grad.shape == rgb.shape

    def test_mathematical_properties(self):
        """Test mathematical properties of the conversion."""
        # Test that the conversion is linear
        rgb1 = torch.tensor([[0.3, 0.4, 0.5]])
        rgb2 = torch.tensor([[0.7, 0.6, 0.5]])

        sh1 = rgb2sh(rgb1)
        sh2 = rgb2sh(rgb2)

        # Average in RGB space
        rgb_avg = (rgb1 + rgb2) / 2
        sh_avg_from_rgb = rgb2sh(rgb_avg)

        # Average in SH space
        sh_avg = (sh1 + sh2) / 2

        # Should be equal due to linearity
        assert torch.allclose(sh_avg_from_rgb, sh_avg, atol=1e-6)
