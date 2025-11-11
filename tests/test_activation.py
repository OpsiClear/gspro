"""Tests for ActivationLUT."""

import pytest
import torch
import torch.nn.functional as F
from pathlib import Path

from gslut import ActivationLUT


@pytest.fixture
def temp_dir(tmp_path):
    """Create temporary directory for LUT storage."""
    return tmp_path / "lut"


@pytest.fixture
def sample_scales_raw():
    """Generate sample scale values (log-space)."""
    return torch.randn(1000) * 2  # Range roughly [-6, 6]


@pytest.fixture
def sample_opacities_raw():
    """Generate sample opacity values (logit-space)."""
    return torch.randn(1000) * 2  # Range roughly [-6, 6]


@pytest.fixture
def sample_quats_raw():
    """Generate sample quaternion values."""
    quats = torch.randn(1000, 4)
    return F.normalize(quats, p=2, dim=-1)


class TestActivationLUT:
    """Test ActivationLUT functionality."""

    def test_initialization_no_lut(self, temp_dir):
        """Test initialization without existing LUT."""
        lut = ActivationLUT(lut_dir=temp_dir, device="cpu")

        assert not lut.is_loaded
        assert lut.exp_centers is None
        assert lut.sigmoid_centers is None
        assert lut.quat_centers is None

    def test_initialization_no_dir(self):
        """Test initialization without specifying directory."""
        lut = ActivationLUT(device="cpu")

        assert not lut.is_loaded
        assert lut.lut_dir is None

    def test_exp_fallback(self, temp_dir, sample_scales_raw):
        """Test exp fallback when LUT not loaded."""
        lut = ActivationLUT(lut_dir=temp_dir, device="cpu")

        # Should fallback to torch.exp
        result = lut.exp(sample_scales_raw)

        # Compare with standard torch.exp
        expected = torch.exp(sample_scales_raw)

        assert torch.allclose(result, expected)

    def test_sigmoid_fallback(self, temp_dir, sample_opacities_raw):
        """Test sigmoid fallback when LUT not loaded."""
        lut = ActivationLUT(lut_dir=temp_dir, device="cpu")

        result = lut.sigmoid(sample_opacities_raw)

        expected = torch.sigmoid(sample_opacities_raw)

        assert torch.allclose(result, expected)

    def test_normalize_fallback(self, temp_dir, sample_quats_raw):
        """Test normalize fallback when LUT not loaded."""
        lut = ActivationLUT(lut_dir=temp_dir, device="cpu")

        result = lut.normalize(sample_quats_raw, dim=-1)

        expected = F.normalize(sample_quats_raw, p=2, dim=-1)

        assert torch.allclose(result, expected)

    def test_build_from_samples(self, temp_dir):
        """Test building LUT from samples."""
        lut = ActivationLUT(
            lut_dir=temp_dir,
            num_clusters_exp=64,
            num_clusters_sigmoid=64,
            num_clusters_quat=32,
            device="cpu",
        )

        # Generate synthetic samples
        scale_samples = torch.linspace(-5, 5, 1000)
        opacity_samples = torch.linspace(-5, 5, 1000)
        quat_samples = torch.randn(1000, 4)
        quat_samples = F.normalize(quat_samples, p=2, dim=-1)

        # Build LUT
        lut.build_from_samples(
            scale_samples=scale_samples,
            opacity_samples=opacity_samples,
            quat_samples=quat_samples,
        )

        # Check that LUT is built
        assert lut.is_loaded
        assert lut.exp_centers is not None
        assert lut.sigmoid_centers is not None
        assert lut.quat_centers is not None
        assert len(lut.exp_centers) == 64
        assert len(lut.sigmoid_centers) == 64
        assert len(lut.quat_centers) == 32

    def test_exp_lut_accuracy(self, temp_dir):
        """Test exp LUT approximation accuracy with linear interpolation."""
        lut = ActivationLUT(
            lut_dir=temp_dir,
            num_clusters_exp=512,
            device="cpu",
            use_linear_interp=True,
        )

        # Generate synthetic data over realistic range
        samples = torch.linspace(-5, 5, 1000)
        lut.build_from_samples(scale_samples=samples)

        # Test accuracy
        test_values = torch.linspace(-5, 5, 100)
        lut_result = lut.exp(test_values)
        true_result = torch.exp(test_values)

        # Check relative error
        relative_error = torch.abs(lut_result - true_result) / (
            torch.abs(true_result) + 1e-6
        )
        mean_error = relative_error.mean()

        # Should be very accurate with linear interpolation
        assert mean_error < 0.01  # <1% mean error

    def test_sigmoid_lut_accuracy(self, temp_dir):
        """Test sigmoid LUT approximation accuracy."""
        lut = ActivationLUT(
            lut_dir=temp_dir,
            num_clusters_sigmoid=512,
            device="cpu",
            use_linear_interp=True,
        )

        # Generate synthetic data over realistic range
        samples = torch.linspace(-6, 6, 1000)
        lut.build_from_samples(opacity_samples=samples)

        # Test accuracy
        test_values = torch.linspace(-6, 6, 100)
        lut_result = lut.sigmoid(test_values)
        true_result = torch.sigmoid(test_values)

        # Check absolute error (sigmoid output is [0,1])
        abs_error = torch.abs(lut_result - true_result)
        mean_error = abs_error.mean()

        # Should be very accurate with linear interpolation
        assert mean_error < 0.01  # <1% mean error

    def test_nearest_neighbor_mode(self, temp_dir):
        """Test nearest neighbor mode (no interpolation)."""
        lut = ActivationLUT(
            lut_dir=temp_dir,
            num_clusters_exp=128,
            device="cpu",
            use_linear_interp=False,
        )

        samples = torch.linspace(-5, 5, 1000)
        lut.build_from_samples(scale_samples=samples)

        # Test with nearest neighbor
        test_values = torch.linspace(-5, 5, 50)
        lut_result = lut.exp(test_values)
        true_result = torch.exp(test_values)

        # Accuracy should be lower than linear interpolation
        relative_error = torch.abs(lut_result - true_result) / (
            torch.abs(true_result) + 1e-6
        )
        mean_error = relative_error.mean()

        # Should still be reasonably accurate
        assert mean_error < 0.1  # <10% mean error

    def test_save_and_load_lut(self, temp_dir):
        """Test saving and loading LUT."""
        # Create and build LUT
        lut1 = ActivationLUT(
            lut_dir=temp_dir, num_clusters_exp=64, device="cpu"
        )

        samples = torch.randn(1000)
        lut1.build_from_samples(scale_samples=samples)
        lut1.save()

        # Check files were created
        assert (temp_dir / "exp_lut.pt").exists()

        # Load LUT in new instance
        lut2 = ActivationLUT(lut_dir=temp_dir, device="cpu")

        assert lut2.is_loaded
        assert lut2.exp_centers is not None
        assert len(lut2.exp_centers) == 64

    def test_load_explicit(self, temp_dir):
        """Test explicit load() call."""
        lut1 = ActivationLUT(num_clusters_exp=32, device="cpu")

        samples = torch.randn(500)
        lut1.build_from_samples(scale_samples=samples)
        lut1.save(lut_dir=temp_dir)

        # Create new instance without auto-load
        lut2 = ActivationLUT(device="cpu")
        assert not lut2.is_loaded

        # Explicitly load
        success = lut2.load(lut_dir=temp_dir)
        assert success
        assert lut2.is_loaded
        assert len(lut2.exp_centers) == 32

    def test_get_stats(self, temp_dir):
        """Test LUT statistics."""
        lut = ActivationLUT(lut_dir=temp_dir, num_clusters_exp=128, device="cpu")

        # Build minimal LUT
        samples = torch.randn(1000)
        lut.build_from_samples(scale_samples=samples)

        stats = lut.get_stats()

        assert stats["is_loaded"]
        assert stats["exp_clusters"] == 128
        assert "exp_range" in stats
        assert stats["use_linear_interp"]

    def test_partial_lut_build(self, temp_dir):
        """Test building only some LUTs."""
        lut = ActivationLUT(lut_dir=temp_dir, device="cpu")

        # Only build exp LUT
        samples = torch.randn(1000)
        lut.build_from_samples(scale_samples=samples)

        assert lut.is_loaded
        assert lut.exp_centers is not None
        assert lut.sigmoid_centers is None
        assert lut.quat_centers is None

        # Exp should use LUT, sigmoid should fallback
        test_values = torch.randn(10)
        exp_result = lut.exp(test_values)
        sigmoid_result = lut.sigmoid(test_values)

        # Both should return valid results
        assert exp_result.shape == test_values.shape
        assert sigmoid_result.shape == test_values.shape
