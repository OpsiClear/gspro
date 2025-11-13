"""
Tests for high-level pipeline API.
"""

import numpy as np
import pytest
import torch

from gslut import ColorPreset, Pipeline, adjust_colors, apply_preset


@pytest.fixture
def sample_colors():
    """Generate sample RGB colors for testing."""
    return np.random.rand(100, 3).astype(np.float32)


@pytest.fixture
def sample_gaussian_data():
    """Generate sample Gaussian data for transform testing."""
    return {
        "means": np.random.randn(100, 3).astype(np.float32),
        "quaternions": np.random.randn(100, 4).astype(np.float32),
        "scales": np.random.rand(100, 3).astype(np.float32) + 0.1,
    }


# ============================================================================
# Pipeline Tests
# ============================================================================


def test_pipeline_basic(sample_colors):
    """Test basic pipeline creation and execution."""
    pipeline = Pipeline().adjust_colors(brightness=1.2)
    result = pipeline(sample_colors)

    assert result.shape == sample_colors.shape
    assert isinstance(result, np.ndarray)


def test_pipeline_chaining(sample_colors):
    """Test chaining multiple operations."""
    pipeline = Pipeline().adjust_colors(brightness=1.2, contrast=1.1)

    result = pipeline(sample_colors)

    assert result.shape == sample_colors.shape
    assert isinstance(result, np.ndarray)
    # Should be in RGB range
    assert result.min() >= 0.0
    assert result.max() <= 1.5  # Brightness=1.2 can push above 1.0


def test_pipeline_with_transforms(sample_gaussian_data):
    """Test pipeline with geometric transformations."""
    pipeline = Pipeline().transform(scale_factor=2.0, translation=[1.0, 0.0, 0.0])

    result = pipeline(sample_gaussian_data)

    assert isinstance(result, dict)
    assert "means" in result
    assert "scales" in result
    # Transform executes: scale -> rotate -> translate
    # Result: means * scale_factor + translation
    expected_means = sample_gaussian_data["means"] * 2.0 + np.array([1.0, 0.0, 0.0])
    np.testing.assert_allclose(result["means"], expected_means, atol=1e-5)
    # Scales should be doubled
    np.testing.assert_allclose(result["scales"], sample_gaussian_data["scales"] * 2.0, atol=1e-5)


def test_pipeline_custom_operation(sample_colors):
    """Test adding custom operations to pipeline."""
    pipeline = Pipeline().custom(lambda x: x * 2.0).adjust_colors(brightness=1.1)

    result = pipeline(sample_colors)

    assert result.shape == sample_colors.shape
    # Custom operation doubles, then brightness multiplies by 1.1
    # But colors are clamped by LUT, so just check shape
    assert isinstance(result, np.ndarray)


def test_pipeline_reset(sample_colors):
    """Test resetting pipeline."""
    pipeline = Pipeline().adjust_colors(brightness=1.2)
    result1 = pipeline(sample_colors)

    pipeline.reset().adjust_colors(brightness=0.8)
    result2 = pipeline(sample_colors)

    # Results should be different
    assert not np.allclose(result1, result2)


def test_pipeline_pytorch(sample_colors):
    """Test pipeline with PyTorch tensors."""
    colors_torch = torch.from_numpy(sample_colors)
    pipeline = Pipeline().adjust_colors(brightness=1.2)

    result = pipeline(colors_torch)

    assert isinstance(result, torch.Tensor)
    assert result.shape == colors_torch.shape


# ============================================================================
# ColorPreset Tests
# ============================================================================


def test_preset_neutral(sample_colors):
    """Test neutral preset (identity with LUT quantization)."""
    preset = ColorPreset.neutral()
    result = preset.apply(sample_colors)

    # Neutral preset still goes through LUT which has quantization error
    np.testing.assert_allclose(result, sample_colors, atol=0.002)


def test_preset_cinematic(sample_colors):
    """Test cinematic preset."""
    preset = ColorPreset.cinematic()
    result = preset.apply(sample_colors)

    assert result.shape == sample_colors.shape
    assert isinstance(result, np.ndarray)


def test_preset_warm(sample_colors):
    """Test warm preset."""
    preset = ColorPreset.warm()
    result = preset.apply(sample_colors)

    assert result.shape == sample_colors.shape
    # Warm preset should generally increase brightness
    assert result.mean() >= sample_colors.mean() * 0.9


def test_preset_cool(sample_colors):
    """Test cool preset."""
    preset = ColorPreset.cool()
    result = preset.apply(sample_colors)

    assert result.shape == sample_colors.shape


def test_preset_vibrant(sample_colors):
    """Test vibrant preset."""
    preset = ColorPreset.vibrant()
    result = preset.apply(sample_colors)

    assert result.shape == sample_colors.shape


def test_preset_muted(sample_colors):
    """Test muted preset."""
    preset = ColorPreset.muted()
    result = preset.apply(sample_colors)

    assert result.shape == sample_colors.shape


def test_preset_dramatic(sample_colors):
    """Test dramatic preset."""
    preset = ColorPreset.dramatic()
    result = preset.apply(sample_colors)

    assert result.shape == sample_colors.shape


def test_preset_to_pipeline(sample_colors):
    """Test converting preset to pipeline."""
    preset = ColorPreset.cinematic()
    pipeline = preset.to_pipeline()

    # Should be able to chain more operations
    assert isinstance(pipeline, Pipeline)

    # Test that we can actually use it
    result = pipeline(sample_colors)
    assert result.shape == sample_colors.shape


def test_preset_pytorch(sample_colors):
    """Test preset with PyTorch tensors."""
    colors_torch = torch.from_numpy(sample_colors)
    preset = ColorPreset.warm()

    result = preset.apply(colors_torch)

    assert isinstance(result, torch.Tensor)
    assert result.shape == colors_torch.shape


# ============================================================================
# High-level Functional API Tests
# ============================================================================


def test_adjust_colors_basic(sample_colors):
    """Test basic color adjustment."""
    result = adjust_colors(sample_colors, brightness=1.2, contrast=1.1)

    assert result.shape == sample_colors.shape
    assert isinstance(result, np.ndarray)


def test_adjust_colors_pytorch(sample_colors):
    """Test adjust_colors with PyTorch."""
    colors_torch = torch.from_numpy(sample_colors)
    result = adjust_colors(colors_torch, brightness=1.2)

    assert isinstance(result, torch.Tensor)


def test_apply_preset_by_name(sample_colors):
    """Test applying preset by name."""
    result = apply_preset(sample_colors, "cinematic")

    assert result.shape == sample_colors.shape


def test_apply_preset_all_names(sample_colors):
    """Test all preset names."""
    presets = ["neutral", "cinematic", "warm", "cool", "vibrant", "muted", "dramatic"]

    for preset_name in presets:
        result = apply_preset(sample_colors, preset_name)
        assert result.shape == sample_colors.shape


def test_apply_preset_invalid_name(sample_colors):
    """Test invalid preset name raises error."""
    with pytest.raises(ValueError, match="Unknown preset"):
        apply_preset(sample_colors, "nonexistent")


def test_apply_preset_with_instance(sample_colors):
    """Test applying preset instance."""
    preset = ColorPreset.warm()
    result = apply_preset(sample_colors, preset)

    assert result.shape == sample_colors.shape


# ============================================================================
# Integration Tests
# ============================================================================


def test_pipeline_full_workflow():
    """Test complete workflow with pipeline."""
    # Generate test data (RGB colors)
    colors = np.random.rand(1000, 3).astype(np.float32)

    # Create pipeline
    pipeline = Pipeline().adjust_colors(
        temperature=0.6,
        brightness=1.15,
        contrast=1.1,
        saturation=1.2,
    )

    result = pipeline(colors)

    assert result.shape == colors.shape
    assert isinstance(result, np.ndarray)
    assert result.min() >= 0.0


def test_multiple_presets_comparison(sample_colors):
    """Test applying multiple presets and comparing."""
    presets = {
        "warm": ColorPreset.warm(),
        "cool": ColorPreset.cool(),
        "vibrant": ColorPreset.vibrant(),
    }

    results = {}
    for name, preset in presets.items():
        results[name] = preset.apply(sample_colors)

    # All should have same shape
    for result in results.values():
        assert result.shape == sample_colors.shape

    # Results should be different
    assert not np.allclose(results["warm"], results["cool"])


def test_pipeline_device_handling():
    """Test pipeline with different devices."""
    colors = np.random.rand(100, 3).astype(np.float32)

    # CPU pipeline
    pipeline_cpu = Pipeline(device="cpu")
    result_cpu = pipeline_cpu.adjust_colors(brightness=1.2)(colors)

    assert isinstance(result_cpu, np.ndarray)

    # CUDA pipeline (if available)
    if torch.cuda.is_available():
        colors_gpu = torch.from_numpy(colors).cuda()
        pipeline_gpu = Pipeline(device="cuda")
        result_gpu = pipeline_gpu.adjust_colors(brightness=1.2)(colors_gpu)

        assert isinstance(result_gpu, torch.Tensor)
        assert result_gpu.device.type == "cuda"
