"""Tests for Color (chained color operations with LUT compilation)."""

import numpy as np
import pytest
from gsply import GSData

from gspro.color.pipeline import Color


@pytest.fixture
def sample_gsdata():
    """Generate sample GSData for testing."""
    rng = np.random.default_rng(42)
    n = 10000

    # Create sample Gaussian data
    means = rng.random((n, 3), dtype=np.float32)
    scales = rng.random((n, 3), dtype=np.float32) * 0.1
    quats = rng.random((n, 4), dtype=np.float32)
    quats = quats / np.linalg.norm(quats, axis=1, keepdims=True)
    opacities = rng.random(n, dtype=np.float32)
    sh0 = rng.random((n, 3), dtype=np.float32)  # RGB colors

    return GSData(
        means=means,
        scales=scales,
        quats=quats,
        opacities=opacities,
        sh0=sh0,
        shN=None,
    )


class TestColor:
    """Test Color functionality."""

    def test_initialization(self):
        """Test Color initialization."""
        pipeline = Color(lut_size=1024)

        assert pipeline.lut_size == 1024
        assert pipeline._compiled_lut is None
        assert pipeline._is_dirty

    def test_method_chaining(self):
        """Test that methods return self for chaining."""
        pipeline = Color()

        # Test chaining
        result = pipeline.temperature(0.6).brightness(1.2).contrast(1.1).gamma(1.05).saturation(1.3)

        assert result is pipeline

    def test_compilation(self, sample_gsdata):
        """Test that compilation creates LUT."""
        pipeline = Color()

        # Before compilation
        assert pipeline._compiled_lut is None
        assert pipeline._is_dirty

        # Add operations and compile
        pipeline.brightness(1.2).contrast(1.1).compile()

        # After compilation
        assert pipeline._compiled_lut is not None
        assert not pipeline._is_dirty
        assert pipeline._compiled_lut.shape == (1024, 3)  # Interleaved LUT

    def test_auto_compilation_on_apply(self, sample_gsdata):
        """Test that apply() auto-compiles if needed."""
        pipeline = Color()
        pipeline.brightness(1.2).contrast(1.1)

        # Not compiled yet
        assert pipeline._compiled_lut is None

        # Apply triggers compilation
        pipeline.apply(sample_gsdata, inplace=False)

        # Now compiled
        assert pipeline._compiled_lut is not None
        assert not pipeline._is_dirty

    def test_recompilation_on_change(self, sample_gsdata):
        """Test that LUT is recompiled when parameters change."""
        pipeline = Color()

        # First compilation
        pipeline.brightness(1.2).compile()
        lut1 = pipeline._compiled_lut.copy()

        # Change parameters
        pipeline.brightness(1.5)
        assert pipeline._is_dirty  # Should be marked dirty

        # Apply triggers recompilation
        pipeline.apply(sample_gsdata, inplace=False)
        lut2 = pipeline._compiled_lut

        # LUTs should be different
        assert not np.allclose(lut1, lut2)

    def test_phase1_operations(self, sample_gsdata):
        """Test Phase 1 (LUT-capable) operations."""
        pipeline = Color()
        original_colors = sample_gsdata.sh0.copy()

        # Apply Phase 1 operations
        result = (
            pipeline.temperature(0.7)
            .brightness(1.2)
            .contrast(1.1)
            .gamma(0.95)
            .apply(sample_gsdata, inplace=False)
        )

        # Result should be different from input
        assert not np.allclose(result.sh0, original_colors)
        # Result should be in valid range
        assert np.all(result.sh0 >= 0.0)
        assert np.all(result.sh0 <= 1.0)

    def test_phase2_operations(self, sample_gsdata):
        """Test Phase 2 (sequential) operations."""
        pipeline = Color()
        original_colors = sample_gsdata.sh0.copy()

        # Apply Phase 2 operations
        result = (
            pipeline.saturation(1.3)
            .shadows(1.1)
            .highlights(0.9)
            .apply(sample_gsdata, inplace=False)
        )

        # Result should be different from input
        assert not np.allclose(result.sh0, original_colors)
        # Result should be in valid range
        assert np.all(result.sh0 >= 0.0)
        assert np.all(result.sh0 <= 1.0)

    def test_combined_phases(self, sample_gsdata):
        """Test combining Phase 1 and Phase 2 operations."""
        pipeline = Color()
        original_colors = sample_gsdata.sh0.copy()

        # Apply both Phase 1 and Phase 2 operations
        result = (
            pipeline.temperature(0.6)  # Phase 1
            .brightness(1.2)  # Phase 1
            .contrast(1.1)  # Phase 1
            .saturation(1.3)  # Phase 2
            .shadows(1.1)  # Phase 2
            .apply(sample_gsdata, inplace=False)
        )

        # Result should be valid
        assert result.sh0.shape == original_colors.shape
        assert np.all(result.sh0 >= 0.0)
        assert np.all(result.sh0 <= 1.0)

    def test_inplace_vs_copy(self, sample_gsdata):
        """Test inplace vs copy behavior."""
        pipeline = Color().brightness(1.2)
        original_colors = sample_gsdata.sh0.copy()

        # Test copy
        result_copy = pipeline.apply(sample_gsdata, inplace=False)
        assert np.allclose(sample_gsdata.sh0, original_colors)  # Original unchanged
        assert result_copy is not sample_gsdata  # Different GSData

        # Test inplace
        result_inplace = pipeline.apply(sample_gsdata, inplace=True)
        assert not np.allclose(sample_gsdata.sh0, original_colors)  # Original modified
        assert result_inplace is sample_gsdata  # Same GSData

    def test_callable_interface(self, sample_gsdata):
        """Test that pipeline can be called as a function."""
        pipeline = Color().brightness(1.2).contrast(1.1)
        original_colors = sample_gsdata.sh0.copy()

        # Should work as a callable
        result = pipeline(sample_gsdata, inplace=False)

        assert result.sh0.shape == original_colors.shape
        assert not np.allclose(result.sh0, original_colors)

    def test_reset(self, sample_gsdata):
        """Test reset functionality."""
        pipeline = Color()

        # Set some operations and compile
        pipeline.brightness(1.2).contrast(1.1).compile()
        assert pipeline._compiled_lut is not None

        # Reset
        pipeline.reset()

        # Should be back to defaults
        params = pipeline.get_params()
        assert params["temperature"] == 0.5
        assert params["brightness"] == 1.0
        assert params["contrast"] == 1.0
        assert params["gamma"] == 1.0
        assert params["saturation"] == 1.0
        assert params["shadows"] == 1.0
        assert params["highlights"] == 1.0
        assert pipeline._compiled_lut is None
        assert pipeline._is_dirty

    def test_get_params(self):
        """Test getting current parameters."""
        pipeline = (
            Color().temperature(0.6).brightness(1.2).contrast(1.1).gamma(0.95).saturation(1.3)
        )

        params = pipeline.get_params()

        assert params["temperature"] == 0.6
        assert params["brightness"] == 1.2
        assert params["contrast"] == 1.1
        assert params["gamma"] == 0.95
        assert params["saturation"] == 1.3
        assert params["shadows"] == 1.0  # Default
        assert params["highlights"] == 1.0  # Default

    def test_parameter_validation(self):
        """Test parameter validation."""
        pipeline = Color()

        # Invalid temperature
        with pytest.raises(ValueError):
            pipeline.temperature(1.5)  # Out of range

        # Invalid brightness
        with pytest.raises(ValueError):
            pipeline.brightness(-0.5)  # Negative

        # Invalid gamma
        with pytest.raises(ValueError):
            pipeline.gamma(0.0)  # Zero

        # Invalid types
        with pytest.raises(TypeError):
            pipeline.brightness("invalid")

    def test_repr(self):
        """Test string representation."""
        pipeline = Color().brightness(1.2).contrast(1.1)
        repr_str = repr(pipeline)

        assert "Color" in repr_str
        assert "bright=1.20" in repr_str  # Uses abbreviated format
        assert "contrast=1.10" in repr_str
        assert "not compiled" in repr_str

        # After compilation
        pipeline.compile()
        repr_str = repr(pipeline)
        assert "compiled" in repr_str
        assert "not compiled" not in repr_str

    def test_extreme_values(self, sample_gsdata):
        """Test with extreme parameter values."""
        pipeline = Color()

        # Apply extreme adjustments
        result = (
            pipeline.temperature(1.0)  # Maximum warm
            .brightness(3.0)  # Very bright
            .contrast(3.0)  # Very high contrast
            .gamma(0.2)  # Very low gamma
            .saturation(3.0)  # Very saturated
            .apply(sample_gsdata, inplace=False)
        )

        # Output should still be clamped to [0, 1]
        assert np.all(result.sh0 >= 0.0)
        assert np.all(result.sh0 <= 1.0)
