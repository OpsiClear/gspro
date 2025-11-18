"""
Quick tests for Filter parameterization.
"""

import numpy as np
import pytest
from gsply import GSData

from gspro import Filter, Param


@pytest.fixture
def sample_gsdata():
    """Create sample GSData for testing."""
    n = 1000
    data = GSData(
        means=np.random.randn(n, 3).astype(np.float32),
        scales=np.random.rand(n, 3).astype(np.float32) * 2.0,
        quats=np.random.randn(n, 4).astype(np.float32),
        opacities=np.random.rand(n).astype(np.float32),
        sh0=np.random.rand(n, 3).astype(np.float32),
        shN=None,
    )
    # Normalize quaternions
    data.quats[:] = data.quats / np.linalg.norm(data.quats, axis=1, keepdims=True)
    return data


class TestFilterTemplate:
    """Test Filter.template() functionality."""

    def test_template_creation(self):
        """Test creating a parameterized filter template."""
        template = Filter.template(
            sphere_radius=Param("r", default=0.8, range=(0.1, 1.0)),
            min_opacity=Param("o", default=0.1, range=(0.0, 1.0)),
        )

        assert len(template._param_map) == 2
        assert "r" in template._param_map
        assert "o" in template._param_map

    def test_template_applies_defaults(self, sample_gsdata):
        """Test that template applies default values."""
        template = Filter.template(min_opacity=Param("o", default=0.5, range=(0.0, 1.0)))

        # Apply without params (should use defaults)
        result = template(sample_gsdata, inplace=False)

        # Should filter out Gaussians with opacity < 0.5
        assert len(result) <= len(sample_gsdata)
        assert len(result) < len(sample_gsdata)  # Some should be filtered

    def test_template_parameter_override(self, sample_gsdata):
        """Test overriding template parameters at runtime."""
        template = Filter.template(min_opacity=Param("o", default=0.1, range=(0.0, 1.0)))

        # Apply with different threshold
        result1 = template(sample_gsdata, inplace=False, params={"o": 0.3})
        result2 = template(sample_gsdata, inplace=False, params={"o": 0.7})

        # Higher threshold should filter more
        assert len(result2) < len(result1)

    def test_template_cache_hit(self, sample_gsdata):
        """Test cache hit with same parameters."""
        template = Filter.template(
            min_opacity=Param("o", default=0.1, range=(0.0, 1.0)),
            max_scale=Param("s", default=2.0, range=(0.1, 10.0)),
        )

        # First call - cache miss
        result1 = template(sample_gsdata, inplace=False, params={"o": 0.3, "s": 1.5})
        cache_size_1 = len(template._filter_cache)

        # Second call with same params - cache hit
        result2 = template(sample_gsdata, inplace=False, params={"o": 0.3, "s": 1.5})
        cache_size_2 = len(template._filter_cache)

        # Cache should not grow (hit)
        assert cache_size_1 == cache_size_2 == 1

        # Results should be identical in size
        assert len(result1) == len(result2)

    def test_template_cache_miss(self, sample_gsdata):
        """Test cache miss with different parameters."""
        template = Filter.template(min_opacity=Param("o", default=0.1, range=(0.0, 1.0)))

        # First call
        template(sample_gsdata, inplace=False, params={"o": 0.2})
        cache_size_1 = len(template._filter_cache)

        # Second call with different params
        template(sample_gsdata, inplace=False, params={"o": 0.4})
        cache_size_2 = len(template._filter_cache)

        # Cache should grow (miss)
        assert cache_size_2 == cache_size_1 + 1 == 2

    def test_template_unknown_parameter(self, sample_gsdata):
        """Test error on unknown parameter."""
        template = Filter.template(min_opacity=Param("o", default=0.1, range=(0.0, 1.0)))

        with pytest.raises(ValueError, match="Unknown parameter 'unknown'"):
            template(sample_gsdata, params={"unknown": 0.5})

    def test_template_parameter_out_of_range(self, sample_gsdata):
        """Test error on parameter out of range."""
        template = Filter.template(min_opacity=Param("o", default=0.5, range=(0.1, 0.9)))

        with pytest.raises(ValueError, match="outside valid range"):
            template(sample_gsdata, params={"o": 1.5})

    def test_non_template_with_params_raises(self, sample_gsdata):
        """Test error when using params on non-template pipeline."""
        pipeline = Filter().min_opacity(0.1)

        with pytest.raises(ValueError, match="not created with template"):
            pipeline(sample_gsdata, params={"o": 0.5})

    def test_template_reset_clears_cache(self):
        """Test that reset() clears the parameter cache."""
        template = Filter.template(min_opacity=Param("o", default=0.1, range=(0.0, 1.0)))

        # Build cache
        data = GSData(
            means=np.random.randn(100, 3).astype(np.float32),
            scales=np.random.rand(100, 3).astype(np.float32),
            quats=np.random.randn(100, 4).astype(np.float32),
            opacities=np.random.rand(100).astype(np.float32),
            sh0=np.random.rand(100, 3).astype(np.float32),
            shN=None,
        )
        data.quats[:] = data.quats / np.linalg.norm(data.quats, axis=1, keepdims=True)

        template(data, params={"o": 0.5})
        assert len(template._filter_cache) > 0

        # Reset
        template.reset()

        # Cache should be empty
        assert len(template._filter_cache) == 0
        assert len(template._param_map) == 0
