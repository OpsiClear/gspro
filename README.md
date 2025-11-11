# gslut - Gaussian Splatting Look-Up Tables

Fast LUT-based operations for 3D Gaussian Splatting, providing optimized color adjustments, activation functions, and SH conversions.

## Features

- **Activation LUTs**: Fast approximations for `exp()`, `sigmoid()`, and `normalize()` using clustered lookup tables with linear interpolation
- **Color LUTs**: Separated per-channel 1D LUTs for fast RGB color adjustments (10x faster than sequential operations)
- **SH Conversions**: Spherical Harmonics (SH0) to RGB color space conversions
- **CPU Optimized**: NumPy-accelerated operations for 2-3x faster CPU processing
- **GPU Ready**: Full PyTorch support for GPU acceleration

## Installation

```bash
pip install gslut
```

### Development Installation

```bash
git clone https://github.com/OpsiClear/gslut.git
cd gslut
pip install -e ".[dev]"
```

## Quick Start

### Color Adjustments

```python
import torch
from gslut import ColorLUT

# Create color LUT
color_lut = ColorLUT(device="cpu")

# Generate sample RGB colors
colors = torch.rand(1000, 3)  # [N, 3] in range [0, 1]

# Apply color adjustments
adjusted = color_lut.apply(
    colors,
    temperature=0.7,    # 0=cool, 0.5=neutral, 1=warm
    brightness=1.2,     # Brightness multiplier
    contrast=1.1,       # Contrast multiplier
    gamma=0.9,          # Gamma correction
    saturation=1.3,     # Saturation adjustment
    shadows=1.1,        # Shadow boost/reduce
    highlights=0.9      # Highlight boost/reduce
)
```

### Activation Functions

```python
import torch
from gslut import ActivationLUT

# Create activation LUT
lut = ActivationLUT(device="cpu", use_linear_interp=True)

# Build LUT from sample data
scale_samples = torch.randn(10000) * 2  # Your scale data
opacity_samples = torch.randn(10000) * 2  # Your opacity data

lut.build_from_samples(
    scale_samples=scale_samples,
    opacity_samples=opacity_samples
)

# Save LUT for future use
lut.save("./lut_cache")

# Use LUT for fast activations
scales_raw = torch.randn(1000)
opacities_raw = torch.randn(1000)

scales = lut.exp(scales_raw)       # Fast exp() approximation
opacities = lut.sigmoid(opacities_raw)  # Fast sigmoid() approximation
```

### SH Conversions

```python
import torch
from gslut import sh2rgb, rgb2sh, SH_C0

# RGB to SH0
rgb_colors = torch.rand(1000, 3)  # RGB in [0, 1]
sh0_coeffs = rgb2sh(rgb_colors)

# SH0 to RGB
rgb_reconstructed = sh2rgb(sh0_coeffs)

# Roundtrip conversion is lossless
assert torch.allclose(rgb_colors, rgb_reconstructed, atol=1e-6)
```

## API Reference

### ColorLUT

Fast color adjustments using separated 1D lookup tables.

```python
class ColorLUT(device="cuda", lut_size=1024)
```

**Methods:**
- `apply(colors, temperature=0.5, brightness=1.0, contrast=1.0, gamma=1.0, saturation=1.0, shadows=1.0, highlights=1.0)` - Apply color adjustments
- `reset()` - Clear LUT cache

**Performance:** 10x faster than sequential operations, 60x faster than 3D LUTs

### ActivationLUT

Clustered lookup tables for activation functions with linear interpolation.

```python
class ActivationLUT(
    lut_dir=None,
    num_clusters_exp=2048,
    num_clusters_sigmoid=2048,
    num_clusters_quat=512,
    device="cuda",
    use_linear_interp=True
)
```

**Methods:**
- `exp(x)` - Fast exp() approximation
- `sigmoid(x)` - Fast sigmoid() approximation
- `normalize(x, dim=-1)` - Fast quaternion normalization
- `build_from_samples(scale_samples, opacity_samples, quat_samples)` - Build LUT from data
- `save(lut_dir)` - Save LUT to disk
- `load(lut_dir)` - Load LUT from disk
- `get_stats()` - Get LUT statistics

**Accuracy:** 0.0002% mean error with 2048 clusters using linear interpolation

### Conversion Functions

```python
sh2rgb(sh: Tensor) -> Tensor
```
Convert SH0 coefficients to RGB colors.

```python
rgb2sh(rgb: Tensor) -> Tensor
```
Convert RGB colors to SH0 coefficients.

```python
get_sh_c0_constant() -> float
```
Get the SH0 normalization constant (0.28209479177387814).

## Performance

### Color LUT Benchmarks

- **Per-channel operations** (Temperature, Brightness, Contrast, Gamma): ~10x faster via 1D LUT
- **CPU optimization**: NumPy path provides 2-3x speedup on CPU
- **No GPU overhead**: All LUT operations on CPU before GPU transfer

### Activation LUT Benchmarks

- **Linear interpolation**: 600x better accuracy than nearest neighbor
- **Mean error**: 0.0002% with 2048 clusters
- **Note**: Modern GPUs have highly optimized exp/sigmoid - LUTs are useful for:
  - Memory-constrained scenarios
  - CPU inference
  - Deterministic reproducibility

## Use Cases

### Gaussian Splatting Rendering

```python
from gslut import ColorLUT, sh2rgb

# Convert SH0 to RGB
sh0_coeffs = load_gaussian_sh0()  # [N, 3]
base_colors = sh2rgb(sh0_coeffs)

# Apply color grading
color_lut = ColorLUT(device="cuda")
final_colors = color_lut.apply(
    base_colors,
    temperature=0.6,
    contrast=1.2,
    saturation=1.1
)
```

### Preprocessing PLY Sequences

```python
from gslut import ActivationLUT

# Build LUT from PLY sequence
lut = ActivationLUT(num_clusters_exp=2048, device="cpu")

# Collect samples from your data
all_scales = []
all_opacities = []
for frame in ply_sequence:
    scales, opacities = load_frame_data(frame)
    all_scales.append(scales)
    all_opacities.append(opacities)

# Build and save LUT
lut.build_from_samples(
    scale_samples=torch.cat(all_scales),
    opacity_samples=torch.cat(all_opacities)
)
lut.save("./my_sequence_lut")
```

## Development

### Running Tests

```bash
pytest tests/ -v
```

### Code Quality

```bash
# Format code
ruff format src/ tests/

# Lint
ruff check src/ tests/

# Fix linting issues automatically
ruff check --fix src/ tests/
```

### Performance Benchmarks

Comprehensive benchmarks comparing NumPy, PyTorch, torch.compile, CPU, and GPU:

```bash
cd benchmarks

# Run all benchmarks
python run_all_benchmarks.py

# Run individual benchmarks
python benchmark_color_lut.py
python benchmark_activation_lut.py
python benchmark_conversions.py
```

See [benchmarks/README.md](benchmarks/README.md) for details and [benchmarks/BENCHMARK_RESULTS.md](benchmarks/BENCHMARK_RESULTS.md) for expected performance characteristics.

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Citation

If you use this library in your research, please cite:

```bibtex
@software{gslut2025,
  title = {gslut: Gaussian Splatting Look-Up Tables},
  author = {OpsiClear},
  year = {2025},
  url = {https://github.com/OpsiClear/gslut}
}
```

## Acknowledgments

This library was extracted from the [universal_4d_viewer](https://github.com/OpsiClear/universal_4d_viewer) project, which provides real-time streaming and rendering of dynamic 4D Gaussian Splatting scenes.

## Related Projects

- [gsplat](https://github.com/nerfstudio-project/gsplat) - Gaussian splatting CUDA kernels
- [nerfstudio](https://github.com/nerfstudio-project/nerfstudio) - NeRF framework
- [universal_4d_viewer](https://github.com/OpsiClear/universal_4d_viewer) - 4D Gaussian viewer
