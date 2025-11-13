<div align="center">

# gspro

### High-Performance Processing for 3D Gaussian Splatting

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)](#testing)

**1,389M colors/sec | 698M Gaussians/sec transforms | 62M Gaussians/sec filtering | Pure NumPy with Numba**

[Features](#features) | [Installation](#installation) | [Quick Start](#quick-start) | [Performance](#performance) | [Documentation](#documentation)

</div>

---

## Overview

**gspro** is a pure Python library for ultra-fast processing of 3D Gaussian Splatting data using Look-Up Tables (LUTs) and Numba-accelerated operations. Built for performance-critical applications, gspro achieves color processing speeds up to 1,722M colors/sec, transform speeds up to 1,593M Gaussians/sec, and filtering speeds up to 447M Gaussians/sec.

**Why gspro?**
- **Blazing Fast**: Zero-copy APIs, LUT-based color ops, Numba JIT with parallel processing
- **Pure Python**: NumPy + Numba (no C++ compilation needed)
- **Composable**: Pipeline API for chaining operations, built-in presets
- **Complete**: Color grading, 3D transforms, spatial filtering all in one library

---

## Features

- **Fastest Color Processing**: Peak performance of 1,722M colors/sec with zero-copy API
  - **100K colors**: 0.072ms (1,389M/s) zero-copy, 0.473ms (211M/s) standard
  - **1M colors**: 0.581ms (1,722M/s)
  - **Operations**: 7 color adjustments (temperature, brightness, contrast, gamma, saturation, shadows, highlights)
  - **Optimizations**: Zero-copy API (6.6x faster), LUT-based processing, nogil=True for true parallelism

- **Fast 3D Transforms**: Up to 1,593M Gaussians/sec for geometric operations
  - **1M Gaussians**: 1.43ms (698M/s) combined transform
  - **500K Gaussians**: 0.31ms (1,593M/s) peak performance
  - **Operations**: translate, rotate, scale, combined transforms
  - **Rotation formats**: quaternion, matrix, axis_angle, euler
  - **Utilities**: Quaternion multiply, format conversions

- **High-Performance Filtering**: 62-447M Gaussians/sec full filtering pipeline
  - **1M Gaussians full filtering**: 16.1ms (62M/s)
  - **Individual operations**: 392-447M Gaussians/sec
  - **Volume filtering**: Sphere and cuboid spatial selection
  - **Property filtering**: Opacity and scale thresholds with AND logic
  - **Optimizations**: Fused kernels, parallel scatter pattern, nogil=True

- **Composable Pipeline**: Chain operations with lazy execution
  - **Built-in presets**: 7 color grading looks (cinematic, warm, cool, vibrant, muted, dramatic)
  - **Functional API**: One-line color adjustments and preset application
  - **Custom operations**: Add user-defined processing steps

- **Pure Python**: NumPy + Numba JIT (no C++ compilation required)
- **Type-safe**: Full type hints for Python 3.10+
- **Production-ready**: Comprehensive test suite, CI/CD pipeline, detailed documentation

---

## Installation

### From PyPI

```bash
pip install gspro
```

### From Source

```bash
git clone https://github.com/OpsiClear/gspro.git
cd gspro
pip install -e .
```

**Requirements:** Python >= 3.10, NumPy >= 1.24.0, Numba >= 0.59.0

---

## Quick Start

### Color Processing

```python
import numpy as np
from gspro import ColorLUT

lut = ColorLUT()
colors = np.random.rand(100_000, 3).astype(np.float32)

# Standard API
result = lut.apply(colors, temperature=0.6, brightness=1.2, saturation=1.3)

# Zero-copy API (6.6x faster, recommended for performance)
out = np.empty_like(colors)
lut.apply_numpy_inplace(colors, out, temperature=0.6, brightness=1.2, saturation=1.3)
```

### 3D Transforms

```python
from gspro import transform

means = np.random.randn(100_000, 3).astype(np.float32)
quaternions = np.random.randn(100_000, 4).astype(np.float32)
quaternions = quaternions / np.linalg.norm(quaternions, axis=1, keepdims=True)
scales = np.random.rand(100_000, 3).astype(np.float32)

# Combined transform (scale -> rotate -> translate)
new_means, new_quats, new_scales = transform(
    means, quaternions, scales,
    scale_factor=2.0,
    rotation=[0.9239, 0, 0, 0.3827],  # quaternion
    translation=[1.0, 0.0, 0.0]
)
```

### Pipeline & Presets

```python
from gspro import ColorPreset, apply_preset

# Use built-in preset
result = ColorPreset.cinematic().apply(colors)

# Or one-line
result = apply_preset(colors, "warm")
```

### Filtering

```python
from gspro.filter import filter_gaussians, calculate_scene_bounds, calculate_recommended_max_scale

# Calculate scene bounds and recommended threshold
bounds = calculate_scene_bounds(positions)
threshold = calculate_recommended_max_scale(scales)

# Apply combined filtering (kwargs like transform module, returns tuple)
new_pos, new_quats, new_scales, new_opac, new_colors = filter_gaussians(
    positions, quaternions, scales, opacities, colors,
    filter_type="sphere",              # or "cuboid" or "none"
    sphere_radius_factor=0.8,          # 80% of scene radius
    opacity_threshold=0.05,            # Remove < 5% opacity
    max_scale=threshold,               # Remove outliers
    scene_bounds=bounds
)

# Use filtered data (tuple return like transform module)
# Unpack only what you need
new_pos, new_quats, *_ = filter_gaussians(positions, quaternions, ...)
```

---

## Complete API Guide

### Color Processing API

#### Basic Usage

```python
from gspro import ColorLUT
import numpy as np

# Create LUT instance (reusable)
lut = ColorLUT(device="cpu", lut_size=1024)

# Prepare colors [N, 3] in range [0, 1]
colors = np.random.rand(100_000, 3).astype(np.float32)

# Method 1: Standard API (returns new array)
result = lut.apply(
    colors,
    temperature=0.6,    # 0=cool, 0.5=neutral, 1=warm
    brightness=1.2,     # Multiplier (1.0=no change)
    contrast=1.1,       # Multiplier (1.0=no change)
    gamma=1.0,          # Gamma correction (1.0=linear)
    saturation=1.3,     # Multiplier (1.0=no change, 0=grayscale)
    shadows=0.9,        # Shadow adjustment (1.0=no change)
    highlights=1.1      # Highlight adjustment (1.0=no change)
)
```

#### Zero-Copy API (Recommended for Performance)

```python
# Pre-allocate output buffer (reuse for multiple frames)
out = np.empty_like(colors)

# Method 2: Zero-copy API (6.6x faster, 1,389 M/s)
lut.apply_numpy_inplace(
    colors, out,
    temperature=0.6,
    brightness=1.2,
    saturation=1.3
)
# Result is written to 'out' in-place
```

#### NumPy-Only Path

```python
# Method 3: NumPy-only (no device conversion)
result = lut.apply_numpy(
    colors,
    temperature=0.6,
    brightness=1.2,
    saturation=1.3
)
```

### Transform API

#### Individual Transforms

```python
from gspro import translate, rotate, scale
import numpy as np

# Gaussian data
means = np.random.randn(100_000, 3).astype(np.float32)
quaternions = np.random.randn(100_000, 4).astype(np.float32)
quaternions = quaternions / np.linalg.norm(quaternions, axis=1, keepdims=True)
scales_arr = np.random.rand(100_000, 3).astype(np.float32)

# Translate
new_means = translate(means, translation=[1.0, 0.0, 0.0])

# Rotate (supports 4 rotation formats)
new_means, new_quats = rotate(
    means, quaternions,
    rotation=[0.9239, 0, 0, 0.3827],  # Quaternion (w, x, y, z)
    rotation_format="quaternion"       # or "matrix", "axis_angle", "euler"
)

# Scale
new_means, new_scales = scale(
    means, scales_arr,
    scale_factor=2.0  # Uniform scale
)
# OR
new_means, new_scales = scale(
    means, scales_arr,
    scale_factor=[2.0, 1.5, 1.0]  # Non-uniform scale
)
```

#### Combined Transform (Recommended)

```python
from gspro import transform

# All-in-one transform (4-5x faster than separate operations)
new_means, new_quats, new_scales = transform(
    means, quaternions, scales_arr,
    translation=[1.0, 0.0, 0.0],
    rotation=[0.9239, 0, 0, 0.3827],
    rotation_format="quaternion",
    scale_factor=2.0
)
```

#### Rotation Format Examples

```python
# Quaternion (w, x, y, z)
rotation = [0.9239, 0, 0, 0.3827]
rotation_format = "quaternion"

# 3x3 Rotation Matrix
rotation = np.array([[1, 0, 0], [0, 0.866, -0.5], [0, 0.5, 0.866]])
rotation_format = "matrix"

# Axis-Angle (axis * angle in radians)
rotation = [0, 0, 1.57]  # 90 degrees around Z
rotation_format = "axis_angle"

# Euler Angles (roll, pitch, yaw in radians)
rotation = [0.1, 0.2, 0.3]
rotation_format = "euler"
```

### Filtering API

#### Complete Example

```python
from gspro.filter import (
    filter_gaussians,
    calculate_scene_bounds,
    calculate_recommended_max_scale
)
import numpy as np

# Gaussian data
positions = np.random.randn(1_000_000, 3).astype(np.float32)
quaternions = np.random.randn(1_000_000, 4).astype(np.float32)
scales = np.random.rand(1_000_000, 3).astype(np.float32)
opacities = np.random.rand(1_000_000).astype(np.float32)
colors = np.random.rand(1_000_000, 3).astype(np.float32)

# Step 1: Calculate scene bounds (run once)
bounds = calculate_scene_bounds(positions)
print(f"Scene center: {bounds.center}")
print(f"Scene size: {bounds.sizes}")

# Step 2: Calculate recommended thresholds (run once)
max_scale_threshold = calculate_recommended_max_scale(scales, percentile=99.5)
print(f"Recommended max_scale: {max_scale_threshold:.4f}")

# Step 3: Apply filtering
filtered_data = filter_gaussians(
    positions,
    quaternions,
    scales,
    opacities,
    colors,
    # Volume filtering (choose one)
    filter_type="sphere",           # or "cuboid" or "none"
    sphere_radius_factor=0.8,       # 80% of scene radius
    # Property filtering
    opacity_threshold=0.05,         # Remove < 5% opacity
    max_scale=max_scale_threshold,  # Remove scale outliers
    # Scene info
    scene_bounds=bounds
)

# Unpack results (tuple like transform API)
new_pos, new_quats, new_scales, new_opac, new_colors = filtered_data
print(f"Filtered: {len(positions)} -> {len(new_pos)} Gaussians")
```

#### Filtering Options

```python
# Option 1: Sphere filtering (keep Gaussians within radius)
filter_type = "sphere"
sphere_center = (0.0, 0.0, 0.0)          # Default: scene center
sphere_radius_factor = 0.8                # 80% of max scene dimension

# Option 2: Cuboid filtering (keep Gaussians within box)
filter_type = "cuboid"
cuboid_center = (0.0, 0.0, 0.0)          # Default: scene center
cuboid_size_factor_x = 0.8                # 80% of X dimension
cuboid_size_factor_y = 0.8                # 80% of Y dimension
cuboid_size_factor_z = 0.8                # 80% of Z dimension

# Option 3: No volume filtering (property filtering only)
filter_type = "none"
```

#### Using apply_filter for Mask Only

```python
from gspro.filter import apply_filter

# Get boolean mask without copying data
mask = apply_filter(
    positions,
    opacities,
    scales,
    filter_type="sphere",
    sphere_radius_factor=0.8,
    opacity_threshold=0.05,
    max_scale=2.5,
    scene_bounds=bounds
)

# Manually apply mask
filtered_positions = positions[mask]
filtered_colors = colors[mask]
```

### Pipeline API

#### Basic Pipeline

```python
from gspro import Pipeline

# Create pipeline
pipeline = Pipeline(device="cpu")

# Add operations
pipeline.adjust_colors(
    temperature=0.6,
    brightness=1.2,
    contrast=1.1,
    saturation=1.3
)

# Execute
result = pipeline(colors)
```

#### Using Presets

```python
from gspro import ColorPreset, apply_preset

# Method 1: Direct preset application
result = ColorPreset.cinematic().apply(colors)

# Method 2: Functional API (one-liner)
result = apply_preset(colors, "warm")

# Available presets
presets = ["neutral", "cinematic", "warm", "cool", "vibrant", "muted", "dramatic"]
```

---

## Performance

### Color Processing (100K colors)

| API | Time | Throughput | Speedup |
|-----|------|------------|---------|
| `apply()` | 0.473 ms | 211 M/s | 1.00x |
| `apply_numpy()` | 0.480 ms | 208 M/s | 0.99x |
| **`apply_numpy_inplace()`** | **0.072 ms** | **1,389 M/s** | **6.57x** |

### Color Batch Scaling (apply_numpy_inplace)

| Batch Size | Time | Throughput |
|------------|------|------------|
| 1K | 0.016 ms | 64 M/s |
| 10K | 0.022 ms | 461 M/s |
| 100K | 0.086 ms | 1,162 M/s |
| 1M | 0.581 ms | 1,722 M/s |

### 3D Transform (1M Gaussians)

| Operation | Time | Throughput |
|-----------|------|------------|
| Combined transform | 1.743 ms | 574 M G/s |

### Transform Batch Scaling

| Batch Size | Time | Throughput |
|------------|------|------------|
| 10K | 0.09 ms | 106 M G/s |
| 100K | 0.12 ms | 863 M G/s |
| 500K | 0.31 ms | 1,593 M G/s |
| 1M | 1.43 ms | 698 M G/s |
| 2M | 7.31 ms | 273 M G/s |

### Filtering Performance (1M Gaussians)

| Operation | Time | Throughput |
|-----------|------|------------|
| Scene bounds (one-time) | 35.7 ms | 28 M/s |
| Recommended scale (one-time) | 6.4 ms | 157 M/s |
| Sphere filter (nogil=True) | 2.5 ms | 405 M/s |
| Cuboid filter (nogil=True) | 4.8 ms | 207 M/s |
| Opacity filter (nogil=True) | 2.6 ms | 392 M/s |
| Scale filter (nogil=True) | 2.2 ms | 447 M/s |
| Combined filter | 3.6 ms | 276 M/s |
| **Full filtering (filter_gaussians)** | **16.1 ms** | **62.1 M/s** |

### Key Performance Highlights

- **Peak Color Processing**: 1,722M colors/sec (1M batch, zero-copy)
- **Peak Transform Speed**: 1,593M Gaussians/sec (500K batch)
- **Peak Filtering Speed**: 447M Gaussians/sec (scale filter)
- **Full Pipeline**: 62.1M Gaussians/sec (complete filtering)
- **Scalability**: Linear scaling from 1K to 2M Gaussians

### Optimization Details

- **Zero-copy APIs**: Direct memory operations without allocation overhead (6.6x speedup)
- **LUT-based processing**: Pre-computed look-up tables for color operations
- **nogil=True**: True parallelism by releasing GIL (+15-37% performance)
- **Fused kernels**: Combined opacity+scale filtering in single pass
- **Parallel processing**: Numba JIT with `prange` for multi-core utilization
- **fastmath optimization**: Aggressive floating-point optimizations on all kernels

---

## API Reference

### ColorLUT

```python
ColorLUT(device="cpu", lut_size=1024)
```

**Methods:**
- `apply(colors, **params)` - Apply color adjustments (211 M/s)
- `apply_numpy()` - NumPy-only path (208 M/s)
- `apply_numpy_inplace(colors, out, **params)` - Zero-copy API (1,389 M/s, 6.6x faster)
- `reset()` - Clear LUT cache

**Parameters:**
- `temperature` (float): 0=cool, 0.5=neutral, 1=warm
- `brightness` (float): Brightness multiplier
- `contrast` (float): Contrast multiplier
- `gamma` (float): Gamma correction
- `saturation` (float): Saturation adjustment
- `shadows` (float): Shadow adjustment
- `highlights` (float): Highlight adjustment

### Transform Functions

```python
translate(means, translation)                    # -> means
rotate(means, quaternions, rotation, ...)        # -> (means, quaternions)
scale(means, scales, scale_factor, ...)          # -> (means, scales)
transform(means, quaternions, scales, ...)       # -> (means, quaternions, scales)
```

**Rotation formats:** "quaternion", "matrix", "axis_angle", "euler"

### Quaternion Utilities

```python
quaternion_multiply(q1, q2)
quaternion_to_rotation_matrix(q)
rotation_matrix_to_quaternion(R)
axis_angle_to_quaternion(axis_angle)
euler_to_quaternion(euler)
quaternion_to_euler(q)
```

### Pipeline

```python
Pipeline(device="cpu")
```

**Methods:**
- `adjust_colors(**params)` - Add color step
- `transform(**params)` - Add transform step
- `custom(func, **kwargs)` - Add custom operation
- `__call__(data)` - Execute pipeline
- `reset()` - Clear operations

### ColorPreset

**Built-in presets:**
- `neutral()` - Identity
- `cinematic()` - Desaturated, high contrast, teal/orange
- `warm()` - Orange tones, boosted brightness
- `cool()` - Blue tones, crisp contrast
- `vibrant()` - Boosted saturation/contrast
- `muted()` - Low saturation, lifted shadows
- `dramatic()` - High contrast, crushed shadows

**Methods:**
- `apply(colors)` - Apply preset
- `to_pipeline()` - Convert to pipeline

### Functional API

```python
adjust_colors(colors, **params)
apply_preset(colors, preset_name)
```

### Filtering

```python
FilterConfig(
    filter_type="none",                # "none", "sphere", or "cuboid"
    sphere_center=(0.0, 0.0, 0.0),
    sphere_radius_factor=1.0,          # 0.0 to 1.0
    cuboid_center=(0.0, 0.0, 0.0),
    cuboid_size_factor_x=1.0,          # 0.0 to 1.0
    cuboid_size_factor_y=1.0,
    cuboid_size_factor_z=1.0,
    opacity_threshold=0.05,            # 0.0 to 1.0
    max_scale=10.0                     # Large value = disabled
)
```

**Functions** (kwargs like transform module):
```python
# Apply filtering to get boolean mask (kwargs like transform)
mask = apply_filter(
    positions, opacities, scales,
    filter_type="sphere",
    sphere_radius_factor=0.8,
    opacity_threshold=0.05,
    max_scale=2.5,
    scene_bounds=bounds
)

# Filter all attributes at once (returns tuple like transform)
new_pos, new_quats, new_scales, new_opac, new_colors = filter_gaussians(
    positions, quaternions, scales, opacities, colors,
    filter_type="sphere",
    sphere_radius_factor=0.8,
    opacity_threshold=0.05,
    scene_bounds=bounds
)

# Calculate scene spatial bounds
bounds = calculate_scene_bounds(positions)
# Returns: SceneBounds(min, max, sizes, max_size, center)

# Calculate recommended max scale threshold
threshold = calculate_recommended_max_scale(scales, percentile=99.5)
```

**FilterConfig** (optional, for convenience):
```python
# Can also use FilterConfig for organizing many parameters
config = FilterConfig(filter_type="sphere", sphere_radius_factor=0.8, ...)
mask = apply_filter(positions, config=config)  # Backward compatible
```

**Filter Types:**
- **none**: No spatial filtering (default)
- **sphere**: Keep Gaussians within radius from center
- **cuboid**: Keep Gaussians within box bounds

**Filter Logic:**
- All filters use AND logic (all conditions must be met)
- Opacity filter: `opacity >= threshold`
- Scale filter: `max(scale_x, scale_y, scale_z) <= max_scale`
- Volume filter: Inside sphere or cuboid

## Example: Full Processing Pipeline

```python
import numpy as np
from gspro import ColorLUT, transform

# Load Gaussian data
colors = load_gaussian_colors()       # [N, 3]
means = load_gaussian_positions()     # [N, 3]
quaternions = load_gaussian_orientations()  # [N, 4]
scales = load_gaussian_scales()       # [N, 3]

# Pre-allocate outputs (zero-copy)
out_colors = np.empty_like(colors)
out_means = np.empty_like(means)
out_quats = np.empty_like(quaternions)
out_scales = np.empty_like(scales)

# Transform geometry (1.479ms for 1M)
transform(
    means, quaternions, scales,
    translation=[1.0, 0.0, 0.0],
    rotation=[0.9239, 0, 0, 0.3827],
    scale_factor=1.5,
    out_means=out_means,
    out_quaternions=out_quats,
    out_scales=out_scales
)

# Color grading (0.099ms for 100K)
lut = ColorLUT()
lut.apply_numpy_inplace(
    colors, out_colors,
    temperature=0.6,
    brightness=1.2,
    contrast=1.1,
    saturation=1.3
)

# Render
render_scene(out_means, out_quats, out_scales, out_colors)
```

---

## Development

### Setup

```bash
# Clone repository
git clone https://github.com/OpsiClear/gspro.git
cd gspro

# Install in development mode
pip install -e .[dev]

# Run tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=gspro --cov-report=html
```

### Project Structure

```
gspro/
├── src/
│   └── gspro/
│       ├── __init__.py        # Public API
│       ├── color.py           # ColorLUT implementation
│       ├── transforms.py      # 3D geometric transforms
│       ├── pipeline.py        # Composable pipeline API
│       ├── filter/            # Filtering system
│       │   ├── __init__.py
│       │   ├── api.py         # High-level filter API
│       │   ├── kernels.py     # Numba-optimized kernels
│       │   └── config.py      # FilterConfig dataclass
│       └── py.typed           # PEP 561 type marker
├── tests/                     # Unit tests
├── benchmarks/                # Performance benchmarks
│   ├── run_all_benchmarks.py
│   ├── benchmark_color_lut.py
│   └── benchmark_large_scale.py
├── .github/                   # CI/CD workflows
│   └── workflows/
│       ├── test.yml
│       ├── build.yml
│       ├── publish.yml
│       └── benchmark.yml
├── pyproject.toml             # Package configuration
└── README.md                  # This file
```

---

## Benchmarking

Run performance benchmarks to measure library performance:

```bash
# Run all benchmarks
cd benchmarks
uv run run_all_benchmarks.py

# Run specific benchmark
uv run benchmark_color_lut.py

# Run large-scale benchmark (1M+ Gaussians)
uv run benchmark_large_scale.py
```

The benchmarks measure:
- **Color processing**: LUT application across different batch sizes
- **Transform performance**: Geometric operations on Gaussians
- **Filtering performance**: Spatial and property-based filtering
- **Scalability**: Performance across varying data sizes

---

## Testing

gspro has comprehensive test coverage with passing tests:

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_color.py -v

# Run with coverage report
pytest tests/ -v --cov=gspro --cov-report=html
```

Test categories:
- Color LUT operations (all adjustment types, caching, edge cases)
- 3D transforms (translation, rotation, scaling, combined)
- Quaternion utilities (conversions, multiplication)
- Pipeline API (composition, presets, custom operations)
- Filtering system (volume filters, property filters, combined logic)

---

## Documentation

For detailed documentation see:
- **OPTIMIZATION_COMPLETE_SUMMARY.md** - Complete optimization history and performance analysis
- **AUDIT_FIXES_SUMMARY.md** - Bug fixes and validation methodology
- **.github/WORKFLOWS.md** - CI/CD pipeline documentation
- **benchmarks/README.md** - Benchmark suite documentation

---

## CI/CD

gspro includes a complete GitHub Actions CI/CD pipeline:

- **Multi-platform testing**: Ubuntu, Windows, macOS
- **Multi-version testing**: Python 3.10, 3.11, 3.12, 3.13
- **Automated benchmarking**: Performance tracking on PRs
- **Build verification**: Wheel building and installation testing
- **PyPI publishing**: Automated release on GitHub Release

See [.github/WORKFLOWS.md](.github/WORKFLOWS.md) for details.

---

## Architecture

**Color Processing:**
- Phase 1: LUT-capable operations (temperature, brightness, contrast, gamma) pre-compiled into 1D LUTs
- Phase 2: Dependent operations (saturation, shadows/highlights) with branchless code
- Single fused Numba kernel with interleaved LUT layout for cache locality
- Zero-copy API eliminates 80% memory allocation overhead

**3D Transforms:**
- Matrix-based operations (scale, rotate, translate)
- Numba-accelerated quaternion operations
- Fused transforms for single-pass processing
- Parallel processing with `prange`

**Filtering System:**
- Fused opacity+scale kernel for 1.95x speedup
- Parallel scatter pattern with prefix sum for lock-free writes
- fastmath optimization on all kernels (5-10% speedup)
- Numba JIT compilation with parallel execution

---

## Contributing

Contributions are welcome! Please see [.github/CONTRIBUTING.md](.github/CONTRIBUTING.md) for guidelines.

**Quick start:**
1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Run tests and benchmarks
5. Submit a pull request

---

## License

MIT License - see [LICENSE](LICENSE) file for details.

---

## Citation

If you use gspro in your research, please cite:

```bibtex
@software{gspro2025,
  author = {OpsiClear},
  title = {gspro: High-Performance Processing for 3D Gaussian Splatting},
  year = {2025},
  url = {https://github.com/OpsiClear/gspro}
}
```

---

## Related Projects

- **gsply**: Ultra-fast Gaussian Splatting PLY I/O library
- **gsplat**: CUDA-accelerated Gaussian Splatting rasterizer
- **nerfstudio**: NeRF training framework with Gaussian Splatting support
- **3D Gaussian Splatting**: Original paper and implementation

---

<div align="center">

**Made with Python and NumPy**

[Report Bug](https://github.com/OpsiClear/gspro/issues) | [Request Feature](https://github.com/OpsiClear/gspro/issues) | [Documentation](.github/WORKFLOWS.md)

</div>
