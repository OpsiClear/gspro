<div align="center">

# gspro

### High-Performance Processing for 3D Gaussian Splatting

[![Python](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)](#testing)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)

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
- **Integrated with gsply**: Built on gsply v0.3.0+ for advanced data management

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
  - **Multi-layer masks**: FilterMasks API with 55x faster Numba-optimized combination (0.026ms vs 1.447ms)
  - **Optimizations**: Fused kernels, parallel scatter pattern, nogil=True, adaptive mask combination

- **Optional Pre-Activation Stage**: Prepare log-domain GSData for downstream GPU/CPU pipelines
  - Fused Numba kernel (`apply_pre_activations`) exponentiates scales, sigmoids opacities, and normalizes quaternions in a single pass
  - 1.3ms for 1M Gaussians on a laptop CPU (≈750M Gaussians/sec), eliminating three separate NumPy sweeps
  - Works in-place with automatic dtype/contiguity fixes for data from `gsply.plyread`

- **Composable Pipeline**: Chain operations with lazy execution
  - **Built-in presets**: 7 color grading looks (cinematic, warm, cool, vibrant, muted, dramatic)
  - **Functional API**: One-line color adjustments and preset application
  - **Custom operations**: Add user-defined processing steps

- **Pure Python**: NumPy + Numba JIT (no C++ compilation required)
- **Type-safe**: Full type hints with Python 3.12+ syntax (PEP 695)
- **Production-ready**: Comprehensive test suite, CI/CD pipeline, pre-commit hooks, detailed documentation

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

### Unified Pipeline (Recommended)

```python
import gsply
from gspro import Pipeline

# Load Gaussian splatting data
data = gsply.plyread("scene.ply")

# Create a processing pipeline
pipeline = (
    Pipeline()
    .within_sphere(radius=0.8)        # Filter: keep 80% of scene
    .min_opacity(0.1)                 # Filter: remove low-opacity Gaussians
    .rotate_quat(quaternion)          # Transform: rotate scene
    .translate([1, 0, 0])             # Transform: move scene
    .brightness(1.2)                  # Color: adjust brightness
    .saturation(1.3)                  # Color: boost saturation
)

# Execute pipeline (inplace modification)
result = pipeline(data, inplace=True)

# Save result
gsply.plywrite("output.ply", result)
```

### Individual Pipelines

```python
import gsply
from gspro import Color, Transform, Filter

# Load data
data = gsply.plyread("scene.ply")

# Apply color adjustments
data = (
    Color()
    .temperature(0.6)
    .brightness(1.2)
    .saturation(1.3)
    (data, inplace=True)  # Callable interface
)

# Apply geometric transforms
data = (
    Transform()
    .rotate_quat(quaternion)
    .translate([1, 0, 0])
    .scale(2.0)
    (data, inplace=True)
)

# Apply filtering
data = (
    Filter()
    .within_sphere(radius=0.8)
    .min_opacity(threshold=0.1)
    (data, inplace=True)
)

# Save
gsply.plywrite("output.ply", data)
```

### GSData Pre-Activation (Optional)

When your training or authoring pipeline stores Gaussians in log-space (log scales, logit opacities, non-normalized quats), fuse the conversion into a single CPU pass before uploading to the renderer:

```python
import gsply
from gspro import apply_pre_activations

data = gsply.plyread("scene_raw_logits.ply")

apply_pre_activations(
    data,
    min_scale=1e-4,
    max_scale=100.0,
    min_quat_norm=1e-8,
    inplace=True,
)
```

`apply_pre_activations` exponentiates + clamps the scales, runs a numerically stable sigmoid on logit opacities, and normalizes quaternions—processing ~1M Gaussians in ≈1.3 ms (≈750M/sec). The helper automatically ensures float32 and contiguous buffers, so it pairs nicely with the zero-copy arrays returned by `gsply.plyread`.

### Using Color Presets

```python
from gspro import ColorPreset

# Apply built-in color grading presets
data = ColorPreset.cinematic()(data, inplace=True)
data = ColorPreset.warm()(data, inplace=True)
data = ColorPreset.vibrant()(data, inplace=True)
```

### Parameterized Pipeline Templates

For workflows requiring runtime parameter variation (animation, A/B testing, interactive tools), use parameterized templates with automatic caching:

```python
from gspro import Color, Filter, Param
import numpy as np

# Create parameterized color template
template = Color.template(
    brightness=Param("b", default=1.2, range=(0.5, 2.0)),
    contrast=Param("c", default=1.1, range=(0.5, 2.0)),
    saturation=Param("s", default=1.2, range=(0.5, 2.0))
)

# Use with different parameter values (automatically cached)
result1 = template(data, params={"b": 1.5, "c": 1.2, "s": 1.4})
result2 = template(data, params={"b": 1.3, "c": 1.0, "s": 1.0})
result3 = template(data, params={"b": 1.5, "c": 1.2, "s": 1.4})  # Cache hit!

# Generate animation frames with parameter variation
frames = []
for i in range(100):
    brightness = 1.0 + 0.2 * np.sin(2 * np.pi * i / 100)
    saturation = 1.0 + 0.3 * np.cos(2 * np.pi * i / 100)

    frame = template(data, inplace=False, params={"b": brightness, "s": saturation, "c": 1.1})
    frames.append(frame)
# Result: 100 frames in ~244ms (2.4ms/frame) with automatic LUT caching

# Filter template for spatial filtering variations
filter_template = Filter.template(
    sphere_radius=Param("r", default=0.8, range=(0.1, 1.0)),
    min_opacity=Param("o", default=0.1, range=(0.0, 1.0))
)

# Vary filter parameters at runtime
result_permissive = filter_template(data, params={"r": 0.9, "o": 0.05})
result_restrictive = filter_template(data, params={"r": 0.5, "o": 0.2})
```

**Performance:**
- Color templates: 2.2ms per call (45M Gaussians/sec)
- Filter templates: 49x faster on cache hits (2.05ms vs 97ms)
- Overhead: ~0.4us per call after optimizations

**When to use templates:**
- Animation with repeated parameter values (>67% cache hit rate)
- A/B testing with multiple iterations per parameter set
- Interactive tools with real-time parameter adjustment

**When NOT to use templates:**
- Single-shot processing with unique parameters
- Exploratory parameter sweeps (no cache benefit)
- Simple 1-2 operation pipelines

---

## Complete API Guide

### Unified Pipeline API

#### Basic Usage

```python
import gsply
from gspro import Pipeline

# Load Gaussian splatting data
data = gsply.plyread("scene.ply")

# Create and configure pipeline
pipeline = (
    Pipeline()
    # Filter operations
    .within_sphere(radius=0.8)
    .min_opacity(0.1)
    .max_scale(2.5)
    # Transform operations
    .translate([1, 0, 0])
    .rotate_quat([1, 0, 0, 0])
    .scale(2.0)
    # Color operations
    .temperature(0.6)
    .brightness(1.2)
    .contrast(1.1)
    .gamma(1.05)
    .saturation(1.3)
    .shadows(1.1)
    .highlights(0.9)
)

# Execute (returns GSData)
result = pipeline(data, inplace=False)

# Or modify in-place
pipeline(data, inplace=True)
```

#### Pipeline Operations

```python
# Color adjustments
pipeline.temperature(0.6)      # 0=cool, 0.5=neutral, 1=warm
pipeline.brightness(1.2)       # Multiplier (1.0=no change)
pipeline.contrast(1.1)         # Multiplier (1.0=no change)
pipeline.gamma(1.05)           # Gamma correction (1.0=linear)
pipeline.saturation(1.3)       # Multiplier (1.0=no change, 0=grayscale)
pipeline.shadows(1.1)          # Shadow adjustment (1.0=no change)
pipeline.highlights(0.9)       # Highlight adjustment (1.0=no change)

# Transform operations
pipeline.translate([1, 0, 0])  # Translation vector
pipeline.rotate_quat(quat)     # Rotation with quaternion
pipeline.rotate_euler(euler)   # Rotation with euler angles
pipeline.rotate_matrix(matrix) # Rotation with matrix
pipeline.rotate_axis_angle(axis, angle)  # Rotation with axis-angle
pipeline.scale(2.0)            # Uniform or per-axis scale
pipeline.set_center([0, 0, 0]) # Set center for rotation/scaling

# Filter operations
pipeline.within_sphere(center=None, radius=0.8)
pipeline.within_box(center=None, size=[0.8, 0.8, 0.8])
pipeline.min_opacity(0.1)
pipeline.max_scale(2.5)
pipeline.bounds(scene_bounds)  # Pre-computed bounds

# Pipeline control
pipeline.apply(data, inplace=True)  # Explicit application
pipeline.reset()               # Clear all operations
len(pipeline)                  # Number of operations
repr(pipeline)                 # String representation
```

### Transform API

#### Basic Transform Operations

```python
import gsply
from gspro import Transform

# Load Gaussian splatting data
data = gsply.plyread("scene.ply")

# Individual operations
data = Transform().translate([1, 0, 0])(data, inplace=True)
data = Transform().rotate_quat([0.9239, 0, 0, 0.3827])(data, inplace=True)
data = Transform().scale(2.0)(data, inplace=True)

# Non-uniform scaling
data = Transform().scale([2.0, 1.5, 1.0])(data, inplace=True)

# Save result
gsply.plywrite("transformed.ply", data)
```

#### Chained Transforms (Recommended)

```python
import gsply
from gspro import Transform

data = gsply.plyread("scene.ply")

# Chain multiple operations (single-pass execution)
data = (
    Transform()
    .translate([1.0, 0.0, 0.0])
    .rotate_quat([0.9239, 0, 0, 0.3827])
    .scale(2.0)
    .set_center([0, 0, 0])  # Set rotation/scale center
    (data, inplace=True)
)

gsply.plywrite("output.ply", data)
```

#### Rotation Format Examples

```python
from gspro import Transform
import numpy as np

# Quaternion (w, x, y, z)
Transform().rotate_quat([0.9239, 0, 0, 0.3827])

# 3x3 Rotation Matrix
rotation_matrix = np.array([[1, 0, 0], [0, 0.866, -0.5], [0, 0.5, 0.866]])
Transform().rotate_matrix(rotation_matrix)

# Axis-Angle (axis and angle in radians)
Transform().rotate_axis_angle([0, 0, 1], 1.57)  # 90 degrees around Z

# Euler Angles (roll, pitch, yaw in radians)
Transform().rotate_euler([0.1, 0.2, 0.3])
```

### Filtering API

#### Basic Filtering

```python
import gsply
from gspro import Filter
from gspro.filter.bounds import calculate_scene_bounds

# Load data
data = gsply.plyread("scene.ply")

# Calculate scene bounds (optional, auto-computed if not provided)
bounds = calculate_scene_bounds(data.means)

# Apply filters
data = (
    Filter()
    .within_sphere(radius=0.8)      # Keep 80% of scene
    .min_opacity(threshold=0.1)     # Remove low-opacity Gaussians
    .max_scale(2.5)                 # Remove large-scale outliers
    .bounds(bounds)                 # Use pre-computed bounds
    (data, inplace=True)
)

gsply.plywrite("filtered.ply", data)
```

#### Filtering Options

```python
import gsply
from gspro import Filter

data = gsply.plyread("scene.ply")

# Option 1: Sphere filtering (keep Gaussians within radius)
data = Filter().within_sphere(
    center=[0, 0, 0],           # Default: scene center
    radius=0.8                  # 80% of max scene dimension
)(data, inplace=True)

# Option 2: Cuboid filtering (keep Gaussians within box)
data = Filter().within_box(
    center=[0, 0, 0],           # Default: scene center
    size=[0.8, 0.8, 0.8]        # 80% of each dimension
)(data, inplace=True)

# Option 3: Property filtering only
data = (
    Filter()
    .min_opacity(threshold=0.05)  # Remove < 5% opacity
    .max_scale(2.5)                # Remove scale outliers
    (data, inplace=True)
)

# Combined filtering
data = (
    Filter()
    .within_sphere(radius=0.8)
    .min_opacity(threshold=0.1)
    .max_scale(2.5)
    (data, inplace=True)
)
```

#### Mask-Based Filtering

The `get_mask()` method allows you to compute boolean masks without copying data, enabling flexible mask composition and inspection:

```python
import gsply
from gspro import Filter

data = gsply.plyread("scene.ply")

# Get mask only (fast - no data copying)
mask = Filter().min_opacity(0.5).max_scale(2.0).get_mask(data)
print(f"Keeping {mask.sum()}/{len(mask)} Gaussians ({mask.sum()/len(mask)*100:.1f}%)")

# Apply mask using GSData slicing
filtered = data[mask]  # or data.copy_slice(mask)

# Combine multiple masks with boolean logic
mask1 = Filter().within_sphere(radius=0.8).get_mask(data)
mask2 = Filter().min_opacity(0.1).get_mask(data)

# AND: Keep only Gaussians that pass both filters
combined_and = mask1 & mask2
result = data[combined_and]

# OR: Keep Gaussians that pass either filter
combined_or = mask1 | mask2
result = data[combined_or]

# NOT: Invert mask to get filtered-out Gaussians
inverse_mask = ~mask1
removed = data[inverse_mask]

# Complex combinations
mask = (
    Filter().min_opacity(0.3).get_mask(data) &
    (Filter().within_sphere(radius=0.8).get_mask(data) |
     Filter().within_box(size=(0.5, 0.5, 0.5)).get_mask(data))
)
result = data[mask]
```

#### Multi-Layer Mask Management (FilterMasks)

For complex filtering scenarios requiring multiple independent mask layers with different combination strategies, use the `FilterMasks` API for managing named mask layers:

```python
import gsply
from gspro import Filter, FilterMasks

data = gsply.plyread("scene.ply")

# Create FilterMasks manager
masks = FilterMasks(data)

# Add multiple named mask layers
masks.add("opacity", Filter().min_opacity(0.3))
masks.add("sphere", Filter().within_sphere(radius=0.8))
masks.add("scale", Filter().max_scale(2.0))

# Inspect mask layers
masks.summary()
# Output:
# Mask Layers (3):
#   opacity: 45,231/100,000 pass (45.2%)
#   sphere:  67,890/100,000 pass (67.9%)
#   scale:   89,123/100,000 pass (89.1%)

# Combine masks with AND logic (all conditions must pass)
combined_and = masks.combine(mode="and")
print(f"{combined_and.sum():,} Gaussians pass all filters")

# Combine masks with OR logic (any condition passes)
combined_or = masks.combine(mode="or")
print(f"{combined_or.sum():,} Gaussians pass any filter")

# Combine specific layers only
mask_subset = masks.combine(mode="and", layers=["opacity", "sphere"])

# Apply masks directly to filter data
filtered = masks.apply(mode="and", inplace=False)
# Or apply only specific layers
filtered = masks.apply(mode="and", layers=["opacity", "scale"], inplace=False)

# Access individual mask layers
opacity_mask = masks["opacity"]  # or masks.get("opacity")

# Remove mask layer
masks.remove("scale")

# Check layer existence
if "opacity" in masks:
    print(f"Opacity layer has {len(masks)} total layers")
```

**Performance:** FilterMasks uses Numba-optimized mask combination with automatic strategy selection:
- **1 layer**: NumPy (0.006ms, lower overhead)
- **2+ layers**: Numba parallel (0.026ms vs 1.447ms numpy, 55x faster)
- **Large-scale (1M Gaussians, 5 layers)**: 0.425ms vs 14.587ms (34x faster)

The mask combination overhead is negligible (3.8% of total filtering time) thanks to Numba optimization, making multi-layer filtering practical for interactive applications.

#### Using Low-Level Utilities

```python
from gspro.filter.bounds import (
    calculate_scene_bounds,
    calculate_recommended_max_scale
)
import gsply

data = gsply.plyread("scene.ply")

# Calculate scene bounds
bounds = calculate_scene_bounds(data.means)
print(f"Scene center: {bounds.center}")
print(f"Scene size: {bounds.sizes}")

# Calculate recommended scale threshold
max_scale = calculate_recommended_max_scale(data.scales, percentile=99.5)
print(f"Recommended max_scale: {max_scale:.4f}")

# Use in filtering
data = (
    Filter()
    .bounds(bounds)
    .max_scale(max_scale)
    (data, inplace=True)
)
```

### Unified Pipeline API (Recommended)

The Pipeline class combines all operations (color, transform, filter) into a single fluent API.

#### Complete Pipeline Example

```python
import gsply
from gspro import Pipeline

# Load data
data = gsply.plyread("scene.ply")

# Create unified pipeline
pipeline = (
    Pipeline()
    # Filtering
    .within_sphere(radius=0.8)
    .min_opacity(0.1)
    .max_scale(2.5)
    # Transforms
    .translate([1, 0, 0])
    .rotate_quat([0.9239, 0, 0, 0.3827])
    .scale(1.5)
    # Color grading
    .temperature(0.6)
    .brightness(1.2)
    .contrast(1.1)
    .saturation(1.3)
)

# Execute pipeline (inplace for zero-copy)
result = pipeline(data, inplace=True)

# Save
gsply.plywrite("output.ply", result)
```

#### Using Color Presets

```python
import gsply
from gspro import ColorPreset, Pipeline

data = gsply.plyread("scene.ply")

# Method 1: Direct preset application
data = ColorPreset.cinematic()(data, inplace=True)

# Method 2: Combine with other operations
pipeline = (
    Pipeline()
    .within_sphere(radius=0.8)
    .translate([1, 0, 0])
)
# Apply preset then other operations
data = ColorPreset.warm()(data, inplace=True)
data = pipeline(data, inplace=True)

# Available presets
# neutral, cinematic, warm, cool, vibrant, muted, dramatic
data = ColorPreset.vibrant()(data, inplace=True)
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

### Pipeline (Unified API)

```python
Pipeline()
```

**Methods (chainable):**

Color operations:
- `temperature(value)` - 0=cool, 0.5=neutral, 1=warm
- `brightness(value)` - Brightness multiplier (1.0=no change)
- `contrast(value)` - Contrast multiplier (1.0=no change)
- `gamma(value)` - Gamma correction (1.0=linear)
- `saturation(value)` - Saturation multiplier (0=grayscale, 1.0=no change)
- `shadows(value)` - Shadow adjustment (1.0=no change)
- `highlights(value)` - Highlight adjustment (1.0=no change)

Transform operations:
- `translate(vector)` - Translation [x, y, z]
- `rotate_quat(quaternion)` - Rotation with quaternion
- `rotate_euler(euler)` - Rotation with euler angles
- `rotate_matrix(matrix)` - Rotation with rotation matrix
- `rotate_axis_angle(axis, angle)` - Rotation with axis-angle
- `scale(factor)` - Uniform or per-axis scale
- `set_center(point)` - Set rotation/scale center [x, y, z]

Filter operations:
- `within_sphere(center=None, radius=0.8)` - Sphere volume filter
- `within_box(center=None, size=[0.8, 0.8, 0.8])` - Cuboid volume filter
- `min_opacity(threshold)` - Minimum opacity threshold (0.0-1.0)
- `max_scale(threshold)` - Maximum scale threshold
- `bounds(scene_bounds)` - Pre-computed scene bounds

Execution:
- `__call__(data, inplace=False)` - Execute pipeline on GSData, returns GSData
- `apply(data, inplace=False)` - Explicit application
- `reset()` - Clear all operations
- `__len__()` - Number of operations
- `__repr__()` - String representation

### Individual Pipeline Classes

```python
Color()      # Color adjustments only
Transform()  # Geometric transforms only
Filter()     # Spatial/property filtering only
```

All classes share the same chainable interface as Pipeline and accept GSData objects.

### ColorPreset

**Built-in presets:**
- `ColorPreset.neutral()` - Identity
- `ColorPreset.cinematic()` - Desaturated, high contrast, teal/orange
- `ColorPreset.warm()` - Orange tones, boosted brightness
- `ColorPreset.cool()` - Blue tones, crisp contrast
- `ColorPreset.vibrant()` - Boosted saturation/contrast
- `ColorPreset.muted()` - Low saturation, lifted shadows
- `ColorPreset.dramatic()` - High contrast, crushed shadows

**Usage:**
```python
data = ColorPreset.cinematic()(data, inplace=True)
```

### Low-Level Utilities

Quaternion operations:
```python
from gspro.transforms import (
    quaternion_multiply,
    quaternion_to_rotation_matrix,
    rotation_matrix_to_quaternion,
    axis_angle_to_quaternion,
    euler_to_quaternion,
    quaternion_to_euler
)
```

Scene bounds:
```python
from gspro.filter.bounds import (
    calculate_scene_bounds,        # -> SceneBounds
    calculate_recommended_max_scale  # -> float
)

bounds = calculate_scene_bounds(positions)
# Returns: SceneBounds(min, max, sizes, max_size, center)

max_scale = calculate_recommended_max_scale(scales, percentile=99.5)
```

**Rotation methods:** `rotate_quat()`, `rotate_matrix()`, `rotate_axis_angle()`, `rotate_euler()`

## Example: Full Processing Pipeline

```python
import gsply
from gspro import Pipeline
from gspro.filter.bounds import calculate_scene_bounds

# Load Gaussian splatting data
data = gsply.plyread("scene.ply")

# Optional: Pre-compute scene bounds for filtering
bounds = calculate_scene_bounds(data.means)

# Create unified pipeline combining all operations
pipeline = (
    Pipeline()
    # Filtering (remove unwanted Gaussians)
    .within_sphere(radius=0.8)      # Keep 80% of scene
    .min_opacity(0.1)               # Remove low-opacity
    .max_scale(2.5)                 # Remove large-scale outliers
    .bounds(bounds)                 # Use pre-computed bounds
    # Geometric transforms
    .translate([1.0, 0.0, 0.0])     # Move scene
    .rotate_quat([0.9239, 0, 0, 0.3827])  # Rotate
    .scale(1.5)                     # Scale up 1.5x
    # Color grading
    .temperature(0.6)               # Cool tones
    .brightness(1.2)                # Increase brightness
    .contrast(1.1)                  # Boost contrast
    .saturation(1.3)                # Vibrant colors
)

# Execute pipeline (inplace for zero-copy performance)
result = pipeline(data, inplace=True)

# Save processed scene
gsply.plywrite("output.ply", result)

# Performance notes:
# - inplace=True: Zero-copy modification for maximum performance
# - Filtering: 62M Gaussians/sec full pipeline
# - Transforms: 698M Gaussians/sec combined operations
# - Colors: 1,389M colors/sec with LUT-based processing
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

# Set up pre-commit hooks (recommended)
pre-commit install

# Run tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=gspro --cov-report=html
```

### Pre-commit Hooks

This project uses [pre-commit](https://pre-commit.com/) to maintain code quality:

```bash
# Install hooks (one-time setup)
pre-commit install

# Run manually on all files
pre-commit run --all-files

# Update hook versions
pre-commit autoupdate
```

The pre-commit hooks will automatically:
- Run ruff linting with auto-fix
- Format code with ruff
- Check for common issues (trailing whitespace, YAML syntax, etc.)
- Validate Python syntax

See [.github/PRE_COMMIT_SETUP.md](.github/PRE_COMMIT_SETUP.md) for detailed setup instructions.

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

- **gsply**: Ultra-fast Gaussian Splatting PLY I/O library (required dependency)
  - v0.3.0+ adds concatenation optimizations (6.15x faster bulk merging)
  - `make_contiguous()` for manual optimization of iterative workflows (100+ operations)
  - Multi-layer mask management (used by FilterMasks in gspro)
  - See [gsply documentation](https://github.com/OpsiClear/gsply) for details
- **gsplat**: CUDA-accelerated Gaussian Splatting rasterizer
- **nerfstudio**: NeRF training framework with Gaussian Splatting support
- **3D Gaussian Splatting**: Original paper and implementation

---

<div align="center">

**Made with Python and NumPy**

[Report Bug](https://github.com/OpsiClear/gspro/issues) | [Request Feature](https://github.com/OpsiClear/gspro/issues) | [Documentation](.github/WORKFLOWS.md)

</div>
