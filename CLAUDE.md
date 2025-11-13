# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

gspro (Gaussian Splatting Processing) is a Python library providing fast LUT-based operations for 3D Gaussian Splatting, including color adjustments and 3D transformations. The library supports both NumPy (CPU) and PyTorch (CPU/GPU) backends with automatic dispatch.

## Development Commands

### Running Scripts
- Use `uv run <script.py>` instead of `python <script.py>`
- Example: `uv run benchmarks/benchmark_color_lut.py`

### Testing
```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_color.py -v

# Run with coverage
pytest tests/ -v --cov=gspro --cov-report=term-missing
```

### Code Quality
```bash
# Format code (auto-fix)
ruff format src/ tests/

# Lint (check only)
ruff check src/ tests/

# Lint with auto-fix
ruff check --fix src/ tests/
```

### Benchmarks
```bash
cd benchmarks

# Run all benchmarks
uv run run_all_benchmarks.py

# Run individual benchmarks
uv run benchmark_color_lut.py
```

### Building Package
```bash
# Build wheel
python -m build

# Check package
twine check dist/*
```

## Architecture

### Core Modules

**src/gspro/color.py** - `ColorLUT` class
- Separated 1D LUTs for RGB color adjustments (10x faster than sequential ops)
- Per-channel operations: temperature, brightness, contrast, gamma, saturation, shadows, highlights
- CPU-optimized via NumPy path (2-3x faster on CPU)

**src/gspro/transforms.py** - 3D geometric transformations
- Dual implementation: NumPy and PyTorch with automatic dispatch based on input type
- Fused `transform()` operation using 4x4 homogeneous transformation matrices
- **NEW: Output buffer support** - Optional `out_*` parameters for 2x performance gain
- **NEW: `transform_fast()`** - Fast-path function with minimal overhead (2.09x speedup)
- **NEW: Contiguity checks** - Auto-converts non-contiguous arrays to prevent 45% slowdown
- Rotation formats: quaternion, matrix, axis_angle, euler
- Quaternion utilities for orientation manipulation
- Individual functions (`translate()`, `rotate()`, `scale()`) available for direct use
- Numba-optimized NumPy backend with automatic fallback

**src/gspro/numba_ops.py** - Numba JIT-compiled kernels
- `quaternion_multiply_single_numba()`: 200x faster than pure NumPy
- `quaternion_multiply_batched_numba()`: Batched quaternion multiplication
- `apply_transform_matrix_numba()`: 3x faster matrix application
- Elementwise operations with parallel processing
- Automatic warmup on import to avoid first-call compilation overhead
- Graceful fallback when Numba not installed

**src/gspro/pipeline.py** - High-level composable API
- `Pipeline` class: Chainable operations using fused transformations
- Single `.transform()` method with all parameters (scale, rotate, translate)
- `ColorPreset` class: Pre-configured color grading looks (cinematic, warm, cool, vibrant, muted, dramatic)
- Functional API: `adjust_colors()`, `apply_preset()`

**src/gspro/utils.py** - Shared utilities
- `linear_interp_1d()`: Fast 1D linear interpolation
- `nearest_neighbor_1d()`: Nearest neighbor lookup

### Design Patterns

1. **Automatic Backend Dispatch**: All transform functions check input type (np.ndarray vs torch.Tensor) and dispatch to appropriate implementation
2. **Lazy Pipeline Execution**: Pipeline operations are only executed when called, allowing flexible composition
3. **Type Preservation**: Functions preserve input types (NumPy in -> NumPy out, Torch in -> Torch out)
4. **Separated LUT Architecture**: ColorLUT uses per-channel 1D LUTs instead of 3D LUTs for 10x performance gain
5. **Fused Matrix Transforms**: Transform operations use pre-computed 4x4 homogeneous matrices for 2-3x speedup

### Test Structure

Tests mirror the source structure:
- `tests/test_color.py`: ColorLUT tests
- `tests/test_transforms.py`: PyTorch transform tests
- `tests/test_transforms_numpy.py`: NumPy transform tests
- `tests/test_pipeline.py`: High-level API tests
- `tests/test_numba_ops.py`: Numba kernel correctness and integration tests

## Key Considerations

### Type Hints
- Use Python 3.12+ type syntax (e.g., `list[int]` not `List[int]`)
- Use union syntax `X | Y` not `Union[X, Y]`
- Be explicit with array/tensor types: `np.ndarray | torch.Tensor`

### Logging vs Printing
- Use Python logging module, not print() statements
- Exception: Benchmark scripts print results for readability

### NumPy/PyTorch Compatibility
- When adding new transforms, implement both `_func_numpy()` and `_func_torch()` versions
- Public API dispatches based on input type: `isinstance(input, np.ndarray)`
- Test both backends thoroughly

### Performance Notes
- ColorLUT: Always 10x+ faster via separated 1D LUTs
- Transform: Fused 4x4 matrix approach (single matmul vs 3 operations)
- **NEW: Output buffer optimizations:**
  - Buffer reuse: 2.07x faster (eliminates allocation overhead)
  - Fast-path function: 2.09x faster (eliminates validation overhead)
  - Contiguity checks: Prevents 45% slowdown from non-contiguous arrays
- Numba JIT compilation:
  - Quaternion operations: 200x faster
  - Elementwise operations: 9-54x faster
  - Matrix multiplication: Uses NumPy BLAS (OpenBLAS/MKL) - faster than Numba
- **Updated benchmark results (1M Gaussians):**
  - PyTorch CPU: ~13.0ms (77M Gaussians/sec)
  - NumPy baseline: ~17.0ms (59M Gaussians/sec) - 1.31x slower
  - **NumPy optimized: ~8.2ms (121M Gaussians/sec) - 1.59x FASTER than PyTorch!**
  - NumPy ultra-fast: ~8.2ms (123M Gaussians/sec) with `transform_fast()`
  - NumPy without Numba: ~53ms (falls back gracefully)
- Numba is optional: Falls back to pure NumPy if not installed
- Small datasets (<10K): NumPy + Numba is 4x faster than PyTorch
- **For batch processing (100 frames): Optimizations save ~0.88 seconds total**

### Ruff Configuration
- Line length: 100 chars
- Target: Python 3.10+
- Ignores: E501 (line too long), N803/N806/N812 (uppercase variables like R for rotation matrix)
- transforms.py: Union syntax allowed for type aliases

## CI/CD

GitHub Actions workflow (.github/workflows/ci.yml):
- **test**: Runs on Ubuntu/Windows/macOS with Python 3.10/3.11/3.12
- **lint**: Checks formatting and linting with ruff
- **build**: Builds package and validates with twine

## Common Patterns

### Adding a New Transform Function
1. Implement `_func_numpy(...)` in transforms.py
2. Implement `_func_torch(...)` in transforms.py
3. Add public API dispatcher that checks `isinstance(input, np.ndarray)`
4. Add tests in both test_transforms.py and test_transforms_numpy.py
5. Update __init__.py exports
6. Update README.md examples

### Creating Pipelines
```python
# Color pipeline
color_pipeline = (
    Pipeline(device="cuda")
    .adjust_colors(brightness=1.2, contrast=1.1, saturation=1.3)
)
result = color_pipeline(rgb_colors)

# Transform pipeline (fused matrix approach)
transform_pipeline = Pipeline().transform(
    scale_factor=2.0,
    rotation=quaternion,
    translation=[1.0, 0.0, 0.0],
    center=[0.0, 0.0, 0.0]
)
result = transform_pipeline(gaussian_data)  # dict with means, quaternions, scales
```
