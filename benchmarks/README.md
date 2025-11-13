# gslut Benchmarks

Performance benchmarks for CPU-optimized gslut library.

## Quick Start

```bash
# Run all benchmarks
cd benchmarks
uv run run_all_benchmarks.py

# Run individual benchmarks
uv run benchmark_transform.py
uv run benchmark_color.py
```

## Benchmarks

### 1. Transform Benchmark (`benchmark_transform.py`)

Tests 3D Gaussian transform performance with fused Numba kernel.

**Features:**
- Fused kernel: Single parallel loop combining all operations
- Custom matrix multiply: 9x faster than BLAS for 3x3 matrices
- Memory locality: Process each Gaussian completely before moving to next
- Output buffer reuse: Pre-allocated arrays eliminate allocation overhead

**Tests:**
- 1M Gaussians with 100 iterations (main test)
- Batch size scaling (10K to 2M)
- Real-world use cases (animation, real-time rendering)

**Expected results:**
- Time: ~1.5ms per 1M Gaussians
- Throughput: 600-700M Gaussians/sec
- Peak: 1.6-2.3 billion G/s at 500K batch size

### 2. Color LUT Benchmark (`benchmark_color.py`)

Tests color adjustment performance using separated 1D LUTs.

**Features:**
- Separated 1D LUTs per channel: 10x faster than sequential operations
- CPU-optimized via NumPy: 2-3x faster than PyTorch CPU
- 7 color operations: temperature, brightness, contrast, gamma, saturation, shadows, highlights

**Tests:**
- Various batch sizes (1K to 1M points)
- Individual operations
- Different LUT resolutions (256 to 4096 bins)

**Expected results:**
- Time: ~38ms per 1M colors (CPU)
- Throughput: 26M colors/sec
- Individual ops: 30-54M points/sec

## Performance Summary

### Transform Performance (1M Gaussians)

```
Time:       1.5 ms
Throughput: 658 M Gaussians/sec
Max FPS:    633 FPS (1.58ms per frame)
```

**Batch size scaling:**
```
   10K:   0.05 ms (185 M G/s)
  100K:   0.11 ms (936 M G/s)
  500K:   0.23 ms (2,146 M G/s)  <- Peak performance
    1M:   1.97 ms (507 M G/s)
    2M:   5.70 ms (351 M G/s)
```

### Color LUT Performance (CPU)

```
     1K:   0.33 ms (3.1 M/s)
    10K:   0.92 ms (10.9 M/s)
   100K:   3.37 ms (29.7 M/s)
     1M:  37.96 ms (26.3 M/s)
```

## Understanding Results

### Performance Metrics

- **ms (milliseconds)**: Time per operation
- **G/s (Gaussians/sec)**: Throughput for transforms (higher is better)
- **points/sec**: Throughput for color LUT (higher is better)

### Key Optimizations

**Transform:**
- Fused Numba kernel: Combines all operations in single parallel loop
- Custom matmul: 9x faster than BLAS for 3x3 matrices
- Memory locality: Process each Gaussian completely
- Parallel execution: Distributes work across all CPU cores

**ColorLUT:**
- Separated 1D LUTs: One per channel (R, G, B)
- CPU-optimized: Fast NumPy path for CPU processing
- LUT caching: Reuse LUTs when parameters unchanged

## System Requirements

- Python 3.10+
- NumPy 1.24+
- Numba 0.58+ (for transform optimizations)
- PyTorch 2.0+ (for color LUT)
- Multi-core CPU recommended
- 8GB+ RAM for large batches

## Sample Output

### Transform Benchmark

```
================================================================================
3D GAUSSIAN TRANSFORM BENCHMARK
Testing with 1,000,000 Gaussians, 100 iterations
================================================================================

Results (1M Gaussians):
  Time:       1.580 ms +/- 0.162 ms
  Throughput: 632.7M Gaussians/sec

================================================================================
BATCH SIZE SCALING
================================================================================
N=   10,000:   0.05 ms ( 185.2M G/s)
N=  100,000:   0.11 ms ( 935.9M G/s)
N=  500,000:   0.23 ms (2146.2M G/s)
N=1,000,000:   1.97 ms ( 507.2M G/s)
N=2,000,000:   5.70 ms ( 351.1M G/s)
```

### Color LUT Benchmark

```
================================================================================
COLOR LUT BENCHMARK (CPU)
================================================================================

Batch Size: 100,000 points
  Time:       3.373 +/- 0.335 ms
  Throughput: 29,650,890 points/sec

================================================================================
INDIVIDUAL OPERATIONS (100K points)
================================================================================

Temperature only:
  Time:       1.954 +/- 0.092 ms
  Throughput: 51,176,627 points/sec

All operations:
  Time:       3.344 +/- 0.220 ms
  Throughput: 29,907,275 points/sec
```

## Interpretation

### Transform Performance

**Optimal batch size: 500K Gaussians**
- Below 100K: Overhead dominates
- 100K-500K: Best throughput (1-2 billion G/s)
- Above 1M: Cache thrashing reduces performance

**Real-world usage:**
- 1M Gaussians @ 633 FPS: Real-time capable
- Animation (100 frames): 158ms total
- Well below 16.67ms target for 60 FPS rendering

### Color LUT Performance

**Throughput characteristics:**
- Individual ops: 30-54M points/sec
- All operations: 26-30M points/sec
- LUT resolution has minimal impact (256-4096)

## Troubleshooting

### Slow Transform Performance

Check Numba installation:
```bash
pip install numba
```

Verify fused kernel is active (should see "Numba available: True" in output).

### Out of Memory

Reduce batch sizes in benchmark scripts:
```python
batch_sizes = [10_000, 100_000, 500_000]  # Instead of up to 2M
```

### Slow Benchmarks

Reduce iterations:
```python
NUM_ITERATIONS = 50  # Instead of 100
```

## Documentation

See detailed performance analysis:
- `CURRENT_PERFORMANCE_SUMMARY.md`: Comprehensive performance report
- `FINAL_OPTIMIZATION_SUMMARY.md`: Optimization journey and results
- `CORRECTNESS_VERIFICATION.md`: Verification of fused kernel correctness

## Contributing

To add a new benchmark:

1. Create `benchmark_<feature>.py`
2. Follow the existing pattern:
   - Use `benchmark_function()` helper if needed
   - Test multiple batch sizes
   - Print clear, formatted results
   - Use ASCII characters only (no Unicode)
3. Add to `run_all_benchmarks.py`
4. Update this README

## License

MIT License - see parent directory LICENSE file.
