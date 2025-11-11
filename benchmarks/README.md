# gslut Benchmarks

Comprehensive performance benchmarks comparing NumPy, PyTorch, torch.compile, CPU, and GPU implementations.

## Quick Start

```bash
# Run all benchmarks
cd benchmarks
python run_all_benchmarks.py

# Run individual benchmarks
python benchmark_color_lut.py
python benchmark_activation_lut.py
python benchmark_conversions.py
```

## Benchmarks

### 1. ColorLUT Benchmark (`benchmark_color_lut.py`)

Compares color adjustment performance across:
- **Regular PyTorch** (CPU/GPU)
- **torch.compile** (CPU/GPU)
- **NumPy optimization** (CPU only)

Tests:
- Full color pipeline (all 7 adjustments)
- Individual operations (temperature, brightness, etc.)
- Different LUT resolutions (256 to 4096 bins)
- Various batch sizes (1K to 1M points)

Expected results:
- GPU: 5-10x faster than CPU for large batches
- torch.compile: 1.2-2x speedup on GPU
- NumPy (CPU): 2-3x faster than pure PyTorch

### 2. ActivationLUT Benchmark (`benchmark_activation_lut.py`)

Compares activation function approximations:
- **Native torch.exp/sigmoid** (baseline)
- **LUT with linear interpolation**
- **LUT with nearest neighbor**
- **torch.compile optimized**

Tests:
- exp() and sigmoid() operations
- Different cluster counts (128 to 4096)
- Accuracy vs performance tradeoff
- Various batch sizes (10K to 10M values)

Expected results:
- **Modern GPUs**: Native is usually faster (highly optimized exp/sigmoid)
- **CPU**: LUT can be competitive for very large batches
- **Accuracy**: 0.0002% error with 2048 clusters + interpolation
- **Use case**: Best for CPU inference or deterministic results

### 3. Conversion Benchmark (`benchmark_conversions.py`)

Tests SH <-> RGB conversion performance:
- **Regular implementation**
- **torch.compile optimized**
- **CPU vs GPU**

Tests:
- rgb2sh() and sh2rgb() operations
- Roundtrip conversion accuracy
- Different tensor shapes (1D to 5D)
- Different data types (float16, float32, float64)
- Memory transfer overhead (CPU <-> GPU)

Expected results:
- GPU: 10-100x faster than CPU
- torch.compile: 1.5-3x speedup
- Small batches: CPU faster due to transfer overhead
- Large batches: GPU much faster

## Understanding Results

### Performance Metrics

- **ms (milliseconds)**: Time per operation
- **points/sec**: Throughput (higher is better)
- **x speedup**: Speedup vs baseline
- **% error**: Mean relative error for LUT approximations

### When to Use Each Method

#### ColorLUT
- **CPU (small batches < 10K)**: Use NumPy-optimized ColorLUT
- **CPU (large batches)**: Use NumPy-optimized ColorLUT
- **GPU (any size)**: Use regular ColorLUT on GPU

#### ActivationLUT
- **Modern GPUs**: Stick with native torch.exp/sigmoid (faster)
- **CPU inference**: Consider LUT for large batches
- **Deterministic results**: Use LUT (reproducible across platforms)
- **Memory-constrained**: Use LUT (reduces bandwidth)

#### Conversions
- **Small batches (< 100K)**: CPU (avoid transfer overhead)
- **Large batches (> 100K)**: GPU (transfer overhead amortized)
- **Repeated operations**: Keep data on GPU

### torch.compile Notes

- **Warmup required**: 10-20 iterations for compilation
- **Best for**: Repeated operations, large batches
- **Overhead**: First call is slow (compilation)
- **PyTorch 2.0+ only**: Check with `hasattr(torch, 'compile')`

## Sample Output

```
ColorLUT Benchmark: NumPy vs PyTorch vs torch.compile
================================================================================

--- Batch Size: 100,000 points ---

Device: CPU
  Regular:           12.345 ± 0.234 ms
                      8,101,852 points/sec
  torch.compile:      8.123 ± 0.156 ms
                     12,312,312 points/sec
                          1.52x speedup

Device: CUDA
  Regular:            1.234 ± 0.023 ms
                     81,037,277 points/sec
  torch.compile:      0.723 ± 0.012 ms
                    138,313,131 points/sec
                          1.71x speedup

Summary: Performance Comparison
================================================================================
CPU vs GPU Speedup:
  100,000 points:  10.00x faster on GPU
```

## System Requirements

- PyTorch >= 2.0.0 (for torch.compile)
- CUDA-capable GPU (optional, for GPU benchmarks)
- 8GB+ RAM recommended
- ~5-10 minutes to run all benchmarks

## Troubleshooting

### CUDA Out of Memory
Reduce batch sizes in the benchmark scripts:
```python
batch_sizes = [1000, 10000, 100000]  # Instead of [... 1000000]
```

### torch.compile Errors
Update PyTorch to 2.0+ or skip torch.compile tests:
```bash
pip install --upgrade torch
```

### Slow Benchmarks
Reduce iterations:
```python
benchmark_function(func, warmup=5, iterations=50)  # Instead of 100
```

## Contributing

To add a new benchmark:

1. Create `benchmark_<feature>.py`
2. Follow the existing pattern:
   - Use `benchmark_function()` helper
   - Test multiple batch sizes and devices
   - Include torch.compile comparison
   - Print clear, formatted results
3. Add to `run_all_benchmarks.py`
4. Update this README

## License

MIT License - see parent directory LICENSE file.
