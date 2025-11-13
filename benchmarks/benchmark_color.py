"""
Benchmark color processing performance (pure NumPy/Numba, no PyTorch).
"""

import time
import numpy as np
from gspro import ColorLUT

N = 100_000
NUM_ITERATIONS = 100

print("=" * 80)
print("COLOR PROCESSING BENCHMARK (NumPy/Numba)")
print(f"Testing with {N:,} colors, {NUM_ITERATIONS} iterations")
print("=" * 80)

# Setup
colors_np = np.random.rand(N, 3).astype(np.float32)
out_colors_np = np.empty_like(colors_np)

lut = ColorLUT()

# Test parameters
params = {
    "temperature": 0.6,
    "brightness": 1.2,
    "contrast": 1.1,
    "saturation": 1.3,
    "shadows": 1.1,
    "highlights": 0.9,
}

# Warmup
print("\nWarming up...")
for _ in range(20):
    lut.apply_numpy_inplace(colors_np, out_colors_np, **params)

# Benchmark apply_numpy_inplace (zero-copy)
print(f"\nBenchmarking apply_numpy_inplace() - {NUM_ITERATIONS} iterations...")
times_inplace = []
for _ in range(NUM_ITERATIONS):
    start = time.perf_counter()
    lut.apply_numpy_inplace(colors_np, out_colors_np, **params)
    times_inplace.append((time.perf_counter() - start) * 1000)

mean_time_inplace = np.mean(times_inplace)
std_time_inplace = np.std(times_inplace)

print(f"\nResults (apply_numpy_inplace - zero-copy):")
print(f"  Time:       {mean_time_inplace:.3f} ms +/- {std_time_inplace:.3f} ms")
print(f"  Throughput: {N/mean_time_inplace*1000/1e6:.0f}M colors/sec")

# Benchmark apply_numpy (with allocation)
print(f"\nBenchmarking apply_numpy() - {NUM_ITERATIONS} iterations...")
times_numpy = []
for _ in range(NUM_ITERATIONS):
    start = time.perf_counter()
    result = lut.apply_numpy(colors_np, **params)
    times_numpy.append((time.perf_counter() - start) * 1000)

mean_time_numpy = np.mean(times_numpy)
std_time_numpy = np.std(times_numpy)

print(f"\nResults (apply_numpy - with allocation):")
print(f"  Time:       {mean_time_numpy:.3f} ms +/- {std_time_numpy:.3f} ms")
print(f"  Throughput: {N/mean_time_numpy*1000/1e6:.0f}M colors/sec")

# Benchmark apply (standard API)
print(f"\nBenchmarking apply() - {NUM_ITERATIONS} iterations...")
times_apply = []
for _ in range(NUM_ITERATIONS):
    start = time.perf_counter()
    result = lut.apply(colors_np, **params)
    times_apply.append((time.perf_counter() - start) * 1000)

mean_time_apply = np.mean(times_apply)
std_time_apply = np.std(times_apply)

print(f"\nResults (apply - standard API):")
print(f"  Time:       {mean_time_apply:.3f} ms +/- {std_time_apply:.3f} ms")
print(f"  Throughput: {N/mean_time_apply*1000/1e6:.0f}M colors/sec")

# Speedup comparison
print("\n" + "=" * 80)
print("SPEEDUP COMPARISON")
print("=" * 80)
print(f"apply_numpy_inplace vs apply_numpy: {mean_time_numpy/mean_time_inplace:.2f}x faster")
print(f"apply_numpy_inplace vs apply:       {mean_time_apply/mean_time_inplace:.2f}x faster")

# Test different batch sizes
print("\n" + "=" * 80)
print("BATCH SIZE SCALING")
print("=" * 80)

batch_sizes = [1_000, 10_000, 100_000, 1_000_000]

for N_test in batch_sizes:
    colors_test = np.random.rand(N_test, 3).astype(np.float32)
    out_test = np.empty_like(colors_test)

    # Warmup
    for _ in range(5):
        lut.apply_numpy_inplace(colors_test, out_test, **params)

    times_test = []
    for _ in range(20):
        start = time.perf_counter()
        lut.apply_numpy_inplace(colors_test, out_test, **params)
        times_test.append((time.perf_counter() - start) * 1000)

    test_time = np.mean(times_test)
    throughput = N_test / test_time * 1000 / 1e6

    print(f"N={N_test:>9,}: {test_time:>7.3f} ms ({throughput:>5.0f}M colors/s)")

print("\n" + "=" * 80)
