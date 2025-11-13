"""
Benchmark 3D Gaussian transform performance (CPU with fused Numba kernel).
"""

import time
import numpy as np
from gspro import transform

N = 1_000_000
NUM_ITERATIONS = 100

print("=" * 80)
print("3D GAUSSIAN TRANSFORM BENCHMARK")
print(f"Testing with {N:,} Gaussians, {NUM_ITERATIONS} iterations")
print("=" * 80)

# Setup
means_np = np.random.randn(N, 3).astype(np.float32)
quats_np = np.random.randn(N, 4).astype(np.float32)
quats_np = quats_np / np.linalg.norm(quats_np, axis=1, keepdims=True)
scales_np = np.random.rand(N, 3).astype(np.float32)

translation_np = np.array([1.0, 2.0, 3.0], dtype=np.float32)
rotation_np = np.array([0.9239, 0.0, 0.0, 0.3827], dtype=np.float32)
scale_factor = 2.0

out_means_np = np.empty_like(means_np)
out_quats_np = np.empty_like(quats_np)
out_scales_np = np.empty_like(scales_np)

# Warmup
print("\nWarming up...")
for _ in range(20):
    transform(means_np, quats_np, scales_np,
              translation=translation_np, rotation=rotation_np, scale_factor=scale_factor,
              out_means=out_means_np, out_quaternions=out_quats_np, out_scales=out_scales_np)

# Benchmark
print(f"Benchmarking {NUM_ITERATIONS} iterations...")
times = []
for _ in range(NUM_ITERATIONS):
    start = time.perf_counter()
    transform(means_np, quats_np, scales_np,
              translation=translation_np, rotation=rotation_np, scale_factor=scale_factor,
              out_means=out_means_np, out_quaternions=out_quats_np, out_scales=out_scales_np)
    times.append((time.perf_counter() - start) * 1000)

mean_time = np.mean(times)
std_time = np.std(times)

print(f"\nResults (1M Gaussians):")
print(f"  Time:       {mean_time:.3f} ms +/- {std_time:.3f} ms")
print(f"  Throughput: {N/mean_time*1000/1e6:.1f}M Gaussians/sec")

# Test different batch sizes
print("\n" + "=" * 80)
print("BATCH SIZE SCALING")
print("=" * 80)

batch_sizes = [10_000, 100_000, 500_000, 1_000_000, 2_000_000]

for N_test in batch_sizes:
    means_test = np.random.randn(N_test, 3).astype(np.float32)
    quats_test = np.random.randn(N_test, 4).astype(np.float32)
    quats_test = quats_test / np.linalg.norm(quats_test, axis=1, keepdims=True)
    scales_test = np.random.rand(N_test, 3).astype(np.float32)

    out_means_test = np.empty_like(means_test)
    out_quats_test = np.empty_like(quats_test)
    out_scales_test = np.empty_like(scales_test)

    # Warmup
    for _ in range(5):
        transform(means_test, quats_test, scales_test,
                  translation=translation_np, rotation=rotation_np, scale_factor=scale_factor,
                  out_means=out_means_test, out_quaternions=out_quats_test, out_scales=out_scales_test)

    times_test = []
    for _ in range(20):
        start = time.perf_counter()
        transform(means_test, quats_test, scales_test,
                  translation=translation_np, rotation=rotation_np, scale_factor=scale_factor,
                  out_means=out_means_test, out_quaternions=out_quats_test, out_scales=out_scales_test)
        times_test.append((time.perf_counter() - start) * 1000)

    test_time = np.mean(times_test)
    throughput = N_test / test_time * 1000 / 1e6

    print(f"N={N_test:>9,}: {test_time:>6.2f} ms ({throughput:>6.1f}M G/s)")

# Real-world use case
print("\n" + "=" * 80)
print("REAL-WORLD USE CASES")
print("=" * 80)

print(f"\nAnimation processing (1M Gaussians):")
print(f"  1 frame:    {mean_time:.1f} ms")
print(f"  100 frames: {mean_time * 100:.0f} ms ({mean_time * 100 / 1000:.2f}s)")
print(f"  1000 frames: {mean_time * 1000 / 1000:.1f}s")

print(f"\nReal-time rendering:")
print(f"  Max FPS:    {1000 / mean_time:.0f} FPS")
print(f"  Frame time: {mean_time:.2f} ms (target: 16.67ms for 60 FPS)")

print("\n" + "=" * 80)
