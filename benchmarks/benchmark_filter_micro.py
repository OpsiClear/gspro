"""
Micro-benchmark for filter_gaussians optimization.

Focused benchmark to measure performance improvements.
"""

import logging
import time

import numpy as np

from gspro.filter import (
    calculate_scene_bounds,
    filter_gaussians,
)

# Suppress logging
logging.getLogger("gspro.filter.api").setLevel(logging.WARNING)


def generate_data(n: int):
    """Generate sample data."""
    np.random.seed(42)
    positions = np.random.randn(n, 3).astype(np.float32) * 2.0
    quaternions = np.random.randn(n, 4).astype(np.float32)
    quaternions = quaternions / np.linalg.norm(quaternions, axis=1, keepdims=True)
    scales = np.random.rand(n, 3).astype(np.float32) * 2.0
    opacities = np.random.rand(n).astype(np.float32)
    colors = np.random.rand(n, 3).astype(np.float32)
    return positions, quaternions, scales, opacities, colors


def main():
    print("=" * 80)
    print("FILTER_GAUSSIANS MICRO-BENCHMARK")
    print("=" * 80)

    n = 1_000_000
    iterations = 200

    print(f"\nGenerating {n:,} Gaussians...")
    positions, quaternions, scales, opacities, colors = generate_data(n)
    bounds = calculate_scene_bounds(positions)

    print(f"Warming up ({iterations} iterations)...")
    for _ in range(iterations):
        filter_gaussians(
            positions,
            quaternions,
            scales,
            opacities,
            colors,
            filter_type="sphere",
            sphere_radius_factor=0.8,
            opacity_threshold=0.05,
            max_scale=2.0,
            scene_bounds=bounds,
        )

    print(f"\nBenchmarking ({iterations} iterations)...")
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        new_pos, new_quats, new_scales, new_opac, new_colors = filter_gaussians(
            positions,
            quaternions,
            scales,
            opacities,
            colors,
            filter_type="sphere",
            sphere_radius_factor=0.8,
            opacity_threshold=0.05,
            max_scale=2.0,
            scene_bounds=bounds,
        )
        end = time.perf_counter()
        times.append((end - start) * 1000)

    avg_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)
    max_time = np.max(times)
    throughput = n / (avg_time / 1000) / 1e6

    print("\nResults:")
    print(f"  Mean:       {avg_time:.3f} ms +/- {std_time:.3f} ms")
    print(f"  Min:        {min_time:.3f} ms")
    print(f"  Max:        {max_time:.3f} ms")
    print(f"  Throughput: {throughput:.1f}M Gaussians/sec")
    print(f"  Kept:       {len(new_pos):,} / {n:,} ({len(new_pos)/n*100:.1f}%)")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
