"""
Detailed benchmark comparing optimization improvements.

Tests individual filter operations to measure impact of:
- fastmath=True on filter kernels
- Fused opacity+scale filter kernel
"""

import logging
import time

import numpy as np

from gspro.filter import (
    apply_filter,
    calculate_scene_bounds,
    filter_gaussians,
)

logging.getLogger("gspro.filter.api").setLevel(logging.WARNING)


def generate_data(n: int, pass_rate: float = 0.94):
    """Generate sample data."""
    np.random.seed(42)
    positions = np.random.randn(n, 3).astype(np.float32) * 2.0
    quaternions = np.random.randn(n, 4).astype(np.float32)
    quaternions = quaternions / np.linalg.norm(quaternions, axis=1, keepdims=True)
    scales = np.random.rand(n, 3).astype(np.float32) * 2.0
    opacities = np.random.rand(n).astype(np.float32)
    colors = np.random.rand(n, 3).astype(np.float32)

    return positions, quaternions, scales, opacities, colors


def benchmark_operation(name, func, iterations=200):
    """Benchmark a specific operation."""
    times = []

    for _ in range(iterations):
        start = time.perf_counter()
        func()
        end = time.perf_counter()
        times.append((end - start) * 1000)

    times_arr = np.array(times)
    return {
        "name": name,
        "mean": np.mean(times_arr),
        "std": np.std(times_arr),
        "min": np.min(times_arr),
        "p50": np.percentile(times_arr, 50),
        "p95": np.percentile(times_arr, 95),
        "max": np.max(times_arr),
    }


def main():
    print("=" * 80)
    print("OPTIMIZATION IMPACT BENCHMARK")
    print("=" * 80)

    n = 1_000_000
    iterations = 200
    warmup = 100

    print(f"\nGenerating {n:,} Gaussians...")
    positions, quaternions, scales, opacities, colors = generate_data(n)
    bounds = calculate_scene_bounds(positions)

    # Test configurations
    tests = [
        {
            "name": "Sphere filter only (fastmath)",
            "func": lambda: apply_filter(
                positions,
                filter_type="sphere",
                sphere_radius_factor=0.8,
                scene_bounds=bounds,
            ),
        },
        {
            "name": "Opacity filter only (fastmath)",
            "func": lambda: apply_filter(
                positions,
                opacities=opacities,
                opacity_threshold=0.05,
            ),
        },
        {
            "name": "Scale filter only (fastmath)",
            "func": lambda: apply_filter(
                positions,
                scales=scales,
                max_scale=2.0,
            ),
        },
        {
            "name": "Opacity + Scale (fused kernel)",
            "func": lambda: apply_filter(
                positions,
                opacities=opacities,
                scales=scales,
                opacity_threshold=0.05,
                max_scale=2.0,
            ),
        },
        {
            "name": "Full filtering (sphere + fused)",
            "func": lambda: apply_filter(
                positions,
                opacities=opacities,
                scales=scales,
                filter_type="sphere",
                sphere_radius_factor=0.8,
                opacity_threshold=0.05,
                max_scale=2.0,
                scene_bounds=bounds,
            ),
        },
        {
            "name": "Full filter_gaussians pipeline",
            "func": lambda: filter_gaussians(
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
            ),
        },
    ]

    results = []
    for test in tests:
        print(f"\n{test['name']}:")
        print(f"  Warming up ({warmup} iterations)...")

        # Warmup
        for _ in range(warmup):
            test["func"]()

        print(f"  Benchmarking ({iterations} iterations)...")
        result = benchmark_operation(test["name"], test["func"], iterations)
        results.append(result)

        print(f"  Mean:  {result['mean']:.3f} ms +/- {result['std']:.3f} ms")
        print(f"  Min:   {result['min']:.3f} ms")
        print(f"  P50:   {result['p50']:.3f} ms")
        print(f"  P95:   {result['p95']:.3f} ms")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print(f"\n{'Operation':<40} {'Mean':>10} {'Min':>10} {'P50':>10} {'P95':>10}")
    print("-" * 80)
    for r in results:
        print(
            f"{r['name']:<40} {r['mean']:>8.2f}ms {r['min']:>8.2f}ms "
            f"{r['p50']:>8.2f}ms {r['p95']:>8.2f}ms"
        )

    # Calculate throughput
    print(f"\n{'Operation':<40} {'Throughput':>15}")
    print("-" * 80)
    for r in results:
        throughput = n / (r["mean"] / 1000) / 1e6
        print(f"{r['name']:<40} {throughput:>10.1f} M/s")

    print("\n" + "=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)

    # Compare fused vs unfused
    opacity_only = next(r for r in results if "Opacity filter only" in r["name"])
    scale_only = next(r for r in results if "Scale filter only" in r["name"])
    fused = next(r for r in results if "Opacity + Scale (fused" in r["name"])

    unfused_time = opacity_only["mean"] + scale_only["mean"]
    speedup = unfused_time / fused["mean"]

    print("\nFused kernel performance:")
    print(f"  Opacity only:       {opacity_only['mean']:.2f}ms")
    print(f"  Scale only:         {scale_only['mean']:.2f}ms")
    print(f"  Unfused total:      {unfused_time:.2f}ms")
    print(f"  Fused (actual):     {fused['mean']:.2f}ms")
    print(f"  Speedup:            {speedup:.2f}x ({(speedup - 1) * 100:.1f}% faster)")

    # Full pipeline comparison
    full_pipeline = next(r for r in results if "Full filter_gaussians" in r["name"])
    print("\nFull pipeline performance:")
    print(f"  Mean:               {full_pipeline['mean']:.2f}ms")
    print(f"  Throughput:         {n / (full_pipeline['mean'] / 1000) / 1e6:.1f} M/s")
    print(f"  Min time:           {full_pipeline['min']:.2f}ms")
    print(f"  Best throughput:    {n / (full_pipeline['min'] / 1000) / 1e6:.1f} M/s")


if __name__ == "__main__":
    main()
