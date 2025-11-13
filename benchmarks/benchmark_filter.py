"""
Benchmark filtering performance.

Tests filtering operations at various scales.
"""

import logging
import time

import numpy as np

from gspro.filter import (
    apply_filter,
    calculate_recommended_max_scale,
    calculate_scene_bounds,
    filter_gaussians,
)

# Suppress logging for cleaner output
logging.getLogger("gspro.filter.api").setLevel(logging.WARNING)


def generate_gaussian_data(n: int):
    """Generate sample Gaussian data."""
    np.random.seed(42)

    positions = np.random.randn(n, 3).astype(np.float32) * 2.0
    quaternions = np.random.randn(n, 4).astype(np.float32)
    quaternions = quaternions / np.linalg.norm(quaternions, axis=1, keepdims=True)
    scales = np.random.rand(n, 3).astype(np.float32) * 2.0
    opacities = np.random.rand(n).astype(np.float32)
    colors = np.random.rand(n, 3).astype(np.float32)

    # Add some outliers
    outlier_idx = np.random.choice(n, size=int(n * 0.01), replace=False)
    scales[outlier_idx] = 10.0

    return {
        "positions": positions,
        "quaternions": quaternions,
        "scales": scales,
        "opacities": opacities,
        "colors": colors,
    }


def benchmark_scene_bounds(n: int = 1_000_000, iterations: int = 100):
    """Benchmark scene bounds calculation."""
    print("\n" + "=" * 80)
    print(f"SCENE BOUNDS CALCULATION ({n:,} Gaussians, {iterations} iterations)")
    print("=" * 80)

    data = generate_gaussian_data(n)
    positions = data["positions"]

    # Warmup
    for _ in range(10):
        calculate_scene_bounds(positions)

    # Benchmark
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        bounds = calculate_scene_bounds(positions)
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms

    avg_time = np.mean(times)
    std_time = np.std(times)

    print(f"Time:       {avg_time:.3f} ms +/- {std_time:.3f} ms")
    print(f"Throughput: {n / (avg_time / 1000) / 1e6:.1f}M Gaussians/sec")


def benchmark_recommended_scale(n: int = 1_000_000, iterations: int = 100):
    """Benchmark recommended max scale calculation."""
    print("\n" + "=" * 80)
    print(f"RECOMMENDED MAX SCALE ({n:,} Gaussians, {iterations} iterations)")
    print("=" * 80)

    data = generate_gaussian_data(n)
    scales = data["scales"]

    # Warmup
    for _ in range(10):
        calculate_recommended_max_scale(scales)

    # Benchmark
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        threshold = calculate_recommended_max_scale(scales)
        end = time.perf_counter()
        times.append((end - start) * 1000)

    avg_time = np.mean(times)
    std_time = np.std(times)

    print(f"Time:       {avg_time:.3f} ms +/- {std_time:.3f} ms")
    print(f"Throughput: {n / (avg_time / 1000) / 1e6:.1f}M Gaussians/sec")


def benchmark_opacity_filter(n: int = 1_000_000, iterations: int = 100):
    """Benchmark opacity filtering."""
    print("\n" + "=" * 80)
    print(f"OPACITY FILTER ({n:,} Gaussians, {iterations} iterations)")
    print("=" * 80)

    data = generate_gaussian_data(n)
    positions = data["positions"]
    opacities = data["opacities"]

    # Warmup
    for _ in range(10):
        apply_filter(positions, opacities=opacities, opacity_threshold=0.1)

    # Benchmark
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        mask = apply_filter(positions, opacities=opacities, opacity_threshold=0.1)
        end = time.perf_counter()
        times.append((end - start) * 1000)

    avg_time = np.mean(times)
    std_time = np.std(times)
    kept_pct = mask.sum() / len(mask) * 100

    print(f"Time:       {avg_time:.3f} ms +/- {std_time:.3f} ms")
    print(f"Throughput: {n / (avg_time / 1000) / 1e6:.1f}M Gaussians/sec")
    print(f"Kept:       {kept_pct:.1f}% of Gaussians")


def benchmark_scale_filter(n: int = 1_000_000, iterations: int = 100):
    """Benchmark scale filtering."""
    print("\n" + "=" * 80)
    print(f"SCALE FILTER ({n:,} Gaussians, {iterations} iterations)")
    print("=" * 80)

    data = generate_gaussian_data(n)
    positions = data["positions"]
    scales = data["scales"]

    # Warmup
    for _ in range(10):
        apply_filter(positions, scales=scales, max_scale=2.0)

    # Benchmark
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        mask = apply_filter(positions, scales=scales, max_scale=2.0)
        end = time.perf_counter()
        times.append((end - start) * 1000)

    avg_time = np.mean(times)
    std_time = np.std(times)
    kept_pct = mask.sum() / len(mask) * 100

    print(f"Time:       {avg_time:.3f} ms +/- {std_time:.3f} ms")
    print(f"Throughput: {n / (avg_time / 1000) / 1e6:.1f}M Gaussians/sec")
    print(f"Kept:       {kept_pct:.1f}% of Gaussians")


def benchmark_sphere_filter(n: int = 1_000_000, iterations: int = 100):
    """Benchmark sphere volume filtering."""
    print("\n" + "=" * 80)
    print(f"SPHERE FILTER ({n:,} Gaussians, {iterations} iterations)")
    print("=" * 80)

    data = generate_gaussian_data(n)
    positions = data["positions"]

    # Pre-calculate bounds (one-time cost)
    bounds = calculate_scene_bounds(positions)

    # Warmup
    for _ in range(10):
        apply_filter(
            positions,
            filter_type="sphere",
            sphere_radius_factor=0.5,
            scene_bounds=bounds,
        )

    # Benchmark
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        mask = apply_filter(
            positions,
            filter_type="sphere",
            sphere_radius_factor=0.5,
            scene_bounds=bounds,
        )
        end = time.perf_counter()
        times.append((end - start) * 1000)

    avg_time = np.mean(times)
    std_time = np.std(times)
    kept_pct = mask.sum() / len(mask) * 100

    print(f"Time:       {avg_time:.3f} ms +/- {std_time:.3f} ms")
    print(f"Throughput: {n / (avg_time / 1000) / 1e6:.1f}M Gaussians/sec")
    print(f"Kept:       {kept_pct:.1f}% of Gaussians")


def benchmark_cuboid_filter(n: int = 1_000_000, iterations: int = 100):
    """Benchmark cuboid volume filtering."""
    print("\n" + "=" * 80)
    print(f"CUBOID FILTER ({n:,} Gaussians, {iterations} iterations)")
    print("=" * 80)

    data = generate_gaussian_data(n)
    positions = data["positions"]

    # Pre-calculate bounds (one-time cost)
    bounds = calculate_scene_bounds(positions)

    # Warmup
    for _ in range(10):
        apply_filter(
            positions,
            filter_type="cuboid",
            cuboid_size_factor_x=0.5,
            cuboid_size_factor_y=0.5,
            cuboid_size_factor_z=0.5,
            scene_bounds=bounds,
        )

    # Benchmark
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        mask = apply_filter(
            positions,
            filter_type="cuboid",
            cuboid_size_factor_x=0.5,
            cuboid_size_factor_y=0.5,
            cuboid_size_factor_z=0.5,
            scene_bounds=bounds,
        )
        end = time.perf_counter()
        times.append((end - start) * 1000)

    avg_time = np.mean(times)
    std_time = np.std(times)
    kept_pct = mask.sum() / len(mask) * 100

    print(f"Time:       {avg_time:.3f} ms +/- {std_time:.3f} ms")
    print(f"Throughput: {n / (avg_time / 1000) / 1e6:.1f}M Gaussians/sec")
    print(f"Kept:       {kept_pct:.1f}% of Gaussians")


def benchmark_combined_filter(n: int = 1_000_000, iterations: int = 100):
    """Benchmark combined filtering (all filters)."""
    print("\n" + "=" * 80)
    print(f"COMBINED FILTER ({n:,} Gaussians, {iterations} iterations)")
    print("=" * 80)

    data = generate_gaussian_data(n)
    positions = data["positions"]
    opacities = data["opacities"]
    scales = data["scales"]

    # Pre-calculate bounds (one-time cost)
    bounds = calculate_scene_bounds(positions)

    # Warmup
    for _ in range(10):
        apply_filter(
            positions,
            opacities=opacities,
            scales=scales,
            filter_type="sphere",
            sphere_radius_factor=0.8,
            opacity_threshold=0.05,
            max_scale=2.0,
            scene_bounds=bounds,
        )

    # Benchmark
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        mask = apply_filter(
            positions,
            opacities=opacities,
            scales=scales,
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
    kept_pct = mask.sum() / len(mask) * 100

    print(f"Time:       {avg_time:.3f} ms +/- {std_time:.3f} ms")
    print(f"Throughput: {n / (avg_time / 1000) / 1e6:.1f}M Gaussians/sec")
    print(f"Kept:       {kept_pct:.1f}% of Gaussians")


def benchmark_filter_gaussians(n: int = 1_000_000, iterations: int = 100):
    """Benchmark filter_gaussians convenience function."""
    print("\n" + "=" * 80)
    print(f"FILTER_GAUSSIANS ({n:,} Gaussians, {iterations} iterations)")
    print("=" * 80)

    data = generate_gaussian_data(n)

    # Pre-calculate bounds (one-time cost)
    bounds = calculate_scene_bounds(data["positions"])

    # Warmup
    for _ in range(10):
        filter_gaussians(
            data["positions"],
            data["quaternions"],
            data["scales"],
            data["opacities"],
            data["colors"],
            filter_type="sphere",
            sphere_radius_factor=0.8,
            opacity_threshold=0.05,
            max_scale=2.0,
            scene_bounds=bounds,
        )

    # Benchmark
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        new_pos, new_quats, new_scales, new_opac, new_colors = filter_gaussians(
            data["positions"],
            data["quaternions"],
            data["scales"],
            data["opacities"],
            data["colors"],
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
    kept_pct = len(new_pos) / n * 100

    print(f"Time:       {avg_time:.3f} ms +/- {std_time:.3f} ms")
    print(f"Throughput: {n / (avg_time / 1000) / 1e6:.1f}M Gaussians/sec")
    print(f"Kept:       {kept_pct:.1f}% of Gaussians")


def benchmark_batch_scaling():
    """Benchmark different batch sizes."""
    print("\n" + "=" * 80)
    print("BATCH SIZE SCALING (Combined Filter)")
    print("=" * 80)

    batch_sizes = [10_000, 100_000, 500_000, 1_000_000, 2_000_000]

    for n in batch_sizes:
        data = generate_gaussian_data(n)
        bounds = calculate_scene_bounds(data["positions"])

        # Warmup
        for _ in range(5):
            apply_filter(
                data["positions"],
                opacities=data["opacities"],
                scales=data["scales"],
                filter_type="sphere",
                sphere_radius_factor=0.8,
                opacity_threshold=0.05,
                max_scale=2.0,
                scene_bounds=bounds,
            )

        # Benchmark
        times = []
        for _ in range(20):
            start = time.perf_counter()
            mask = apply_filter(
                data["positions"],
                opacities=data["opacities"],
                scales=data["scales"],
                filter_type="sphere",
                sphere_radius_factor=0.8,
                opacity_threshold=0.05,
                max_scale=2.0,
                scene_bounds=bounds,
            )
            end = time.perf_counter()
            times.append((end - start) * 1000)

        avg_time = np.mean(times)
        throughput = n / (avg_time / 1000) / 1e6

        print(f"N={n:>9,}: {avg_time:6.2f} ms ({throughput:6.1f}M G/s)")


def main():
    """Run all benchmarks."""
    print("=" * 80)
    print("GSPRO FILTERING PERFORMANCE BENCHMARKS")
    print("=" * 80)

    # One-time operations
    benchmark_scene_bounds(n=1_000_000, iterations=100)
    benchmark_recommended_scale(n=1_000_000, iterations=100)

    # Individual filters
    benchmark_opacity_filter(n=1_000_000, iterations=100)
    benchmark_scale_filter(n=1_000_000, iterations=100)
    benchmark_sphere_filter(n=1_000_000, iterations=100)
    benchmark_cuboid_filter(n=1_000_000, iterations=100)

    # Combined filtering
    benchmark_combined_filter(n=1_000_000, iterations=100)
    benchmark_filter_gaussians(n=1_000_000, iterations=100)

    # Scaling
    benchmark_batch_scaling()

    print("\n" + "=" * 80)
    print("ALL BENCHMARKS COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    main()
