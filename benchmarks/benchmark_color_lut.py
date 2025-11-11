"""
Benchmark ColorLUT: NumPy vs PyTorch vs torch.compile on CPU and GPU
"""

import time
from typing import Callable

import numpy as np
import torch

from gslut import ColorLUT


def benchmark_function(func: Callable, warmup: int = 10, iterations: int = 100) -> float:
    """Benchmark a function and return average time in milliseconds."""
    # Warmup
    for _ in range(warmup):
        func()

    # Benchmark
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        func()
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms

    return np.mean(times), np.std(times)


def benchmark_color_lut():
    """Benchmark ColorLUT on different backends."""
    print("=" * 80)
    print("ColorLUT Benchmark: NumPy vs PyTorch vs torch.compile")
    print("=" * 80)

    batch_sizes = [1000, 10000, 100000, 1000000]
    devices = ["cpu"]

    # Check if CUDA is available
    if torch.cuda.is_available():
        devices.append("cuda")

    results = {}

    for batch_size in batch_sizes:
        print(f"\n--- Batch Size: {batch_size:,} points ---")

        for device in devices:
            print(f"\nDevice: {device.upper()}")

            # Generate test data
            colors = torch.rand(batch_size, 3, device=device)

            # Test 1: Regular ColorLUT
            lut = ColorLUT(device=device, lut_size=1024)

            def test_regular():
                result = lut.apply(
                    colors,
                    temperature=0.7,
                    brightness=1.2,
                    contrast=1.1,
                    gamma=0.9,
                    saturation=1.3,
                    shadows=1.1,
                    highlights=0.9,
                )
                if device == "cuda":
                    torch.cuda.synchronize()
                return result

            mean_time, std_time = benchmark_function(test_regular)
            throughput = (batch_size / mean_time) * 1000  # points per second
            results[f"{device}_regular_{batch_size}"] = mean_time

            print(f"  Regular:        {mean_time:8.3f} ± {std_time:6.3f} ms")
            print(f"                  {throughput:12,.0f} points/sec")

            # Test 2: torch.compile (PyTorch 2.0+)
            if hasattr(torch, "compile"):
                try:
                    # Create a wrapper function to compile
                    def apply_compiled(colors):
                        return lut.apply(
                            colors,
                            temperature=0.7,
                            brightness=1.2,
                            contrast=1.1,
                            gamma=0.9,
                            saturation=1.3,
                            shadows=1.1,
                            highlights=0.9,
                        )

                    compiled_apply = torch.compile(apply_compiled, mode="reduce-overhead")

                    def test_compiled():
                        result = compiled_apply(colors)
                        if device == "cuda":
                            torch.cuda.synchronize()
                        return result

                    # Extra warmup for compiled version
                    for _ in range(20):
                        test_compiled()

                    mean_time_compiled, std_time_compiled = benchmark_function(
                        test_compiled, warmup=5, iterations=100
                    )
                    throughput_compiled = (batch_size / mean_time_compiled) * 1000
                    speedup = mean_time / mean_time_compiled
                    results[f"{device}_compiled_{batch_size}"] = mean_time_compiled

                    print(f"  torch.compile:  {mean_time_compiled:8.3f} ± {std_time_compiled:6.3f} ms")
                    print(f"                  {throughput_compiled:12,.0f} points/sec")
                    print(f"                  {speedup:8.2f}x speedup")

                except Exception as e:
                    print(f"  torch.compile:  FAILED ({e})")

    # Summary comparison
    print("\n" + "=" * 80)
    print("Summary: Performance Comparison")
    print("=" * 80)

    if "cpu" in devices and "cuda" in devices:
        print("\nCPU vs GPU Speedup:")
        for batch_size in batch_sizes:
            cpu_time = results.get(f"cpu_regular_{batch_size}")
            gpu_time = results.get(f"cuda_regular_{batch_size}")
            if cpu_time and gpu_time:
                speedup = cpu_time / gpu_time
                print(f"  {batch_size:>8,} points: {speedup:6.2f}x faster on GPU")

    if hasattr(torch, "compile"):
        print("\ntorch.compile Speedup:")
        for device in devices:
            print(f"\n  {device.upper()}:")
            for batch_size in batch_sizes:
                regular_time = results.get(f"{device}_regular_{batch_size}")
                compiled_time = results.get(f"{device}_compiled_{batch_size}")
                if regular_time and compiled_time:
                    speedup = regular_time / compiled_time
                    print(f"    {batch_size:>8,} points: {speedup:6.2f}x speedup")


def benchmark_individual_operations():
    """Benchmark individual color operations."""
    print("\n" + "=" * 80)
    print("Individual Operation Benchmark")
    print("=" * 80)

    batch_size = 100000
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")

    operations = {
        "Temperature only": {"temperature": 0.7},
        "Brightness only": {"brightness": 1.5},
        "Contrast only": {"contrast": 1.5},
        "Gamma only": {"gamma": 0.8},
        "Saturation only": {"saturation": 1.5},
        "Shadows only": {"shadows": 1.3},
        "Highlights only": {"highlights": 0.7},
        "All operations": {
            "temperature": 0.7,
            "brightness": 1.2,
            "contrast": 1.1,
            "gamma": 0.9,
            "saturation": 1.3,
            "shadows": 1.1,
            "highlights": 0.9,
        },
    }

    for device in devices:
        print(f"\n{device.upper()}:")
        colors = torch.rand(batch_size, 3, device=device)
        lut = ColorLUT(device=device)

        for op_name, params in operations.items():
            def test_op():
                result = lut.apply(colors, **params)
                if device == "cuda":
                    torch.cuda.synchronize()
                return result

            mean_time, std_time = benchmark_function(test_op)
            throughput = (batch_size / mean_time) * 1000

            print(f"  {op_name:20s}: {mean_time:8.3f} ± {std_time:6.3f} ms")
            print(f"  {'':20s}  {throughput:12,.0f} points/sec")


def benchmark_lut_sizes():
    """Benchmark different LUT resolutions."""
    print("\n" + "=" * 80)
    print("LUT Resolution Benchmark")
    print("=" * 80)

    batch_size = 100000
    lut_sizes = [256, 512, 1024, 2048, 4096]
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")

    for device in devices:
        print(f"\n{device.upper()}:")
        colors = torch.rand(batch_size, 3, device=device)

        for lut_size in lut_sizes:
            lut = ColorLUT(device=device, lut_size=lut_size)

            def test_lut():
                result = lut.apply(colors, brightness=1.2, contrast=1.1)
                if device == "cuda":
                    torch.cuda.synchronize()
                return result

            mean_time, std_time = benchmark_function(test_lut)
            throughput = (batch_size / mean_time) * 1000
            memory_kb = (lut_size * 3 * 4) / 1024  # 3 channels, 4 bytes per float

            print(f"  LUT {lut_size:4d}: {mean_time:8.3f} ± {std_time:6.3f} ms")
            print(f"  {'':11s}  {throughput:12,.0f} points/sec ({memory_kb:.1f} KB)")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("gslut ColorLUT Performance Benchmark")
    print("=" * 80)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"torch.compile available: {hasattr(torch, 'compile')}")

    benchmark_color_lut()
    benchmark_individual_operations()
    benchmark_lut_sizes()

    print("\n" + "=" * 80)
    print("Benchmark Complete!")
    print("=" * 80)
