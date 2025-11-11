"""
Benchmark SH/RGB Conversions: CPU vs GPU vs torch.compile
"""

import time
from typing import Callable

import numpy as np
import torch

from gslut import rgb2sh, sh2rgb


def benchmark_function(func: Callable, warmup: int = 10, iterations: int = 100) -> tuple[float, float]:
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


def benchmark_conversions():
    """Benchmark sh2rgb and rgb2sh conversions."""
    print("=" * 80)
    print("SH/RGB Conversion Benchmark")
    print("=" * 80)

    batch_sizes = [10000, 100000, 1000000, 10000000]
    devices = ["cpu"]

    if torch.cuda.is_available():
        devices.append("cuda")

    for batch_size in batch_sizes:
        print(f"\n--- Batch Size: {batch_size:,} colors ---")

        for device in devices:
            print(f"\nDevice: {device.upper()}")

            # Generate test data
            rgb_colors = torch.rand(batch_size, 3, device=device)
            sh_coeffs = torch.randn(batch_size, 3, device=device)

            # === RGB2SH ===
            print("\n  rgb2sh():")

            def test_rgb2sh():
                result = rgb2sh(rgb_colors)
                if device == "cuda":
                    torch.cuda.synchronize()
                return result

            mean_time, std_time = benchmark_function(test_rgb2sh)
            throughput = (batch_size / mean_time) * 1000

            print(f"    Regular:        {mean_time:8.3f} ± {std_time:6.3f} ms")
            print(f"                    {throughput:12,.0f} conversions/sec")

            # torch.compile
            if hasattr(torch, "compile"):
                try:
                    rgb2sh_compiled = torch.compile(rgb2sh, mode="reduce-overhead")

                    def test_rgb2sh_compiled():
                        result = rgb2sh_compiled(rgb_colors)
                        if device == "cuda":
                            torch.cuda.synchronize()
                        return result

                    # Extra warmup
                    for _ in range(20):
                        test_rgb2sh_compiled()

                    mean_time_compiled, std_time_compiled = benchmark_function(
                        test_rgb2sh_compiled, warmup=5, iterations=100
                    )
                    throughput_compiled = (batch_size / mean_time_compiled) * 1000
                    speedup = mean_time / mean_time_compiled

                    print(f"    torch.compile:  {mean_time_compiled:8.3f} ± {std_time_compiled:6.3f} ms")
                    print(f"                    {throughput_compiled:12,.0f} conversions/sec")
                    print(f"                    {speedup:8.2f}x speedup")

                except Exception as e:
                    print(f"    torch.compile:  FAILED ({e})")

            # === SH2RGB ===
            print("\n  sh2rgb():")

            def test_sh2rgb():
                result = sh2rgb(sh_coeffs)
                if device == "cuda":
                    torch.cuda.synchronize()
                return result

            mean_time, std_time = benchmark_function(test_sh2rgb)
            throughput = (batch_size / mean_time) * 1000

            print(f"    Regular:        {mean_time:8.3f} ± {std_time:6.3f} ms")
            print(f"                    {throughput:12,.0f} conversions/sec")

            # torch.compile
            if hasattr(torch, "compile"):
                try:
                    sh2rgb_compiled = torch.compile(sh2rgb, mode="reduce-overhead")

                    def test_sh2rgb_compiled():
                        result = sh2rgb_compiled(sh_coeffs)
                        if device == "cuda":
                            torch.cuda.synchronize()
                        return result

                    # Extra warmup
                    for _ in range(20):
                        test_sh2rgb_compiled()

                    mean_time_compiled, std_time_compiled = benchmark_function(
                        test_sh2rgb_compiled, warmup=5, iterations=100
                    )
                    throughput_compiled = (batch_size / mean_time_compiled) * 1000
                    speedup = mean_time / mean_time_compiled

                    print(f"    torch.compile:  {mean_time_compiled:8.3f} ± {std_time_compiled:6.3f} ms")
                    print(f"                    {throughput_compiled:12,.0f} conversions/sec")
                    print(f"                    {speedup:8.2f}x speedup")

                except Exception as e:
                    print(f"    torch.compile:  FAILED ({e})")

            # === ROUNDTRIP ===
            print("\n  Roundtrip (rgb2sh + sh2rgb):")

            def test_roundtrip():
                sh = rgb2sh(rgb_colors)
                rgb = sh2rgb(sh)
                if device == "cuda":
                    torch.cuda.synchronize()
                return rgb

            mean_time, std_time = benchmark_function(test_roundtrip)
            throughput = (batch_size / mean_time) * 1000

            print(f"    Regular:        {mean_time:8.3f} ± {std_time:6.3f} ms")
            print(f"                    {throughput:12,.0f} conversions/sec")


def benchmark_tensor_shapes():
    """Benchmark different tensor shapes."""
    print("\n" + "=" * 80)
    print("Tensor Shape Benchmark")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device.upper()}")

    shapes = [
        ((1000000, 3), "1D batch"),
        ((1000, 1000, 3), "2D image-like"),
        ((100, 100, 100, 3), "3D volumetric"),
        ((10, 10, 10, 10, 10, 3), "5D nested"),
    ]

    for shape, description in shapes:
        total_points = np.prod(shape[:-1])
        rgb_colors = torch.rand(shape, device=device)

        def test_conversion():
            sh = rgb2sh(rgb_colors)
            rgb = sh2rgb(sh)
            if device == "cuda":
                torch.cuda.synchronize()
            return rgb

        mean_time, std_time = benchmark_function(test_conversion)
        throughput = (total_points / mean_time) * 1000

        print(f"\n  {description:20s} {str(shape):30s}")
        print(f"    {mean_time:8.3f} ± {std_time:6.3f} ms")
        print(f"    {throughput:12,.0f} conversions/sec")


def benchmark_dtype_comparison():
    """Benchmark different data types."""
    print("\n" + "=" * 80)
    print("Data Type Benchmark")
    print("=" * 80)

    batch_size = 1000000
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device.upper()}")

    dtypes = [
        (torch.float32, "float32"),
        (torch.float64, "float64"),
        (torch.float16, "float16 (half)") if device == "cuda" else None,
    ]

    for dtype_info in dtypes:
        if dtype_info is None:
            continue

        dtype, name = dtype_info
        print(f"\n{name}:")

        try:
            rgb_colors = torch.rand(batch_size, 3, device=device, dtype=dtype)

            def test_conversion():
                sh = rgb2sh(rgb_colors)
                rgb = sh2rgb(sh)
                if device == "cuda":
                    torch.cuda.synchronize()
                return rgb

            mean_time, std_time = benchmark_function(test_conversion)
            throughput = (batch_size / mean_time) * 1000

            print(f"  Roundtrip:      {mean_time:8.3f} ± {std_time:6.3f} ms")
            print(f"                  {throughput:12,.0f} conversions/sec")

        except Exception as e:
            print(f"  FAILED: {e}")


def benchmark_memory_transfer():
    """Benchmark CPU <-> GPU memory transfer overhead."""
    if not torch.cuda.is_available():
        print("\nSkipping memory transfer benchmark (CUDA not available)")
        return

    print("\n" + "=" * 80)
    print("CPU <-> GPU Memory Transfer Benchmark")
    print("=" * 80)

    batch_sizes = [10000, 100000, 1000000, 10000000]

    for batch_size in batch_sizes:
        print(f"\nBatch Size: {batch_size:,} colors")

        # CPU -> GPU
        rgb_cpu = torch.rand(batch_size, 3, device="cpu")

        def test_cpu_to_gpu():
            rgb_gpu = rgb_cpu.to("cuda")
            torch.cuda.synchronize()
            return rgb_gpu

        transfer_time, _ = benchmark_function(test_cpu_to_gpu)
        bandwidth = (batch_size * 3 * 4 / transfer_time) * 1000 / (1024**3)  # GB/s

        print(f"  CPU -> GPU:     {transfer_time:8.3f} ms ({bandwidth:.2f} GB/s)")

        # Conversion on GPU
        rgb_gpu = rgb_cpu.to("cuda")

        def test_conversion_gpu():
            sh = rgb2sh(rgb_gpu)
            rgb = sh2rgb(sh)
            torch.cuda.synchronize()
            return rgb

        conversion_time, _ = benchmark_function(test_conversion_gpu)

        print(f"  GPU conversion: {conversion_time:8.3f} ms")

        # GPU -> CPU
        def test_gpu_to_cpu():
            result_cpu = rgb_gpu.to("cpu")
            torch.cuda.synchronize()
            return result_cpu

        transfer_back_time, _ = benchmark_function(test_gpu_to_cpu)
        bandwidth_back = (batch_size * 3 * 4 / transfer_back_time) * 1000 / (1024**3)

        print(f"  GPU -> CPU:     {transfer_back_time:8.3f} ms ({bandwidth_back:.2f} GB/s)")

        total_time = transfer_time + conversion_time + transfer_back_time
        cpu_only_time = benchmark_function(
            lambda: sh2rgb(rgb2sh(rgb_cpu)), warmup=10, iterations=100
        )[0]

        print(f"  Total (GPU):    {total_time:8.3f} ms (with transfers)")
        print(f"  CPU only:       {cpu_only_time:8.3f} ms")
        print(f"  Conclusion:     {'GPU faster' if total_time < cpu_only_time else 'CPU faster'}")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("gslut Conversion Performance Benchmark")
    print("=" * 80)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"torch.compile available: {hasattr(torch, 'compile')}")

    benchmark_conversions()
    benchmark_tensor_shapes()
    benchmark_dtype_comparison()
    benchmark_memory_transfer()

    print("\n" + "=" * 80)
    print("Benchmark Complete!")
    print("=" * 80)
