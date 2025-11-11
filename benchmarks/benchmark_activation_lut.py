"""
Benchmark ActivationLUT: LUT vs Native vs torch.compile on CPU and GPU
"""

import time
from typing import Callable

import numpy as np
import torch

from gslut import ActivationLUT


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


def benchmark_activation_lut():
    """Benchmark ActivationLUT vs native operations."""
    print("=" * 80)
    print("ActivationLUT Benchmark: LUT vs Native vs torch.compile")
    print("=" * 80)

    batch_sizes = [10000, 100000, 1000000, 10000000]
    devices = ["cpu"]

    if torch.cuda.is_available():
        devices.append("cuda")

    results = {}

    for batch_size in batch_sizes:
        print(f"\n--- Batch Size: {batch_size:,} values ---")

        for device in devices:
            print(f"\nDevice: {device.upper()}")

            # Generate test data
            scales_raw = torch.randn(batch_size, device=device) * 2
            opacities_raw = torch.randn(batch_size, device=device) * 2

            # === EXP BENCHMARK ===
            print("\n  exp() operation:")

            # Native PyTorch
            def test_native_exp():
                result = torch.exp(scales_raw)
                if device == "cuda":
                    torch.cuda.synchronize()
                return result

            mean_time, std_time = benchmark_function(test_native_exp)
            throughput = (batch_size / mean_time) * 1000
            results[f"{device}_native_exp_{batch_size}"] = mean_time

            print(f"    Native:         {mean_time:8.3f} ± {std_time:6.3f} ms")
            print(f"                    {throughput:12,.0f} values/sec")

            # LUT with linear interpolation
            lut = ActivationLUT(device=device, num_clusters_exp=2048, use_linear_interp=True)
            samples = torch.linspace(-6, 6, 10000, device=device)
            lut.build_from_samples(scale_samples=samples)

            def test_lut_exp():
                result = lut.exp(scales_raw)
                if device == "cuda":
                    torch.cuda.synchronize()
                return result

            mean_time_lut, std_time_lut = benchmark_function(test_lut_exp)
            throughput_lut = (batch_size / mean_time_lut) * 1000
            speedup = mean_time / mean_time_lut
            results[f"{device}_lut_exp_{batch_size}"] = mean_time_lut

            print(f"    LUT (interp):   {mean_time_lut:8.3f} ± {std_time_lut:6.3f} ms")
            print(f"                    {throughput_lut:12,.0f} values/sec")
            print(f"                    {speedup:8.2f}x vs native")

            # LUT without interpolation
            lut_nn = ActivationLUT(device=device, num_clusters_exp=2048, use_linear_interp=False)
            lut_nn.build_from_samples(scale_samples=samples)

            def test_lut_nn_exp():
                result = lut_nn.exp(scales_raw)
                if device == "cuda":
                    torch.cuda.synchronize()
                return result

            mean_time_nn, std_time_nn = benchmark_function(test_lut_nn_exp)
            throughput_nn = (batch_size / mean_time_nn) * 1000
            speedup_nn = mean_time / mean_time_nn

            print(f"    LUT (nearest):  {mean_time_nn:8.3f} ± {std_time_nn:6.3f} ms")
            print(f"                    {throughput_nn:12,.0f} values/sec")
            print(f"                    {speedup_nn:8.2f}x vs native")

            # torch.compile (if available)
            if hasattr(torch, "compile"):
                try:
                    exp_compiled = torch.compile(torch.exp, mode="reduce-overhead")

                    def test_compiled_exp():
                        result = exp_compiled(scales_raw)
                        if device == "cuda":
                            torch.cuda.synchronize()
                        return result

                    # Extra warmup
                    for _ in range(20):
                        test_compiled_exp()

                    mean_time_compiled, std_time_compiled = benchmark_function(
                        test_compiled_exp, warmup=5, iterations=100
                    )
                    throughput_compiled = (batch_size / mean_time_compiled) * 1000
                    speedup_compiled = mean_time / mean_time_compiled

                    print(f"    torch.compile:  {mean_time_compiled:8.3f} ± {std_time_compiled:6.3f} ms")
                    print(f"                    {throughput_compiled:12,.0f} values/sec")
                    print(f"                    {speedup_compiled:8.2f}x vs native")

                except Exception as e:
                    print(f"    torch.compile:  FAILED ({e})")

            # === SIGMOID BENCHMARK ===
            print("\n  sigmoid() operation:")

            # Native PyTorch
            def test_native_sigmoid():
                result = torch.sigmoid(opacities_raw)
                if device == "cuda":
                    torch.cuda.synchronize()
                return result

            mean_time, std_time = benchmark_function(test_native_sigmoid)
            throughput = (batch_size / mean_time) * 1000

            print(f"    Native:         {mean_time:8.3f} ± {std_time:6.3f} ms")
            print(f"                    {throughput:12,.0f} values/sec")

            # LUT
            def test_lut_sigmoid():
                result = lut.sigmoid(opacities_raw)
                if device == "cuda":
                    torch.cuda.synchronize()
                return result

            mean_time_lut, std_time_lut = benchmark_function(test_lut_sigmoid)
            throughput_lut = (batch_size / mean_time_lut) * 1000
            speedup = mean_time / mean_time_lut

            print(f"    LUT (interp):   {mean_time_lut:8.3f} ± {std_time_lut:6.3f} ms")
            print(f"                    {throughput_lut:12,.0f} values/sec")
            print(f"                    {speedup:8.2f}x vs native")


def benchmark_lut_cluster_sizes():
    """Benchmark different cluster counts."""
    print("\n" + "=" * 80)
    print("LUT Cluster Size Benchmark")
    print("=" * 80)

    batch_size = 1000000
    cluster_counts = [128, 256, 512, 1024, 2048, 4096]
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")

    for device in devices:
        print(f"\n{device.upper()}:")
        scales_raw = torch.randn(batch_size, device=device) * 2
        samples = torch.linspace(-6, 6, 10000, device=device)

        # Baseline: native exp
        def test_native():
            result = torch.exp(scales_raw)
            if device == "cuda":
                torch.cuda.synchronize()
            return result

        native_time, _ = benchmark_function(test_native)
        print(f"  Native exp:      {native_time:8.3f} ms (baseline)\n")

        for num_clusters in cluster_counts:
            lut = ActivationLUT(device=device, num_clusters_exp=num_clusters, use_linear_interp=True)
            lut.build_from_samples(scale_samples=samples)

            def test_lut():
                result = lut.exp(scales_raw)
                if device == "cuda":
                    torch.cuda.synchronize()
                return result

            mean_time, std_time = benchmark_function(test_lut)
            speedup = native_time / mean_time
            memory_kb = (num_clusters * 2 * 4) / 1024  # 2 arrays, 4 bytes per float

            # Calculate accuracy
            with torch.no_grad():
                lut_result = lut.exp(scales_raw[:1000])
                true_result = torch.exp(scales_raw[:1000])
                rel_error = torch.abs(lut_result - true_result) / (torch.abs(true_result) + 1e-6)
                mean_error = rel_error.mean().item() * 100

            print(f"  {num_clusters:4d} clusters: {mean_time:8.3f} ± {std_time:6.3f} ms")
            print(f"  {'':15s}  {speedup:6.2f}x vs native, {mean_error:.4f}% error, {memory_kb:.1f} KB")


def benchmark_accuracy_vs_performance():
    """Benchmark accuracy vs performance tradeoff."""
    print("\n" + "=" * 80)
    print("Accuracy vs Performance Tradeoff")
    print("=" * 80)

    batch_size = 100000
    device = "cuda" if torch.cuda.is_available() else "cpu"

    scales_raw = torch.randn(batch_size, device=device) * 2
    samples = torch.linspace(-6, 6, 10000, device=device)

    # Native baseline
    def test_native():
        result = torch.exp(scales_raw)
        if device == "cuda":
            torch.cuda.synchronize()
        return result

    native_time, _ = benchmark_function(test_native)
    native_result = torch.exp(scales_raw)

    print(f"\nDevice: {device.upper()}")
    print(f"Native exp: {native_time:.3f} ms (baseline, 0% error)\n")

    configs = [
        (256, True, "256 clusters + interp"),
        (256, False, "256 clusters + nearest"),
        (512, True, "512 clusters + interp"),
        (1024, True, "1024 clusters + interp"),
        (2048, True, "2048 clusters + interp"),
        (4096, True, "4096 clusters + interp"),
    ]

    for num_clusters, use_interp, label in configs:
        lut = ActivationLUT(
            device=device, num_clusters_exp=num_clusters, use_linear_interp=use_interp
        )
        lut.build_from_samples(scale_samples=samples)

        def test_lut():
            result = lut.exp(scales_raw)
            if device == "cuda":
                torch.cuda.synchronize()
            return result

        mean_time, std_time = benchmark_function(test_lut)
        speedup = native_time / mean_time

        # Calculate accuracy
        with torch.no_grad():
            lut_result = lut.exp(scales_raw)
            rel_error = torch.abs(lut_result - native_result) / (torch.abs(native_result) + 1e-6)
            mean_error = rel_error.mean().item() * 100
            max_error = rel_error.max().item() * 100

        print(f"{label:25s}: {mean_time:7.3f} ms, {speedup:5.2f}x speedup")
        print(f"{'':25s}  Mean error: {mean_error:.4f}%, Max error: {max_error:.4f}%")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("gslut ActivationLUT Performance Benchmark")
    print("=" * 80)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"torch.compile available: {hasattr(torch, 'compile')}")

    benchmark_activation_lut()
    benchmark_lut_cluster_sizes()
    benchmark_accuracy_vs_performance()

    print("\n" + "=" * 80)
    print("Benchmark Complete!")
    print("=" * 80)
